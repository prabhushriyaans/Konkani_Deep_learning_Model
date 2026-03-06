import json
import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings(
    "ignore",
    message="The PyTorch API of nested tensors"
)

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running on: {device}")

# ============================================================
# CONFIG
# ============================================================

DATA_FILE = "text_konkani.json"

MAX_LEN = 32
EMBED_DIM = 64
NUM_HEADS = 4

BATCH_SIZE = 16
EPOCHS = 400
LR = 5e-4

TEMPERATURE = 1.5

# ============================================================
# TOKENIZATION (FOR NN MODELS)
# ============================================================

def tokenize(text):
    text = text.lower().strip()
    tokens = []
    for w in text.split():
        tokens.append(w)
        if len(w) > 3:
            tokens.append(w[:3])
            tokens.append(w[-3:])
    return tokens

def encode_text(text, vocab):
    ids = [vocab.get(t, 1) for t in tokenize(text)][:MAX_LEN]
    return ids + [0] * (MAX_LEN - len(ids))

# ============================================================
# DATA
# ============================================================

def load_data():
    if not os.path.exists(DATA_FILE):
        return {"negative": [], "neutral": [], "positive": []}
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_new_data(text, label):
    if not text:
        return
    data = load_data()
    if text not in data[label]:
        data[label].append(text)
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

# ============================================================
# TRANSFORMER MODEL
# ============================================================

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.pos_emb = nn.Parameter(torch.zeros(1, MAX_LEN, EMBED_DIM))

        enc = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=128,
            dropout=0.25,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=1)
        self.fc = nn.Linear(EMBED_DIM, 3)

    def forward(self, x):
        mask = (x == 0)
        x = self.embedding(x) + self.pos_emb
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = x.sum(dim=1) / (~mask).sum(dim=1, keepdim=True).clamp(min=1)
        return self.fc(x)

# ============================================================
# CNN MODEL
# ============================================================

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)

        self.conv3 = nn.Conv1d(EMBED_DIM, 64, 3, padding=1)
        self.conv5 = nn.Conv1d(EMBED_DIM, 64, 5, padding=2)

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 3)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        c3 = torch.relu(self.conv3(x))
        c5 = torch.relu(self.conv5(x))
        p3 = torch.max(c3, dim=2)[0]
        p5 = torch.max(c5, dim=2)[0]
        x = self.dropout(torch.cat([p3, p5], dim=1))
        return self.fc(x)

# ============================================================
# TRAINING ALL MODELS
# ============================================================

def train_models():
    global vocab, transformer, cnn, svm, tfidf

    data = load_data()
    texts, labels, tokens = [], [], []
    label_map = {"negative": 0, "neutral": 1, "positive": 2}

    for k, v in data.items():
        for t in v:
            texts.append(t)
            labels.append(label_map[k])
            tokens.extend(tokenize(t))

    if len(texts) < 10:
        print("Dataset too small.")
        return False

    # ================= VOCAB =================
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, tok in enumerate(sorted(set(tokens))):
        vocab[tok] = i + 2

    X = torch.tensor([encode_text(t, vocab) for t in texts])
    Y = torch.tensor(labels)

    loader = DataLoader(
        TensorDataset(X, Y),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # ================= MODELS =================
    transformer = TransformerClassifier(len(vocab)).to(device)
    cnn = CNNClassifier(len(vocab)).to(device)

    class_weights = torch.tensor([1.1, 1.0, 1.1]).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05
    )

    opt_t = optim.Adam(transformer.parameters(), lr=LR, weight_decay=1e-3)
    opt_c = optim.Adam(cnn.parameters(), lr=LR, weight_decay=1e-3)

    print(f"🔥 Training Transformer + CNN for {EPOCHS} epochs on {len(texts)} samples...")

    # ================= EPOCH LOOP =================
    for epoch in range(EPOCHS):
        transformer.train()
        cnn.train()

        total_loss_t = 0.0
        total_loss_c = 0.0

        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)

            # ---- Transformer ----
            opt_t.zero_grad()
            out_t = transformer(bx)
            loss_t = criterion(out_t, by)
            loss_t.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            opt_t.step()
            total_loss_t += loss_t.item()

            # ---- CNN ----
            opt_c.zero_grad()
            out_c = cnn(bx)
            loss_c = criterion(out_c, by)
            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(cnn.parameters(), 1.0)
            opt_c.step()
            total_loss_c += loss_c.item()

        # ---- LOGGING ----
        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"T-Loss: {total_loss_t/len(loader):.4f} | "
                f"C-Loss: {total_loss_c/len(loader):.4f}"
            )

    # ================= SVM (SEPARATE) =================
    print("🔥 Training SVM...")
    tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=2
    )

    X_tfidf = tfidf.fit_transform(texts)

    svm = CalibratedClassifierCV(
        LinearSVC(),
        method="sigmoid"
    )
    svm.fit(X_tfidf, labels)

    print("✅ Hybrid Brain Ready.\n")
    return True

# ============================================================
# INTERACTIVE LOOP
# ============================================================

if __name__ == "__main__":
    trained = train_models()
    print("--- KONKANI TRIPLE HYBRID AI (TRANSFORMER + CNN + SVM) ---")

    last_text = ""

    while True:
        text = input("\nYou: ").strip()
        if not text or text.lower() == "exit":
            break

        if text.lower().startswith("wrong "):
            label = text.split()[-1].lower()
            if label in ["negative", "neutral", "positive"]:
                save_new_data(last_text, label)
                print("📝 Corrected. Retraining...")
                trained = train_models()
            continue

        last_text = text
        if not trained:
            continue

        inp_nn = torch.tensor([encode_text(text, vocab)], device=device)

        with torch.no_grad():
            p_t = torch.softmax(transformer(inp_nn) / TEMPERATURE, dim=1)[0]
            p_c = torch.softmax(cnn(inp_nn) / TEMPERATURE, dim=1)[0]

        p_s = torch.tensor(
            svm.predict_proba(tfidf.transform([text]))[0]
        )

        confs = [
            ("Transformer", torch.max(p_t).item(), torch.argmax(p_t).item()),
            ("CNN", torch.max(p_c).item(), torch.argmax(p_c).item()),
            ("SVM", torch.max(p_s).item(), torch.argmax(p_s).item()),
        ]

        model, conf, idx = max(confs, key=lambda x: x[1])
        tags = ["Negative", "Neutral", "Positive"]

        print(f"AI: {tags[idx]} ({conf*100:.1f}% | {model})")