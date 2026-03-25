import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

# --- CONFIG ---
DATA_FILE = "text_konkani.json"
MAX_LEN, EMBED_DIM, BATCH_SIZE, EPOCHS, LR = 32, 64, 16, 400, 5e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- TOKENIZER ---
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

# --- MODEL ARCHITECTURES ---
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        layer = nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=4, dim_feedforward=128, dropout=0.3, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.fc = nn.Linear(EMBED_DIM, 3)
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.conv = nn.Conv1d(EMBED_DIM, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64, 3)
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = torch.max(x, dim=2)[0]
        return self.fc(x)

# --- TRAINING LOGIC ---
def train():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found!")
        return

    with open(DATA_FILE, encoding="utf-8") as f:
        data_json = json.load(f)

    texts, labels, vocab = [], [], {"<PAD>": 0, "<UNK>": 1}
    for idx, cat in enumerate(["negative", "neutral", "positive"]):
        for t in data_json[cat]:
            texts.append(t)
            labels.append(idx)
            for tok in tokenize(t):
                if tok not in vocab: vocab[tok] = len(vocab)

    # 1. Train SVM
    print("Training SVM...")
    tfidf = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
    x_tfidf = tfidf.fit_transform(texts)
    svm = CalibratedClassifierCV(LinearSVC(dual=False)).fit(x_tfidf, labels)
    with open("svm_hybrid.pkl", "wb") as f:
        pickle.dump((svm, tfidf), f)

    # 2. Train Neural Networks
    input_ids = torch.tensor([encode_text(t, vocab) for t in texts])
    label_ts = torch.tensor(labels)
    loader = DataLoader(TensorDataset(input_ids, label_ts), batch_size=BATCH_SIZE, shuffle=True)

    t_model = TransformerClassifier(len(vocab)).to(device)
    c_model = CNNClassifier(len(vocab)).to(device)
    optimizer = optim.Adam(list(t_model.parameters()) + list(c_model.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"Training NN Models for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        t_model.train(); c_model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(t_model(bx), by) + criterion(c_model(bx), by)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 1 == 0: print(f"Epoch {epoch+1} complete.")

    torch.save({"transformer": t_model.state_dict(), "cnn": c_model.state_dict(), "vocab": vocab}, "nn_hybrid.pth")
    print("✅ All models saved successfully!")

if __name__ == "__main__":
    train()