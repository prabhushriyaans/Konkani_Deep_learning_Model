import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os

# --- NEW: HUGGING FACE TOKENIZERS ---
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# --- CONFIG ---
DATA_FILE = "text_konkani_boosted.json"
MAX_LEN, EMBED_DIM, BATCH_SIZE, EPOCHS, LR = 32, 64, 512, 400, 5e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encode_text(text, tokenizer):
    ids = tokenizer.encode(str(text).lower().strip()).ids[:MAX_LEN]
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
        print(f"Error: {DATA_FILE} not found! Did you run the injector script?")
        return

    with open(DATA_FILE, encoding="utf-8") as f:
        data_json = json.load(f)

    texts, labels = [], []
    for idx, cat in enumerate(["negative", "neutral", "positive"]):
        for t in data_json[cat]:
            texts.append(t)
            labels.append(idx)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train_length:{len(train_texts)} Test_length:{len(test_texts)}")
    
    # --- TRAIN BPE TOKENIZER ---
    print("Training Sub-word BPE Tokenizer on Konkani data...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]"], vocab_size=5000)
    tokenizer.train_from_iterator(train_texts, trainer)
    tokenizer.save("konkani_bpe.json")
    vocab_size = tokenizer.get_vocab_size()
    print(f"BPE Tokenizer saved. Vocab size: {vocab_size}")

    # --- TRAIN SVM ---
    print("Training SVM...")
    # We still use basic split for TFIDF, but BPE for Neural Networks
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b") 
    x_train_tfidf = tfidf.fit_transform(train_texts)
    x_test_tfidf = tfidf.transform(test_texts)
    
    svm = CalibratedClassifierCV(LinearSVC(dual=False)).fit(x_train_tfidf, train_labels)
    with open("svm_hybrid.pkl", "wb") as f:
        pickle.dump((svm, tfidf), f)

    # --- TRAIN NEURAL NETWORKS ---
    input_ids = torch.tensor([encode_text(t, tokenizer) for t in train_texts])
    label_ts = torch.tensor(train_labels)
    loader = DataLoader(TensorDataset(input_ids, label_ts), batch_size=BATCH_SIZE, shuffle=True)

    t_model = TransformerClassifier(vocab_size).to(device)
    c_model = CNNClassifier(vocab_size).to(device)
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
        if (epoch+1) % 50 == 0: print(f"Epoch {epoch+1} complete.")

    torch.save({"transformer": t_model.state_dict(), "cnn": c_model.state_dict(), "vocab_size": vocab_size}, "nn_hybrid.pth")
    
    # --- EVALUATION ---
    print("\n--- TEST DATA EVALUATION ---")
    t_model.eval(); c_model.eval()
    test_ids = torch.tensor([encode_text(t, tokenizer) for t in test_texts])
    hybrid_preds = []
    confidences = [] # Tracking confidence scores
    
    with torch.no_grad():
        for i in range(len(test_texts)):
            p_s = torch.tensor(svm.predict_proba(x_test_tfidf[i])[0]).to(device)
            inp_nn = test_ids[i].unsqueeze(0).to(device)
            p_t = torch.softmax(t_model(inp_nn), dim=1)[0]
            p_c = torch.softmax(c_model(inp_nn), dim=1)[0]
            
            # Combine probabilities based on weights
            final_p = (0.4 * p_t) + (0.3 * p_c) + (0.3 * p_s)
            
            # Record the highest probability (confidence) and the predicted label
            conf, pred = torch.max(final_p, dim=0)
            hybrid_preds.append(pred.item())
            confidences.append(conf.item())

    # Print the Precision/Recall/F1-score table
    print(classification_report(test_labels, hybrid_preds, target_names=["Negative", "Neutral", "Positive"]))
    
    # Calculate and print Average Confidence
    avg_conf = sum(confidences) / len(confidences)
    print(f"Average Model Confidence: {avg_conf * 100:.2f}%\n")

if __name__ == "__main__":
    train()