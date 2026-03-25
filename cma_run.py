import torch
import torch.nn as nn
import pickle
import os

# Copying class definitions so torch can load them
MAX_LEN, EMBED_DIM = 32, 64
TEMPERATURE = 1.0  # Reduced from 1.5 for better confidence

def tokenize(text):
    text = text.lower().strip()
    tokens = []
    for w in text.split():
        tokens.append(w)
        if len(w) > 3:
            tokens.append(w[:3]); tokens.append(w[-3:])
    return tokens

def encode_text(text, vocab):
    ids = [vocab.get(t, 1) for t in tokenize(text)][:MAX_LEN]
    return ids + [0] * (MAX_LEN - len(ids))

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

# --- LOAD MODELS ---
if not os.path.exists("nn_hybrid.pth") or not os.path.exists("svm_hybrid.pkl"):
    print("Error: Model files not found. Run cma_train.py first!")
    exit()

nn_data = torch.load("nn_hybrid.pth", map_location="cpu")
vocab = nn_data["vocab"]
t_model = TransformerClassifier(len(vocab))
c_model = CNNClassifier(len(vocab))
t_model.load_state_dict(nn_data["transformer"])
c_model.load_state_dict(nn_data["cnn"])
t_model.eval(); c_model.eval()

with open("svm_hybrid.pkl", "rb") as f:
    svm, tfidf = pickle.load(f)

print("--- KONKANI HYBRID AI LOADED ---")

while True:
    text = input("\nYou: ").strip()
    if not text or text.lower() == "exit": break

    inp_nn = torch.tensor([encode_text(text, vocab)])
    with torch.no_grad():
        p_t = torch.softmax(t_model(inp_nn) / TEMPERATURE, dim=1)[0]
        p_c = torch.softmax(c_model(inp_nn) / TEMPERATURE, dim=1)[0]
    
    p_s = torch.tensor(svm.predict_proba(tfidf.transform([text]))[0])

    # Combined Probability (Weighted Ensemble)
    final_p = (0.4 * p_t) + (0.3 * p_c) + (0.3 * p_s)
    labels = ["Negative", "Neutral", "Positive"]
    idx = torch.argmax(final_p).item()

    print(f"AI: {labels[idx]} ({final_p[idx]*100:.1f}%)")