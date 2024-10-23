import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import streamlit as st

# Load dataset (using Paul Graham essays or any other dataset)
with open('paul_graham_essays.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Preprocess text (removing special characters except periods)
text = re.sub(r'[^a-zA-Z0-9 .]', '', text)  # Using raw string to avoid invalid escape sequence warning
text = text.lower().strip()

# Tokenize into words and build vocabulary
words = text.split()
vocab = sorted(set(words))
stoi = {w: i+1 for i, w in enumerate(vocab)}  # Start from 1, 0 reserved for padding
stoi['.'] = 0
itos = {i: w for w, i in stoi.items()}

# Define device for PyTorch (use GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create context length (block_size) and X, Y pairs for word prediction
block_size = 5  # Number of words for context
X, Y = [], []
for i in range(len(words) - block_size):
    context = words[i:i+block_size]
    target = words[i+block_size]
    X.append([stoi[w] for w in context])
    Y.append(stoi[target])

X = torch.tensor(X).to(device)
Y = torch.tensor(Y).to(device)

# Word Embedding model
class NextWord(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.lin1(x))  # Activation function
        x = self.lin2(x)
        return x

# Initialize the model
emb_dim = 64
hidden_size = 1024
model = NextWord(block_size, len(stoi), emb_dim, hidden_size).to(device)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=0.001)

# Training the model
batch_size = 256
for epoch in range(1000):  # Max epochs
    for i in range(0, X.shape[0], batch_size):
        x_batch = X[i:i+batch_size]
        y_batch = Y[i:i+batch_size]
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        opt.step()
        opt.zero_grad()
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

# Generating text using the trained model
def generate_text(model, itos, stoi, block_size, seed_text, max_len=20):
    context = [stoi.get(w, 0) for w in seed_text.split()[-block_size:]]
    generated_text = seed_text
    for _ in range(max_len):
        x = torch.tensor([context]).to(device)
        y_pred = model(x)
        next_word = itos[torch.argmax(y_pred, dim=1).item()]
        generated_text += ' ' + next_word
        context = context[1:] + [stoi[next_word]]
        if next_word == '.':
            break
    return generated_text

# Test generation
print(generate_text(model, itos, stoi, block_size, "the meaning of life", max_len=50))

# Visualize embeddings using t-SNE
def plot_tsne_embeddings(model, itos):
    embeddings = model.emb.weight.detach().cpu().numpy()
    tsne = TSNE(n_components=2)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 10))
    for i, label in enumerate(itos.values()):
        x, y = reduced_embeddings[i]
        plt.scatter(x, y)
        plt.text(x+0.05, y+0.05, label)
    plt.show()

plot_tsne_embeddings(model, itos)

# Streamlit app for text generation
st.title("MLP-Based Next-Word Prediction")
st.write("Enter a seed text, and the model will predict the next few words.")

seed_text = st.text_input("Seed Text", "the meaning of life")
num_words = st.slider("Number of words to generate", min_value=1, max_value=50, value=10)

generated_text = generate_text(model, itos, stoi, block_size, seed_text, max_len=num_words)
st.write(f"Generated Text: {generated_text}")

# Controls for modifying hyperparameters
st.sidebar.header("Model Controls")
block_size = st.sidebar.slider("Context Length", 1, 10, block_size)
emb_dim = st.sidebar.slider("Embedding Dimension", 32, 256, emb_dim)
hidden_size = st.sidebar.slider("Hidden Layer Size", 512, 2048, hidden_size)
