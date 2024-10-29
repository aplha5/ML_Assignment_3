import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # for making figures
import streamlit as st
import re
import os
from torch.optim.lr_scheduler import StepLR

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load the text file
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Load your specific text data
paul_graham_text = load_text('paul_graham_essays.txt')

def generate_word_prediction_dataset(text, block_size=5, print_limit=20):
    """
    Generates a dataset for word-based prediction from input text.
    
    Args:
        text (str): Input text to process.
        block_size (int): Number of words used as context to predict the next word.
        print_limit (int): Number of (context, target) pairs to print for visualization.

    Returns:
        X (torch.Tensor): Input tensor containing contexts.
        Y (torch.Tensor): Output tensor containing target word indices.
        stoi (dict): String-to-index mapping of words.
        itos (dict): Index-to-string mapping of words.
    """
    # Step 1: Clean and split text into words
    text = re.sub(r'[^a-zA-Z0-9 .]', '', text).lower().strip()
    words = text.split()

    # Step 2: Create vocabulary and mappings
    vocabulary = set(words)
    stoi = {word: i + 1 for i, word in enumerate(vocabulary)}
    stoi["."] = 0  # Sentence-end marker
    itos = {i: word for word, i in stoi.items()}
    itos[0] = "."  # Ensure "." is included in `itos`

    X, Y = [], []  # Inputs and targets
    count = 0  # Counter for visualization

    # Step 3: Generate (X, Y) pairs using a sliding window
    for i in range(len(words) - block_size):
        context = words[i:i + block_size]
        target = words[i + block_size]
        X.append([stoi[word] for word in context])
        Y.append(stoi[target])

        # Print context-target pairs (limited by `print_limit`)
        if count < print_limit:
            print(' '.join(context), '--->', target)
            count += 1

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    print(f"Dataset generated with {len(X)} samples")
    return X, Y, stoi, itos

X, Y, stoi, itos = generate_word_prediction_dataset(paul_graham_text, block_size=5, print_limit=20)
print("X tensor shape:", X.shape)
print("y tensor shape:", Y.shape)

class NextWord(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size, activation_fn='ReLU'):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, vocab_size)
        if activation_fn == 'ReLU':
            self.activation = nn.ReLU()
        elif activation_fn == 'Tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Unsupported activation function. Use 'ReLU' or 'Tanh'.")

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)
        x = self.activation(self.lin1(x))  # Apply activation function
        x = self.lin2(x)
        return x

def save_model(model, embedding_size, context_length, activation_fn):
    model_name = f"model_emb{embedding_size}_ctx{context_length}_act{activation_fn}.pth"
    torch.save(model.state_dict(), model_name)
    print(f"Model saved as {model_name}")

def generate_text(model, itos, stoi, block_size, input_sentence, max_len=100):
    input_indices = [stoi.get(word, 0) for word in input_sentence.split()]  
    context = [0] * max(0, block_size - len(input_indices)) + input_indices[-block_size:]
    generated_text = input_sentence.strip() + ' '
    
    for _ in range(max_len):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        next_word = itos[ix]
        generated_text += next_word + ' '
        context = context[1:] + [ix]
    
    generated_text = generated_text.replace(' .', '.')
    
    return generated_text.strip()

# Training parameters
loss_fn = nn.CrossEntropyLoss()
hidden_size = 1024
batch_size = 4096
print_every = 50
num_epochs = 100
st.title("Next Word Prediction Model")
st.write("Generate text using a trained next-word prediction model.")

# User inputs
block_size = st.selectbox("Select context length:", [5, 10, 15])
embedding_size = st.selectbox("Select embedding size:", [64, 128])
activation_function = st.selectbox("Select activation function:", ["ReLU", "Tanh"])
input_sentence = st.text_input("Enter a sentence:", "The meaning of life")
num_words_to_predict = st.number_input("Number of words to predict:", min_value=1, value=100)

for emb_dim in [embedding_size]:
    for block_size in [block_size]:
        for activation_function in [activation_function]:
            X_tensor, y_tensor, stoi, itos = generate_word_prediction_dataset(paul_graham_text, block_size=block_size, print_limit=10)
            model = NextWord(block_size, len(stoi), emb_dim, hidden_size, activation_function).to(device)
            
            opt = torch.optim.AdamW(model.parameters(), lr=0.01)
            scheduler = StepLR(opt, step_size=50, gamma=0.1)
            X_tensor.to(device)
            y_tensor.to(device)

            for epoch in range(num_epochs):
                for i in range(0, X_tensor.shape[0], batch_size):
                    x = X_tensor[i:i + batch_size].to(device)
                    y = y_tensor[i:i + batch_size].to(device)
                    y_pred = model(x)
                    loss = loss_fn(y_pred, y)
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
                scheduler.step()
                if (epoch + 1) % print_every == 0:
                    print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
            save_model(model, emb_dim, block_size, activation_function)

# Generate sample text
input_sentence = "The meaning of life"
prediction=generate_text(model, itos, stoi, block_size, input_sentence, 1000)
print(prediction)
st.write("Generated Text:", prediction)