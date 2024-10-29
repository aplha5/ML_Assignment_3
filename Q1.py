import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load text data
with open('paul_graham_essays.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()

# Remove special characters except for full stops
text = re.sub(r"[^a-zA-Z0-9 \.]", "", text)

# Split into sentences on full stops and tokenize
sentences = text.split(".")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
sequences = tokenizer.texts_to_sequences(sentences)
# Define context length
context_length = 10

X, y = [], []
for seq in sequences:
    for i in range(context_length, len(seq)):
        context = seq[i - context_length:i]
        target = seq[i]
        X.append(context)
        y.append(target)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten

embedding_dim = 64  # Adjustable embedding size
hidden_layer_size = 1024  # Adjustable hidden layer size

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=context_length),
    Flatten(),
    Dense(hidden_layer_size, activation='relu'),  # Change activation function as desired
    Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
epochs = 600
model.fit(X, y, epochs=epochs, batch_size=64)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import streamlit as st
# Get the embedding weights from the model
embeddings = model.layers[0].get_weights()[0]

# Apply t-SNE to reduce embeddings to 2D
tsne = TSNE(n_components=2, random_state=42)
reduced_embeddings = tsne.fit_transform(embeddings)

# Plot a subset of words for clarity
plt.figure(figsize=(10, 10))
for i, word in enumerate(word_index):
    if i < 100:  # Show only the first 100 words
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
        plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
st.pyplot(plt)


# Define prediction function
def predict_next_words(input_text, k=3, context_length=5):
    sequence = tokenizer.texts_to_sequences([input_text])[0]
    for _ in range(k):
        padded_sequence = pad_sequences([sequence[-context_length:]], maxlen=context_length, padding='pre')
        pred_word_id = np.argmax(model.predict(padded_sequence), axis=-1)[0]
        sequence.append(pred_word_id)
    return tokenizer.sequences_to_texts([sequence])[0]

# Streamlit Interface
st.title("MLP-Based Next-Word Prediction")

# Input text box
input_text = st.text_input("Enter a starting text:", "")

# Prediction controls
k = st.slider("Number of words to predict", min_value=1, max_value=10, value=3)
context_length = st.selectbox("Context length", options=[5, 10, 15], index=0)
embedding_dim = st.selectbox("Embedding dimension", options=[64, 128], index=0)
activation_fn = st.selectbox("Activation function", options=["relu", "tanh"], index=0)
random_seed = st.number_input("Random seed", min_value=0, max_value=1000, value=42)

# Display prediction
if input_text:
    prediction = predict_next_words(input_text, k, context_length)
    st.write("Generated Text:", prediction)

