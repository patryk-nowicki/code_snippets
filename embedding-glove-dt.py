import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Load pre-trained GloVe embeddings
embeddings_index = {}
with open('path/to/glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Define the input data (a list of sentences)
sentences = ["The cat sat on the mat", "The dog ran in the park"]

# Create a tokenizer to convert the sentences into a sequence of words
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

# Create a matrix of GloVe embeddings for the words in our vocabulary
num_words = len(word_index) + 1
embedding_dim = 100
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Convert the sentences to sequences of word indices
sequences = tokenizer.texts_to_sequences(sentences)

# Pad the sequences to ensure they are all the same length
max_len = max([len(seq) for seq in sequences])
padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len, padding='post')

# Train a decision tree classifier on top of the word embeddings
X = embedding_matrix[padded_sequences.flatten()].reshape(-1, max_len*embedding_dim)
y = np.array([0, 1])  # Example labels
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Make predictions on new sentences
new_sentences = ["The cat chased the mouse", "The dog slept in the sun"]
new_sequences = tokenizer.texts_to_sequences(new_sentences)
new_padded_sequences = keras.preprocessing.sequence.pad_sequences(new_sequences, maxlen=max_len, padding='post')
new_X = embedding_matrix[new_padded_sequences.flatten()].reshape(-1, max_len*embedding_dim)
predictions = clf.predict(new_X)
print(predictions)
