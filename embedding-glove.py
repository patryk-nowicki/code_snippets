import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding

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

# Define the embedding layer using the GloVe embedding matrix
embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embedding_matrix],
                            trainable=False)

# Convert the sentences to sequences of word indices
sequences = tokenizer.texts_to_sequences(sentences)

# Pad the sequences to ensure they are all the same length
max_len = max([len(seq) for seq in sequences])
padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len, padding='post')

# Pass the padded sequences through the embedding layer to get the word embeddings
word_embeddings = embedding_layer(padded_sequences)

# Print the shape of the word embeddings tensor
print(word_embeddings.shape)



'''
You can download pre-trained GloVe embeddings from the Stanford NLP website: https://nlp.stanford.edu/projects/glove/

On the website, you will find multiple pre-trained GloVe embeddings for different corpus sizes and embedding dimensions. For example, glove.6B.zip contains pre-trained embeddings on a 6 billion token corpus with embedding dimensions of 50, 100, 200, and 300.

Once you have downloaded the pre-trained GloVe embeddings, extract the zip file and you should see a text file with a name like glove.6B.100d.txt. This file contains the pre-trained embeddings for a 100-dimensional space.

You can then use the path to this file in your code, as shown in the previous example. Make sure to replace path/to/glove.6B.100d.txt in the example code with the actual path to the file on your system.

'''
