import gensim.downloader as api

# Load pre-trained Word2Vec embeddings
model = api.load("word2vec-google-news-300")

# Define the input data (a list of sentences)
sentences = ["The cat sat on the mat", "The dog ran in the park"]

# Create a tokenizer to convert the sentences into a sequence of words
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

# Create a matrix of Word2Vec embeddings for the words in our vocabulary
num_words = len(word_index) + 1
embedding_dim = 300
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if word in model.wv.vocab:
        embedding_matrix[i] = model.wv[word]

# Define the embedding layer using the Word2Vec embedding matrix
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
