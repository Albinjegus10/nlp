# from gensim.models import Word2Vec
#
# # Example sentences
# sentences = [['I', 'love', 'natural', 'language', 'processing'],
#              ['Word', 'embeddings', 'are', 'powerful'],
#              ['Machine', 'learning', 'is', 'interesting']]
#
# # Train the Word2Vec model
# model = Word2Vec(sentences, min_count=1)
#
# # Get the word vector for a specific word
# word = 'language'
# word_vector = model.wv[word]
# print(f"Vector representation of '{word}': {word_vector}")
#
# # Find similar words
# similar_words = model.wv.most_similar('language')
# print(f"Words similar to 'language': {similar_words}")


from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.utils import pad_sequences
from keras.datasets import imdb
import numpy as np

# Set the maximum number of words to consider in the vocabulary
max_words = 10000

# Load the IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

# Set the maximum length of the input sequences
max_sequence_length = 500

# Pad the sequences to have the same length
x_train = pad_sequences(x_train, maxlen=max_sequence_length)
x_test = pad_sequences(x_test, maxlen=max_sequence_length)

# Load pre-trained word embeddings
embedding_dim = 100
#embedding_matrix = np.load('path_to_embedding_matrix.npy'), weights=[embedding_matrix]

# Create the model
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_sequence_length, trainable=True))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=1, batch_size=128, validation_data=(x_test, y_test))

