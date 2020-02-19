import keras
from keras.preprocessing.text import Tokenizer
import json
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding

'''
test_file = open('./patents.json', 'r')

patents = json.load(test_file)
titles = (patents['patents'])

abstracts = []
for item in titles:
    temp = item.values()[0]
    abstracts.append(temp)

print(abstracts[0])
tokenizer = Tokenizer(num_words=None, filters='#$%&()*+-<=>@[\\]^_`{|}~\t\n', lower = False, split = ' ')

tokenizer.fit_on_texts(abstracts)
text = tokenizer.texts_to_sequences(abstracts)

print(text[100])

idx_word = tokenizer.index_word
print(idx_word)

features = []
labels = []

training_length = 50

# Iterate through the sequences of tokens
for seq in text:

    # Create multiple training examples from each sequence
    for i in range(training_length, len(seq)):
        # Extract the features and label
        extract = seq[i - training_length:i + 1]

        # Set the features and label
        features.append(extract[:-1])
        labels.append(extract[-1])

print(labels[:15])
num_words = len(idx_word) + 1
label_array = np.zeros((len(features), num_words), dtype=np.int32)

for example_index, word_index in enumerate(labels):
    label_array[example_index, word_index] = 1

print(label_array.shape)

model = Sequential()

model.add(
    Embedding(input_dim=num_words,
              input_length = training_length,
              output_dim=100,
              weights=[embedding_matrix],
              trainable=False,
              mask_zero=True))
model.add(Masking(mask_value=0.0))

# Recurrent layer
model.add(LSTM(64, return_sequences=False,
               dropout=0.1, recurrent_dropout=0.1))

# Fully connected layer
model.add(Dense(64, activation='relu'))

# Dropout for regularization
model.add(Dropout(0.5))

# Output layer
model.add(Dense(num_words, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
'''
g = np.loadtxt('/Users/ashwinr/Downloads/glove.6B/glove.6B.100d.txt', dtype='str', comments=None)
vectors = g[:, 1:].astype('float')
words = g[:, 0]

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v2) * np.linalg.norm(v1))

word_lookup = {word: vector for word, vector in zip(words, vectors)}
print(cosine_similarity(word_lookup['neuronal'], word_lookup['neural']))
#print(np.dot(word_lookup['neural'], word_lookup['neuronal']))


'''
# New matrix to hold word embeddings
embedding_matrix = np.zeros((num_words, vectors.shape[1]))

for i, word in enumerate(idx_word.keys()):
    # Look up the word embedding
    vector = word_lookup.get(word, None)

    # Record in matrix
    if vector is not None:
        embedding_matrix[i + 1, :] = vector
'''