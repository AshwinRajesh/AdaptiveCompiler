import keras
from keras.preprocessing.text import Tokenizer
import json
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding


samples = ["Set total to zero", "Assign 5 to variable x", "Add 3 to 4",
            "Assign 0 to num", "Display the word 'hello'", "Print 'hi'",
            "Set string to 'hi'", "Multiply 6 by 5", "Set x to false",
            "Bob is equal to 10", "Display 'yes'", "Subtract 10 by 8",
            "Print the variable bob", "Output '4' to the screen",
            "Set the variable count to 5", "Assign 'hi' to greeting",
            "Divide 24 by 6", "Print the sentence 'Hello there!'"]

tokenizer = Tokenizer(num_words=None, filters='#$%&()*+-<=>@[\\]^_`{|}~\t\n', lower = False, split = ' ')

tokenizer.fit_on_texts(samples)
text = tokenizer.texts_to_sequences(samples)

print(text)

labels = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0],
          [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0],
          [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0],
          [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0],
          [0, 0, 1, 0], [0, 0, 0, 1]])

commands = ["set", "new", "arithmetic", "display"]

idx_word = tokenizer.index_word
num_words = len(idx_word) + 1

label_array = np.zeros(len(text), num_words)

g = np.loadtxt('/Users/ashwinr/Downloads/glove.6B/glove.6B.100d.txt', dtype='str', comments=None)
vectors = g[:, 1:].astype('float')
words = g[:, 0]

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v2) * np.linalg.norm(v1))

word_lookup = {word: vector for word, vector in zip(words, vectors)}

#print(np.dot(word_lookup['neural'], word_lookup['neuronal']))

# New matrix to hold word embeddings
embedding_matrix = np.zeros((num_words, vectors.shape[1]))

for i, word in enumerate(idx_word.keys()):
    # Look up the word embedding
    vector = word_lookup.get(word, None)

    # Record in matrix
    if vector is not None:
        embedding_matrix[i + 1, :] = vector

model = Sequential()

model.add(
    Embedding(input_dim=num_words,
              input_length = len(text),
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

print("Training")
history = model.fit(text, labels, epochs=150)


def set():
    pass

def new():
    pass

def arithmetic():
    pass

def display():
    pass

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


g = np.loadtxt('/Users/ashwinr/Downloads/glove.6B/glove.6B.100d.txt', dtype='str', comments=None)
vectors = g[:, 1:].astype('float')
words = g[:, 0]

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v2) * np.linalg.norm(v1))

word_lookup = {word: vector for word, vector in zip(words, vectors)}
print(cosine_similarity(word_lookup['neuronal'], word_lookup['neural']))
#print(np.dot(word_lookup['neural'], word_lookup['neuronal']))


# New matrix to hold word embeddings
embedding_matrix = np.zeros((num_words, vectors.shape[1]))

for i, word in enumerate(idx_word.keys()):
    # Look up the word embedding
    vector = word_lookup.get(word, None)

    # Record in matrix
    if vector is not None:
        embedding_matrix[i + 1, :] = vector
'''
