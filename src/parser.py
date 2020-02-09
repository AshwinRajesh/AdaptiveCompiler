import keras
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=None,
                     filters='#$%&()*+-<=>@[\\]^_`{|}~\t\n',
                     lower = False, split = ' ')

list = [["Hello there", "Hi"], ["Hello how are you"], ["Hello there"]]


tokenizer.fit_on_texts(list)
text = tokenizer.texts_to_sequences(list)
print(text)