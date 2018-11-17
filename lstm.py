from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import numpy

filename = "input.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
raw_text = [i for i in raw_text if (i.isalnum() or i in ['\n', ' ']) and i not in ['ë', 'ô']]

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print("Total characters:", n_chars)
print("Total vocabulary:", n_vocab)
print("Characters:")
print(chars)

seq_length = 50
dataX = []
dataY = []
for i in range(0, n_chars - seq_length):
    seq_in = raw_text[i:i+seq_length]
    seq_out = raw_text[i+seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total patterns:", n_patterns)

X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(n_vocab)
y = np_utils.to_categorical(dataY)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath = "lstm_weights_{epoch:02d}-{loss:.5f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=1, mode='min')
callbacks_list = [checkpoint]

# model.load_weights("lstm_weights_50-1.56724.hdf5")

model.fit(X, y, epochs=50, batch_size=128, callbacks=callbacks_list)
