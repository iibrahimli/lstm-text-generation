from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
import numpy

filename = "input.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print("Total characters:", n_chars)
print("Total vocabulary:", n_vocab)

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

# for loading weights
filename = "name_of_your_weights_checkpoint_file"
model.load_weights(filename)
print("Loaded weights")

model.compile(loss='categorical_crossentropy', optimizer='adam')

start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]

for i in range(5000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    pred = model.predict(x)
    index = numpy.argmax(pred)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print(result, sep='', end='')
    pattern.append(index)
    pattern = pattern[1:]
