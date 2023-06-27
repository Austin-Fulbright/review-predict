import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.regularizers import l2
from keras.initializers import GlorotUniform
import gender_edit
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load the dataset into a pandas DataFrame
data = pd.read_csv('prep_data.csv')


X_train, X_test, y_train, y_test = train_test_split(data['preprocessed_review'], data['review_type'], test_size=0.2, random_state=42)

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to a fixed length
max_sequence_length = 100
X_train_padded = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_sequence_length)

# Convert the target variable to numpy arrays
y_train_np = np.array(y_train)
y_test_np = np.array(y_test)

# Build the RNN model
embedding_dim = 100

model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, embedding_dim, input_length=max_sequence_length))
model.add(LSTM(128, kernel_initializer=GlorotUniform(seed=42), recurrent_regularizer=l2(1e-5)))
model.add(Dropout(0.5)) # Add dropout layer to prevent overfitting
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
batch_size = 64
epochs = 3

model.fit(X_train_padded, y_train_np, batch_size=batch_size, epochs=epochs, validation_data=(X_test_padded, y_test_np))

# Evaluate the model
loss, accuracy = model.evaluate(X_test_padded, y_test_np)
print("Model loss:", loss)
print("Model accuracy:", accuracy)
