from sklearn.model_selection import train_test_split
from tokenizer import tokenize_and_pad_sequences





code_padded_sequences = tokenize_and_pad_sequences(code_tokens, max_sequence_length)
documentation_padded_sequences = tokenize_and_pad_sequences(doc_tokens, max_sequence_length)

# Create the training data
X = np.array(code_padded_sequences)
y = np.array(documentation_padded_sequences)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
from keras.callbacks import EarlyStopping
def train_Seq2Seq(model, X_train, y_train):
    #prepare decoder data for training Seq2Seq

    decoder_input_data = np.zeros_like(y_train)
    decoder_input_data = np.array(decoder_input_data)
    decoder_input_data[:, 1:] = y_train[:, :-1]
    decoder_input_data[:, 0] = code_word_index['<start>']

    early_stopping = EarlyStopping(monitor='accuracy', patience=3, restore_best_weights=True)
    history = model.fit([X_train, decoder_input_data], np.expand_dims(y_train, -1), batch_size=8, epochs=5, validation_split=0.2)

    model.save('./trained_models/text2code.keras')