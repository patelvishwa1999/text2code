from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

def BasicModel(latent_dim, documentation_word_index, code_word_index):
    # Seq2seq Model

    # Vocabulary sizes
    DOC_VOCAB = len(documentation_word_index)
    CODE_VOCAB = len(code_word_index)

    # Encoder
    encoder_inputs = Input(shape=(None,))
    enc_emb = Embedding(input_dim=DOC_VOCAB, output_dim=latent_dim)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None,))
    dec_emb_layer = Embedding(input_dim=CODE_VOCAB, output_dim=latent_dim)
    dec_emb = dec_emb_layer(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
    decoder_dense = Dense(CODE_VOCAB, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compiling the model
    learning_rate_schedule = ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.9
    )

    # Create the Adam optimizer with the learning rate schedule
    optimizer = Adam(learning_rate=learning_rate_schedule, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model



def AttentionModel(latent_dim, documentation_word_index, code_word_index):
    # Seq2seq Model with Attention Mechanism

    # Vocabulary sizes
    DOC_VOCAB = len(documentation_word_index)
    CODE_VOCAB = len(code_word_index)

    # Encoder
    encoder_inputs = Input(shape=(None,))
    enc_emb = Embedding(input_dim=DOC_VOCAB, output_dim=latent_dim)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
    encoder_states = [state_h, state_c]

    # Decoder with Attention
    decoder_inputs = Input(shape=(None,))
    dec_emb_layer = Embedding(input_dim=CODE_VOCAB, output_dim=latent_dim)
    dec_emb = dec_emb_layer(decoder_inputs)
    
    # Attention layer
    attention = Attention()([decoder_inputs, encoder_outputs])
    
    # Combine attention output and decoder input
    decoder_combined_context = concatenate([dec_emb, attention], axis=-1)

    # Decoder LSTM
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_combined_context, initial_state=encoder_states)

    # Dense layer for output
    decoder_dense = Dense(CODE_VOCAB, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compiling the model
    learning_rate_schedule = ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.9
    )

    # Create the Adam optimizer with the learning rate schedule
    optimizer = Adam(learning_rate=learning_rate_schedule, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model