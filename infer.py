
def create_inference_encoder_model(model):
    # Extract the encoder part from the full model
    encoder_inputs = model.input[0]
    encoder_outputs, state_h, state_c = model.layers[2].output  # Assuming the LSTM layer is at index 2
    encoder_states = [state_h, state_c]
    encoder_model = Model(encoder_inputs, encoder_states)

    return encoder_model

def create_inference_decoder_model(model):
    # Extract the decoder part from the full model

    # Decoder input
    decoder_inputs = model.input[1]
    # Decoder states input
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    # Decoder embedding
    dec_emb_layer = model.layers[3]  # Assuming the Embedding layer is at index 3
    dec_emb = dec_emb_layer(decoder_inputs)

    # Decoder LSTM
    decoder_lstm = model.layers[4]  # Assuming the LSTM layer is at index 4
    decoder_outputs, state_h, state_c = decoder_lstm(dec_emb, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    # Dense layer for output
    decoder_dense = model.layers[5]  # Assuming the Dense layer is at index 5
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the decoder model
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return decoder_model



def create_attention_encoder_model(model):
    # Extract the encoder part from the full model
    encoder_inputs = model.input[0]
    encoder_outputs, state_h, state_c = model.layers[4].output  # Assuming the LSTM layer is at index 4
    encoder_states = [state_h, state_c]
    encoder_model = Model(encoder_inputs, encoder_states)
    
    return encoder_model


def create_attention_decoder_model(model):
    # Extract the decoder part from the full model

    # Decoder input
    decoder_inputs = model.input[1]
    # Decoder states input
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    # Decoder embedding
    dec_emb_layer = model.layers[5]  # Assuming the Embedding layer is at index 5
    dec_emb = dec_emb_layer(decoder_inputs)
    
    # Attention layer
    attention_layer = model.layers[6]  # Assuming the Attention layer is at index 6
    attention_out = attention_layer([decoder_inputs, model.layers[3].output])  # Assuming Attention input is at index 3
    
    # Combine attention output and decoder input
    decoder_combined_context = concatenate([dec_emb, attention_out], axis=-1)

    # Decoder LSTM
    decoder_lstm = model.layers[7]  # Assuming the LSTM layer is at index 7
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_combined_context, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    # Dense layer for output
    decoder_dense = model.layers[8]  # Assuming the Dense layer is at index 8
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the decoder model
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return decoder_model


    # Function to generate a sequence
def generate_sequence(input_sequence, encoder_model, decoder_model):
    # Encode the input sequence
    states_value = encoder_model.predict(input_sequence)

    # Initialize the decoder input sequence with a start token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = code_word_index['<start>']

    # Generate the output sequence
    stop_condition = False
    decoded_sequence = []

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        print("Output TOKENS : ", output_tokens)
        flat_tokens = output_tokens.flatten()
        output_token = np.argmax(flat_tokens)
        print("Output TOKEN :",output_token)
        output_word = reverse_code_word_index[output_token]
        print("Output WORD :",output_word)
        # Append the sampled token to the decoded sequence
        decoded_sequence.append(output_word)
        if output_word == '<end>' or output_word == '<pad>' or len(decoded_sequence) > 50:
          stop_condition = True

        
        # Update the target sequence for the next iteration
        target_seq[0, 0] = output_token
        
        # Update states
        states_value = [h, c]

    return decoded_sequence