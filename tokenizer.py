from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



def tokenize_and_pad_sequences(text_tokens, max_sequence_length):
    """
    Tokenizes the input text tokens, creates vocabulary and sequences, and pads the sequences to a fixed length.

    Parameters:
    - text_tokens (list of lists): List of lists containing text tokens.
    - max_sequence_length (int): The maximum length for padding sequences.

    Returns:
    - tokenizer (Tokenizer): The fitted Tokenizer.
    - word_index (dict): Vocabulary word index.
    - reverse_word_index (dict): Reverse vocabulary index.
    - padded_sequences (numpy array): Padded sequences of text tokens.
    """
    # Tokenizer to create vocabulary and sequences
    tokenizer = Tokenizer(oov_token="<oov>")
    tokenizer.fit_on_texts(text_tokens)

    # Vocabulary and Reverse Vocabulary
    word_index = tokenizer.word_index
    reverse_word_index = tokenizer.index_word
    reverse_word_index[0] = '<pad>'
    word_index['<pad>'] = 0

    # Convert text tokens to sequences of numbers using the word index
    sequences = tokenizer.texts_to_sequences(text_tokens)

    # Pad sequences to a fixed length
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

    return tokenizer, word_index, reverse_word_index, padded_sequences



