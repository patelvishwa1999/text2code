def evaluate_model_in_chunks(model, X_test, y_test, chunk_size=100):
    total_loss = 0
    total_accuracy = 0
    total_chunks = int(np.ceil(len(X_test) / chunk_size))

    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        X_test_chunk = X_test[start_idx:end_idx]
        y_test_chunk = y_test[start_idx:end_idx]
        np_zeros_chunk = np.zeros_like(y_test_chunk)

        loss, accuracy = model.evaluate([X_test_chunk, np_zeros_chunk], np.expand_dims(y_test_chunk, -1), batch_size=1, verbose=0)
        total_loss += loss
        total_accuracy += accuracy

    avg_loss = total_loss / total_chunks
    avg_accuracy = total_accuracy / total_chunks
    return avg_loss, avg_accuracy

from nltk.translate.bleu_score import corpus_bleu

def calculate_bleu_scores(y_references, y_preds):
    """
    Calculate BLEU scores for a list of reference and candidate sentences,
    and returns average BLEU SCORE

    Parameters:
    - y_references (list of lists): List of reference sentences, where each reference is a list of tokens.
    - y_preds (list of lists): List of candidate sentences, where each candidate is a list of tokens.

    Returns:
    - bleu_scores : Average of - BLEU scores for each reference-candidate pair.
    """
    bleu_scores = []

    for i in range(len(y_preds)):
        reference = y_references[i]
        candidate = y_preds[i]
        bleu_score = corpus_bleu([reference], [candidate])
        bleu_scores.append(bleu_score)
    
    avg_bleu_score = sum(bleu_scores)/len(bleu_scores)

    return avg_bleu_score