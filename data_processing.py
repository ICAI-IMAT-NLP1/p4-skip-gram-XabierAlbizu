from typing import List, Tuple, Dict, Generator
from collections import Counter
import torch
import numpy as np
import random

try:
    from src.utils import tokenize
except ImportError:
    from utils import tokenize


def load_and_preprocess_data(infile: str) -> List[str]:
    """
    Load text data from a file and preprocess it using a tokenize function.

    Args:
        infile (str): The path to the input file containing text data.

    Returns:
        List[str]: A list of preprocessed and tokenized words from the input text.
    """
    with open(infile) as file:
        text = file.read()  # Read the entire file
    
    tokens: List[str] = tokenize(text)
    return tokens
    

def create_lookup_tables(words: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create lookup tables for vocabulary.

    Args:
        words: A list of words from which to create vocabulary.

    Returns:
        A tuple containing two dictionaries. The first dictionary maps words to integers (vocab_to_int),
        and the second maps integers to words (int_to_vocab).
    """
    # TODO
    word_counts: Counter = Counter(words)
    # Sorting the words from most to least frequent in text occurrence.
    word_counts = list(word_counts.keys())

    # Create int_to_vocab and vocab_to_int dictionaries.
    vocab_to_int: Dict[str, int] = {}
    int_to_vocab: Dict[int, str] = {}
    for i, w in enumerate(word_counts):
        vocab_to_int[w] = i
        int_to_vocab[i] = w

    return vocab_to_int, int_to_vocab


def subsample_words(words: List[str], vocab_to_int: Dict[str, int], threshold: float = 1e-5) -> Tuple[List[int], Dict[str, float]]:
    """
    Perform subsampling on a list of word integers using PyTorch, aiming to reduce the 
    presence of frequent words according to Mikolov's subsampling technique. This method 
    calculates the probability of keeping each word in the dataset based on its frequency, 
    with more frequent words having a higher chance of being discarded. The process helps 
    in balancing the word distribution, potentially leading to faster training and better 
    representations by focusing more on less frequent words.
    
    Args:
        words (list): List of words to be subsampled.
        vocab_to_int (dict): Dictionary mapping words to unique integers.
        threshold (float): Threshold parameter controlling the extent of subsampling.

        
    Returns:
        List[int]: A list of integers representing the subsampled words, where some high-frequency words may be removed.
        Dict[str, float]: Dictionary associating each word with its frequency.
    """
    # TODO
    # Convert words to integers
    word_apps_counter = Counter(words)
    freqs: Dict[str, float] = {w: 1-torch.sqrt(threshold/torch.tensor([word_apps_counter[w]])) for w in words}

    train_words = [vocab_to_int[word] for word in words if np.random.rand() > freqs[word]]

    return train_words, freqs

def get_target(words: List[str], idx: int, window_size: int = 5) -> List[str]:
    """
    Get a list of words within a window around a specified index in a sentence.

    Args:
        words (List[str]): The list of words from which context words will be selected.
        idx (int): The index of the target word.
        window_size (int): The maximum window size for context words selection.

    Returns:
        List[str]: A list of words selected randomly within the window around the target word.
    """
    # TODO
    r = min(random.randint(1,window_size), idx, len(words)-1-idx)
    target_words: List[str] = [words[i] for i in range(idx-r, r+idx+1)]
    target_words.remove(words[idx])

    return target_words

def get_batches(words: List[int], batch_size: int, window_size: int = 5) -> Generator[Tuple[List[int], List[int]], None, None]:
    """Generate batches of word pairs for training.

    This function creates a generator that yields tuples of (inputs, targets),
    where each input is a word, and targets are context words within a specified
    window size around the input word. This process is repeated for each word in
    the batch, ensuring only full batches are produced.

    Args:
        words: A list of integer-encoded words from the dataset.
        batch_size: The number of words in each batch.
        window_size: The size of the context window from which to draw context words.

    Yields:
        A tuple of two lists:
        - The first list contains input words (repeated for each of their context words).
        - The second list contains the corresponding target context words.
    """

    # TODO
    for idx in range(0, len(words), batch_size):
        inputs_ = [words[i] for i in range(idx, min(idx+window_size+1, len(words)))]
        targets, inputs = [], []
        for i in range(len(inputs_)):
            targets_aux = get_target(inputs_,i,window_size)
            for t in targets_aux:
                targets.append(t)
                inputs.append(inputs_[i])
        yield inputs, targets

def cosine_similarity(embedding: torch.nn.Embedding, valid_size: int = 16, valid_window: int = 100, device: str = 'cpu'):
    """Calculates the cosine similarity of validation words with words in the embedding matrix.

    This function calculates the cosine similarity between some random words and
    embedding vectors. Through the similarities, it identifies words that are
    close to the randomly selected words.

    Args:
        embedding: A PyTorch Embedding module.
        valid_size: The number of random words to evaluate.
        valid_window: The range of word indices to consider for the random selection.
        device: The device (CPU or GPU) where the tensors will be allocated.

    Returns:
        A tuple containing the indices of valid examples and their cosine similarities with
        the embedding vectors.

    Note:
        sim = (a . b) / |a||b| where `a` and `b` are embedding vectors.
    """

    # TODO
    #nums
    indxs = torch.randint(low=0, high=valid_window, size=(valid_size,))
    valid_examples = embedding(indxs)
    nums = torch.abs(valid_examples @ embedding.weight.T)

    #dens
    valid_examples_norms = torch.norm(valid_examples, dim=0)
    embedding_norms = torch.norm(embedding.weight, dim=0)
    dens = valid_examples_norms @ embedding_norms.T

    similarities = nums / dens

    return valid_examples, similarities