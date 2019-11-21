import logging
from collections import defaultdict

import numpy as np


PAD_INDEX = 0
PAD_SYMBOL = '<pad>'
OOV_INDEX = 1
OOV_SYMBOL = '<oov>'

logger = logging.getLogger(__name__)


class Vocab:

    """
    vocab file - use other_symbols for adding additional symbols to the vocab.
    """

    def __init__(self, sentences, vocab_size=None, add_pad=True,
                 add_unk=True, other_symbols=[], min_freq=0):
        self.vocab, self.reverse_vocab = __class__.generate_vocab(
            sentences, vocab_size, add_pad, add_unk, other_symbols,
            min_freq)
        """
        note: sentences can be a list of str or even a stream.
        """
        logger.info('vocab size: {}'.format(len(self.reverse_vocab)))

    def size(self):
        return len(self.reverse_vocab)

    def get_tokens(self):
        return self.reverse_vocab

    def encode_sentence(self, words, as_numpy=True, pad_to=None):
        """

        :param words: list of tokens - i.e., list of str
        :param as_numpy:
        :param pad_to:
        :return:
        """
        result = [self.vocab[word] if word in self.vocab else OOV_INDEX for word in words]
        if pad_to is not None:
            assert pad_to >= len(words), \
                'padded length {} < real length {}'.format(pad_to, len(result))
            result += [0] * (pad_to - len(result))
        if as_numpy:
            result = np.array(result, dtype=np.int)
        return result

    def encode_sentences(self, sentences, as_numpy=True, pad=True):
        """

        :param sentences: list of sentences, i.e., list of list of tokens (str)
        :param as_numpy:
        :param pad:
        :return:
        """
        if pad:
            max_length = max([len(sentence) for sentence in sentences])
        else:
            max_length = None
        result = [
            self.encode_sentence(sentence, as_numpy=False, pad_to=max_length) for
            sentence in sentences]
        if as_numpy:
            result = np.array(result, dtype=np.int)
        return result

    def decode_sentence(self, ids):
        return [self.reverse_vocab[id] for id in ids]

    def is_in_vocab(self, word):
        return word in self.vocab

    @staticmethod
    def generate_vocab(sentences, vocab_size, add_pad, add_oov,
                       other_symbols, min_freq):
        word_freq = defaultdict(int)
        for sentence in sentences:
            for word in sentence.split():
                word_freq[word] += 1
        ordered_keys = [
            k for k in sorted(word_freq, key=word_freq.get, reverse=True) if
            word_freq.get(k) >= min_freq]
        if vocab_size is not None:
            ordered_keys = ordered_keys[:vocab_size]
        if add_pad:
            ordered_keys.insert(PAD_INDEX, PAD_SYMBOL)
            assert ordered_keys[PAD_INDEX] == PAD_SYMBOL
        if add_oov:
            ordered_keys.insert(OOV_INDEX, OOV_SYMBOL)
            assert ordered_keys[OOV_INDEX] == OOV_SYMBOL
        for symbol in other_symbols:
            ordered_keys.append(symbol)
        reverse_vocab = ordered_keys
        vocab = {}
        for i, word in enumerate(reverse_vocab):
            vocab[word] = i
        return vocab, reverse_vocab
