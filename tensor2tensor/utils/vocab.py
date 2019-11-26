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
    Creates a vocabulary over an iterable of sentences (a sentence = str).
    Tokenization is just a plain space split.
    """

    def __init__(self, sentences, max_vocab_size=None, add_pad=True,
                 add_unk=True, append_additional_symbols=[], insert_additional_symbols={},
                 min_freq=0):
        """

        :param sentences: iterables over the sentences - every element is a string
        :param max_vocab_size: int
        :param add_pad: bool
        :param add_unk: bool
        :param append_additional_symbols: list of token (str). Every token will be appended
                                            at the end of the vocabulary IF it is NOT already
                                            in the vocab.
        :param min_freq: min frequency for a token to be added to the vocabulary.
        """
        self.vocab, self.reverse_vocab = Vocab._generate_vocab(
            sentences, max_vocab_size, add_pad, add_unk, append_additional_symbols,
            insert_additional_symbols, min_freq)

        logger.info('vocabulary size: {}'.format(len(self.reverse_vocab)))

    def size(self):
        return len(self.reverse_vocab)

    def get_tokens(self):
        return self.reverse_vocab

    def encode_sentence(self, words, as_numpy=True, pad_to=None):
        """
        Encode a sentence.

        :param words: list of tokens - i.e., list of str
        :param as_numpy: return a numpy array/matrix
        :param pad_to: will pad up to this length
        :return:
        """
        result = [self.vocab.get(word, OOV_INDEX) for word in words]
        if pad_to is not None:
            assert pad_to >= len(words), \
                'padded length {} < real length {}'.format(pad_to, len(result))
            result += [0] * (pad_to - len(result))
        if as_numpy:
            result = np.array(result, dtype=np.int)
        return result

    def encode_sentences(self, sentences, as_numpy=True, pad=True):
        """
        Encode a list of sentences. (list of str)

        :param sentences: list of sentences, i.e., list of list of tokens (str)
        :param as_numpy: return a numpy array/matrix
        :param pad: will add pad symbol
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
    def _generate_vocab(sentences, vocab_size, add_pad, add_oov, append_additional_symbols,
                        insert_additional_symbols, min_freq):
        """
        Main code impl. to generate the vocab.
        :return:
        """
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
        for symbol, position in insert_additional_symbols.items():
            if symbol not in ordered_keys:
                ordered_keys.insert(position, symbol)
        for symbol in append_additional_symbols:
            if symbol not in ordered_keys:
                ordered_keys.append(symbol)
        reverse_vocab = ordered_keys
        vocab = {}
        for i, word in enumerate(reverse_vocab):
            vocab[word] = i
        return vocab, reverse_vocab
