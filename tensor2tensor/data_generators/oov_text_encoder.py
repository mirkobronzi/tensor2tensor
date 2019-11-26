from tensor2tensor.data_generators.text_encoder import TokenTextEncoder


class OOVTokenTextEncoder(TokenTextEncoder):

  def encode(self, s):
    """Converts a space-separated string of tokens to a list of ids."""
    sentence = s
    tokens = sentence.strip().split()
    if self._replace_oov is not None:
      tokens = [t if t in self._token_to_id else self._replace_oov
                for t in tokens]
    ret = [self._token_to_id.get(tok, 2) for tok in tokens]
    return ret[::-1] if self._reverse else ret