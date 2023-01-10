from text.symbols import get_symbols, get_symbol_to_id, get_id_to_symbol



def text_to_symbol(text, symbol_version="default"):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  _symbol_to_id = get_symbol_to_id(symbol_version)
  sequence = [_symbol_to_id[symbol] for symbol in text.split()]
  return sequence


def cleaned_text_to_sequence(cleaned_text, symbol_version="default"):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  _symbol_to_id = get_symbol_to_id(symbol_version)
  sequence = [_symbol_to_id[symbol] for symbol in cleaned_text.split()]
  return sequence


def sequence_to_text(sequence, symbol_version="default"):
  '''Converts a sequence of IDs back to a string'''
  _id_to_symbol = get_id_to_symbol(symbol_version)
  result = ''
  for symbol_id in sequence:
    s = _id_to_symbol[symbol_id]
    result += s
  return result
