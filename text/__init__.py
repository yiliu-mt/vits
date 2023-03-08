import re
from text import cleaners
from text.symbols import get_symbol_to_id, get_id_to_symbol

_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


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
  sequence = [_symbol_to_id["@" + symbol] for symbol in cleaned_text.split()]
  return sequence


def sequence_to_text(sequence, symbol_version="default"):
  '''Converts a sequence of IDs back to a string'''
  _id_to_symbol = get_id_to_symbol(symbol_version)
  result = ''
  for symbol_id in sequence:
    s = _id_to_symbol[symbol_id]
    result += s
  return result


def text_to_sequence(text, cleaner_names, symbol_version="v1"):
  """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

  The text can optionally have ARPAbet sequences enclosed in curly braces embedded
  in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

  Args:
    text: string to convert to a sequence
    cleaner_names: names of the cleaner functions to run the text through

  Returns:
    List of integers corresponding to the symbols in the text
  """
  sequence = []
  _symbol_to_id = get_symbol_to_id(symbol_version)

  # Check for curly braces and treat their contents as ARPAbet:
  while len(text):
    m = _curly_re.match(text)

    if not m:
      sequence += _symbols_to_sequence(_clean_text(text, cleaner_names), _symbol_to_id)
      break
    sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names), _symbol_to_id)
    sequence += _arpabet_to_sequence(m.group(2), _symbol_to_id)
    text = m.group(3)

  return sequence


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception("Unknown cleaner: %s" % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(symbols, symbol_to_id):
  return [symbol_to_id[s] for s in symbols if _should_keep_symbol(s, symbol_to_id)]


def _arpabet_to_sequence(text, symbol_to_id):
  # TODO: add @ before the chinese phonemes
  # return _symbols_to_sequence(["@" + s for s in text.split()], symbol_to_id)
  return _symbols_to_sequence([s for s in text.split()], symbol_to_id)


def _should_keep_symbol(s, symbol_to_id):
  return s in symbol_to_id and s != "_" and s != "~"
