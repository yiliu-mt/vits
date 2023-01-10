from text import cmudict, pinyin, pinyin_v2

_pad = "_"
_pause = ["sil", "eos", "sp", "#0", "#1", "#2", "#3", "#4", "sp0", "sp1", "sp2", "sp3"]
_initials = [
    "^",
    "b",
    "c",
    "ch",
    "d",
    "f",
    "g",
    "h",
    "j",
    "k",
    "l",
    "m",
    "n",
    "p",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "x",
    "z",
    "zh",
    # from aishell3
    "y",
    "w",
    "rr",
]
_tones = ["1", "2", "3", "4", "5"]
_finals = [
    "a",
    "ai",
    "an",
    "ang",
    "ao",
    "e",
    "ei",
    "en",
    "eng",
    "er",
    "i",
    "ia",
    "ian",
    "iang",
    "iao",
    "ie",
    "ii",
    "iii",
    "in",
    "ing",
    "iong",
    "iou",
    "o",
    "ong",
    "ou",
    "u",
    "ua",
    "uai",
    "uan",
    "uang",
    "uei",
    "uen",
    "ueng",
    "uo",
    "v",
    "van",
    "ve",
    "vn",
    "io",
]
symbols_default = [_pad] + _pause + _initials + [i + j for i in _finals for j in _tones]


_special = "-"
_punctuation = "!'(),.:;? "
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["sp", "spn", "sil", "#0", "#1", "#2", "#3", "#4"]
_silences_v2 = ["sp0", "sp1", "sp2", "sp3"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = cmudict.valid_symbols
_pinyin = pinyin.valid_symbols
_pinyin_v2 = pinyin_v2.valid_symbols

symbols_v2_1 = (
        [_pad]
        + list(_special)
        + list(_punctuation)
        + list(_letters)
        + _silences
        + _silences_v2
        + _arpabet
        + _pinyin
        + _pinyin_v2
)


# Mappings from symbol to numeric ID and vice versa:
symbol_to_id_default = {s: i for i, s in enumerate(symbols_default)}
id_to_symbol_default = {i: s for i, s in enumerate(symbols_default)}

symbol_to_id_v2_1 = {s: i for i, s in enumerate(symbols_v2_1)}
id_to_symbol_v2_1 = {i: s for i, s in enumerate(symbols_v2_1)}


def get_symbols(symbol_version):
    if symbol_version == "default":
        return symbols_default
    elif symbol_version == "v2.1":
        return symbols_v2_1
    else:
        raise NotImplemented(f"Symbol version {symbol_version} not implemented")
        

def get_symbol_to_id(symbol_version):
    if symbol_version == "default":
        return symbol_to_id_default
    elif symbol_version == "v2.1":
        return symbol_to_id_v2_1
    else:
        raise NotImplemented(f"Symbol version {symbol_version} not implemented")


def get_id_to_symbol(symbol_version):
    if symbol_version == "default":
        return id_to_symbol_default
    elif symbol_version == "v2.1":
        return id_to_symbol_v2_1
    else:
        raise NotImplemented(f"Symbol version {symbol_version} not implemented")
    