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

# TODO(Yi Liu): the symbol should be updated
symbols = [_pad] + _pause + _initials + [i + j for i in _finals for j in _tones]
