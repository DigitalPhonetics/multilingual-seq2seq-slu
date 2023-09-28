#!/usr/bin/env python3

import sys
from transformers import AutoTokenizer

idir = sys.argv[1]

tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

langs = {x.split("_")[0]: x for x in tokenizer.lang_code_to_id.keys()}

with open(f"{idir}/utt2spk") as utt2spk, open(f"{idir}/utt2lid", "w") as utt2lid:
    for line in utt2spk:
        utt = line.split()[0]

        if "common_voice" in utt:
            lang = utt[142:144]
        else:
            lang = "zh"

        utt2lid.write(f"{utt} {tokenizer.lang_code_to_id[langs[lang]]}\n")
