#!/usr/bin/env python3

import os
import re
import sys

import pandas as pd

idir = sys.argv[1]
odir = sys.argv[2]

os.makedirs(odir, exist_ok=True)

dir_dict = {
    "train": "slue-voxpopuli_fine-tune.tsv",
    "val": "slue-voxpopuli_dev.tsv",
    "test": "slue-voxpopuli_dev.tsv",  # output labels for test are not public, so we use devel
}

ontonotes_to_combined_label = {
    "GPE": "PLACE",
    "LOC": "PLACE",
    "CARDINAL": "QUANT",
    "MONEY": "QUANT",
    "ORDINAL": "QUANT",
    "PERCENT": "QUANT",
    "QUANTITY": "QUANT",
    "ORG": "ORG",
    "DATE": "WHEN",
    "TIME": "WHEN",
    "NORP": "NORP",
    "PERSON": "PERSON",
    "LAW": "LAW",
}

for x in dir_dict:
    with open(os.path.join(odir, f"{x}.source"), "w", encoding="utf-8") as source, open(
        os.path.join(odir, f"{x}.target"), "w", encoding="utf-8"
    ) as target:
        transcript_df = pd.read_csv(os.path.join(idir, dir_dict[x]), sep="\t")
        for row in transcript_df.values:
            transcript = row[2]
            entities = []
            if str(row[6]) != "None":
                for slot in row[6].split("], "):
                    ent_type = (
                        slot.split(",")[0]
                        .replace("[", "")
                        .replace("]", "")
                        .replace('"', "")
                        .replace("'", "")
                    )
                    if ent_type in ontonotes_to_combined_label:
                        ent_type = ontonotes_to_combined_label[ent_type]
                    else:
                        continue
                    fill_start = int(
                        slot.split(",")[1]
                        .replace("[", "")
                        .replace("]", "")
                        .replace('"', "")
                        .replace("'", "")
                        .replace(" ", "")
                    )
                    fill_len = int(
                        slot.split(",")[2]
                        .replace("[", "")
                        .replace("]", "")
                        .replace('"', "")
                        .replace("'", "")
                        .replace(" ", "")
                    )
                    filler = transcript[fill_start : fill_start + fill_len]
                    entities.append(
                        {
                            "type": ent_type,
                            "filler": filler,
                            "filler_start": fill_start,
                            "filler_end": fill_start + fill_len,
                        }
                    )
            new_transcript = transcript[:]
            for entity in entities:
                new_transcript = (
                    new_transcript[: entity["filler_start"]]
                    + entity["type"]
                    + " FILL "
                    + entity["filler"]
                    + " SEP "
                    + new_transcript[entity["filler_end"] :]
                )

            source.write(re.sub(r"\s+", " ", transcript) + "\n")
            target.write(re.sub(r"\s+", " ", new_transcript) + "\n")
