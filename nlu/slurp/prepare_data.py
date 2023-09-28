#!/usr/bin/env python3

import json
import os
import sys
import re

idir = sys.argv[1]
odir = sys.argv[2]

for subset in ["train", "devel", "test"]:
    os.makedirs(odir, exist_ok=True)

    with open(os.path.join(idir, "dataset", "slurp", subset + ".jsonl")) as meta, open(
        os.path.join(odir, f"{subset}.source"), "w", encoding="utf-8"
    ) as source, open(
        os.path.join(odir, f"{subset}.target"), "w", encoding="utf-8"
    ) as target:
        for line in meta:
            prompt = json.loads(line.strip())
            transcript = prompt["sentence"]
            sentence_annotation = prompt["sentence_annotation"]
            num_entities = sentence_annotation.count("[")
            entities = []
            for slot in range(num_entities):
                ent_type = (
                    sentence_annotation.split("[")[slot + 1]
                    .split("]")[0]
                    .split(":")[0]
                    .strip()
                )
                filler = (
                    sentence_annotation.split("[")[slot + 1]
                    .split("]")[0]
                    .split(":")[1]
                    .strip()
                )
                entities.append({"type": ent_type, "filler": filler})
            sortednames = sorted(entities, key=lambda x: x["type"].lower())
            predict_sent = prompt["scenario"] + "_" + prompt["action"]
            for k in sortednames:
                predict_sent += " SEP " + k["type"] + " FILL " + k["filler"].lower()
            predict_sent += " SEP " + transcript
            words = "{}".format(predict_sent)
            source.write(f"{transcript}\n")
            target.write(f"{words}\n")
