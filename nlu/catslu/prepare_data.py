#!/usr/bin/env python3

# Copyright 2021  Sujay Suresh Kumar
#           2021  Carnegie Mellon University
# Apache 2.0

import os
import sys
from pathlib import Path
import json
import string as string_lib

if len(sys.argv) != 3:
    print("Usage: python data_prep.py [catslu_root] [output_dir]")
    sys.exit(1)

catslu_root = sys.argv[1]
odir = sys.argv[2]

BLACKLIST_IDS = ["map-df61ee397d015314dfde80255365428b_4b3d3b2f332793052a000014-1"]


catslu_root_path = Path(catslu_root)

catslu_traindev = Path(os.path.join(catslu_root_path, "catslu_traindev", "data"))
catslu_traindev_domain_dirs = [
    f for f in catslu_traindev.iterdir() if f.is_dir()  # and f.name == "map"
]

catslu_test = Path(os.path.join(catslu_root_path, "catslu_test", "data"))
catslu_test_domain_dirs = [
    f for f in catslu_test.iterdir() if f.is_dir()  # and f.name == "map"
]


def _process_data(data):
    global BLACKLIST_IDS
    sources = []
    targets = []
    for dialogue in data:
        for utterance in dialogue["utterances"]:
            utt_id = "{}-{}-{}".format(
                domain_dir.parts[-1], utterance["wav_id"], utterance["utt_id"]
            )

            manual_transcript = utterance["manual_transcript"]
            manual_transcript = manual_transcript.replace("(unknown)", "")
            manual_transcript = manual_transcript.replace("(side)", "")
            manual_transcript = manual_transcript.replace("(dialect)", "")
            manual_transcript = manual_transcript.replace("(robot)", "")
            manual_transcript = manual_transcript.replace("(noise)", "")

            if utt_id in BLACKLIST_IDS or manual_transcript == "":
                continue

            transcript = []

            for semantic in utterance["semantic"]:
                if len(semantic) == 2:
                    transcript.append(f"{semantic[0]} FILL {semantic[1]}")
                elif len(semantic) == 3:
                    transcript.append(f"{semantic[0]}_{semantic[1]} FILL {semantic[2]}")

            # transcript.append(manual_transcript)

            sources.append(manual_transcript + "\n")
            if len(transcript) > 0:
                targets.append(" SEP ".join(transcript) + "\n")
            else:
                targets.append("SEP\n")

    return sources, targets


os.makedirs(odir, exist_ok=True)

with open(
    os.path.join(odir, "train.source"), "w", encoding="utf-8"
) as train_source_f, open(
    os.path.join(odir, "train.target"), "w", encoding="utf-8"
) as train_target_f, open(
    os.path.join(odir, "val.source"), "w", encoding="utf-8"
) as dev_source_f, open(
    os.path.join(odir, "val.target"), "w", encoding="utf-8"
) as dev_target_f:

    for domain_dir in catslu_traindev_domain_dirs:
        with open(os.path.join(domain_dir, "train.json")) as fp:
            train_data = json.load(fp)

        train_sources, train_targets = _process_data(train_data)

        train_source_f.writelines(train_sources)
        train_target_f.writelines(train_targets)

        with open(os.path.join(domain_dir, "development.json")) as fp:
            dev_data = json.load(fp)

        dev_sources, dev_targets = _process_data(dev_data)

        dev_source_f.writelines(dev_sources)
        dev_target_f.writelines(dev_targets)

with open(
    os.path.join(odir, "test.source"), "w", encoding="utf-8"
) as test_source_f, open(
    os.path.join(odir, "test.target"), "w", encoding="utf-8"
) as test_target_f:
    for domain_dir in catslu_test_domain_dirs:
        with open(os.path.join(domain_dir, "test.json")) as fp:
            test_data = json.load(fp)

        test_sources, test_targets = _process_data(test_data)

        test_source_f.writelines(test_sources)
        test_target_f.writelines(test_targets)
