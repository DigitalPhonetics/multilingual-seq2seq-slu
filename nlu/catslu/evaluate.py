#!/usr/bin/env python3

import sys
import re

utts = {"gold": [], "pred": []}
filename = {"gold": sys.argv[1], "pred": sys.argv[2]}

full = "_full" in filename["gold"]

for data in ["gold", "pred"]:
    with open(filename[data]) as f:
        for line in f:
            parts = line.strip().split(" SEP ")

            if full:
                parts = parts[:-1]

            semantic = []

            for part in parts:
                if " FILL " not in part:
                    continue
                act, value = part.split(" FILL ", maxsplit=1)
                act = "".join(act.split()).lower()
                value = "".join(value.split()).lower()
                if "_" in act:
                    act, slot = act.split("_", maxsplit=1)
                    semantic.append([act.strip(), slot.strip(), value.strip()])
                else:
                    semantic.append([act.strip(), value.strip()])
            utts[data].append(set([tuple(i) for i in semantic]))

total_utter_number = 0
correct_utter_number = 0
TP, FP, FN = 0, 0, 0

for anno_utt, pred_utt in zip(utts["gold"], utts["pred"]):
    anno_semantics = set([tuple(item) for item in anno_utt])
    pred_semantics = set([tuple(item) for item in pred_utt])

    total_utter_number += 1
    if anno_semantics == pred_semantics:
        correct_utter_number += 1

    TP += len(anno_semantics & pred_semantics)
    FN += len(anno_semantics - pred_semantics)
    FP += len(pred_semantics - anno_semantics)

print(f"Precision: {100 * TP / (TP + FP)}")
print(f"Recall: {100 * TP / (TP + FN)}")
print(f"F1: {100 * 2 * TP / (2 * TP + FN +FP)}")
print(f"Accuracy: {100 * correct_utter_number / total_utter_number}")
