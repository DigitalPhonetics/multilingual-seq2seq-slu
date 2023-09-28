#!/usr/bin/env python3

import jiwer
import sys

filenames = {"ref": "", "hyp": ""}
filenames["ref"] = sys.argv[1]
filenames["hyp"] = sys.argv[2]

slots = {"concept": {"ref": [], "hyp": []}, "concept_value": {"ref": [], "hyp": []}}

for f in ["ref", "hyp"]:
    with open(filenames[f], encoding="utf-8") as file:
        for line in file:
            if " SEP " in line:
                sem = [x.strip().replace(" ", "_") for x in line.split(" SEP ")[:-1]]
                concept = " ".join([x.split("_FILL_", maxsplit=1)[0] for x in sem])
                concept_value = " ".join(sem)
            else:
                concept = ""
                concept_value = ""
            slots["concept"][f].append(concept)
            slots["concept_value"][f].append(concept_value)

slots_tuples = {"concept": [], "concept_value": []}

for i in range(len(slots["concept"]["ref"])):
    if slots["concept"]["ref"][i] != "":
        slots_tuples["concept"].append((slots["concept"]["ref"][i], slots["concept"]["hyp"][i]))
        slots_tuples["concept_value"].append((slots["concept_value"]["ref"][i], slots["concept_value"]["hyp"][i]))

cer = jiwer.wer([x[0] for x in slots_tuples["concept"]], [x[1] for x in slots_tuples["concept"]])
cver = jiwer.wer([x[0] for x in slots_tuples["concept_value"]], [x[1] for x in slots_tuples["concept_value"]])

print(f"{cer};{cver}")
