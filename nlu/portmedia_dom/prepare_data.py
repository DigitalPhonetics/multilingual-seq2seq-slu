#!/usr/bin/env python3

import glob
import os
import sys
import xml.etree.ElementTree as ET

idir = sys.argv[1]
odir = sys.argv[2]

os.makedirs(odir, exist_ok=True)

files = sorted(glob.glob(os.path.join(idir, "PMDOM2FR_00", "PMDOM2FR", "BLOCK*", "*.xml")))

subsets = {
    "train": files[0:400],
    "dev": files[400:500],
    "test": files[500:700],
}

for subset in subsets.keys():
    with open(os.path.join(odir, f"{subset}.source"), "w", encoding="utf-8") as source, open(
        os.path.join(odir, f"{subset}.target"), "w", encoding="utf-8"
    ) as target:
        for file in subsets[subset]:
            xml_root = ET.parse(file).getroot()
            compere = ""

            for speaker in xml_root.findall("./Speakers/Speaker"):
                if speaker.get("name").lower().startswith("compère"):
                    compere = speaker.get("id")
                    break

            for turn in xml_root.findall(".//Turn"):
                if turn.get("speaker") == compere:
                    continue

                transcription = " ".join(sum([t.split() for t in turn.itertext()], []))
                transcription = transcription.replace("(", "")
                transcription = transcription.replace(")", "")
                transcription = transcription.replace("*", "")

                if transcription == "":
                    continue

                slots = [
                    e.get("concept") + " FILL " + e.get("valeur")
                    for e in turn.findall("./SemDebut")
                    if e.get("concept") != "null"
                ]

                source.write(f"{transcription}\n")
                target.write(f"{' SEP '.join(slots + [transcription])}\n")
