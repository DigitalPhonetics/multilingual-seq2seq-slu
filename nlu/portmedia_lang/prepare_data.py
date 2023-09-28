#!/usr/bin/env python3

import glob
import os
import sys
import xml.etree.ElementTree as ET
from num2words import num2words

idir = sys.argv[1]
odir = sys.argv[2]

os.makedirs(odir, exist_ok=True)

files = sorted(glob.glob(os.path.join(idir, "PMLANG3IT_00", "PMLANG3IT", "BLOCK*", "*.xml")))

subsets = {
    "train": files[0:304],
    "dev": files[304:404],
    "test": files[404:604],
}

for subset in subsets.keys():
    with open(os.path.join(odir, f"{subset}.source"), "w", encoding="utf-8") as source, open(
        os.path.join(odir, f"{subset}.target"), "w", encoding="utf-8"
    ) as target:
        for file in subsets[subset]:
            with open(file) as f:
                lines = f.readlines()

            if file[-13:] == "08730_305.xml":
                lines[0] = "<" + lines[0]
            elif file[-13:] == "08730_675.xml":
                lines[0] = lines[0][1:]

            content = "".join(lines)

            xml_root = ET.fromstring(content)
            compere = ""

            for speaker in xml_root.findall("./Speakers/Speaker"):
                if speaker.get("name").lower().startswith("compère"):
                    compere = speaker.get("id")
                    break

            for turn in xml_root.findall(".//Turn"):
                if turn.get("speaker") == compere:
                    continue


                words = sum([t.split() for t in turn.itertext()], [])

                for w in range(len(words)):
                    if words[w].isdigit():
                        words[w] = num2words(int(words[w]), lang="it")
                    elif words[w].isdecimal():
                        words[w] = num2words(float(words[w]), lang="it")

                transcription = " ".join(words)
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
