#!/usr/bin/env python3

import os
import sys
import xml.etree.ElementTree as ET

files = {
    "train": ["lot1", "lot2", "lot3", "lot4"],
    "val": ["testHC_a_blanc"],
    "test": ["testHC"],
}

idir = sys.argv[1]
odir = sys.argv[2]

os.makedirs(odir, exist_ok=True)

data_dir = os.path.join(idir, "MEDIA1FR", "DATA")

for subset in files.keys():
    with open(
        os.path.join(odir, f"{subset}.source"), "w", encoding="utf-8"
    ) as source, open(
        os.path.join(odir, f"{subset}.target"), "w", encoding="utf-8"
    ) as target:
        for part in files[subset]:
            xml_root = ET.parse(os.path.join(data_dir, f"media_{part}.xml")).getroot()
            for turn in xml_root.findall(".//turn[@speaker='spk']"):
                turn_id = turn.get("id")

                slots = []

                for sem in turn.findall("./semAnnotation[@withContext='false']/sem"):
                    # xpath sem[@mode!='null'] does not work for some reason
                    if sem.get("mode") == "null":
                        continue
                    specifier = sem.get("specif")
                    if specifier == "Relative-reservation":
                        specifier = "-relative-reservation"
                    concept = sem.get("concept") + specifier
                    value = sem.get("value")
                    if value == "":
                        slots.append(concept)
                    else:
                        slots.append(f"{concept} FILL {value}")

                transcription = "".join(
                    [x.strip() for x in turn.find("./transcription").itertext()]
                )
                transcription = transcription.replace("(", "")
                transcription = transcription.replace(")", "")
                transcription = transcription.replace("*", "")

                if transcription == "":
                    continue

                source.write(f"{transcription}\n")
                target.write(f"{' SEP '.join(slots + [transcription])}\n")
