#!/usr/bin/env python3

import argparse
import json
import logging
import re
import sys

sys.path.append("/resources/asr-data/slurp/scripts/evaluation")

from metrics import ErrorMetric
from metrics.distance import CharDistance
from util import format_results

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)

logger = logging.getLogger(__name__)


def example_is_valid(example, intents_set, slots_set):
    line = example["line"]
    intent, sentence = "", ""

    if "|" in line:
        intent, sentence = line.split("|", maxsplit=1)
    else:
        return False

    if len(intent) == 0 or len(sentence) == 0:
        return False

    in_slot = False
    for c in sentence:
        if c == "[":
            if in_slot == False:
                in_slot = True
            else:
                return False
        elif c == "]":
            if in_slot == True:
                in_slot = False
            else:
                return False

    if in_slot == True:
        return False

    if example["intent"] not in intents_set:
        return False

    for entity in example["entities"]:
        if entity["type"] not in slots_set:
            return False

    return True


def parse_examples(filenames):
    examples = {"pred": {}, "gold": {}}

    for data in ["gold", "pred"]:
        with open(filenames[data]) as f:
            i = 0
            for line in f:
                i += 1

                line = line.strip()

                intent = line.split()[0]
                entities = []

                for entity in line.split(" SEP ")[1:-1]:
                    if len(entity.split(" FILL")) != 2:
                        continue
                    ent_type = entity.split(" FILL")[0].strip()
                    ent_val = entity.split(" FILL")[1].strip().replace(" ", "")
                    ent_val = ent_val.replace("▁", " ").strip().replace("'", "'")
                    entities.append({"type": ent_type, "filler": ent_val})

                i_str = str(i)
                examples[data][i_str] = {"scenario": "", "intent": intent, "entities": entities}

    return examples


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SLURPS evaluation script modified to work with plain text input")
    parser.add_argument("ref", required=True, type=str, help="Reference file")
    parser.add_argument("hyp", type=str, required=True, help="Hypothesis file")
    parser.add_argument("--average", type=str, default="micro", help="The averaging modality {micro, macro}.")
    parser.add_argument("--full", action="store_true", help="Print the full results, including per-label metrics.")
    parser.add_argument("--errors", action="store_true", help="Print TPs, FPs, and FNs in each row.")
    parser.add_argument(
        "--table-layout",
        type=str,
        default="fancy_grid",
        help="The results table layout {fancy_grid (DEFAULT), csv, tsv}.",
    )
    parser.add_argument(
        "--n-best",
        type=str,
        default="first",
        choices=["first", "valid", "oracle"],
        help="Method to select a hypothesis from n-best.",
    )

    args = parser.parse_args()

    # logger.info("Loading data")
    filenames = {"pred": args.hyp, "gold": args.ref}

    examples = parse_examples(filenames)

    intents_set = set()
    slots_set = set()

    if args.n_best == "valid":
        for example in examples["gold"]:
            intents_set.add(example["intent"])
            for entity in example["entities"]:
                slots_set.add(entity["type"])

    n_gold_examples = len(examples["gold"])
    n_pred_examples = len(examples["pred"])
    assert n_pred_examples % n_gold_examples == 0
    n_best = int(n_pred_examples / n_gold_examples)
    char_distance = CharDistance()
    examples_pred_selected = dict()

    if args.n_best == "first":
        for gold_id in list(examples["gold"]):
            pred_index = (int(gold_id) - 1) * n_best + 1
            examples_pred_selected[gold_id] = examples["pred"][str(pred_index)]
    elif args.n_best == "valid":
        for gold_id in list(examples["gold"]):
            start = (int(gold_id) - 1) * n_best + 1
            for i in range(start, start + n_best):
                pred_example = examples["pred"][str(i)]
                if example_is_valid(pred_example, intents_set, slots_set) or i == start + n_best:
                    del pred_example["line"]
                    examples_pred_selected[gold_id] = pred_example
                    break
    elif args.n_best == "oracle":
        for gold_id in list(examples["gold"]):
            gold_example = json.dumps(examples["gold"][gold_id])
            min_char_distance = 1000000.0
            start = (int(gold_id) - 1) * n_best + 1
            for i in range(start, start + n_best):
                pred_example = json.dumps(examples["pred"][str(i)])
                example_char_distance = char_distance(gold_example, pred_example)
                if example_char_distance < min_char_distance:
                    min_char_distance = example_char_distance
                    examples_pred_selected[gold_id] = examples["pred"][str(i)]

    examples["pred"] = examples_pred_selected

    # logger.info("Initializing metrics")
    intent_f1 = ErrorMetric.get_instance(metric="f1", average=args.average)
    full_f1 = ErrorMetric.get_instance(metric="f1", average=args.average)
    span_f1 = ErrorMetric.get_instance(metric="span_f1", average=args.average)
    distance_metrics = {}
    for distance in ["word", "char"]:
        distance_metrics[distance] = ErrorMetric.get_instance(
            metric="span_distance_f1", average=args.average, distance=distance
        )
    slu_f1 = ErrorMetric.get_instance(metric="slu_f1", average=args.average)

    for gold_id in list(examples["gold"]):
        if gold_id in examples["pred"]:
            gold_example = examples["gold"].pop(gold_id)
            pred_example = examples["pred"].pop(gold_id)
            intent_f1(gold_example["intent"], pred_example["intent"])
            span_f1(gold_example["entities"], pred_example["entities"])
            full_f1(json.dumps(gold_example), json.dumps(pred_example))
            for distance, metric in distance_metrics.items():
                metric(gold_example["entities"], pred_example["entities"])

    # logger.info("Results:")
    results = full_f1.get_metric()
    print(
        format_results(
            results=results, label="exact match", full=args.full, errors=args.errors, table_layout=args.table_layout
        ),
        "\n",
    )

    results = intent_f1.get_metric()
    print(
        format_results(
            results=results, label="intent", full=args.full, errors=args.errors, table_layout=args.table_layout
        ),
        "\n",
    )

    results = span_f1.get_metric()
    print(
        format_results(
            results=results, label="entities", full=args.full, errors=args.errors, table_layout=args.table_layout
        ),
        "\n",
    )

    for distance, metric in distance_metrics.items():
        results = metric.get_metric()
        slu_f1(results)
        print(
            format_results(
                results=results,
                label="entities (distance {})".format(distance),
                full=args.full,
                errors=args.errors,
                table_layout=args.table_layout,
            ),
            "\n",
        )
    results = slu_f1.get_metric()
    print(
        format_results(
            results=results, label="SLU F1", full=args.full, errors=args.errors, table_layout=args.table_layout
        ),
        "\n",
    )

    # logger.warning("Gold examples not predicted: {} (out of {})".format(len(examples["gold"]), n_gold_examples))
