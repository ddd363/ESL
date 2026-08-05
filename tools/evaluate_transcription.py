#!/usr/bin/env python3
"""Evaluate ASR transcripts against human references for the ESL app."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ERROR_TYPES = {"asr_substitution", "asr_deletion", "asr_insertion", "genuine_learner_error"}
WORD_PATTERN = re.compile(r"\b[\w']+\b")


def normalized_words(text):
    return WORD_PATTERN.findall((text or "").lower())


def word_error_counts(reference, hypothesis):
    reference_words = normalized_words(reference)
    hypothesis_words = normalized_words(hypothesis)
    rows = len(reference_words) + 1
    columns = len(hypothesis_words) + 1
    matrix = [[(0, 0, 0, 0) for _ in range(columns)] for _ in range(rows)]

    for row in range(1, rows):
        matrix[row][0] = (row, 0, row, 0)
    for column in range(1, columns):
        matrix[0][column] = (column, 0, 0, column)

    for row in range(1, rows):
        for column in range(1, columns):
            if reference_words[row - 1] == hypothesis_words[column - 1]:
                matrix[row][column] = matrix[row - 1][column - 1]
                continue

            substitution = matrix[row - 1][column - 1]
            deletion = matrix[row - 1][column]
            insertion = matrix[row][column - 1]
            candidates = [
                (substitution[0] + 1, substitution[1] + 1, substitution[2], substitution[3]),
                (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]),
                (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1),
            ]
            matrix[row][column] = min(candidates, key=lambda counts: counts[0])

    errors, substitutions, deletions, insertions = matrix[-1][-1]
    return {
        "reference_words": len(reference_words),
        "errors": errors,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }


def read_csv(path, required_columns):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = required_columns - columns
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        return list(reader)


def load_metadata(path):
    if not path:
        return {}
    with Path(path).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a JSON object.")
    return metadata


def source_transcripts(path):
    rows = read_csv(path, {"clip_id", "transcript"})
    transcripts = {}
    for row in rows:
        clip_id = (row["clip_id"] or "").strip()
        if not clip_id:
            raise ValueError(f"{path} contains a blank clip_id.")
        if clip_id in transcripts:
            raise ValueError(f"{path} contains duplicate clip_id: {clip_id}")
        transcripts[clip_id] = row["transcript"] or ""
    return transcripts


def parse_source(value):
    try:
        name, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Sources must use NAME=PATH.") from error
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Sources must use NAME=PATH.")
    return name.strip(), Path(path.strip())


def summarize_source(manifest, transcripts, annotations, source):
    by_cohort = defaultdict(Counter)
    totals = Counter()
    manifest_ids = {row["clip_id"] for row in manifest}
    missing_ids = manifest_ids - set(transcripts)
    extra_ids = set(transcripts) - manifest_ids
    if missing_ids or extra_ids:
        detail = []
        if missing_ids:
            detail.append(f"missing: {', '.join(sorted(missing_ids))}")
        if extra_ids:
            detail.append(f"unknown: {', '.join(sorted(extra_ids))}")
        raise ValueError(f"{source} predictions do not match manifest ({'; '.join(detail)}).")

    for row in manifest:
        counts = word_error_counts(row["reference_transcript"], transcripts[row["clip_id"]])
        totals.update(counts)
        by_cohort[row["cohort"]].update(counts)

    false_learner_errors = 0
    annotated_substitutions = 0
    for annotation in annotations:
        if annotation["source"] != source:
            continue
        if annotation["error_type"] == "asr_substitution":
            annotated_substitutions += 1
            if annotation["presented_as_learner_error"].strip().lower() == "true":
                false_learner_errors += 1

    def metrics(counter):
        reference_words = counter["reference_words"]
        return {
            **dict(counter),
            "wer": (counter["errors"] / reference_words) if reference_words else None,
        }

    return {
        "overall": metrics(totals),
        "by_cohort": {cohort: metrics(counter) for cohort, counter in sorted(by_cohort.items())},
        "false_learner_error_rate": (
            false_learner_errors / annotated_substitutions if annotated_substitutions else None
        ),
        "false_learner_errors": false_learner_errors,
        "manually_annotated_asr_substitutions": annotated_substitutions,
    }


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--source", action="append", required=True, type=parse_source)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = read_csv(args.manifest, {"clip_id", "cohort", "reference_transcript"})
    annotations = read_csv(
        args.annotations,
        {"clip_id", "source", "error_type", "presented_as_learner_error"},
    )
    manifest_ids = {row["clip_id"] for row in manifest}
    for annotation in annotations:
        if annotation["clip_id"] not in manifest_ids:
            raise ValueError(f"Annotation references unknown clip_id: {annotation['clip_id']}")
        if annotation["error_type"] not in ERROR_TYPES:
            raise ValueError(f"Unsupported error_type: {annotation['error_type']}")
        if annotation["presented_as_learner_error"].strip().lower() not in {"true", "false"}:
            raise ValueError("presented_as_learner_error must be true or false.")

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "manifest_sha256": file_sha256(args.manifest),
            "annotations_sha256": file_sha256(args.annotations),
        },
        "metadata": load_metadata(args.metadata),
        "sources": {},
    }
    for source, path in args.source:
        result["inputs"][f"{source}_predictions_sha256"] = file_sha256(path)
        result["sources"][source] = summarize_source(
            manifest, source_transcripts(path), annotations, source
        )

    output = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()