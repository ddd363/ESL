# Transcription Evaluation

This harness measures the live Deepgram path and batch Replicate path against
human references without treating unusual learner language as an ASR error.

Keep audio, transcripts, and annotations local. The repository only contains
templates; evaluation data is ignored by Git.

## Prepare Local Data

Copy `manifest.template.csv` to `manifest.csv`. Each clip needs a stable ID,
a cohort (`native`, `esl`, or `code_switching`), and a literal human reference
transcript. The reference must preserve genuine learner errors.

Create one predictions CSV per source with `clip_id,transcript`. Create an
`annotations.csv` file for manual adjudication. Add one row for every reviewed
ASR discrepancy and use `asr_substitution` only when the human reference shows
the student said a different word. Set `presented_as_learner_error=true` only
when that ASR substitution could be treated as a learner error in feedback.

`genuine_learner_error` rows document language errors actually present in the
audio; they are never counted as false learner errors.

## Run

```bash
/Users/spc/miniforge3/bin/conda run -n project1 python tools/evaluate_transcription.py \
  --manifest evaluation/manifest.csv \
  --annotations evaluation/annotations.csv \
  --source deepgram=evaluation/predictions/deepgram.csv \
  --source replicate=evaluation/predictions/replicate.csv \
  --metadata evaluation/run-metadata.json \
  --output evaluation/results/baseline.json
```

Record the model, language, keyterms, endpointing, smart-format setting, and
recording date in `run-metadata.json`. Run baseline and each configuration
change separately. Compare overall and ESL-cohort WER together with
`false_learner_error_rate`; do not select a configuration using WER alone.

The false learner-error metric is intentionally manual. It measures the share
of reviewed ASR substitutions that could be presented to a teacher as a real
student error. It is `null` until substitutions are annotated.