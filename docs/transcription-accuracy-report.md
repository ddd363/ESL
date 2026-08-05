# Live Transcription Accuracy — Analysis & Recommendations

Scope: the Deepgram live-streaming path used by the ESL app (`app.py`'s
`DeepgramStreamingClient` + `components/audio_recorder/index.html`), which
feeds both the on-screen live transcript and, when selected, the transcript
used for grammar/IELTS feedback (`analysis_transcript_choice = "Deepgram"`).

**Symptom under investigation:** Deepgram confidently transcribing the
*wrong* word, not audio going missing. That distinction matters — a dropped
network chunk produces a gap or truncation, not a fluent, well-formed wrong
word. The sections below are ordered to put the causes that actually produce
word-substitution errors first; a few structural issues found along the way
(§5-§8) don't target this symptom directly but are included for completeness.

---

## 1. No `model` or `language` pinned — the single biggest lever for substitution errors

**Where:** `app.py:431-441` (`DeepgramStreamingClient._build_url`)

```python
params = {
    "encoding": "linear16",
    "sample_rate": str(self.sample_rate),
    "channels": "1",
    "interim_results": "true",
    "punctuate": "true",
    "smart_format": "true",
    "diarize": "true",
}
```

There is no `model` and no `language` key here. Every stream rides whatever
Deepgram's account-level default model happens to be, which:
- can change without any code change on your side, so accuracy behavior is
  not reproducible over time;
- for **accented, non-native, disfluent speech** — exactly this app's
  population — model choice is the dominant factor in whether the acoustic
  model correctly matches a spoken phoneme sequence to the intended word
  versus a phonetically-similar but wrong common English word. Older/smaller
  models lean harder on their language-model prior (i.e. "which word is most
  likely in English overall"), which is precisely what produces confident
  wrong-word substitutions on unfamiliar accents or word choices — the model
  isn't uncertain, it's just resolving ambiguous acoustics toward the wrong
  familiar word.

**Recommendation:** Explicitly set `model=nova-3` (Deepgram's most accurate
current general model; fall back to `nova-2` if unavailable on the account)
and `language=en` (or `multi` if students code-switch between English and
another language mid-answer — Deepgram's multilingual mode exists
specifically for that case, and forcing `en` on genuinely mixed-language
audio is itself a source of confident wrong-word output). This is the
highest-value single change here: it directly targets acoustic
misrecognition rather than a downstream symptom of it.

## 2. No keyword/keyterm boosting — the model has no bias toward this app's domain vocabulary

Deepgram's general model's word choice is driven by general-English word
frequency. Any word that's uncommon in general speech but common in this
app's domain (IELTS-topic vocabulary, band-descriptor terms, recurring
classroom vocabulary, student/place names) is competing against far more
frequent, phonetically-similar everyday words — and loses by default. This
is a direct, mechanical cause of "confidently wrong word" errors, separate
from accent.

Deepgram supports exactly this correction, currently unused here:
- `nova-2`: `keywords=<term>:<intensity>` query params.
- `nova-3`: `keyterm=<term>` (Keyterm Prompting — more effective on the newer
  model).

**Recommendation:** Maintain a short, curated list of recurring domain terms
and pass them as keyterms/keywords on every connection. Low effort, and it
targets exactly the words most likely to be misidentified and most likely to
matter to the downstream grading (since the OpenAI feedback prompt requires
quoting exact transcript words — a boosted, correctly-recognized domain word
is one fewer place where a transcription error masquerades as a language
error in the final report).

## 3. `smart_format` can rewrite words in ways that read as "misidentified" even when the acoustic recognition was correct

**Where:** `app.py:438` (`smart_format: true`)

`smart_format` post-processes recognized words into formatted forms —
spelling "for" as "4," normalizing dates, currency, etc. For an ESL Speaking
context, a student saying a word in conversational form that Deepgram
reformats can look, in the transcript, exactly like a wrong-word
substitution even though the underlying recognition was correct. This
matters here specifically because the feedback prompt (`DEFAULT_OPENAI_SYSTEM_PROMPT`,
`app.py:37`) treats the transcript as ground truth and quotes it verbatim —
a reformatted token quoted back as "the student's exact words" can produce
feedback about a word the student never actually said in that form.

**Recommendation:** Test with `smart_format=false` (keeping `punctuate=true`
only) on a sample of recordings and compare. If smart formatting isn't
adding clear value for spoken-exam transcripts, turning it off removes a
source of transcript/acoustic mismatch. If some of its behaviors are wanted,
Deepgram exposes finer-grained flags (`numerals`, `dates`, etc.) instead of
the all-or-nothing `smart_format` toggle.

## 4. Confidence scores are computed but discarded — misrecognitions are indistinguishable from certain output

**Where:** `_on_message`, `app.py:557-619` — `alt.get("confidence")` is read
and written only to the diagnostic event log (`app.py:605`); it never
reaches `_format_deepgram_alt_with_speakers` (`app.py:353`), the live
transcript, or the text handed to the OpenAI grading call.

This doesn't reduce the *rate* of misidentified words, but it's the reason
misidentifications are invisible once they happen: every word is presented
with equal confidence, so a low-probability guess reads exactly like a
certain transcription, both to a teacher looking at the transcript and to
the OpenAI prompt that grades it as if it were verified fact.

**Recommendation:** Thread `confidence` per word through into the rendered
transcript (e.g. a subtle underline/highlight below some threshold, similar
in spirit to the existing LanguageTool annotation styling in
`_build_lt_annotated_html`, `app.py:2211`) so low-confidence spans are
visibly flagged rather than presented as fact. At minimum, consider
excluding very-low-confidence words from the strict verbatim-quote
requirement in `DEFAULT_OPENAI_SYSTEM_PROMPT` (`app.py:49`) — grading a
misheard word as a grammar error is a transcription bug wearing a language
bug's clothes.

---

## Secondary findings (structural — not direct causes of word substitution)

These came up during the review and are worth fixing, but they don't
explain "confidently wrong word" errors specifically — they're either about
lost audio (a different symptom) or about maintainability/robustness.

**§5 — Audio is proxied through the Streamlit component value channel.**
Streamlit custom components communicate via a single "latest value wins"
slot, not a queue (`components/audio_recorder/index.html:421` →
`app.py:2614`). A slow rerun can silently overwrite an in-flight audio chunk
before Python reads it — that produces a gap/truncation in the transcript,
not a wrong word. The codebase's heavy defensive instrumentation
(`bytes_sent_total` counters, `DEEPGRAM_NOT_EMITTING_TRANSCRIPTS` hard-fail
log, `check_deepgram_health()` forced restarts) indicates this has been a
real problem before, just a different one than the one asked about here.

**§6 — Deprecated main-thread audio capture, native (non-16kHz) sample rate.**
`ScriptProcessorNode` (`components/audio_recorder/index.html:309`) runs
audio processing on the main thread and is deprecated in favor of
`AudioWorklet`; under thread contention this can produce glitches/dropped
buffers. Audio is also shipped at the browser's native rate (44.1/48 kHz)
rather than downsampled to 16 kHz, which is mostly a bandwidth/§5-risk
concern, not an accuracy one on its own — Deepgram accepts arbitrary rates.

**§7 — `deepgram-sdk` is a declared dependency but unused.** `requirements.txt`
lists it, but `app.py:389-784` hand-rolls the WebSocket protocol with
`websocket-client` instead. Not wrong, but it means protocol nuances the
official SDK already handles (reconnect/backoff, auth refresh, framing
changes) have to be independently maintained here.

**§8 — Diarization enabled on interim results; no endpointing tuning for
ESL pause patterns.** Per-word speaker labels are less stable on
`interim_results` than on finals. Separately, default endpointing assumes
native-speaker pause timing; ESL speakers pausing mid-sentence to retrieve
vocabulary risk Deepgram closing an utterance early, fragmenting a clause
into two segments. This affects segmentation/coherence quality, not
per-word accuracy.

---

## Priority order

| # | Fix | Targets misidentification directly? | Effort | Expected impact |
|---|-----|---|--------|------------------|
| 1 | Pin `model=nova-3`, `language=en`/`multi` explicitly | **Yes** | Low | **Highest** |
| 2 | Keyterm/keyword boosting for domain vocabulary | **Yes** | Low | High |
| 3 | Reassess/disable `smart_format`, use granular flags instead | **Yes** | Low | Medium |
| 4 | Surface confidence scores; exclude low-confidence words from strict-quote grading | **Yes** (visibility, not rate) | Low-Medium | Medium |
| 8 | Endpointing/`utterance_end_ms` tuning for ESL pause patterns | No (segmentation) | Low | Medium (coherence grading) |
| 5 | Fix Streamlit component audio channel (dropped-audio risk) | No (different symptom) | High / Low mitigation | High for the drop symptom |
| 6 | AudioWorklet migration + 16 kHz downsampling | No (mostly mitigates §5) | Medium | Medium |
| 7 | Resolve unused `deepgram-sdk` dependency | No | Low | Low (maintainability) |

## Suggested validation approach

Record a small fixed test set — a few native speakers, a few
accented/ESL speakers, and include the actual domain vocabulary you'd want
boosted — and get a manual reference transcript. Run it through the current
pipeline and count substitution errors specifically (word present in output,
audio wasn't missing, but the word is wrong) versus deletions (word simply
absent). That split confirms whether the fix priority above is right before
and after applying §1-§4, and separates this symptom cleanly from the
dropped-audio risk in §5.
