# 11. Scene-aware expressive narration (2026-10-18)

**Goal:** a narrator who sounds like an engaged human storyteller. Delivery follows the story, while the voice
identity, the words and intelligibility stay fixed. Everything here is free and local, and runs on a Colab T4.

## 1. Audit of the narration system before this work

| Aspect | State | Problem |
|---|---|---|
| Engine | Kokoro-82M (kokoro-onnx), one stock voice per character; Indic Parler-TTS for Telugu | Kokoro has **no expression control**, only speed. |
| "Tones" | per line: speed ±5 %, **pitch shift ±0.6 semitone with `asetrate`**, ±1 dB gain, optional tremolo/"breathy" filters, pause after | Emotion was **faked by DSP**, which the brief forbids. |
| Unit | one request per line (cast) / per sentence (narrator) | fine (1–3 sentences) |
| Checks | Whisper speech check (word/letter error) + retries | no pace, gap, clipping, pitch or **identity** checks; no record of what was actually applied |
| Story awareness | the author's tone label per line | no beat context, no arc, no smoothing |
| Timeline | narration-first (audio lengths set shot lengths) | pauses only *after* a line; no pause *before* a reveal |

## 2. Candidates (licences verified on the Hugging Face API, APIs from the model cards and PyPI)

| Candidate | Licence | Expression control | Fit for this pipeline | Decision |
|---|---|---|---|---|
| **Qwen3-TTS-12Hz-1.7B-CustomVoice** | Apache-2.0 | `instruct` (natural language) ✅; 9 fixed speakers (English natives: **aiden**, **ryan**); 10 languages, no Telugu | `qwen-tts` pins `transformers==4.57.3`, so it runs in an isolated environment (same pattern as the Telugu voice); FlashAttention and bf16 are optional, SDPA works | **Implemented** as the expressive English backend |
| Qwen3-TTS 0.6B CustomVoice | Apache-2.0 | **none** (model card: no instruction control) | same | Used only for CPU plumbing tests |
| CosyVoice-300M-Instruct / Fun-CosyVoice3-0.5B | Apache-2.0 | instruct ✅ | **no pip package** (repo clone, Python 3.10-era dependencies) | Deferred: install too fragile for Colab without a measured gain |
| Chatterbox (original) | MIT | `exaggeration` + `cfg_weight` scalars only, English | `chatterbox-tts` pins **`torch==2.6.0`** and `transformers==5.2.0`, so it needs its own torch (~2.5 GB more); outputs carry a Perth watermark | Deferred: coarser control, heaviest install |
| **Kokoro-82M** | Apache-2.0 | speed only | already integrated, fast, CPU or GPU | **Kept** as the reliable fallback; reported honestly as "pace + pauses only" |

## 3. Measured on this machine (CPU, no GPU: T4 numbers come from `VOICE_BENCH`)

**Qwen3-TTS 1.7B CustomVoice in bf16 on CPU** (speaker `ryan`, same sentence, same seed; only the instruction
changes):

| Take | Length | Pace (words/s) | Median pitch | RMS | Words heard (Whisper) |
|---|---|---|---|---|---|
| no instruction | 3.76 s | 3.28 | 161 Hz | −23.6 dB | exact |
| "quiet, restrained suspense, slowing slightly…" | 5.76 s | 2.14 | 126 Hz | −23.5 dB | exact |
| "genuinely astonished and energetic…" | 3.76 s | 3.28 | 214 Hz | −21.6 dB | **"But" → "Well"** |

- **The instructions work:** suspense is slower and lower; astonishment is higher and louder.
- **Risk 1, identity drift:** ECAPA similarity between these takes of the same speaker is 0.58–0.72, the range
  where different but similar Kokoro voices fall. Strong styles can make the narrator sound like another person.
- **Risk 2, changed words:** the strong excited take changed a word. The word check caught it.
- **Speed (CPU only, not representative):** model load 349 s the first time and 14 s from cache; 238–739 s per take.
  The 0.6B model in fp32 needs 41 s for 4.2 s of audio.

**Narrator identity (ECAPA, SpeechBrain 1.1.2), calibrated on Kokoro** (6 voices, 4 sentences, 2 speeds):
- Chunk-vs-chunk pairs **overlap**: same voice ≥ 0.692, different voices up to 0.759 (af_heart vs af_bella).
- **Leave-one-out against the narrator's other chunks separates cleanly:** same voice ≥ 0.856, closest other voice
  ≤ 0.737. Threshold: **0.80**.
- SpeechBrain 1.0.3 is **broken** on torchaudio ≥ 2.9 (`list_audio_backends` was removed); 1.1.2 works.

**Kokoro A/B on the benchmark script** (hook → setup → build-up → suspense question → reveal → conclusion):
- neutral 25.8 s, expressive 27.4 s (pace and meaningful pauses);
- the same voice in every chunk of both versions (minimum LOO similarity 0.864);
- at most 9 % of words missing per line with Whisper tiny (the word gate is ≤ 15 %).

**Voice QC false alarms found and fixed:** an exact word comparison flagged berth/birth, travelling/traveling,
AM/"a m" and waitlisted/"weight listed". Such substitutions are now letter-compared and recorded as "heard as".
Genuinely missing or repeated words are still caught.

## 4. Decisions

1. **Backend:**
   - `VOICE_BACKEND = "kokoro"` stays the default until `VOICE_BENCH = True` has measured both backends on the
     T4: time, VRAM, words, identity, and a blind A/B listen.
   - `"qwen3-tts"` is ready to switch on for the narrator, with `NARRATOR_QWEN_SPEAKER = "aiden"` or `"ryan"` and
     `QWEN_TTS_DTYPE = "float32"` (the T4 has no fast bf16; fp16 is untested).
   - Other characters keep their Kokoro voices; Telugu keeps Parler.
2. **Expression is never faked with DSP.** With a direction, the tone pitch shift, gain and tremolo are off.
   Character pitch stays, because it is identity.
3. **Intensity is conservative:**
   - profile ranges (e.g. science explainer 0.15–0.6);
   - **`EXPRESSIVE_MAX_INTENSITY = 0.5`** for instruction-following voices, because of the measured identity
     drift and changed word;
   - peaks are rare and dramatic runs are limited.
4. **Identity is measured, not assumed.** After all lines are made, each one is compared with the narrator's other
   chunks (LOO ≥ 0.80). An outlier from an expressive backend gets one calmer retake with a new seed, kept only if
   it is closer to the narrator and the words are right. Otherwise it is flagged for listening.
5. **Honest mapping.** Each line's report says, per field, whether it was handled as `native`, `instruction`,
   `description`, `silence` or `unsupported`. Nothing unsupported is claimed as applied.

## 5. What was built

| Brief's module | Here |
|---|---|
| `voice_direction.py` + `prosody_planner.py` | `engine/lib/voice_direction.py`: `VoiceDirection` (Pydantic: 12 emotions + compatible blends, intensity, pace, energy, emphasis words that must be in the line, pauses before/after, style instruction, transition, role); 6 voice profiles; story beats across shots; direction from the plan → author tone → narrative role and sentence function (never from "!" or keywords); optional local-LLM director with validated JSON; `smooth()` arc rules; neutral baseline |
| `expressive_tts.py` | `engine/lib/expressive_tts.py` (capability table, mapping, report) + `engine/lib/qwen_tts_worker.py` (isolated worker, pinned lock) + `QwenVoice` in the narration cell |
| `voice_consistency.py` + `voice_quality_control.py` | `engine/lib/voice_qc.py`: per-chunk measurements and verdicts; letter-aware word diff; ECAPA `SpeakerCheck` + leave-one-out `consistency()` |
| `narration_assembler.py` | timeline cell (pauses before/after from the direction; 8 ms fades at silent edges; no per-chunk loudness normalisation; final loudness in the render) + `assemble()` for the benchmark |

**Music:** under mostly serious or empathetic narration the music steps back 4 dB. It never rises because the voice
is excited, and the sidechain ducking keeps every word clear.

**Benchmark (`VOICE_BENCH = True`):** the same 6-line script for each backend, neutral vs expressive.
- Audio: `narration_{neutral,expressive}_{kokoro,qwen3-tts}.wav`.
- Reports: `voice_direction.json`, `prosody_report.json`, `voice_qc_report.json` (time, real-time factor, peak VRAM,
  per-chunk checks).
- A **blind A/B** player with the key saved in `blind_test_key.json`.

**Video runs:**
- Without the benchmark: `render/voice_direction.json`, `prosody_report.json`, `voice_qc_report.json`.
- The topic notebook uses the same cast cells, so it gets all of this too.

**Tests:**
- `tests/test_voice.py` (24 checks): rules, smoothing, validation, honest mapping, QC on planted defects, a real
  Kokoro A/B with ECAPA and Whisper.
- `tests/test_voice_backend.py` (6 checks): the `QwenVoice` class with a stand-in worker.
- The end-to-end notebook runs pass (expressive narration; voice benchmark).

## 6. Not verified yet (needs a Colab T4 and people listening)

- **T4 runtime, peak VRAM and fp32-vs-fp16 behaviour** of Qwen3-TTS 1.7B: run `VOICE_BENCH = True`.
- **Perceived naturalness and engagement:** listen to the blind A/B in the benchmark. A larger blind test with
  several listeners is needed before claiming "more engaging". **Retention can only be judged from YouTube
  analytics.**
- **The 0.80 identity threshold for Qwen3-TTS:** it was calibrated on Kokoro. The benchmark's `voice_qc_report.json`
  shows the real distribution for Qwen; adjust `voice_qc.TH` if needed.
- **Expression in Telugu (Parler descriptions):** its model card says emotion is "not extensively tested" for Telugu.

## 7. Addendum: the director's own (milder) instructions on Qwen3-TTS 1.7B, measured

Same sentence, speaker `ryan`, seed 7, bf16 on CPU. The instructions are exactly what `voice_direction.instruction()`
produces at intensity 0.45.

| Take | Length | Pace | Median pitch | Pitch spread | Words heard |
|---|---|---|---|---|---|
| no instruction | 3.76 s | 3.28 w/s | 161 Hz | 3.31 semitones | exact |
| suspenseful 0.45 | 4.40 s | 2.86 w/s | 180 Hz | **5.05 semitones** | exact |
| surprised 0.45 | 4.08 s | 3.05 w/s | 149 Hz | 2.54 semitones | **opening "But" dropped** |

- **Effect size:** moderate and in the intended direction (slower, with a livelier contour for suspense), smaller
  than the strong test instructions. That is the intended size for a narrator.
- **Dropped word:** the "surprised"/"excited" styles dropped the opening "But" in both strengths tried. The word
  check catches it and the line gets the calmer retake. **Watch reveal lines in the benchmark.**
- **Identity:** plain vs styled 0.71–0.74; leave-one-out over these three takes 0.77–0.81. Qwen samples, so its
  take-to-take variation is larger than deterministic Kokoro's. **Thresholds are now per backend
  (`voice_qc.SPEAKER_TH`): Kokoro 0.80 (calibrated), Qwen3-TTS and Parler 0.72 (provisional).** Recalibrate from the
  T4 benchmark.

**Tests at the end of this work:** 12 suites, **252 checks**, all passing (including the online licence re-check).
End-to-end notebook runs pass: expressive Kokoro narration, and `VOICE_BENCH` with Kokoro plus a stand-in Qwen worker.

## 8. Addendum: two friends talking (2026-10-18)

**Finding:** the narrator rules of this document were also being applied to Arjun and Meera.
- Pauses after lines of 450–900 ms gave 0.45–1.18 s gaps between turns.
- Smoothing ran across both speakers.

**Change:** story plans now use `voice_direction.dialogue_directions`.
- Conversational intentions.
- Response gaps of 120–950 ms by situation.
- Pause after = 0.
- Per-speaker smoothing.
- Friend-style Qwen instructions.

**Identity threshold:** now depends on clip length.
- Under 2 s: not judged.
- 2–3.5 s: 0.70.
- 3.5 s and longer: the per-backend threshold.
- Calibrated on 37 dialogue lines: other-voice max 0.18.

**Details:** `docs/13_NATURAL_VOICES_AND_STORYTELLING.md`.
