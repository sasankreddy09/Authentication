# 14. Natural Telugu dialogue + Gemini free-tier TTS (3D glasses video) (2026-10-19)

**The brief:** `claude.md`, "Natural Telugu Dialogue + Google AI Studio Free-Tier Voice Generation".

**Scope (user instruction):** only the 3D glasses video changes.
- **New:** `colab/stories/three_d_friends_te_v2/`, `colab/three_d_friends_te_v2_plan.json`, `colab/colab_three_d_friends_te_v2.ipynb`.
- **Shared engine:** received only small hooks that do nothing unless a plan asks for them.
- **Other videos:** no other plan or notebook was rebuilt or edited.

**Status words used below:** PASS, FAIL, NOT_TESTED, RESOURCE_UNAVAILABLE, FREE_QUOTA_EXHAUSTED, ACCESS_REQUIRED, MANUAL_REVIEW_REQUIRED, POLICY_BLOCKED.

## 1. Validation of the brief (official Google pages read on 2026-10-19)

| Brief's claim | Official source | Result |
|---|---|---|
| `gemini-3.8-flash-tts` / `-lite-tts` support Telugu + 2 speakers | speech-generation | **Confirmed** (both) |
| `client.interactions.create`, `speech_metadata` (speaker, style), `speech_config.speakers` | speech-generation (code samples) | **Confirmed**, implemented exactly |
| Unary audio is WAV (don't add a 2nd header) | speech-generation | **Confirmed**: WAV, 44-byte RIFF, 24 kHz, mono, 16-bit; raw PCM only when streaming / l16 |
| `google-genai >= 2.25.0` | speech-generation; PyPI | **Confirmed**; latest 2.31.0, Apache-2.0 |
| Standard Free Tier free of charge | pricing | **Confirmed** for Standard and Priority. **Batch/Flex are NOT free** → no service tier is ever set |
| Free tier data use | pricing, terms | "Used to improve our products: **Yes**" on free tier; human review possible; "do not submit sensitive, confidential or personal information" |
| Confirm quotas for the real project | rate-limits | Free-tier TTS limits are **not published per model**; per project, shown in AI Studio; exceeding → 429 RESOURCE_EXHAUSTED |
| Billing status can't be verified from a key | billing | **Confirmed**: no billing-status API documented. The tier is shown in AI Studio → Projects → "Billing Tier" |
| Up to 2 speakers, prebuilt voices | speech-generation | Confirmed; 30 prebuilt voices listed (names only; no official gender list) |
| Terms | terms | 18+ only. Free use is "for developers building … professional or business purposes". Google claims no ownership of output. EEA/UK/CH users need paid services (not applicable to India) |

## 2. Architecture audit (relevant parts before this work)

- **Telugu dialogue:**
  - `stories/three_d_friends_te_story.json` was a translation of the old English story (`source_text` = English claim, `build_story_plan.py --ground`).
  - The topic Story Director refused automatic Telugu stories (English-only NLI).
- **Voices:**
  - `speak()` → Kokoro / Qwen3-TTS / Indic Parler (Telugu, own environment, gated, `HF_TOKEN`).
  - One request per line; per-line sidecars are the cache.
  - The Telugu speech check is multilingual Whisper (letter error ≤ 35 %).
- **Timeline:** narration-first; each line's WAV loudness drives that character's mouth. Separate visual timeline.
- **Gaps in that state:**
  - No Telugu-specific dialogue critic.
  - Pronunciation rules were not per backend (Parler's `3D → త్రీడీ` applied to all).
  - No hosted TTS path, budget or two-speaker splitting.

**Baseline tests before editing:** all suites passing (348 offline + 1 online, from the previous round). The voice and Telugu suites were re-run after the CUDA/Parler fixes: 26/24/6 PASS.

## 3. Telugu dialogue: weaknesses found and the new writer + critic

**Weaknesses of the old Telugu script.** The new critic found 11, and my reading agrees. A native speaker must confirm.

- **Bookish words** (T1):
  - `చదునైన` (flat), `దృశ్యం`, `దిశల`, `ద్వారా`.
  - `చిత్రం / చిత్రాలు` (in conversation people say `బొమ్మ / బొమ్మలు`).
- **Calques** (T8): "ఒక తెలివైన జత గ్లాసెస్" ("a clever pair of glasses") and "సైన్స్ కోసం!" ("for science!").

**New writer** (`engine/lib/telugu_dialogue.py`):
- `SYSTEM`: the brief's Telugu screenplay-writer instructions.
- Wired into the Story Director's dialogue stage when `language == "te"`. Same schema: every factual turn adds `source_text`, the claim in plain English.

**New Telugu critic** (rules, with turn ids):

| Rule | Detects |
|---|---|
| T0 | not Telugu |
| T1 | bookish/translated vocabulary (with spoken alternatives) |
| T2 | formulaic openings |
| T3 | long speeches |
| T4 | too much English |
| T5 | stage directions or tags inside spoken text |
| T6 | the same reaction word repeated (వావ్, అద్భుతం, అరే, నిజంగానా …) |
| T7 | Arjun's questions starting the same way |
| T8 | English idioms translated word for word |

**New script** (`stories/three_d_friends_te_v2/story.json`):
- Written natively from the v2 premise. Arjun ducks and spills popcorn. He guesses "tiny screens", and Meera says "టెస్ట్ చేద్దాం" and has him close one eye. Then projector + filters, circular polarization, and the popcorn callback.
- The critic caught one issue in my own draft (three Arjun questions starting with "అంటే"); it was fixed.
- **Result:** 0 issues; 178 words, ~68 s estimated.
- **Side by side:** `colab/samples/telugu/dialogue_before_after.md`, with a native-speaker rating table.

## 4. Telugu fact verification

**Rule:** the English NLI model never reads Telugu.

1. Each factual Telugu turn carries the canonical English claim (`source_text`). That English text is checked against the cited Wikipedia fact with the existing NLI: **9/9 PASS**.
2. `localization_check(te, en, glossary)`: the Telugu line must contain the Telugu rendering of every key term of the English claim (glossary in the story JSON), and must keep its numbers (none lost or added).
   - Anything else is UNKNOWN, recorded in `plan["review"]`.
   - A required QC check, "Telugu factual lines keep the verified claims", then stays UNKNOWN → **NOT APPROVED (MANUAL_REVIEW_REQUIRED)**.
   - **Result for the 3D script: 5/5 PASS.**

## 5. Gemini free-tier integration (`engine/lib/gemini_tts.py`)

- **Preflight** (no synthesis call; fails closed):
  - enabled;
  - an approved free model (`FREE_TTS_MODELS`);
  - `GEMINI_FREE_TIER_CONFIRMED = True` (manual — a key proves nothing);
  - the key in the Colab secret `GEMINI_API_KEY`;
  - SDK ≥ 2.25.0;
  - the daily video budget;
  - calls left for the video;
  - the model visible to the key (one `models.get` metadata call).
- **Requests:**
  - only the approved Telugu dialogue text plus short style words (privacy);
  - no service tier (never Batch/Flex/Priority);
  - prebuilt voices only;
  - ≤ 2 speakers.
- **Responses:**
  - The WAV is validated (RIFF, non-empty, no double header) and saved **unchanged** in `PROJ/gemini_audio/<hash>.wav`, with its real format.
  - It is resampled only afterwards, for the renderer.
- **Cache:** keyed by model + speakers + exact transcript + styles. An identical transcript is never requested twice.
- **Budget** (local safety limits, not Google's quota):
  - `MAX_VIDEOS_PER_DAY = 1`, `MAX_VIDEO_DURATION_SECONDS = 120`, `MAX_GEMINI_TTS_CALLS_PER_VIDEO = 6`, `MAX_GEMINI_RETRIES_PER_SEGMENT = 1`, `GEMINI_FREE_TIER_ONLY = True`.
  - Ledger: `MyDrive/ai-video/_gemini_usage.json`, recording every request with project, model, speakers, transcript hash, time, status, retry and cache hit.
- **Errors:**
  - 429 is retried once, only when the server asks for a delay of ≤ 60 s; otherwise FREE_QUOTA_EXHAUSTED.
  - 401/403 → ACCESS_REQUIRED; 404 → POLICY_BLOCKED.
  - The key is redacted from every message, log and file.
- **Registry:** `registry.HOSTED_EXCEPTIONS` lists the two models as a **proprietary hosted free-tier TTS exception**, separate from the open models. `hosted_ok(model, use)` allows `tts` only. `LICENSE_STRICT` stays True for everything local.

## 6. Two-speaker speech and per-turn alignment

**Requests:**
- `GEMINI_MODE = "scene"`: consecutive scenes are grouped into as few two-speaker requests as fit (≈ 40 s each), keeping 2 calls of the budget for repairs. The 3D Telugu video's 6 scenes → **4 requests**.
- `"turn"`: one request per line.

**Splitting** (`engine/lib/turn_align.py`):
1. Find the pauses (20 ms frames, > 35 dB under the loud parts, ≥ 120 ms).
2. Work out where each speaker change should be. With the recogniser's word times this comes from the transcript letters (difflib); without them, from each turn's share of letters.
3. Snap each change to the nearest pause.
4. Check the order and that each turn's length is plausible.

If any cut has no pause nearby, the result is **UNRESOLVED**: the chunk is remade per turn, and no timestamp is invented.

**Per-turn checks:**
- The letter error of the words heard inside each turn (same Telugu threshold as Parler, 35 %).
- One per-turn retry (without style) on failure.
- Speaker attribution: each turn's voice embedding must be closest to its own friend's other turns.

**All or nothing:** if any line can't be made reliably, the **whole video uses local Indic Parler**. Arjun never changes voice mid-video.

**Kept from the model's own delivery:**
- The pauses Gemini put between turns are used as the turn gaps; the first turn of each chunk keeps the director's gap.
- Each turn becomes its own `narration/<id>.wav` and sidecar, so mouths, subtitles, identity checks and caching work unchanged.

## 7. Voices for Arjun and Meera

**Gemini (provisional; audition not run — needs your key):**
- Arjun = `Puck`, Meera = `Kore` (the pair used in Google's two-speaker example), stored as `gemini_voice` with `gemini_voice_status: "provisional"` in the Character Bible.
- `TELUGU_VOICE_AUDITION = True` plays 4 candidate pairs on the same Telugu exchange: Puck/Kore, Achird/Sulafat, Fenrir/Aoede, Zubenelgenubi/Vindemiatrix (4 calls).
- Google doesn't document voice genders, so you confirm by listening.

**Parler (unchanged fallback):** Prakash / Lalitha with their descriptions.

**Pronunciation per backend:**
- Parler reads `3D` as `త్రీడీ`.
- Gemini gets the text as written (Gemini's handling of "3D" in Telugu is to be confirmed by listening).
- Subtitles always keep the original text.

## 8. Changed / added files

**New (3D glasses Telugu):**
- `colab/stories/three_d_friends_te_v2/story.json`
- `colab/three_d_friends_te_v2_plan.json`
- `colab/colab_three_d_friends_te_v2.ipynb`
- `colab/scripts/three_d_friends_te_v2_script.md`
- `colab/samples/telugu/dialogue_before_after.md`

**New library modules:**
- `engine/lib/gemini_tts.py`
- `engine/lib/turn_align.py`
- `engine/lib/telugu_dialogue.py`

**New tests:**
- `tests/test_gemini_telugu.py`: 49 checks, all mocked, plus one real-speech check.
- `tests/run_te_gemini_nb.py`: end-to-end harness.

**Additive hooks in shared files** (inactive unless a plan sets `tts_backend: gemini_free` or the settings select it):

| File | Hook |
|---|---|
| `engine/cells/cast/settings.py` | the Telugu/Gemini settings block (default `TELUGU_TTS_BACKEND = "auto"`) |
| `engine/cells/cast/narration.py` | Gemini voice pass, Telugu benchmark/audition, `gemini:` voices in `line_settings` |
| `engine/cells/narrator/narration.py` | `gemini:` route in `speak()` |
| `engine/cells/shared/qc.py` | Telugu localization + Gemini checks |
| `engine/lib/registry.py` | `HOSTED_EXCEPTIONS`, `hosted_ok` |
| `engine/lib/characters.py` | `gemini_voice` fields |
| `engine/lib/story.py` | Telugu writer prompt when `language == "te"` |
| `colab/build_story_plan.py` | Telugu critic + localization + `tts_backend` for non-English stories |

## 9. Tests (executed here, CPU, no API key)

- **`test_gemini_telugu.py`: 49 PASS**, covering:
  - free-tier confirmation; missing key; invalid/paid model; old SDK;
  - daily and per-video budgets; 429 with a short delay, a long delay or none; 403; key redaction;
  - cache reuse; WAV / PCM / double-header / empty; the request shape (speakers, `speech_metadata`, no service tier, no names in the text);
  - chunk planning; splitting; no-pause → UNRESOLVED; implausible → UNRESOLVED;
  - the whole voice pass (19 lines, ≤ 4 requests, gaps kept, per-character audio); per-turn repair; quota → whole-video fallback; unclear attribution → fallback; unheard words → fallback;
  - the Telugu critic (old 11 / new 0, formulas, tags, English share, repeated reactions); localization PASS/UNKNOWN; pronunciation per backend; Bible voices; plan; Gemini scope (no other Google generation API anywhere).
- **Real-speech splitting:** two Kokoro voices in one waveform, Whisper small.en word times → cuts within **0.08 s** of the true turn ends.
- **End to end, Telugu v2 notebook (CPU, draft):**
  - **Stand-in Gemini:**
    - Preflight 8/8 PASS; **4 API calls**, all 19 turns split, attributed and heard.
    - Telugu facts PASS; the Gemini QC checks PASS; video rendered.
    - Two lines failed the identity check only because the stand-in deliberately changes pitch per line.
  - **No API key:** preflight ACCESS_REQUIRED → "Telugu voices: local Indic Parler-TTS" (stand-in worker) → the video is made.

## 10. Not done / not verifiable here

1. **No real Gemini call was made** (no key here). Not measured yet:
   - real API call counts and real Telugu output;
   - the alignment on real Gemini audio;
   - the voice audition;
   - the benchmark (Parler vs Flash scene vs Flash per-turn vs Lite).

   The real path is `TELUGU_VOICE_BENCH = True` / `TELUGU_VOICE_AUDITION = True` in Colab. That makes ≤ 14 calls, within `MAX_GEMINI_BENCH_CALLS`.
2. **No native-speaker listening yet.** Whether Gemini is better than Parler, and whether the new dialogue sounds natural, must be decided by Telugu speakers (the listening sheets are written by the notebook). **Nothing here claims Gemini sounds better.**
3. **Free-tier quota per project is unknown** until AI Studio shows it. The local budget is conservative.
4. **Publishing conditions:** read Google's terms yourself before publishing. Free use is "for developers building … professional or business purposes", 18+, data may be reviewed.
5. **Telugu automatic topic stories:** the writer and critic exist in the library. The topic notebook was not rebuilt (out of scope), so automatic Telugu stories there remain off.
6. **The old Telugu 3D notebook** (`colab_three_d_friends_te_v1.ipynb`) is left as it was.

## 11. How to use (Colab)

1. **Free-tier project:**
   - Open https://aistudio.google.com → API keys. Make a key in a project **without billing**.
   - Check AI Studio → **Projects**: the "Billing Tier" must say **Free**. Never link a billing account to that project.
2. **Secret:** in Colab → 🔑 Secrets, add `GEMINI_API_KEY` with notebook access on. Keep `HF_TOKEN` too: Parler is the fallback.
3. In `colab_three_d_friends_te_v2.ipynb`, set `GEMINI_FREE_TIER_CONFIRMED = True` (only after step 1). Then Run all.
4. **First time, recommended:**
   - Set `TELUGU_VOICE_AUDITION = True`, run, and pick the best pair.
   - Then set `TELUGU_VOICE_BENCH = True` to compare with Parler.
   - Native speakers fill `render/telugu_voice_bench/listening_sheet.md`.
   - Set both flags back to False.
5. **Outputs:**
   - `render/gemini_report.json` (preflight, chunks, alignment, calls)
   - `render/gemini_usage_today.json`
   - `render/qc_report.json`
   - `final.mp4`

## 12. Addendum: REST instead of the google-genai SDK (2026-10-19)

**What went wrong:** on Colab, `pip install "google-genai>=2.25.0"` printed
`google-colab 1.0.0 requires google-auth==2.49.0, but you have google-auth 2.62.0`.

**Why:**
- Every google-genai version with the new TTS support (2.25.0–2.31.0, checked on PyPI) requires `google-auth >= 2.56`.
- Colab's own `google-colab` package pins `google-auth == 2.49.0`, so the conflict can't be avoided in Colab's environment.
- It risks Colab features such as Secrets and Drive mounting.

**Fix:**
- The notebook calls the documented REST form of the same API, with Colab's own `requests`: `POST https://generativelanguage.googleapis.com/v1beta/interactions`, key in the `x-goog-api-key` header, audio read from the `model_output` step.
- **Nothing is installed.** The preflight no longer checks an SDK version.
- Two-speaker requests now include `"mode": "conversational"`, as in Google's official example.

**Tests:** `test_gemini_telugu.py` 56 PASS. The REST checks cover request URL/body, key only in the header, audio extraction, 429 JSON with retryDelay, 403, and a 404 from the model lookup treated as NOT_TESTED rather than blocking.

**If you already ran the pip install:** Runtime → **Disconnect and delete runtime** (restores Colab's google-auth), then open the rebuilt notebook and Run all. Don't run the pip install again.

## 13. Addendum: first real Gemini run, voices chosen (2026-10-19)

**Result:** the user ran `TELUGU_VOICE_AUDITION = True` on Colab. That was the first real free-tier Gemini request, through the REST client, and it succeeded. The user chose **Arjun = Achird, Meera = Sulafat**. Both are saved in the Character Bible (`gemini_voice_status: "auditioned"`); the plan is rebuilt as `plan_version` 2, so Drive's old plan with the provisional Puck/Kore is replaced and backed up.

**Bug found and fixed:**
- **Cause:** audition/benchmark requests are logged under `bench:<project>`, and the daily limit (`MAX_VIDEOS_PER_DAY = 1`) counted that as today's video.
- **Effect:** the real video run would have been refused the same day.
- **Fix:** `bench:` entries are never counted as videos, including entries already in the Drive ledger. Covered by a test.

## 14. Addendum: first full Gemini run spent its budget (2026-10-19)

**What happened on Colab:**
- Preflight PASS, speech check on the GPU.
- Then: "Gemini not used: FREE_QUOTA_EXHAUSTED: local budget: 6 calls for this video used", and a fallback to Parler, which was blocked (HF account `sasank-209` hasn't accepted the model's terms).

**Likely causes:**
1. The real two-speaker ("conversational") audio has no clear pause at some speaker changes, so the splitter refused the chunk and went per line.
2. Whisper's weak Telugu marked lines "not heard" and each got a repair request.

Either way, calls were spent on work that couldn't be finished.

**Improvements:**
- **Budget-aware repairs:** if there aren't enough calls to repair every unclear line, the conversation take is kept and those lines are flagged for listening (`flagged=True` → QC lists them). No calls are spent.
- **Budget-aware fallback:** an unsplittable chunk goes per line only if enough calls remain; otherwise it falls back at once.
- **Quick replies:** with no pause, the cut is made at the quietest point within ±0.15 s of where the recogniser's words change speaker. It is recorded as `cuts_without_pause`, and the speaker check confirms it. Minimum pause is now 80 ms (was 120).
- **Cached audio:** a used-up budget no longer blocks the preflight. Audio already made (cached on Drive) is reused for free; every new request is still refused over the budget.
- **Per-chunk report:** the notebook prints each chunk's result (split / why / cut without pause / repaired / flagged / per line).

**Tests:** `test_gemini_telugu.py` 61 PASS.

## 15. Addendum: v3 makes Gemini opt-in under strict open source (2026-10-20)

- The v3 plan (`three_d_friends_te_v3`) sets `strict_open_source: true` and `tts_backend: "indic_parler"`. Gemini is used only when the user sets `TELUGU_TTS_BACKEND = "gemini_free"`.
- The v2 notebook is unchanged: it was not rebuilt, and its plan still says `gemini_free` with `auto`.
- Kept as before:
  - REST client (no SDK);
  - Achird / Sulafat voices;
  - confirmed Free tier;
  - 6 calls per video;
  - scene-level cached requests;
  - quota exhaustion ends in fallback or a stop, never a paid route.
- New in `run_voices`:
  - `asr_repairs=False` (v3 voice policy): lines the recogniser doubts are kept and flagged for listening; no repair request is made.
  - `err_fn`: the bilingual comparison for code-mixed text.
- The transcript form (`display` vs `telugu_script`) is decided by `TELUGU_PRON_BENCH`, which costs 2 calls from the separate benchmark budget.
