"""Background music from a licensed library, arranged by chapter.

Why: the old score was synthesised drones and plucks; it sounded amateur and didn't follow the story.
Research (see docs/08 section 22): explainer music should be simple, instrumental, mid-tempo, sit ~12–18 dB
under the voice, and change only at chapter boundaries with smooth crossfades.

Library: Kevin MacLeod (incompetech.com), Creative Commons Attribution 4.0. incompetech's FAQ: "Can I use your
music for YouTube videos? Yes, AND you can monetize the videos. Be sure to credit me." The credit text below is
added to the YouTube description automatically. Tracks were chosen from incompetech's own catalogue data
(feel / bpm / description). Download URLs were checked (HTTP 200) on 2026-10-15.

Pure Python: arrange() and credits() are unit-tested locally (tests/test_musiclib.py).
"""
import urllib.parse

BASE = "https://incompetech.com/music/royalty-free/mp3-royaltyfree/"
LICENSE = "Creative Commons: By Attribution 4.0 https://creativecommons.org/licenses/by/4.0/"

# title -> catalogue facts (from incompetech's pieces.json) + the moods we use it for
TRACKS = {
    "Easy Lemon":     {"bpm": 82,  "bytes": 5057420, "feel": "Bright, Calming, Relaxed",
                       "note": "\"Takes voice-over very well\" (catalogue)", "moods": ["calm", "explain", "curious"]},
    "Wallpaper":      {"bpm": 92,  "bytes": 8804339, "feel": "Bouncy, Bright, Calming, Uplifting",
                       "note": "\"Very very clean mix that is non-offensive\" (catalogue)", "moods": ["explain", "busy", "calm"]},
    "Investigations": {"bpm": 94,  "bytes": 3745876, "feel": "Humorous, Mysterious",
                       "note": "light mystery: hooks and 'where is it?' moments", "moods": ["mystery", "curious", "tension"]},
    "Carefree":       {"bpm": 96,  "bytes": 6566713, "feel": "Bouncy, Bright, Calming, Uplifting",
                       "note": "\"Nothing says 'It'll be okay' quite like a ukulele\" (catalogue)", "moods": ["joy", "payoff", "resolution"]},
    "Inspired":       {"bpm": 120, "bytes": 9156393, "feel": "Bright, Relaxed, Calming, Uplifting",
                       "note": "\"corporate films of intricate industrial processes\" (catalogue)", "moods": ["process", "busy"]},
    "Life of Riley":  {"bpm": 102, "bytes": 7533831, "feel": "Bright, Relaxed, Uplifting",
                       "note": "cheery mid-tempo", "moods": ["joy", "busy"]},
    "Local Forecast - Elevator": {"bpm": 82, "bytes": 7570404, "feel": "Bouncy, Bright, Grooving",
                       "note": "waiting / elevator feel", "moods": ["waiting", "reflective"]},
    "Thinking Music": {"bpm": 125, "bytes": 6270785, "feel": "Dark, Humorous, Suspenseful",
                       "note": "comedic 'hard thinking'", "moods": ["thinking", "tension"]},
}
DEFAULT_BY_MOOD = {"mystery": "Investigations", "curious": "Investigations", "tension": "Investigations",
                   "calm": "Easy Lemon", "explain": "Easy Lemon", "busy": "Wallpaper", "process": "Wallpaper",
                   "waiting": "Local Forecast - Elevator", "reflective": "Easy Lemon", "joy": "Carefree",
                   "payoff": "Carefree", "resolution": "Carefree", "awe": "Easy Lemon", "sad": "Easy Lemon"}


def url(title):
    return BASE + urllib.parse.quote(title) + ".mp3"


def arrange(plan, timeline, total):
    """Music cues: one track per chapter, crossfading at chapter boundaries.

    plan["music"]["cues"]: [{"from": "sh01", "track": "Investigations", "gain_db": 0, "offset": 0}, ...]
    Without cues: a new cue starts wherever the shot's "chapter" changes, using the chapter's mood.
    timeline: [{"shot_id", "start", "dur", "mood"}]. Returns [{"track", "start", "end", "gain_db", "offset"}]."""
    music = plan.get("music", {})
    start_of = {r["shot_id"]: r["start"] for r in timeline}
    shots = {s["shot_id"]: s for s in plan.get("shots", [])}
    cues = []
    if music.get("cues"):
        for c in music["cues"]:
            if c["from"] not in start_of:
                raise ValueError(f"music cue starts at unknown shot {c['from']}")
            if c["track"] not in TRACKS:
                raise ValueError(f"music track '{c['track']}' is not in the licensed library {sorted(TRACKS)}")
            cues.append({"track": c["track"], "start": start_of[c["from"]], "gain_db": c.get("gain_db", 0.0),
                         "offset": c.get("offset", 0.0)})
    else:
        # no cues in the plan: one track per chapter if the plan has chapters, otherwise follow the scene moods;
        # either way a track plays at least MIN_CUE seconds (switching more often feels restless)
        MIN_CUE = 12.0
        runs = []
        for r in timeline:
            sh = shots.get(r["shot_id"], {})
            mood = sh.get("music_mood") or r.get("mood", "calm")
            key = sh.get("chapter") or DEFAULT_BY_MOOD.get(mood, "Easy Lemon")
            if runs and runs[-1]["key"] == key:
                runs[-1]["end"] = r["start"] + r["dur"]
            else:
                runs.append({"key": key, "track": DEFAULT_BY_MOOD.get(mood, "Easy Lemon"), "start": r["start"],
                             "end": r["start"] + r["dur"]})
        merged_runs = []
        for run in runs:
            if merged_runs and run["end"] - run["start"] < MIN_CUE:
                merged_runs[-1]["end"] = run["end"]            # too short: the previous track keeps playing
            else:
                merged_runs.append(run)
        if len(merged_runs) > 1 and merged_runs[0]["end"] - merged_runs[0]["start"] < MIN_CUE:
            merged_runs[1]["start"] = merged_runs[0]["start"]; merged_runs.pop(0)
        cues = [{"track": r["track"], "start": r["start"], "gain_db": 0.0, "offset": 0.0} for r in merged_runs]
    cues.sort(key=lambda c: c["start"])
    merged = []                                         # two chapters in a row with the same track: keep playing
    for c in cues:
        if merged and merged[-1]["track"] == c["track"] and merged[-1]["gain_db"] == c["gain_db"]:
            continue
        merged.append(c)
    for a, b in zip(merged, merged[1:]):
        a["end"] = b["start"]
    if merged:
        merged[0]["start"] = 0.0
        merged[-1]["end"] = total
    return merged


def credits(cues):
    """The attribution text CC BY 4.0 requires (incompetech's format), one block per track used."""
    seen, out = set(), []
    for c in cues:
        if c["track"] not in seen:
            seen.add(c["track"])
            out.append(f"\"{c['track']}\" Kevin MacLeod (incompetech.com)\nLicensed under {LICENSE}")
    return "\n".join(out)
