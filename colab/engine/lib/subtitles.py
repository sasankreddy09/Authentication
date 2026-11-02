"""Word-highlighted ("karaoke") subtitles as an ASS file, burned in by FFmpeg/libass in the final encode — no frame
is drawn in Python. Each word lights up when it is spoken (timed from the speech recogniser's word timestamps);
numbers and key terms are emphasised. Short cues, safe margins, sizes relative to the frame (any resolution).
For scripts libass may not shape (Telugu), the notebook draws subtitles with Pillow instead (render cell).
"""
from __future__ import annotations

import re


def ass_time(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    h, cs = divmod(cs, 360000); m, cs = divmod(cs, 6000); s, cs = divmod(cs, 100)
    return f"{h}:{m:02}:{s:02}.{cs:02}"


def ass_color(hex_rgb: str, alpha: int = 0) -> str:
    """'#RRGGBB' → ASS &HAABBGGRR."""
    h = hex_rgb.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def _esc(w: str) -> str:
    return w.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def write_ass(path, cues, W, H, font="DejaVu Sans", colors=None, emphasis=(), accent="#FFC43D", dim="#BFC6D0"):
    """cues: [(start, end, [(word, word_start_s), ...], speaker)]. Words already spoken are shown in the speaker's
    colour (white for the narrator), words still to come slightly dimmed; emphasised words (numbers, key terms)
    in the accent colour and bold. Returns the number of dialogue lines."""
    colors = colors or {}
    size, outline, margin = round(H * 0.055), max(2, round(H / 360)), round(H * 0.09)
    emph = {e.lower() for e in emphasis}
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},{ass_color('#FFFFFF')},{ass_color(dim)},{ass_color('#000000')},{ass_color('#000000', 0x80)},-1,0,0,0,100,100,0,0,1,{outline},0,2,{round(W * 0.06)},{round(W * 0.06)},{margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for start, end, words, speaker in cues:
        if not words:
            continue
        col = colors.get(speaker)
        parts = [f"{{\\1c{ass_color(col)}}}" if col and col.upper() != "#FFFFFF" else ""]
        t_prev = start
        for i, (w, ws) in enumerate(words):
            ws = min(max(ws, t_prev), end)
            if i == 0 and ws > start:                    # silence before the first word: nothing lit yet
                parts.append(f"{{\\k{round((ws - start) * 100)}}}")
            nxt = words[i + 1][1] if i + 1 < len(words) else end
            dur = max(1, round((max(nxt, ws) - ws) * 100))
            key = re.sub(r"[^\w]", "", w.lower())
            if key in emph or re.search(r"\d", w):
                parts.append(f"{{\\k{dur}\\1c{ass_color(accent)}\\b1}}{_esc(w)}{{\\1c{ass_color(col or '#FFFFFF')}}} ")
            else:
                parts.append(f"{{\\k{dur}}}{_esc(w)} ")
            t_prev = ws
        lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,{speaker},0,0,0,,{''.join(parts).rstrip()}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(head + "\n".join(lines) + "\n")
    return len(lines)
