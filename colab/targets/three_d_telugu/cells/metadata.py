# ==== 14. YouTube text (paste these into YouTube Studio) ====
md_ = PLAN["metadata"]
voices = ", ".join(f"{c['name']} ({c['voice']})" for c in CHARS.values())
sources = sorted({u for c in PLAN.get("claims", []) for u in c["sources"]})
desc = (md_["summary"] + "\n\n"
        + ("Sources:\n" + "\n".join(sources) + "\n\n" if sources else "")
        + (md_["disclaimer"] + "\n" if md_.get("disclaimer") else "")
        + f"Voices: Indic Parler-TTS stock Telugu speakers: {voices}.\n"
        "About this video: the voices are AI-generated (synthetic)" + ("" if all(sh.get("recipe") in ("card", "stock", "ui", "journey", "diagram") for sh in SHOTS) else " and so is the animation") + ". "
        "No real people or voices were imitated.")
if md_.get("image_credits"):                         # CC BY images: title, author, source, licence (TASL)
    desc += "\n\nImages (Wikimedia Commons):\n" + "\n".join(md_["image_credits"])
if MUSIC_CREDITS:                                    # CC BY 4.0 requires this credit (incompetech's format)
    desc += "\n\nMusic:\n" + MUSIC_CREDITS
meta = {"title": md_["title"], "description": desc, "tags": md_["tags"],
        "category": md_.get("category", "Film & Animation"), "visibility": "Private",
        "altered_or_synthetic_content": "Yes",
        "made_for_kids": md_.get("audience", "Decide honestly in YouTube Studio")}
(PROJ / "render" / "metadata.json").write_text(json.dumps(meta, indent=2))
print("TITLE:\n" + meta["title"] + "\n\nDESCRIPTION:\n" + desc + "\n\nTAGS:\n" + ", ".join(meta["tags"]))
print(f"\nUpload as PRIVATE, answer YES to 'Altered or synthetic content'. Audience: {meta['made_for_kids']}.")
