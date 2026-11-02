# ==== 14. YouTube text (paste these into YouTube Studio) ====
md_ = PLAN["metadata"]
sources = sorted({u for c in PLAN.get("claims", []) for u in c["sources"]})
about = md_.get("disclaimer") or ("About this video: the narration is synthetic (text-to-speech) and the visuals are "
                                  "AI-generated illustration.")
desc = ((md_.get("summary") or PLAN["title"]) + "\n\n"
        + ("Sources:\n" + "\n".join(sources) + "\n\n" if sources else "") + about)
if MUSIC_CREDITS:                                    # CC BY 4.0 requires this credit (incompetech's format)
    desc += "\n\nMusic:\n" + MUSIC_CREDITS
meta = {"title": md_["title"], "description": desc, "tags": md_["tags"],
        "category": md_.get("category", "Science & Technology"), "visibility": "Private",
        "altered_or_synthetic_content": "Yes"}
(PROJ / "render" / "metadata.json").write_text(json.dumps(meta, indent=2))
print("TITLE:\n" + meta["title"] + "\n\nDESCRIPTION:\n" + desc + "\n\nTAGS:\n" + ", ".join(meta["tags"]))
print("\nUpload as PRIVATE and answer YES to 'Altered or synthetic content'.")
