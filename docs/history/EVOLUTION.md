# How the repository evolved

This account separates Git evidence from dates written inside design documents. Most video material arrived in bulk commits, so document dates do not establish that each described stage was separately implemented and tested on that date.

| Period / evidence | Architecture or behavior |
|---|---|
| Early Git history, `462fd1c` through `e0e0cb1` | Create Next App, Next.js authentication/dashboard, Supabase and website styling. This was a separate application, not the video engine. |
| Early research and the legacy Claude prompt | Broad topic-to-video automation, model research and runtime-generated-code ideas. See `ai_video_automation_research.md` and `../prompts/claude-legacy.md`. |
| Oct 11 design documents | Fixed Python executor consuming validated JSON replaced runtime code generation as the design principle. Voice-first timing, bounded short shots, caches and staged checks were planned. The archived architecture's “no implementation yet” describes this historical stage. |
| Later logical-consistency work | Typed scene state, positions, camera side, ownership and events; VACE/layout-conditioned shots; deterministic checks and repair/review paths. |
| Telugu/resource work | Indic Parler in an isolated process, Telugu font shaping, narration caches, GPU/CPU routing and limits for small Colab machines. |
| Topic/story/voice work | Research and fact checks, Arjun/Meera story grammar, expressive delivery and multilingual speech diagnostics. |
| Gemini/Tenglish work | Explicit optional hosted-TTS path and natural Telugu–English dialogue experiments. These options remain in historical snapshots; the current target selects local Parler. |
| Storybook work | Procedural layered character rigs, versioned environments, scene camera layouts, pose/face animation and asset hashes. The checked-in `v1` characters are provisional examples. |
| Character Studio / cinematic work | Reference-conditioned image generation, reviewed RGBA atlas assets, immutable versions, explicit design/animation review, character-only delivery and fail-closed artwork selection. |
| Git `5a742b5`, `316d2e1`, `4f25ceb` | Bulk additions of notebooks, docs, plans, extracted cells, partial assets and reports. The mixed tree contained website code and video material; some documented engine paths did not exist. |
| Git `9be0a4b` | Cinematic workflow, reference-conditioned generation/cache fixes, guarded atlas rendering, prototype review tooling and CPU fixture tests added on a feature branch. It did not create approved production artwork. |
| Current reorganization | Website removed; canonical editable modules restored under `colab/engine/lib`; builders, notebooks, plans, assets, stories and reports grouped under `colab`; only the latest Telugu 3D video upgraded and given a conservative Colab rendering preset. |

The prior production pipeline was already more advanced than the early architecture document: the latest Storybook notebook embedded 34 modules, and Character Studio included its extra image-generation helpers. The new source tree makes that implementation visible instead of treating the notebook JSON as the only source.

Historical QC is retained under `colab/reports/`. The Storybook report recorded 28 PASS and 3 UNKNOWN results with overall approval false; unresolved artwork/speech review should not be rewritten as successful production approval. New synthetic test movies verify plumbing, not the user's intended finished artwork.

The last supplied cinematic master prompt is retained as [reference input](../prompts/cinematic-character-master.md). Its original character-only scope preceded the user's explicit request to remove the website and reorganize the repository. Current behavior and instructions are documented in [the current architecture](../02_ARCHITECTURE.md), not inferred from older proposals.
