# ==== 1. Your topic (everything below is made from it, with free local models) ====
TOPIC          = "How do black holes form?"     # ← type your question or topic here
TARGET_SECONDS = 60                             # about 60 s = ~150 spoken words
WIKI_CONTACT   = ""                             # ← your e-mail or a web page: Wikimedia asks every program that reads
                                                #   Wikipedia/Commons for contact details (used only in request headers)
PLANNER_MODEL  = "Qwen/Qwen3-4B-Instruct-2507"  # Apache-2.0, ~8 GB on the GPU; "Qwen/Qwen3-1.7B" if memory is short
NARRATOR_VOICE = "af_heart"                     # Kokoro stock voice (e.g. af_heart, am_michael, bf_emma)
USE_LICENSED_IMAGES = True                      # real images from Wikimedia Commons (public domain, CC0, CC BY)
ALLOW_SHAREALIKE = False                        # also CC BY-SA images (the whole video may then have to be BY-SA)
LICENSE_STRICT = True                           # True: refuse models marked "review" in the licence registry
STORY_MODE     = "two_friends_cinematic"        # two original 2D friends discuss the topic | "narrator": one narrator
LANGUAGE       = "en"                           # "en" (Kokoro voices) | "te" (Indic Parler-TTS; needs HF_TOKEN)
MAX_STORY_PREMISES  = 2                         # story concepts proposed before the best supported one is chosen
MAX_STORY_REVISIONS = 1                         # targeted rewrites of scenes the critic finds weak
STORY_BENCH    = False                          # True: old vs new story writer on 3 topics (same facts), then stop
import re as _re
TOPIC_PROJECT = "topic_" + _re.sub(r"[^a-z0-9]+", "_", TOPIC.lower()).strip("_")[:40]
