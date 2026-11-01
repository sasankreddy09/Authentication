# 07 — YouTube Automation

**Status:** Phase 1 design. No credentials have been requested or created, and nothing has been uploaded.
**Date:** 2026-10-11. API facts were re-checked today against developers.google.com and support.google.com.

Labels:
- **[O]** official, **[C]** community.
- **[UNVERIFIED]** means the official sources conflict or are silent. Test it before relying on it.

---

## 1. Summary

| Topic | Decision |
|---|---|
| API | YouTube Data API v3 via `google-api-python-client` (2.201.0 on 2026-09-30 [O]) |
| Auth | OAuth 2.0 **Desktop app** client, loopback redirect via `google-auth-oauthlib` `InstalledAppFlow.run_local_server()` (1.5.0 [O]). One-time browser consent on the Ubuntu laptop |
| Scopes | `youtube.upload` (upload) + `youtube.force-ssl` (captions, thumbnails, updates). Least privilege: request `youtube.force-ssl` only if captions are uploaded via the API |
| Token storage | `~/.config/ai-video-studio/youtube_token.json` (mode 600) or the OS keyring. **Never** in the repo, never in Colab |
| Upload | Resumable `videos.insert`, chunked (multiple of 256 KB), session URI persisted for resume |
| Default privacy | **private**. Public or scheduled needs a separate explicit command after human review |
| Disclosure | `status.containsSyntheticMedia = true` for realistic AI visuals, plus a plain-language note in the description |
| Idempotency | `ref:<project_id>` marker + `publish_record.json` + stored session URI → no duplicate uploads |
| Quota | ~100 uploads/day in the dedicated upload bucket; 10,000 units/day for everything else [O]. Far above this project's needs |

---

## 2. API facts (verified 2026-10-11)

### 2.1 Quota [O]

- **2025-12-04:** the upload cost dropped "from approximately 1600 units to approximately 100 units" ([revision history](https://developers.google.com/youtube/v3/revision_history)).
- **2026-06-01:** `videos.insert` and `search.list` moved to **their own quota buckets**. The `videos.insert` page now says: *"100 calls per day. A call to this method has a quota cost of 1 unit in the Video Uploads quota bucket."* ([videos.insert](https://developers.google.com/youtube/v3/docs/videos/insert))
- The quota page's summary text still says "1600" in one place. Trust the per-method table and the revision history ([quota costs](https://developers.google.com/youtube/v3/determine_quota_cost)).
- Quotas reset at **midnight Pacific Time**. More quota is requested through the *Audit and Quota Extension Form*.

| Method | Quota cost | Used for |
|---|---|---|
| `videos.insert` | 1 unit, Video Uploads bucket (100/day) | Upload |
| `thumbnails.set` | ~50 | Custom thumbnail |
| `captions.insert` | 400 | SRT caption track |
| `videos.update` | 50 | Change privacy, schedule or metadata later |
| `playlistItems.insert` | 50 | Add to a playlist |
| `videos.list` | 1 | Check processing status, duplicate check |

**Per video:** 1 upload unit + ~500 general units, so about 20 fully processed videos/day before the general bucket runs out.

### 2.2 Private-by-default restriction for unaudited projects — conflicting docs

- The **2020-07-28** revision-history entry is still published: videos uploaded *"from unverified API projects created after 28 July 2020 will be restricted to private viewing mode"* until the project passes an audit [O].
- The **current** `videos.insert` page says the opposite: *"Videos uploaded from unverified API projects are not restricted to private viewing mode."* [O] No revision-history entry announces the change.
- **[UNVERIFIED in practice.]** This pipeline uploads **private by default anyway**, so it is not blocked either way.
- When you later want API-driven public or scheduled publishing:
  1. Test one upload.
  2. If YouTube locks it private, submit the [audit form](https://developers.google.com/youtube/v3/guides/quota_and_compliance_audits).
- **Fallback that always works:** upload private via the API, then switch to public by hand in YouTube Studio.

### 2.3 Fields and limits [O] ([videos resource](https://developers.google.com/youtube/v3/docs/videos))

| Field | Rule |
|---|---|
| `snippet.title` | ≤ 100 characters; `<` and `>` not allowed |
| `snippet.description` | ≤ 5000 **bytes** (UTF-8), no `<` or `>` |
| `snippet.tags` | ≤ 500 characters total; commas count; a tag with a space counts as if quoted (+2) |
| `snippet.categoryId` | 27 = Education, 28 = Science & Technology [C — confirm with `videoCategories.list?regionCode=US`] |
| `snippet.defaultLanguage`, `defaultAudioLanguage` | `en` |
| `status.privacyStatus` | `private` / `unlisted` / `public`. The API default is **public**, so the pipeline always sets it explicitly |
| `status.publishAt` | ISO 8601. Only valid when `privacyStatus=private` and the video has never been published. On `videos.update`, send `privacyStatus=private` too. A past time publishes immediately |
| `status.selfDeclaredMadeForKids` | Required decision (false for these documentaries unless aimed at children) |
| `status.containsSyntheticMedia` | Boolean disclosure of realistic altered or synthetic content (added 2024-10-30) |
| `status.license`, `status.embeddable` | `youtube` / `creativeCommon`; boolean |

**Files:**
- Up to **256 GB or 12 hours**, whichever is less.
- Videos **longer than 15 minutes** need a phone-verified account ([help](https://support.google.com/youtube/answer/71673)) [O].

**Thumbnails ([thumbnails.set](https://developers.google.com/youtube/v3/docs/thumbnails/set)):**
- JPEG or PNG, max **50 MB** (raised from 2 MB on 2026-09-14) [O]. Use 1280×720 (16:9) or larger.
- A 403 means the channel cannot set custom thumbnails. YouTube Help says custom thumbnails need a verified or feature-eligible channel [UNVERIFIED this pass]. Verify the channel once at youtube.com/verify.

### 2.4 Resumable upload protocol [O] ([guide](https://developers.google.com/youtube/v3/guides/using_resumable_upload_protocol))

- Each chunk size must be a **multiple of 256 KB**, except the last chunk.
- An interrupted upload returns **HTTP 308 Resume Incomplete**. Query the status with an empty PUT and `Content-Range: bytes */<total>`. The `Range` response header shows the bytes received; continue from the next byte.
- Retry **500/502/503/504** with exponential backoff and jitter, and honour `Retry-After`.
- **Never** retry `quotaExceeded` or `uploadLimitExceeded`. Stop and report until the next Pacific midnight.
- `MediaFileUpload(path, chunksize=8*1024*1024, resumable=True)` in google-api-python-client implements this. Store `request.resumable_uri` after the first chunk to resume across process restarts.

### 2.5 OAuth facts [O] ([OAuth 2.0](https://developers.google.com/identity/protocols/oauth2))

- **Out-of-band (copy/paste code) flow:** blocked since 2023-01-31. Use the **loopback** flow (`http://localhost:<port>`), which is what `run_local_server()` does.
- **"Testing" publishing status:** for an external app in Testing, refresh tokens for non-basic scopes **expire after 7 days**. You would re-authorize weekly.
- **Unused tokens:** refresh tokens also expire after **6 months unused**, and are revoked when you change your password if Gmail scopes are involved (not the case here).
- **Token cap:** max **100 refresh tokens per Google account per client ID**; the oldest are revoked silently. Don't re-authorize in a loop.
- **Verification:** there is no explicit "personal use" exemption from verification ([help](https://support.google.com/cloud/answer/7454865)). Options for a single-user tool:
  - **(a) Stay in Testing**, with yourself as a test user. Simple, but you re-authorize every 7 days. The CLI detects `invalid_grant` and tells you to run `main.py auth youtube`.
  - **(b) Set "In production" without verification.** You click through the "Google hasn't verified this app" warning yourself. The cap is 100 users. Community reports say tokens then do not expire after 7 days [UNVERIFIED].
  - **Recommendation:** start with (a) during development. Move to (b) once the pipeline is stable.

---

## 3. One-time setup on Ubuntu (walkthrough)

You do these steps yourself, in the browser. The pipeline never automates account or console actions.

1. **Google Cloud project**
   1. Go to <https://console.cloud.google.com/> → *New project* → name `ai-video-studio`.
   2. Open *APIs & Services → Library* → enable **YouTube Data API v3**.
2. **OAuth consent screen** (*Google Auth Platform → Branding / Audience*)
   1. User type **External**. App name, your email as support and developer contact.
   2. Under *Audience → Test users*, add the Google account that owns the YouTube channel.
   3. Under *Data access*, add the scopes `https://www.googleapis.com/auth/youtube.upload` and `https://www.googleapis.com/auth/youtube.force-ssl`.
3. **OAuth client**
   1. Go to *Clients → Create client → Application type: **Desktop app*** → download the JSON.
   2. Store it outside the repo:

      ```bash
      mkdir -p ~/.config/ai-video-studio && chmod 700 ~/.config/ai-video-studio
      mv ~/Downloads/client_secret_*.json ~/.config/ai-video-studio/client_secret.json
      chmod 600 ~/.config/ai-video-studio/client_secret.json
      ```

4. **Channel:** verify the channel (phone) at <https://www.youtube.com/verify>. This enables custom thumbnails and uploads longer than 15 minutes.
5. **Authorize once:**

   ```bash
   python main.py auth youtube
   # → opens the browser on http://localhost:<random port>, you pick the channel's account and approve
   # → writes ~/.config/ai-video-studio/youtube_token.json (mode 600)
   # → prints the channel title + id so you can confirm it is the right channel
   ```

   - **Headless or SSH:** run `ssh -L 8080:localhost:8080 laptop` and use `main.py auth youtube --port 8080`, then open the printed URL in your local browser.
6. **Test:** `python main.py publish --project projects/_smoke_test --privacy private` uploads a 5-second test clip, records it, and you delete it in Studio afterwards.

---

## 4. Publishing workflow

```text
final_16x9.mp4 + metadata.json + thumbnail.png + subtitles.srt
            │
            ▼
[1] Pre-flight (no network)
    - qc_report.json: all hard checks passed
    - approval.json exists, approved_by set, sha256 matches final_16x9.mp4
    - metadata within limits (title ≤100 chars, description ≤5000 bytes, tags ≤500 chars, no < >)
    - containsSyntheticMedia set when plan.disclosure says so (cannot be turned off by flags)
    - privacy = requested value; 'public' requires --confirm-public
            │
            ▼
[2] Duplicate check
    - publish_record.json has video_id? → stop ("already uploaded", print URL)
    - pending session_uri stored? → resume that upload instead of starting a new one
    - optional: videos.list over your recent uploads, looking for "ref:<project_id>"
            │
            ▼
[3] videos.insert (resumable, 8 MB chunks)
    - save session_uri to publish_state.json after the first response
    - backoff on 5xx; stop on quota errors
    - on 201: write video_id immediately (before anything else can fail)
            │
            ▼
[4] thumbnails.set     (non-fatal on 403: warn "channel not eligible")
[5] captions.insert    (SRT, language en, name "English"; non-fatal)
[6] playlistItems.insert (optional)
            │
            ▼
[7] publish_record.json: video_id, watch URL, studio URL, privacy, publishAt,
    quota units used, file sha256, metadata hash, timestamps, API responses (trimmed)
```

### 4.1 Reference implementation sketch (Phase 3 code, shown for design review)

```python
# publishing/youtube.py (sketch)
from pathlib import Path
import json, time, random
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

CFG = Path.home() / ".config/ai-video-studio"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload",
          "https://www.googleapis.com/auth/youtube.force-ssl"]
RETRYABLE = {500, 502, 503, 504}
FATAL_REASONS = {"quotaExceeded", "uploadLimitExceeded", "forbidden"}

def authorize(port: int = 0) -> Credentials:
    flow = InstalledAppFlow.from_client_secrets_file(CFG / "client_secret.json", SCOPES)
    creds = flow.run_local_server(port=port, open_browser=True)
    tok = CFG / "youtube_token.json"
    tok.write_text(creds.to_json()); tok.chmod(0o600)
    return creds

def load_creds() -> Credentials:
    creds = Credentials.from_authorized_user_file(CFG / "youtube_token.json", SCOPES)
    if not creds.valid:
        creds.refresh(Request())             # invalid_grant → tell user to run `main.py auth youtube`
        (CFG / "youtube_token.json").write_text(creds.to_json())
    return creds

def upload(project: Path, privacy: str = "private", publish_at: str | None = None) -> dict:
    meta = json.loads((project / "07_publish/metadata.json").read_text())
    record_path = project / "07_publish/publish_record.json"
    if record_path.exists():
        return json.loads(record_path.read_text())          # idempotent: already uploaded
    yt = build("youtube", "v3", credentials=load_creds())
    status = {"privacyStatus": privacy,
              "selfDeclaredMadeForKids": meta["self_declared_made_for_kids"],
              "containsSyntheticMedia": meta["contains_synthetic_media"]}
    if publish_at:
        status.update(privacyStatus="private", publishAt=publish_at)
    body = {"snippet": {k: meta[k] for k in ("title", "description", "tags")}
                       | {"categoryId": meta["category_id"],
                          "defaultLanguage": meta["default_language"],
                          "defaultAudioLanguage": meta["default_audio_language"]},
            "status": status}
    media = MediaFileUpload(project / "05_render/final_16x9.mp4",
                            mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    # (production code also persists req.resumable_uri to resume across process restarts)
    response, attempt = None, 0
    while response is None:
        try:
            _, response = req.next_chunk()
        except HttpError as e:
            reason = _reason(e)
            if reason in FATAL_REASONS or e.resp.status not in RETRYABLE or attempt >= 8:
                raise
            attempt += 1
            time.sleep(min(64, 2 ** attempt) + random.random())
    record = {"video_id": response["id"],
              "url": f"https://www.youtube.com/watch?v={response['id']}",
              "studio_url": f"https://studio.youtube.com/video/{response['id']}/edit",
              "privacy": status["privacyStatus"], "publish_at": publish_at,
              "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    record_path.write_text(json.dumps(record, indent=2))    # write before thumbnail/captions
    return record
```

---

## 5. Security rules

| Rule | How it is enforced |
|---|---|
| No secrets in code or Git | Credentials live only in `~/.config/ai-video-studio/` (mode 700/600) or the keyring. `.gitignore` covers `client_secret*.json`, `*token*.json`, `.env`, `secrets/`. A `gitleaks` pre-commit hook blocks accidental commits |
| No secrets on Colab or GPU workers | The publisher runs only on the laptop. Job files never contain tokens |
| Least privilege | Only the `youtube.upload` and `youtube.force-ssl` scopes. No `youtube` (full manage) scope |
| Revocation | `main.py auth youtube --revoke` calls Google's revoke endpoint and deletes the local token. You can also revoke at <https://myaccount.google.com/permissions> |
| Audit trail | `publish_record.json` + `logs/pipeline.jsonl` record every API call (method, status, quota units) with **no** token values |
| Logging hygiene | The logger redacts `Authorization` headers and `access_token`/`refresh_token` keys |

---

## 6. Safeguards against unwanted publication

1. **Private is the default** everywhere: CLI default, config default, and API body always explicit.
2. **Two-step publication.** `publish --privacy public` requires `--confirm-public` **and** an `approval.json` whose `sha256` matches the current final file. Re-rendering invalidates the approval.
3. **Scheduling** (`--publish-at`) uploads private with `publishAt`. You can cancel it in Studio before that time.
4. **Disclosure cannot be skipped.** If `plan.disclosure.contains_synthetic_media` is true, the publisher refuses to send `false`.
5. **Rate cap.** `settings.yaml: publishing.max_uploads_per_day` defaults to 2. This guards against bugs and against the spam policy below.
6. **Quota errors stop the run** with a clear message. They are never retried in a loop.

---

## 7. YouTube policies for AI-generated content (as of 2026-10-11)

### 7.1 Altered or synthetic content disclosure [O] ([policy](https://support.google.com/youtube/answer/14328491))

- YouTube requires creators *"to disclose when they use AI to meaningfully alter or generate photorealistic content."* Examples that require disclosure include content that *"generates a realistic scene that didn't actually occur."*
- **Exempt:** non-realistic content, minor aesthetic edits, and *"production assistance, like using generative AI tools to create or improve a video outline, script, thumbnail, title, or infographic."*
- **Labels:** for photorealistic AI content, a label may appear in the player. For non-photorealistic or animated content, it appears in the expanded description. YouTube may also apply labels itself (for example from C2PA metadata), and creators cannot adjust those.
- **Penalty:** creators who consistently fail to disclose may get labels applied manually, content removed, or suspension from the YouTube Partner Program.

**For this project:**
- Photoreal historical reconstructions and synthetic narration → set `containsSyntheticMedia=true`.
- Add a description line such as "Narration is synthetic (text-to-speech). Historical scenes are AI-generated reconstructions for illustration."

### 7.2 Monetization: "inauthentic content" [O] ([YPP policy](https://support.google.com/youtube/answer/1311392))

- **Rename:** on 2025-07-15 the policy was renamed from "repetitious content" to "inauthentic content".
- **Requirement:** content must *"not be mass-produced, generic, repetitive, or manipulative."*
- **Not allowed:** *"similar or repetitive content with low educational value, commentary, narratives, or minimal variation across videos."*
- **Automated tools:** *"If you use automated tools or templates ... the final product must still demonstrate your creative vision and provide educational or entertainment value."*
- **AI personas:** these are not allowed for sensitive topics when presented as human experts.

### 7.3 Spam policy [O] ([policy](https://support.google.com/youtube/answer/2801973))

> "Automated or synthetic mass-production: Using automated tools or AI to churn out high volumes of similar content with minimal changes... we don't allow use of these tools to flood our platform with repetitive content."

The example given closely matches a naive version of this pipeline: *"Channels that use the exact same background music and repetitive AI generated imagery across many videos, with each video reading out an AI-generated script."*

### 7.4 2026 context [C]

- YouTube's CEO letter (January 2026) committed to reducing "low-quality AI content", and press reported channel terminations for "AI slop" ([Variety](https://au.variety.com/2026/digital/news/youtube-channels-using-ai-tools-reduce-slop-neal-mohan-letter-32190/), [Tubefilter](https://tubefilter.com/2026/01/29/youtube-ai-slop-channel-crackdown-bans/)).
- **Likeness detection** was opened to all creators aged 18+ around May 2026. Do not generate recognizable real people.

### 7.5 What the pipeline does about it

| Policy risk | Design response |
|---|---|
| Undisclosed synthetic media | Disclosure flag is enforced; description note is automatic |
| Mass-produced / repetitive | Volume cap; varied style presets, voices and music; research-backed, topic-specific scripts; human review before publishing; no "same template every day" mode |
| Low educational value | Research stage with sources; critic checks for substance; sources listed in the description |
| Misleading historical imagery | Reconstruction labelling; contested claims presented as theories; no fake "archival footage" look |
| Real-person likeness | No generation of identifiable real people; named historical figures appear only as statues, art-style illustrations or text |
| Copyrighted music or footage | Only licensed, CC0 or self-generated audio with permissive licenses; `licenses.json` per project |

The pipeline does **not** bypass quotas, the audit, disclosure or any other platform control.

---

## 8. Sources

- [videos.insert](https://developers.google.com/youtube/v3/docs/videos/insert) · [videos resource](https://developers.google.com/youtube/v3/docs/videos) · [quota costs](https://developers.google.com/youtube/v3/determine_quota_cost) · [revision history](https://developers.google.com/youtube/v3/revision_history) · [resumable upload](https://developers.google.com/youtube/v3/guides/using_resumable_upload_protocol) · [audit and quota extension](https://developers.google.com/youtube/v3/guides/quota_and_compliance_audits) · [thumbnails.set](https://developers.google.com/youtube/v3/docs/thumbnails/set) · [captions.insert](https://developers.google.com/youtube/v3/docs/captions/insert)
- [OAuth 2.0](https://developers.google.com/identity/protocols/oauth2) · [OOB migration](https://developers.google.com/identity/protocols/oauth2/resources/oob-migration) · [app verification](https://support.google.com/cloud/answer/7454865)
- [Synthetic content disclosure](https://support.google.com/youtube/answer/14328491) · [YPP policies](https://support.google.com/youtube/answer/1311392) · [Spam policy](https://support.google.com/youtube/answer/2801973) · [Upload limits](https://support.google.com/youtube/answer/71673)
- PyPI: [google-api-python-client](https://pypi.org/project/google-api-python-client/) · [google-auth-oauthlib](https://pypi.org/project/google-auth-oauthlib/)
