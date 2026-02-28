"""
tools/content_tools.py – Content creation tools including Veo 3 video generation.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from crewai.tools import tool

# ── output directories ────────────────────────────────────────────────────────
_BASE      = Path(__file__).parent.parent / "outputs"
_VIDEOS    = _BASE / "videos"
_CONTENT   = _BASE / "content"
_REPORTS   = _BASE / "reports"

for _d in (_VIDEOS, _CONTENT, _REPORTS):
    _d.mkdir(parents=True, exist_ok=True)

_SESSION_FILE = _CONTENT / "session_content.json"


# ── session sidecar helpers ───────────────────────────────────────────────────

def _update_session_content(entry: dict) -> None:
    """Append a content entry to the session sidecar JSON file."""
    entries: list = []
    if _SESSION_FILE.exists():
        try:
            entries = json.loads(_SESSION_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            entries = []
    entries.append(entry)
    _SESSION_FILE.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def _gemini(prompt: str, system: str = "", _timeout: int = 130) -> str:
    """Call Gemini 2.5 Flash.
    Hard wall-clock cap = 130s. On timeout/error returns a fallback string
    so the agent keeps moving – the pipeline never crashes.
    """
    import concurrent.futures

    def _call() -> str:
        try:
            from config import gemini_llm
            from langchain_core.messages import HumanMessage, SystemMessage

            if gemini_llm is None:
                return "Gemini unavailable – proceeding with available information."
            msgs = []
            if system:
                msgs.append(SystemMessage(content=system))
            msgs.append(HumanMessage(content=prompt))
            resp = gemini_llm.invoke(msgs)
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception as exc:
            print(f"[ContentTools] Gemini error (will return fallback): {exc}", flush=True)
            return "Unable to generate response at this time – proceeding with available context."

    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(_call)
    try:
        result = fut.result(timeout=_timeout)
        ex.shutdown(wait=False)
        return result
    except concurrent.futures.TimeoutError:
        ex.shutdown(wait=False)
        print(f"[ContentTools] Gemini hard timeout ({_timeout}s) – returning fallback", flush=True)
        return "Response timed out – proceeding with available context from prior steps."


# ── tools ─────────────────────────────────────────────────────────────────────

@tool("generate_video_content")
def generate_video_content(
    prompt: str,
    title: str,
    aspect_ratio: str = "16:9",
    resolution: str = "720p",
) -> str:
    """Generate an AI marketing video using Google Veo 3.

    Args:
        prompt: Detailed video generation prompt describing the scene, mood, and action.
        title: Descriptive title for the video asset.
        aspect_ratio: Video aspect ratio – '16:9' (landscape) or '9:16' (portrait/Reels).
        resolution: Output resolution – '720p' or '1080p'.

    Returns:
        Path to the saved MP4 file, or an error message.
    """
    try:
        from google import genai
        from google.genai import types
        from config import GOOGLE_API_KEY
    except ImportError as exc:
        return f"[Video generation unavailable – missing dependency: {exc}]"

    if not GOOGLE_API_KEY:
        return "[Video generation unavailable – GOOGLE_API_KEY not set]"

    print(f"[Content Maker] Submitting Veo 3 job for '{title}'…", flush=True)

    client = genai.Client(api_key=GOOGLE_API_KEY)
    try:
        operation = client.models.generate_videos(
            model="veo-3.1-fast-generate-preview",
            prompt=prompt,
            config=types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                resolution=resolution,
            ),
        )
    except Exception as exc:
        return f"[Veo 3 job submission failed: {exc}]"

    # Poll up to 10 minutes (40 × 15 s)
    for i in range(40):
        time.sleep(15)
        try:
            operation = client.operations.get(operation)
        except Exception:
            pass
        elapsed = (i + 1) * 15
        if operation.done:
            print(f"[Content Maker] Video '{title}' done after {elapsed}s", flush=True)
            break
        print(f"[Content Maker] Video '{title}' generating… ({elapsed}s elapsed)", flush=True)

    if not operation.done:
        return f"[Video generation timed out after 600s for '{title}']"

    try:
        video_data = operation.response.generated_videos[0].video
        ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c if c.isalnum() else "_" for c in title)
        filename   = f"{safe_title}_{ts}.mp4"
        filepath   = _VIDEOS / filename

        # Strategy 1: use the SDK's authenticated download (handles auth automatically)
        saved = False
        try:
            client.files.download(file=video_data)
            if hasattr(video_data, "save"):
                video_data.save(str(filepath))
                saved = True
        except Exception as _e1:
            print(f"[Content Maker] SDK download failed ({_e1}), trying fallback…", flush=True)

        # Strategy 2: raw bytes on the object
        if not saved and hasattr(video_data, "data") and video_data.data:
            filepath.write_bytes(video_data.data)
            saved = True

        # Strategy 3: authenticated HTTP request with API key
        if not saved and hasattr(video_data, "uri") and video_data.uri:
            import urllib.request
            uri = video_data.uri
            if "?" in uri:
                uri = f"{uri}&key={GOOGLE_API_KEY}"
            else:
                uri = f"{uri}?key={GOOGLE_API_KEY}"
            req = urllib.request.Request(uri, headers={"X-Goog-Api-Key": GOOGLE_API_KEY})
            with urllib.request.urlopen(req) as resp:
                filepath.write_bytes(resp.read())
            saved = True

        if not saved:
            return "[Video generation completed but could not download the file]"

        video_url = f"/outputs/videos/{filename}"
        _update_session_content({
            "type":      "video",
            "title":     title,
            "platform":  "multi-platform",
            "content":   prompt,
            "video_url": video_url,
            "timestamp": datetime.now().isoformat(),
        })
        print(f"[Content Maker] Video saved → {filepath}", flush=True)
        return f"Video saved: {video_url}"

    except Exception as exc:
        return f"[Video save error: {exc}]"


@tool("write_ad_copy")
def write_ad_copy(
    brand: str,
    product: str,
    audience: str,
    platform: str,
    tone: str = "professional",
) -> str:
    """Write platform-specific advertising copy.

    Args:
        brand: Brand name.
        product: Product or service being advertised.
        audience: Target audience description.
        platform: Advertising platform (Google, Meta, TikTok, LinkedIn).
        tone: Tone of voice (professional, playful, urgent, inspirational).

    Returns:
        Ad copy variants as a formatted string.
    """
    copy = _gemini(
        prompt=(
            f"Brand: {brand}\nProduct: {product}\nAudience: {audience}\n"
            f"Platform: {platform}\nTone: {tone}\n\n"
            f"Write 3 ad copy variants optimised for {platform}. "
            "Each variant: Headline (max 30 chars for Google/Meta), "
            "Description (max 90 chars), Primary Text/Hook, CTA. "
            "Label each variant clearly."
        ),
        system=f"You are a performance marketing copywriter specialising in {platform} ads.",
    )

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe     = "".join(c if c.isalnum() else "_" for c in brand)
    filename = f"ad_copy_{safe}_{platform}_{ts}.md"
    (_CONTENT / filename).write_text(copy, encoding="utf-8")

    _update_session_content({
        "type":      "ad_copy",
        "title":     f"{brand} – {platform} Ad Copy",
        "platform":  platform,
        "content":   copy,
        "video_url": "",
        "timestamp": datetime.now().isoformat(),
    })

    return f"Ad copy saved to outputs/content/{filename}\n\n{copy}"


@tool("generate_social_posts")
def generate_social_posts(
    brand: str,
    campaign: str,
    platforms: str,
    num_posts: int = 5,
) -> str:
    """Generate platform-specific social media posts with hashtags.

    Args:
        brand: Brand name.
        campaign: Campaign name or brief description.
        platforms: Comma-separated list of platforms (Instagram, Twitter, LinkedIn, TikTok).
        num_posts: Number of posts to generate per platform.

    Returns:
        Path to saved JSON file + preview of posts.
    """
    platform_list = [p.strip() for p in platforms.split(",") if p.strip()]
    all_posts: dict = {}

    for platform in platform_list:
        raw = _gemini(
            prompt=(
                f"Brand: {brand}\nCampaign: {campaign}\nPlatform: {platform}\n\n"
                f"Generate {num_posts} unique social media posts for {platform}. "
                "Each post: caption text, 5–10 relevant hashtags, emoji, engagement CTA. "
                "Format as JSON array: [{\"post\": \"...\", \"hashtags\": [...], \"cta\": \"...\"}]"
            ),
            system=f"You are a social media manager expert in {platform} content strategy.",
        )
        # try to parse JSON, fallback to raw text
        try:
            import re
            json_match = re.search(r"\[.*\]", raw, re.DOTALL)
            posts = json.loads(json_match.group()) if json_match else [{"post": raw}]
        except Exception:
            posts = [{"post": raw}]

        all_posts[platform] = posts

        _update_session_content({
            "type":      "social_post",
            "title":     f"{brand} – {platform} Posts",
            "platform":  platform,
            "content":   json.dumps(posts, ensure_ascii=False),
            "video_url": "",
            "timestamp": datetime.now().isoformat(),
        })

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe     = "".join(c if c.isalnum() else "_" for c in brand)
    filename = f"social_posts_{safe}_{ts}.json"
    (_CONTENT / filename).write_text(
        json.dumps(all_posts, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    preview = json.dumps(all_posts, indent=2, ensure_ascii=False)[:1000]
    return f"Social posts saved to outputs/content/{filename}\n\nPreview:\n{preview}"


@tool("create_email_template")
def create_email_template(
    brand: str,
    campaign: str,
    audience: str,
    goal: str,
) -> str:
    """Create an HTML email template with subject line, body, and CTA.

    Args:
        brand: Brand name.
        campaign: Campaign name or type.
        audience: Target audience.
        goal: Email goal (nurture, convert, re-engage, announce).

    Returns:
        Path to saved HTML file + subject line preview.
    """
    html_email = _gemini(
        prompt=(
            f"Brand: {brand}\nCampaign: {campaign}\nAudience: {audience}\nGoal: {goal}\n\n"
            "Write a professional marketing email with: "
            "Subject line (A/B test 2 variants), Preheader text, "
            "HTML body (header, personalised opening, value proposition, body copy, "
            "prominent CTA button, social links, unsubscribe footer). "
            "Output complete HTML."
        ),
        system="You are an email marketing specialist. Write persuasive, accessible HTML emails.",
    )

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe     = "".join(c if c.isalnum() else "_" for c in brand)
    filename = f"email_template_{safe}_{ts}.html"
    (_CONTENT / filename).write_text(html_email, encoding="utf-8")

    _update_session_content({
        "type":      "email",
        "title":     f"{brand} – {campaign} Email",
        "platform":  "email",
        "content":   html_email[:500],
        "video_url": "",
        "timestamp": datetime.now().isoformat(),
    })

    return f"Email template saved to outputs/content/{filename}\n\nPreview:\n{html_email[:400]}"


@tool("save_content_session")
def save_content_session(entry_json: str) -> str:
    """Append a content entry to the session sidecar JSON file.

    Args:
        entry_json: JSON string with keys: type, title, platform, content, video_url, timestamp.

    Returns:
        Confirmation message.
    """
    try:
        entry = json.loads(entry_json)
    except json.JSONDecodeError as exc:
        return f"[Invalid JSON: {exc}]"

    entry.setdefault("timestamp", datetime.now().isoformat())
    _update_session_content(entry)
    return f"Session entry saved: {entry.get('type')} – {entry.get('title')}"


@tool("generate_content_report")
def generate_content_report(summary: str, brand_name: str) -> str:
    """Generate a final content summary markdown report.

    Args:
        summary: Summary of all content created during the session.
        brand_name: Brand name.

    Returns:
        Path to saved report file.
    """
    report = _gemini(
        prompt=(
            f"Brand: {brand_name}\n\nContent created:\n{summary}\n\n"
            "Write a concise Content Production Report (400–600 words) with sections: "
            "## Content Summary, ## Videos Produced, ## Ad Copy, ## Social Media, "
            "## Email Templates, ## Performance Projections, ## Next Steps."
        ),
        system="You are a Content Director summarising campaign content deliverables.",
    )

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe    = "".join(c if c.isalnum() else "_" for c in brand_name)
    path    = _REPORTS / f"content_report_{safe}_{ts}.md"
    path.write_text(report, encoding="utf-8")
    print(f"[ContentTools] Content report saved → {path}", flush=True)
    return f"Content report saved to {path}\n\n{report}"
