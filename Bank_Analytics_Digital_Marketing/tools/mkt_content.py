"""
tools/mkt_content.py – Marketing content creation tools.

Replaced Veo 3.1 video generation (expensive, ~600s polling) with
Gemini image generation (gemini-3.1-flash-image-preview) for promotional
posters — significantly cheaper and near-instant.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crewai.tools import tool

_BASE     = Path(__file__).parent.parent / "outputs"
_POSTERS  = _BASE / "posters"
_CONTENT  = _BASE / "content"
_REPORTS  = _BASE / "reports"

for _d in (_POSTERS, _CONTENT, _REPORTS):
    _d.mkdir(parents=True, exist_ok=True)

_SESSION_FILE = _CONTENT / "session_content.json"


def _update_session_content(entry: dict) -> None:
    entries: list = []
    if _SESSION_FILE.exists():
        try:
            entries = json.loads(_SESSION_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            entries = []
    entries.append(entry)
    _SESSION_FILE.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def _gemini(prompt: str, system: str = "", _timeout: int = 130) -> str:
    """Call Gemini 2.5 Flash with hard wall-clock timeout."""
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
            print(f"[ContentTools] Gemini error: {exc}", flush=True)
            return "Unable to generate response at this time."

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


@tool("generate_promotional_poster")
def generate_promotional_poster(
    prompt: str,
    title: str,
    brand_name: str = "Bank",
    campaign_type: str = "Retention",
) -> str:
    """
    Generate an AI promotional poster image using Gemini image generation.
    Replaces Veo 3.1 video generation — significantly cheaper and near-instant.

    Args:
        prompt:        Detailed visual description for the poster (scene, style, colours,
                       typography hints, mood). Be specific about banking/financial visuals.
        title:         Descriptive title for the poster asset.
        brand_name:    Bank or brand name to feature on the poster.
        campaign_type: Campaign theme (e.g. 'Retention', 'Premium Offer', 'Loyalty Reward').

    Returns:
        URL path to the saved PNG file, or an error message.
    """
    try:
        import PIL.Image
        from google import genai
        from io import BytesIO
        from config import GOOGLE_API_KEY
    except ImportError as exc:
        return f"[Poster generation unavailable – missing dependency: {exc}]"

    if not GOOGLE_API_KEY:
        return "[Poster generation unavailable – GOOGLE_API_KEY not set]"

    # Build a rich banking-context poster prompt.
    # Resolution: 1K (1024×1024) — $0.067 per image on paid tier.
    full_prompt = (
        f"Create a professional promotional banking poster for '{brand_name}' — {campaign_type} campaign. "
        f"{prompt} "
        "Style: modern financial brand aesthetic, premium feel, trust-inspiring colour palette "
        "(deep blue, gold accents), clean typography. "
        "Include subtle banking motifs (upward charts, shield icon, handshake). "
        "Suitable for digital channels: website, email header, social media banner. "
        "Output image resolution: 1024x1024 pixels (1K), square format, high detail."
    )

    print(f"[Content Maker] Generating 1K promotional poster for '{title}'…", flush=True)

    try:
        from google.genai import types as _gtypes

        client   = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=[full_prompt],
            config=_gtypes.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )
    except Exception as exc:
        return f"[Poster generation failed: {exc}]"

    # Extract the image from the response parts
    image_data = None
    caption    = ""
    for part in response.candidates[0].content.parts:
        if part.text is not None:
            caption = part.text
        elif part.inline_data is not None:
            image_data = part.inline_data.data

    if not image_data:
        # Return caption/description if no image was produced
        fallback_msg = caption or "No image data returned by the model."
        print(f"[Content Maker] No image bytes — model returned: {fallback_msg[:200]}", flush=True)
        return f"[Poster generation: no image data] {fallback_msg}"

    # Save the PNG
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c if c.isalnum() else "_" for c in title)
    filename   = f"poster_{safe_title}_{ts}.png"
    filepath   = _POSTERS / filename

    TARGET_SIZE = (1024, 1024)   # 1K — $0.067 per image on paid tier

    try:
        image = PIL.Image.open(BytesIO(image_data))
        # Ensure exactly 1024×1024 — resize only if the model returned a different size
        if image.size != TARGET_SIZE:
            image = image.resize(TARGET_SIZE, PIL.Image.LANCZOS)
        image.save(str(filepath), format="PNG", optimize=False)
        actual_size = f"{image.size[0]}×{image.size[1]}"
    except Exception:
        # Fallback: write raw bytes if PIL can't open
        filepath.write_bytes(image_data)
        actual_size = "unknown"

    poster_url = f"/outputs/posters/{filename}"

    _update_session_content({
        "type":       "poster",
        "title":      title,
        "platform":   "multi-platform",
        "content":    caption or full_prompt,
        "poster_url": poster_url,
        "size":       actual_size,
        "timestamp":  datetime.now().isoformat(),
    })

    print(f"[Content Maker] Poster saved ({actual_size}) → {filepath}", flush=True)
    if caption:
        print(f"[Content Maker] Model caption: {caption[:200]}", flush=True)

    return (
        f"Promotional poster saved: {poster_url} ({actual_size})"
        + (f"\nCaption: {caption}" if caption else "")
    )


@tool("write_ad_copy")
def write_ad_copy(brand: str, product: str, audience: str, platform: str, tone: str = "professional") -> str:
    """
    Write platform-specific advertising copy.

    Args:
        brand:    Brand name.
        product:  Product or service being advertised.
        audience: Target audience description.
        platform: Advertising platform (Google, Meta, TikTok, LinkedIn).
        tone:     Tone of voice (professional, playful, urgent, inspirational).

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
def generate_social_posts(brand: str, campaign: str, platforms: str, num_posts: int = 5) -> str:
    """
    Generate platform-specific social media posts with hashtags.

    Args:
        brand:     Brand name.
        campaign:  Campaign name or brief description.
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
    (_CONTENT / filename).write_text(json.dumps(all_posts, indent=2, ensure_ascii=False), encoding="utf-8")

    preview = json.dumps(all_posts, indent=2, ensure_ascii=False)[:1000]
    return f"Social posts saved to outputs/content/{filename}\n\nPreview:\n{preview}"


@tool("create_email_template")
def create_email_template(brand: str, campaign: str, audience: str, goal: str) -> str:
    """
    Create an HTML email template with subject line, body, and CTA.

    Args:
        brand:    Brand name.
        campaign: Campaign name or type.
        audience: Target audience.
        goal:     Email goal (nurture, convert, re-engage, announce).

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
    """
    Append a content entry to the session sidecar JSON file.

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
    """
    Generate a final content summary markdown report.

    Args:
        summary:    Summary of all content created during the session.
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

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() else "_" for c in brand_name)
    path = _REPORTS / f"content_report_{safe}_{ts}.md"
    path.write_text(report, encoding="utf-8")
    print(f"[ContentTools] Content report saved → {path}", flush=True)
    return f"Content report saved to {path}\n\n{report}"
