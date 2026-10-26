
import asyncio
import datetime as dt
import email.utils
import html
import json
import os
import re
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET

import edge_tts
import feedparser
from openai import OpenAI

# ============================================================
# WELCOME TO PARADISE — DAILY NEWS PODCAST
# ============================================================

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "docs"
EPISODES = SITE / "episodes"

TIMEZONE = ZoneInfo("Europe/Paris")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
VOICE = os.getenv("VOICE", "en-GB-RyanNeural")
BASE_URL = os.getenv("PODCAST_BASE_URL", "").rstrip("/")
MAX_EPISODES = 5

# GitHub manual runs can use FORCE_RUN=true or FORCE_RUN=1.
FORCE_RUN = os.getenv("FORCE_RUN", "").lower() in (
    "true", "1", "yes"
)

FEEDS = {
    "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "France 24": "https://www.france24.com/en/rss",
    "Guardian World": "https://www.theguardian.com/world/rss",
    "Guardian Business": "https://www.theguardian.com/uk/business/rss",
    "Guardian France": "https://www.theguardian.com/world/france/rss",
}


def clean(value):
    value = re.sub(r"<[^>]+>", " ", str(value or ""))
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


# ============================================================
# STEP 1 — COLLECT CURRENT NEWS
# ============================================================

def collect_news(now):
    stories = []

    for source, url in FEEDS.items():
        try:
            feed = feedparser.parse(
                url,
                request_headers={
                    "User-Agent": "WelcomeToParadise/1.0"
                },
            )

            for entry in feed.entries[:15]:
                timestamp = (
                    entry.get("published_parsed")
                    or entry.get("updated_parsed")
                )

                if not timestamp:
                    continue

                published = dt.datetime(
                    *timestamp[:6],
                    tzinfo=dt.timezone.utc,
                )

                age_hours = (
                    now.astimezone(dt.timezone.utc) - published
                ).total_seconds() / 3600

                if not -2 <= age_hours <= 40:
                    continue

                title = clean(entry.get("title"))
                link = entry.get("link", "")

                if not title or not link:
                    continue

                stories.append({
                    "source": source,
                    "title": title,
                    "summary": clean(
                        entry.get("summary", "")
                    )[:750],
                    "url": link,
                    "published": published.isoformat(),
                })

        except Exception as error:
            print(f"Feed warning ({source}): {error}")

    unique = {}

    for story in stories:
        key = re.sub(
            r"[^a-z0-9]", "", story["title"].lower()
        )[:85]
        unique.setdefault(key, story)

    result = sorted(
        unique.values(),
        key=lambda story: story["published"],
        reverse=True,
    )[:65]

    print(f"Collected {len(result)} news articles")

    if len(result) < 8:
        raise RuntimeError(
            "Insufficient recent news articles. "
            "Podcast publication cancelled."
        )

    return result


# ============================================================
# STEP 2 — WRITE THE JOURNALISTIC NARRATION
# ============================================================

def generate_script(stories, now):
    api_key = os.environ["OPENAI_API_KEY"].strip()
client = OpenAI(api_key=api_key)

    indexed_stories = [
        {"index": i, **story}
        for i, story in enumerate(stories)
    ]

    prompt = f"""
You are the editor and presenter of an English-language
personal news podcast called WELCOME TO PARADISE.

DATE:
{now.strftime("%A %d %B %Y")}

EDITORIAL OBJECTIVE:
Create a sophisticated, informative 10–15-minute news
briefing for a business professional based in France.

The listener works in international premium beverages,
spirits, FMCG and hospitality.

EDITORIAL BALANCE:
- Important international developments
- France and European affairs
- Business, economics and financial markets
- Consumer goods, hospitality and premium beverages
- One deeper analytical story

Select approximately 4–6 consequential stories.

Prefer fewer stories with deeper context rather than
a large number of superficial headlines.

For each story explain:
- What happened
- Why it matters
- Relevant background supported by the inputs
- Potential implications
- What to watch next

FACTUAL ACCURACY:
Use ONLY the provided RSS headlines and excerpts for
claims about current events.

These are summaries, not full articles.

Never invent:
- Facts or figures
- Quotes
- Election outcomes
- Corporate announcements
- Dates or developments
- Details not supported by the supplied sources

Attribute important claims to the news outlet.

Clearly distinguish sourced facts from interpretation.
If the information is insufficient, omit the story.

Do not force a spirits or hospitality story if there
is no relevant reporting.

PRESENTATION STYLE:
- Professional British broadcast journalist
- Intelligent, calm and engaging
- Natural spoken English
- Smooth transitions
- Subtle personality, without sensationalism
- No bullet points in the spoken narration
- No markdown or URLs in the narration

OPENING:
Begin exactly with:

"Welcome to Paradise!"

Then introduce the date and the day's most important
stories in a natural broadcast-news style.

ENDING:
Finish with the most important things to watch next.

TARGET LENGTH:
1,200–1,600 spoken words, provided sufficient reporting.
Do not pad the script with unsupported information.

OUTPUT:
Return valid JSON with exactly these fields:

"title": episode title
"narration": complete spoken script
"summary": short written summary
"source_indices": integer indices of articles used

SOURCE ARTICLES:
{json.dumps(indexed_stories, ensure_ascii=False)}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.25,
    )

    data = json.loads(
        response.choices[0].message.content
    )

    narration = data.get("narration", "").strip()

    if len(narration.split()) < 650:
        raise RuntimeError(
            "Generated narration is too short."
        )

    used_sources = []

    for index in data.get("source_indices", []):
        if isinstance(index, int) and 0 <= index < len(stories):
            if stories[index] not in used_sources:
                used_sources.append(stories[index])

    if len(used_sources) < 3:
        raise RuntimeError(
            "Not enough traceable news sources."
        )

    print(
        f"Script generated: {len(narration.split())} words"
    )

    return data, used_sources


# ============================================================
# STEP 3 — GENERATE FREE BRITISH-ENGLISH AUDIO
# ============================================================

async def generate_audio(narration, output_file):
    sentences = re.split(
        r"(?<=[.!?])\s+", narration
    )

    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) > 2300 and current:
            chunks.append(current.strip())
            current = ""

        current += sentence + " "

    if current.strip():
        chunks.append(current.strip())

    parts = []

    for index, chunk in enumerate(chunks):
        part = output_file.with_name(
            f"_audio_part_{index:02d}.mp3"
        )

        print(
            f"Generating audio segment {index + 1}/{len(chunks)}"
        )

        speaker = edge_tts.Communicate(
            chunk,
            VOICE,
            rate="-4%",
        )

        await speaker.save(str(part))

        if not part.exists() or part.stat().st_size < 1000:
            raise RuntimeError(
                f"Audio segment {index + 1} failed."
            )

        parts.append(part)

    manifest = output_file.with_suffix(".txt")

    manifest.write_text(
        "".join(
            f"file '{part.resolve()}'\n"
            for part in parts
        ),
        encoding="utf-8",
    )

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(manifest),
                "-c:a", "libmp3lame",
                "-b:a", "64k",
                str(output_file),
            ],
            check=True,
        )
    finally:
        manifest.unlink(missing_ok=True)

        for part in parts:
            part.unlink(missing_ok=True)

    print(f"Audio generated: {output_file}")


# ============================================================
# STEP 4 — CREATE THE PODCAST RSS FEED AND WEB PAGE
# ============================================================

def publish_episode(now, data, sources, audio_file):
    SITE.mkdir(parents=True, exist_ok=True)
    EPISODES.mkdir(parents=True, exist_ok=True)

    date = now.strftime("%Y-%m-%d")

    metadata = {
        "date": date,
        "title": clean(data["title"]),
        "summary": clean(data["summary"]),
        "sources": sources,
        "audio": f"episodes/{date}.mp3",
        "created": now.isoformat(),
    }

    metadata_file = EPISODES / f"{date}.json"

    metadata_file.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    episodes = []

    for file in EPISODES.glob("*.json"):
        try:
            episodes.append(
                json.loads(file.read_text(encoding="utf-8"))
            )
        except Exception:
            continue

    episodes.sort(
        key=lambda episode: episode["date"],
        reverse=True,
    )

    # Keep a small archive for free GitHub Pages hosting.
    for old in episodes[MAX_EPISODES:]:
        (EPISODES / f"{old['date']}.json").unlink(
            missing_ok=True
        )
        (EPISODES / f"{old['date']}.mp3").unlink(
            missing_ok=True
        )

    episodes = episodes[:MAX_EPISODES]

    # Generate RSS XML
    ITUNES = "http://www.itunes.com/dtds/podcast-1.0"
    ET.register_namespace("itunes", ITUNES)

    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")

    channel_fields = {
        "title": "Welcome to Paradise — Daily Global Briefing",
        "link": BASE_URL + "/",
        "description": (
            "An English-language personal daily news podcast. "
            "AI summaries based on attributed news-feed excerpts."
        ),
        "language": "en",
    }

    for tag, value in channel_fields.items():
        ET.SubElement(channel, tag).text = value

    ET.SubElement(
        channel, f"{{{ITUNES}}}explicit"
    ).text = "false"

    for episode in episodes:
        item = ET.SubElement(channel, "item")

        ET.SubElement(
            item, "title"
        ).text = episode["title"]

        source_text = "\n".join(
            f"{source['source']}: {source['url']}"
            for source in episode["sources"]
        )

        ET.SubElement(
            item, "description"
        ).text = episode["summary"] + "\n\n" + source_text

        audio_url = BASE_URL + "/" + episode["audio"]
        audio_path = SITE / episode["audio"]

        ET.SubElement(
            item,
            "enclosure",
            url=audio_url,
            length=str(audio_path.stat().st_size),
            type="audio/mpeg",
        )

        ET.SubElement(
            item, "guid", isPermaLink="false"
        ).text = episode["date"]

        created = dt.datetime.fromisoformat(
            episode["created"]
        )

        ET.SubElement(
            item, "pubDate"
        ).text = email.utils.format_datetime(created)

    ET.ElementTree(rss).write(
        SITE / "feed.xml",
        encoding="utf-8",
        xml_declaration=True,
    )

    # Generate a simple website with an audio player.
    articles = []

    for episode in episodes:
        source_links = "".join(
            (
                '<li><a href="'
                + html.escape(source["url"], quote=True)
                + '">'
                + html.escape(
                    source["source"] + ": " + source["title"]
                )
                + "</a></li>"
            )
            for source in episode["sources"]
        )

        articles.append(f"""
        <article>
            <h2>{html.escape(episode["title"])}</h2>
            <p>{episode["date"]}</p>
            <audio controls preload="none"
                   src="{html.escape(episode["audio"])}">
            </audio>
            <p>{html.escape(episode["summary"])}</p>
            <details>
                <summary>Sources</summary>
                <ul>{source_links}</ul>
            </details>
        </article>
        """)

    page = f"""
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport"
              content="width=device-width, initial-scale=1">
        <title>Welcome to Paradise</title>
        <style>
            body {{
                background: #111820;
                color: #f4f4f4;
                font-family: system-ui, sans-serif;
                max-width: 750px;
                margin: 50px auto;
                padding: 20px;
                line-height: 1.6;
            }}
            h1 {{ font-size: 2.5rem; }}
            article {{
                border-top: 1px solid #445;
                padding: 25px 0;
            }}
            audio {{ width: 100%; }}
            a {{ color: #9dcfff; }}
        </style>
    </head>
    <body>
        <h1>Welcome to Paradise</h1>
        <p>Your daily global news briefing</p>
        <p><a href="feed.xml">Subscribe to the podcast</a></p>
        <p>AI-generated summaries of news-feed excerpts.
        Consult the linked reporting for verification.</p>
        {''.join(articles)}
    </body>
    </html>
    """

    (SITE / "index.html").write_text(
        page,
        encoding="utf-8",
    )

    (SITE / ".nojekyll").touch()

    print("Podcast RSS feed and website generated.")


# ============================================================
# STEP 5 — MAIN AUTOMATION
# ============================================================

def main():
    now = dt.datetime.now(TIMEZONE)

    print(f"Current Bordeaux time: {now.isoformat()}")
    print(f"Manual override: {FORCE_RUN}")

    # Manual GitHub Actions runs bypass the time restriction.
    if not FORCE_RUN:
        if now.weekday() >= 5 or now.hour not in (6, 7):
            print(
                "Skipping: outside weekday morning window."
            )
            return

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Missing OPENAI_API_KEY GitHub secret."
        )

    if not BASE_URL:
        raise RuntimeError(
            "Missing PODCAST_BASE_URL GitHub variable."
        )

    date = now.strftime("%Y-%m-%d")
    audio_file = EPISODES / f"{date}.mp3"

    if audio_file.exists() and not FORCE_RUN:
        print("Today's episode already exists.")
        return

    print("Starting Welcome to Paradise production.")

    stories = collect_news(now)

    data, sources = generate_script(stories, now)

    EPISODES.mkdir(parents=True, exist_ok=True)

    try:
        asyncio.run(
            generate_audio(data["narration"], audio_file)
        )

        publish_episode(
            now, data, sources, audio_file
        )

    except Exception:
        audio_file.unlink(missing_ok=True)
        raise

    print("SUCCESS: Podcast episode created.")
    print(f"MP3: {audio_file}")
    print(f"Feed: {BASE_URL}/feed.xml")


if __name__ == "__main__":
    main()
