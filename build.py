"""Generate a short private-use news podcast from attributed RSS excerpts."""
import asyncio
import datetime as dt
import email.utils
import html
import json
import os
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET

import edge_tts
import feedparser
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'docs'
EPISODES = SITE / 'episodes'
PARIS = ZoneInfo('Europe/Paris')
FEEDS = {
    'BBC World': 'https://feeds.bbci.co.uk/news/world/rss.xml',
    'BBC Business': 'https://feeds.bbci.co.uk/news/business/rss.xml',
    'France 24': 'https://www.france24.com/en/rss',
    'The Guardian World': 'https://www.theguardian.com/world/rss',
    'The Guardian Business': 'https://www.theguardian.com/uk/business/rss',
    'The Guardian France': 'https://www.theguardian.com/world/france/rss',
}
VOICE = os.getenv('VOICE', 'en-GB-RyanNeural')
MODEL = os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')
MAX_EPISODES = int(os.getenv('MAX_EPISODES', '5'))


def clean(s):
    s = re.sub('<[^>]*>', ' ', str(s or ''))
    return re.sub(r'\s+', ' ', html.unescape(s)).strip()


def gather(now):
    stories = []
    for source, url in FEEDS.items():
        try:
            parsed = feedparser.parse(url, request_headers={'User-Agent': 'WelcomeToParadisePersonalPodcast/1.0'})
            for e in parsed.entries[:14]:
                stamp = e.get('published_parsed') or e.get('updated_parsed')
                if not stamp:
                    continue
                published = dt.datetime(*stamp[:6], tzinfo=dt.timezone.utc)
                age = (now.astimezone(dt.timezone.utc) - published).total_seconds() / 3600
                if not -2 <= age <= 40:
                    continue
                title = clean(e.get('title', ''))
                link = e.get('link', '')
                summary = clean(e.get('summary', ''))[:750]
                if title and link:
                    stories.append({'source': source, 'title': title, 'summary': summary,
                                    'url': link, 'published': published.isoformat()})
        except Exception as exc:
            print(f'Warning: {source}: {exc}', file=sys.stderr)
    dedup = {}
    for item in stories:
        key = re.sub(r'[^a-z0-9]', '', item['title'].lower())[:85]
        dedup.setdefault(key, item)
    return sorted(dedup.values(), key=lambda x: x['published'], reverse=True)[:65]


def make_script(stories, now):
    if len(stories) < 8:
        raise RuntimeError(f'Only {len(stories)} recent RSS stories; refusing to publish an unreliable briefing.')
    client = OpenAI()
    prompt = f'''You are the careful editor of a PERSONAL, ENGLISH-LANGUAGE news podcast called Welcome to Paradise.
Today is {now.strftime('%A %d %B %Y')} in Europe/Paris. Use ONLY the following RSS titles and summaries, NOT your background knowledge, for specific current events. They are headlines and excerpts, not complete reporting. Do not invent quotes, numbers, dates, motives, outcomes or details. Do not claim independent verification. Attribute major claims to the named outlet. Where excerpts lack enough information, say so or omit the story. Avoid sensationalism. Choose 4–6 consequential distinct stories; include France/Europe, world affairs, economics, and when supported by the inputs, consumer goods, hospitality or premium drinks. Do NOT force a drinks story when sources don't support one. Give fewer stories with deeper *cautious* context and practical implications, clearly separating inference from sourced fact. The last story should be a thoughtful deeper analysis grounded in the source headlines. No ungrounded predictions. Journalistic, calm British-English presenter style, fluid transitions. Start exactly with "Welcome to Paradise!" and a date-based introduction. End with what to watch next. Aim for 1200–1600 words; if sources are thin, be shorter instead of padding. Plain prose only: no headings, markdown, bullets or URLs in narration.
Return valid JSON ONLY with keys: title (string), narration (string), summary (string, 100-160 words), source_indices (array of integer indices of source items actually used). Do not put citations inside narration. No copyrighted article passages beyond short quotes.
RSS SOURCES:\n{json.dumps([dict(index=i, **s) for i,s in enumerate(stories)], ensure_ascii=False)}'''
    response = client.chat.completions.create(model=MODEL, messages=[{'role': 'user', 'content': prompt}], response_format={'type': 'json_object'}, temperature=0.25)
    data = json.loads(response.choices[0].message.content)
    narration = data.get('narration', '').strip()
    if len(narration.split()) < 650:
        raise RuntimeError('Narration too short; no episode published.')
    used = [stories[i] for i in data.get('source_indices', []) if isinstance(i, int) and 0 <= i < len(stories)]
    if len(used) < 3:
        raise RuntimeError('Not enough traceable sources; no episode published.')
    return data, used


async def synthesize(text, output):
    # Edge TTS is a free, unofficial interface to online voices; availability is not guaranteed.
    # Split on sentence boundaries to keep individual requests manageable.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ''
    for sentence in sentences:
        if len(current) + len(sentence) > 2300 and current:
            chunks.append(current.strip()); current = ''
        current += sentence + ' '
    if current.strip():
        chunks.append(current.strip())
    parts = []
    for i, chunk in enumerate(chunks):
        part = output.with_name(f'_part_{i:02}.mp3')
        await edge_tts.Communicate(chunk, VOICE, rate='-4%').save(str(part))
        if part.stat().st_size < 1000:
            raise RuntimeError(f'Audio part {i} was empty')
        parts.append(part)
    # MP3 stream concatenation is supported by ffmpeg concat demuxer for consistent streams.
    import subprocess
    manifest = output.with_suffix('.txt')
    manifest.write_text(''.join(f"file '{p.resolve()}'\n" for p in parts), encoding='utf-8')
    try:
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(manifest), '-c:a', 'libmp3lame', '-b:a', '64k', str(output)], check=True)
    finally:
        manifest.unlink(missing_ok=True)
        for p in parts: p.unlink(missing_ok=True)


def publish(now, data, sources, audio):
    SITE.mkdir(exist_ok=True); EPISODES.mkdir(exist_ok=True)
    date = now.strftime('%Y-%m-%d')
    meta_path = EPISODES / f'{date}.json'
    metadata = {'date': date, 'title': clean(data['title']), 'summary': clean(data['summary']),
                'sources': sources, 'audio': f'episodes/{date}.mp3', 'created': now.isoformat()}
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    episodes = []
    for file in EPISODES.glob('*.json'):
        try: episodes.append(json.loads(file.read_text(encoding='utf-8')))
        except Exception: pass
    episodes.sort(key=lambda x: x['date'], reverse=True)
    for old in episodes[MAX_EPISODES:]:
        (EPISODES / f"{old['date']}.json").unlink(missing_ok=True)
        (EPISODES / f"{old['date']}.mp3").unlink(missing_ok=True)
    episodes = episodes[:MAX_EPISODES]
    base = os.environ['PODCAST_BASE_URL'].rstrip('/')
    rss = ET.Element('rss', version='2.0', attrib={'xmlns:itunes': 'http://www.itunes.com/dtds/podcast-1.0'})
    ch = ET.SubElement(rss, 'channel')
    for tag, value in [('title', 'Welcome to Paradise — Private Daily Briefing'),
                       ('link', base + '/'), ('description', 'Personal English-language weekday news briefing. RSS excerpts are summarized by AI; verify primary sources.'),
                       ('language', 'en'), ('copyright', 'Personal listening only')]:
        ET.SubElement(ch, tag).text = value
    ET.SubElement(ch, '{http://www.itunes.com/dtds/podcast-1.0}explicit').text = 'false'
    for ep in episodes:
        item = ET.SubElement(ch, 'item')
        ET.SubElement(item, 'title').text = ep['title']
        links = '\n'.join(f"{s['source']}: {s['title']} — {s['url']}" for s in ep['sources'])
        ET.SubElement(item, 'description').text = ep['summary'] + '\n\nSources:\n' + links
        url = base + '/' + ep['audio']
        ET.SubElement(item, 'enclosure', url=url, length=str((SITE / ep['audio']).stat().st_size), type='audio/mpeg')
        ET.SubElement(item, 'guid', isPermaLink='false').text = ep['date']
        d = dt.datetime.fromisoformat(ep['created'])
        ET.SubElement(item, 'pubDate').text = email.utils.format_datetime(d)
    ET.ElementTree(rss).write(SITE / 'feed.xml', encoding='utf-8', xml_declaration=True)
    cards = []
    for ep in episodes:
        links = ''.join(f'<li><a href="{html.escape(s["url"], quote=True)}" rel="noopener noreferrer">{html.escape(s["source"] + ": " + s["title"])}</a></li>' for s in ep['sources'])
        cards.append(f'<article><h2>{html.escape(ep["title"])}</h2><small>{ep["date"]}</small><p>{html.escape(ep["summary"])}</p><audio controls preload="none" src="{html.escape(ep["audio"])}"></audio><details><summary>Sources</summary><ul>{links}</ul></details></article>')
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Welcome to Paradise</title><style>body{{font-family:system-ui,-apple-system,sans-serif;max-width:720px;margin:45px auto;padding:0 20px;background:#10151c;color:#f2f4f6;line-height:1.6}}a{{color:#8fcbff}}article{{border-top:1px solid #394450;padding:20px 0}}audio{{width:100%}}small{{color:#aeb9c4}}details{{margin-top:15px}}h1{{font-size:2.3rem}}</style></head><body><h1>Welcome to Paradise</h1><p>Private-use daily news briefing · <a href="feed.xml">Podcast RSS feed</a></p><p>Automated AI summaries of attributed news-feed excerpts. Check original sources before relying on important claims. This URL is unlisted, not password protected.</p>{''.join(cards)}</body></html>'''
    (SITE / 'index.html').write_text(page, encoding='utf-8')
    (SITE / '.nojekyll').touch()


def main():
    now = dt.datetime.now(PARIS)
    if os.getenv('FORCE_RUN') != '1' and (now.weekday() >= 5 or now.hour not in (6, 7)):
        print(f'Skipping: local time {now.isoformat()} outside weekday morning window')
        return
    date = now.strftime('%Y-%m-%d')
    if (EPISODES / f'{date}.mp3').exists() and os.getenv('FORCE_RUN') != '1':
        print('Episode already exists today; skipping')
        return
    if not os.getenv('OPENAI_API_KEY') or not os.getenv('PODCAST_BASE_URL'):
        raise RuntimeError('Set OPENAI_API_KEY and PODCAST_BASE_URL in GitHub repository secrets/variables.')
    stories = gather(now)
    print(f'Found {len(stories)} recent articles')
    data, sources = make_script(stories, now)
    EPISODES.mkdir(exist_ok=True, parents=True)
    audio = EPISODES / f'{date}.mp3'
    try:
        asyncio.run(synthesize(data['narration'], audio))
        publish(now, data, sources, audio)
    except Exception:
        audio.unlink(missing_ok=True)
        raise
    print(f'Created {audio} ({audio.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
