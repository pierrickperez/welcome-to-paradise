# Welcome to Paradise — your personal weekday news podcast

A low-cost GitHub Actions workflow that creates an English-language news briefing, converts it to MP3 using a free online neural voice, and publishes a podcast RSS feed via GitHub Pages.

## Important limitations

- **Unlisted, not private:** GitHub Pages content is publicly reachable by anyone who knows or discovers its URL. Never place secrets or personal information in the script, feed, or repository.
- **Voice is not guaranteed:** `edge-tts` uses an unofficial interface to an online voice service; it may change, fail, or be rate-limited. Use only if acceptable under the service's terms. A supported paid TTS API is the reliable fallback.
- **Research is RSS-excerpt based:** The script uses recent RSS titles and summaries, not full paywalled articles. It refuses to publish if too few recent items are available. AI can still make mistakes; read linked sources for consequential facts.
- **Schedule is approximate:** GitHub Actions scheduled jobs can run late or be skipped during platform congestion. The script handles Europe/Paris daylight saving, but 08:00 publication is not guaranteed.
- **Storage:** Only the latest five episodes remain in the published site. MP3 files stay in Git history, so repository size grows over time. This is a small personal prototype, not a scalable podcast-hosting service.
- **API billing:** ChatGPT Plus does not include OpenAI API usage. Set a usage budget and alerts on your OpenAI API account.

## Setup (15–30 minutes)

1. Sign up for [GitHub](https://github.com) and create a **public** repository named `welcome-to-paradise`. GitHub Pages on GitHub Free requires a public repository. Do not store any secrets in it.
2. Upload **the contents of this folder**, including `.github/workflows/daily.yml`, `scripts/build.py`, `requirements.txt`, and `docs/.nojekyll`. GitHub's web uploader may hide dotfiles; if `.github` is missing, create `.github/workflows/daily.yml` through **Add file → Create new file** with that path.
3. Go to [OpenAI Platform](https://platform.openai.com/api-keys), create an API key, and set an API spending limit or alerts. The API is separately billed.
4. In the GitHub repository go to **Settings → Secrets and variables → Actions → Secrets → New repository secret**. Name: `OPENAI_API_KEY`. Value: your API key. Never paste the key into a repository file.
5. Under **Settings → Secrets and variables → Actions → Variables**, create `PODCAST_BASE_URL` with value `https://YOUR_GITHUB_USERNAME.github.io/welcome-to-paradise` (replace your actual username and repository name).
6. Go to **Settings → Pages → Build and deployment** and select **Deploy from a branch**, branch `main`, folder `/docs`. Save. If your default branch is named differently, select that branch.
7. Go to **Actions → Daily podcast → Run workflow** to create a test episode. Wait for the job to finish. You can inspect errors in its job log.
8. Open `https://YOUR_GITHUB_USERNAME.github.io/welcome-to-paradise/` (Pages deployment can take a few minutes). Check the audio and source links.
9. In **Apple Podcasts** on iPhone, open Library, then the `...` menu, choose **Follow a Show by URL**, and paste `https://YOUR_GITHUB_USERNAME.github.io/welcome-to-paradise/feed.xml`.
10. Once the test works, the GitHub workflow runs automatically on weekday mornings. GitHub scheduled workflows require Actions to be enabled; inactive public repositories can have schedules disabled after extended inactivity.

## How it works

- RSS feeds from BBC, France 24 and The Guardian supply recent attributed headlines and summaries.
- OpenAI API creates a 10–15 minute English narration, a short summary, and source references.
- `edge-tts` synthesizes a British English voice (`en-GB-RyanNeural`) and ffmpeg joins the audio.
- The script updates `docs/feed.xml`, `docs/index.html` and up to five MP3 files.
- GitHub Actions commits the changes; GitHub Pages publishes them.

## Customization

- Voice: edit `VOICE` in `.github/workflows/daily.yml`. Use a supported `edge-tts` voice name.
- Length and editorial style: edit the `prompt` in `scripts/build.py`.
- Opening: edit `Start exactly with "Welcome to Paradise!"` in the prompt.
- Archive: set `MAX_EPISODES` in the workflow (default 5).
- News sources: edit the `FEEDS` dictionary in `scripts/build.py`.
- Run manually: Actions → Daily podcast → Run workflow. Manual runs force an episode even outside the weekday window.

## Approximate cost

GitHub Actions and Pages may fit within GitHub Free's published usage limits for this small prototype. The online voice is currently usable without a separate paid account, but has no service guarantee. OpenAI API usage is pay-as-you-go, and costs depend on the model and amount of source text. Review current pricing and set billing alerts before activating.

## Security

The feed is accessible to anyone with the URL. Do not include personal or company-confidential information. Rotate the API key if accidentally exposed. Review GitHub Actions and GitHub Pages usage policies and limits before depending on this for long-term hosting.
