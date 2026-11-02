# VONtex

A personal AI workspace. The static frontend is hosted on GitHub Pages; Gemini requests go through a Cloudflare Worker so the Gemini API key is never shipped to the browser.

## Current architecture

- `index.html`: responsive GitHub Pages frontend.
- `worker/src/index.js`: Cloudflare Worker API; validates the app access key and calls Gemini.
- `worker/wrangler.toml`: Wrangler deployment configuration.
- Browser local storage: chat history, memory notes, settings, and the Worker access key.

The backend supports text chat, optional Gemini Google Search grounding with source links, and one attached PDF or text-based document per message. Saved memory notes are passed as context. It does not provide autonomous actions, account login, or cross-device sync.

## 1. Create a Gemini API key

1. Open [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Create an API key for the Gemini API.
3. Keep the key private. Do not paste it into `index.html`, GitHub Pages settings, or any public file.

The Worker currently uses `gemini-3.6-flash`. Gemini free-tier access has model-specific quotas and can change; check the current [Gemini API pricing and limits](https://ai.google.dev/gemini-api/docs/pricing) before relying on it.

## 2. Deploy the Cloudflare Worker

The API workflow registers the account `workers.dev` subdomain `vontex-lokesh2742` if the Cloudflare account has not registered one yet. The Worker URL will be `https://vontex-api.vontex-lokesh2742.workers.dev`.

Install Node.js, then open PowerShell or a terminal in the repository folder:

```powershell
npx wrangler login
npx wrangler deploy --config worker/wrangler.toml
```

Wrangler prints a Worker URL similar to:

```
https://vontex-api.YOUR-SUBDOMAIN.workers.dev
```

Keep this URL for the next step.

## 3. Add secrets to Cloudflare

Create a separate long random app access key. In PowerShell, you can generate one with:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Add the Gemini API key:

```powershell
npx wrangler secret put GEMINI_API_KEY --config worker/wrangler.toml
```

Paste the Gemini key when prompted.

Add the app access key:

```powershell
npx wrangler secret put VONTEX_ACCESS_KEY --config worker/wrangler.toml
```

Paste the long random access key when prompted. Store it somewhere safe. Never commit either secret to Git.

Deploy again after setting secrets:

```powershell
npx wrangler deploy --config worker/wrangler.toml
```

## 4. Connect the website

1. Open your GitHub Pages VONtex website and refresh it after the latest commit deploys.
2. Open **Settings**.
3. Paste your Worker URL into **Cloudflare Worker URL**.
4. Paste the same random access key used for `VONTEX_ACCESS_KEY` into **VONtex access key**.
5. Select **Save connection**, then **Test connection**.
6. Return to Workspace and send a message.

The health check confirms that the Worker is reachable and whether its secrets exist. It does not make a paid request to Gemini.

## Security notes

- The Gemini API key is stored as a Cloudflare Worker secret and is not exposed to the browser.
- The app access key is a lightweight shared-secret gate, not full user authentication. Anyone who knows it can use your Worker quota. Do not share it.
- The access key is stored in this browser's local storage when you save Settings. Avoid using shared/public computers.
- Web research uses Gemini's Google Search grounding when you turn it on for a message. Search results can use additional quota and are returned with source links.
- Attached PDFs and text files are sent to Gemini through the Worker for that request. Attachments are not stored in chat history; the local chat stores the message and filename only. The app accepts one PDF, TXT, Markdown, CSV, or JSON file up to 5 MB per message.
- The Worker only accepts requests from the configured GitHub Pages origin and local development origins listed in `worker/src/index.js`. Update `ALLOWED_ORIGINS` if you use a custom domain.
- Gemini's free tier has quotas and its data-use terms differ from paid tiers. Avoid sending sensitive personal, financial, medical, or confidential information.
- GitHub Pages is static hosting; it cannot safely store private API keys or run this backend.

## Local development

The Worker can be run locally with Wrangler. Create `worker/.dev.vars` (this file is ignored by Git) with:

```dotenv
GEMINI_API_KEY=your_gemini_api_key
VONTEX_ACCESS_KEY=your_random_app_access_key
```

Then run:

```powershell
npx wrangler dev --config worker/wrangler.toml
```

The local frontend origin `http://localhost:5500` is allowed by default.

## Next milestones

1. Streaming responses and improved chat UX.
2. Real file upload and document understanding.
3. Persistent database-backed memory and chat sync.
4. Tool calling with explicit approval before actions.
5. Web research, coding tools, and sandboxed execution.

