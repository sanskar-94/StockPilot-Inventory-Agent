# Connected mode — full setup (Shopify + NIM + Slack + n8n)

Demo mode needs none of this. This guide is only for the live store path: n8n
on a schedule pulling Shopify, the engine reasoning with NVIDIA NIM, and Slack
as the approval gate.

There are exactly **three external APIs**. Everything else is local Python.

---

## 0. Run the engine where n8n can reach it

```bash
source .venv/bin/activate
cp .env.example .env            # fill in NIM_API_KEY (see step 2)
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

If n8n runs on the **same machine**, the engine is `http://localhost:8000`.
If n8n runs in **Docker**, use `http://host.docker.internal:8000` instead, and
update the two `Engine ·` nodes in the workflow.

---

## 1. Shopify Admin API  (read store data + write the order back)

1. In your Shopify admin: **Settings → Apps and sales channels → Develop apps →
   Create an app**.
2. **Configure Admin API scopes** — enable at least:
   `read_products`, `read_inventory`, `read_orders`, `write_draft_orders`.
3. **Install app**, then copy the **Admin API access token** (`shpat_...`).
4. In n8n: **Credentials → New → Header Auth**, name it **`Shopify Admin Token`**:
   - Header **Name**: `X-Shopify-Access-Token`
   - Header **Value**: your `shpat_...` token
5. In the workflow, replace `YOUR-STORE.myshopify.com` in the three Shopify
   nodes with your real store domain.

> Note: Shopify's `products.json` doesn't return unit cost or lead time. The
> workflow defaults cost to 40% of price and lead time to 7 days. To be exact,
> store real costs/lead times as product metafields (or a side CSV) and adjust
> the *Build Engine Payload* node.

---

## 2. NVIDIA NIM  (the reasoning + writing layer)

1. Sign up free at **https://build.nvidia.com**.
2. Pick a model (default `meta/llama-3.1-70b-instruct`) → **Get API Key** →
   copy the `nvapi-...` key.
3. Put it in `.env` next to the engine:
   ```
   NIM_API_KEY=nvapi-xxxxxxxx
   NIM_MODEL=meta/llama-3.1-70b-instruct
   ```
4. **Fallback (recommended)** so a rate limit never stops a run. Any
   OpenAI-style endpoint works (OpenAI, Groq, Together, OpenRouter):
   ```
   FALLBACK_API_KEY=sk-...
   FALLBACK_URL=https://api.openai.com/v1/chat/completions
   FALLBACK_MODEL=gpt-4o-mini
   ```

NIM is called **once per run** from inside the engine (not per SKU), and it
only ranks/groups/summarizes — never changes a number.

---

## 3. Slack  (the approval gate)

1. Create an app at **https://api.slack.com/apps → Create New App → From scratch**.
2. **OAuth & Permissions → Bot Token Scopes**: add `chat:write`. Install to the
   workspace and copy the **Bot User OAuth Token** (`xoxb-...`).
3. Invite the bot to your channel: `/invite @YourBot` in `#inventory`.
4. In n8n: **Credentials → New → Header Auth**, name it **`Slack Bot Token`**:
   - Header **Name**: `Authorization`
   - Header **Value**: `Bearer xoxb-...`  (include the word `Bearer`)
5. Edit the channel in the **Build Slack Message** node (default `#inventory`).

The **Interactivity** request URL is wired in step 5 below (it needs the
workflow's webhook URL, which only exists after import).

---

## 4. Import the workflow into n8n

1. n8n → **Workflows → Import from File** → choose
   `n8n/stockpilot_workflow.json`.
2. Open each `Shopify ·` and `Slack ·` node and pick the credential you created
   from the dropdown (they import as "select credential").
3. **Save**, then toggle the workflow **Active**.

Self-hosted n8n install, for reference:
```bash
npm install -g n8n      # or: npx n8n
n8n                     # opens http://localhost:5678
```

---

## 5. Wire the Slack buttons back to n8n

1. With the workflow **active**, open the **Slack Interactivity Webhook** node
   and copy its **Production URL**
   (e.g. `https://your-n8n-host/webhook/stockpilot-approval`).
2. Slack app → **Interactivity & Shortcuts** → toggle **On** → paste that URL
   into **Request URL** → **Save Changes**.

Now an **Approve** click fetches the engine's cached plan and creates a Shopify
**draft order**; a **Reject** click places nothing. Both post a confirmation
back to Slack.

---

## End-to-end test

1. Engine running (`/health` returns ok).
2. Workflow active, credentials set, store domain replaced.
3. In n8n, open the workflow and click **Execute Workflow** (runs the schedule
   path immediately) — you should see a Block Kit message land in Slack.
4. Click **Approve** — check Shopify **Draft Orders** for a new `stockpilot`-
   tagged draft, and watch the confirmation post in the Slack thread.

---

## Checklist

| What | Where you set it |
|---|---|
| Shopify store domain | three `Shopify ·` nodes (replace `YOUR-STORE`) |
| Shopify Admin token | n8n credential `Shopify Admin Token` |
| NIM API key | engine `.env` (`NIM_API_KEY`) |
| Fallback LLM key | engine `.env` (`FALLBACK_*`) — optional but recommended |
| Slack bot token | n8n credential `Slack Bot Token` |
| Slack channel | `Build Slack Message` node |
| Engine URL | two `Engine ·` nodes (localhost vs host.docker.internal) |
| Slack interactivity URL | Slack app settings ← n8n webhook Production URL |
