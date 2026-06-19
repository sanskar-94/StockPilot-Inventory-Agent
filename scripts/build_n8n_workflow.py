"""Generate n8n/stockpilot_workflow.json deterministically.

Keeping the workflow in code means it is reviewable and regenerable. Run:
    python scripts/build_n8n_workflow.py
It writes a single importable JSON you load via n8n -> Import from File.

The workflow has two entry points (n8n allows several triggers per workflow):
  * a daily Schedule that pulls Shopify -> engine -> posts to Slack, and
  * a Webhook that Slack calls when someone clicks Approve / Reject.
"""
from __future__ import annotations

import json
import os

API_VERSION = "2024-10"
# n8n Cloud cannot reach your laptop — expose the engine with a tunnel
# (cloudflared / ngrok) or a deploy, and paste that public URL here.
ENGINE_URL = "https://YOUR-ENGINE-PUBLIC-URL"
STORE = "https://YOUR-STORE.myshopify.com"     # the connected dev store
SLACK_CHANNEL = "#inventory"                  # <-- edit to your channel
SHOPIFY_CRED = "Shopify Access Token account"  # the native n8n Shopify credential


def http_header_auth(cred_name: str) -> dict:
    return {"httpHeaderAuth": {"id": "REPLACE_ME", "name": cred_name}}


def shopify_cred() -> dict:
    return {"shopifyAccessTokenApi": {"id": "REPLACE_ME", "name": SHOPIFY_CRED}}


def slack_cred() -> dict:
    return {"slackApi": {"id": "REPLACE_ME", "name": "Slack API"}}


def slack_http(url: str, body_expr: str) -> dict:
    """Slack POST that authenticates with the predefined Slack credential, so
    the user just creates one 'Slack API' credential (the xoxb token)."""
    return {"method": "POST", "url": url,
            "authentication": "predefinedCredentialType",
            "nodeCredentialType": "slackApi",
            "sendBody": True, "specifyBody": "json", "jsonBody": body_expr,
            "sendHeaders": True,
            "headerParameters": {"parameters": [
                {"name": "Content-Type", "value": "application/json; charset=utf-8"}]},
            "options": {}}


# ── transform code (kept as readable JS strings) ───────────────────
BUILD_PAYLOAD_JS = r"""
// Flatten the Shopify pull into the engine's expected shape.
// The native Shopify node returns one item per order / product, so read all of
// them with .all() rather than a single { orders: [...] } envelope.
const shopifyOrders   = $('Shopify · Pull Orders').all().map(i => i.json);
const shopifyProducts = $('Shopify · Pull Products').all().map(i => i.json);

const orders = [];
for (const o of shopifyOrders) {
  const date = (o.created_at || '').slice(0, 10);
  for (const li of (o.line_items || [])) {
    if (!li.sku) continue;
    orders.push({
      date,
      sku: li.sku,
      units: li.quantity,
      price: parseFloat(li.price || 0),
      on_promo: (li.discount_allocations && li.discount_allocations.length) ? 1 : 0,
    });
  }
}

// Shopify products.json does not carry unit cost (it lives on inventory_item)
// or lead time. Use safe defaults; override per-SKU via metafields/config later.
const products = [];
for (const p of shopifyProducts) {
  for (const v of (p.variants || [])) {
    if (!v.sku) continue;
    products.push({
      sku: v.sku,
      cost: parseFloat(v.cost || (v.price ? v.price * 0.4 : 0)),
      price: parseFloat(v.price || 0),
      lead_time_days: 7,
      lead_time_std: 2,
      supplier: p.vendor || 'default',
      on_hand: (v.inventory_quantity != null) ? v.inventory_quantity : 0,
      on_order: 0,
      moq: 1,
      order_cost: 50,
      holding_rate: 0.25,
    });
  }
}

return [{ json: { orders, products, use_llm: true, render_charts: true } }];
""".strip()

BUILD_SLACK_JS = r"""
// Turn the engine response into a Slack Block Kit message with buttons.
const r = $json;
const summary = (r.reasoning && r.reasoning.summary) || 'Reorder plan ready.';
const ranked  = (r.reasoning && r.reasoning.ranked_orders) || [];
const top = ranked.slice(0, 5)
  .map(o => `• *${o.sku}* — order *${o.order_qty}*  _(${o.why})_`)
  .join('\n') || 'No SKUs need reordering right now.';

const blocks = [
  { type: 'header', text: { type: 'plain_text', text: '📦 StockPilot — reorder plan' } },
  { type: 'section', text: { type: 'mrkdwn', text: summary } },
  { type: 'section', text: { type: 'mrkdwn', text: top } },
  { type: 'context', elements: [ { type: 'mrkdwn',
      text: `Reorder *${r.reorder_count}* SKU(s) · est. *$${r.total_order_value}*` } ] },
  { type: 'actions', block_id: 'stockpilot_actions', elements: [
      { type: 'button', style: 'primary', action_id: 'approve',
        text: { type: 'plain_text', text: '✅ Approve' }, value: 'approve' },
      { type: 'button', style: 'danger', action_id: 'reject',
        text: { type: 'plain_text', text: '❌ Reject' }, value: 'reject' },
  ]},
];

return [{ json: { channel: '%SLACK_CHANNEL%', text: summary, blocks } }];
""".strip().replace("%SLACK_CHANNEL%", SLACK_CHANNEL)

PARSE_ACTION_JS = r"""
// Slack posts interactive payloads as x-www-form-urlencoded with one 'payload'
// field holding JSON. n8n exposes form fields under $json.body.
// Guard every access so a stray / malformed POST never errors the execution.
const body = $json.body || $json;
let raw = body.payload || $json.payload;
let payload = {};
try { payload = (typeof raw === 'string') ? JSON.parse(raw) : (raw || {}); } catch (e) { payload = {}; }
const action = (payload.actions && payload.actions[0]) || {};
return [{ json: {
  decision: action.value || action.action_id || 'ignore',
  user: (payload.user && (payload.user.username || payload.user.name)) || 'someone',
  response_url: payload.response_url || '',
  channel: (payload.channel && payload.channel.id) || '',
} }];
""".strip()

BUILD_DRAFT_JS = r"""
// Build a Shopify draft order from the engine's latest cached plan.
const plan = $json.plan || [];
const lines = plan.filter(p => p.order_qty > 0).map(p => ({
  title: p.sku,
  quantity: p.order_qty,
  price: p.unit_cost,            // cost basis; this is a purchase draft
}));
const draft_order = {
  draft_order: {
    line_items: lines.length ? lines : [{ title: 'No reorder needed', quantity: 1, price: 0 }],
    note: 'StockPilot auto-draft — approved in Slack',
    tags: 'stockpilot',
  },
};
return [{ json: draft_order }];
""".strip()


def node(id_, name, type_, version, pos, params, creds=None):
    n = {"id": id_, "name": name, "type": type_, "typeVersion": version,
         "position": pos, "parameters": params}
    if creds:
        n["credentials"] = creds
    return n


def sticky(id_, name, pos, w, h, content, color=7):
    return {"id": id_, "name": name, "type": "n8n-nodes-base.stickyNote",
            "typeVersion": 1, "position": pos,
            "parameters": {"width": w, "height": h, "color": color, "content": content}}


def http(method, url, *, body_expr=None, query=None, auth=None, headers=None):
    p = {"method": method, "url": url, "options": {}}
    if auth:
        p["authentication"] = "genericCredentialType"
        p["genericAuthType"] = "httpHeaderAuth"
    else:
        p["authentication"] = "none"
    if query:
        p["sendQuery"] = True
        p["queryParameters"] = {"parameters": [{"name": k, "value": v} for k, v in query.items()]}
    if headers:
        p["sendHeaders"] = True
        p["headerParameters"] = {"parameters": [{"name": k, "value": v} for k, v in headers.items()]}
    if body_expr is not None:
        p["sendBody"] = True
        p["specifyBody"] = "json"
        p["jsonBody"] = body_expr
    return p


def main():
    nodes = []

    # ── main flow ──────────────────────────────────────────────────
    nodes.append(node("schedule", "Daily Schedule 6am", "n8n-nodes-base.scheduleTrigger",
                      1.2, [-100, 0],
                      {"rule": {"interval": [{"field": "hours", "hoursInterval": 24,
                                              "triggerAtHour": 6}]}}))

    nodes.append(node("pullOrders", "Shopify · Pull Orders", "n8n-nodes-base.shopify",
                      1, [140, -80],
                      {"authentication": "accessToken", "resource": "order",
                       "operation": "getAll", "returnAll": True,
                       "options": {"status": "any"}},
                      shopify_cred()))

    nodes.append(node("pullProducts", "Shopify · Pull Products", "n8n-nodes-base.shopify",
                      1, [380, -80],
                      {"authentication": "accessToken", "resource": "product",
                       "operation": "getAll", "returnAll": True, "options": {}},
                      shopify_cred()))

    nodes.append(node("buildPayload", "Build Engine Payload", "n8n-nodes-base.code",
                      2, [620, -80], {"jsCode": BUILD_PAYLOAD_JS}))

    nodes.append(node("engineRun", "Engine · POST /run", "n8n-nodes-base.httpRequest",
                      4.2, [860, -80],
                      http("POST", f"{ENGINE_URL}/run", body_expr="={{ $json }}")))

    nodes.append(node("buildSlack", "Build Slack Message", "n8n-nodes-base.code",
                      2, [1100, -80], {"jsCode": BUILD_SLACK_JS}))

    nodes.append(node("slackPost", "Slack · Post Plan", "n8n-nodes-base.httpRequest",
                      4.2, [1340, -80],
                      slack_http("https://slack.com/api/chat.postMessage", "={{ $json }}"),
                      slack_cred()))

    # ── approval flow ──────────────────────────────────────────────
    nodes.append(node("webhook", "Slack Interactivity Webhook", "n8n-nodes-base.webhook",
                      2, [-100, 320],
                      {"httpMethod": "POST", "path": "stockpilot-approval",
                       "responseMode": "onReceived", "options": {}}))

    nodes.append(node("parseAction", "Parse Slack Action", "n8n-nodes-base.code",
                      2, [140, 320], {"jsCode": PARSE_ACTION_JS}))

    nodes.append(node("ifApproved", "Approved?", "n8n-nodes-base.if", 2, [380, 320],
                      {"conditions": {"options": {"caseSensitive": True, "version": 2},
                                      "combinator": "and",
                                      "conditions": [{
                                          "leftValue": "={{ $json.decision }}",
                                          "rightValue": "approve",
                                          "operator": {"type": "string", "operation": "equals"}}]}}))

    nodes.append(node("getPlan", "Engine · GET /plan", "n8n-nodes-base.httpRequest",
                      4.2, [620, 200], http("GET", f"{ENGINE_URL}/plan")))

    nodes.append(node("buildDraft", "Build Shopify Draft Order", "n8n-nodes-base.code",
                      2, [860, 200], {"jsCode": BUILD_DRAFT_JS}))

    # Native Shopify node has no draft-order op, so write back via HTTP Request
    # but authenticate with the SAME Shopify credential (predefined type).
    nodes.append(node("createDraft", "Shopify · Create Draft Order", "n8n-nodes-base.httpRequest",
                      4.2, [1100, 200],
                      {"method": "POST",
                       "url": f"{STORE}/admin/api/{API_VERSION}/draft_orders.json",
                       "authentication": "predefinedCredentialType",
                       "nodeCredentialType": "shopifyAccessTokenApi",
                       "sendBody": True, "specifyBody": "json", "jsonBody": "={{ $json }}",
                       "options": {}},
                      shopify_cred()))

    nodes.append(node("confirmApproved", "Slack · Confirm Approved", "n8n-nodes-base.httpRequest",
                      4.2, [1340, 200],
                      slack_http("https://slack.com/api/chat.postMessage",
                                 "={{ { channel: $('Parse Slack Action').first().json.channel, "
                                 "text: '✅ Approved by ' + $('Parse Slack Action').first().json.user "
                                 "+ ' — draft order created in Shopify.' } }}"),
                      slack_cred()))

    nodes.append(node("confirmRejected", "Slack · Confirm Rejected", "n8n-nodes-base.httpRequest",
                      4.2, [620, 440],
                      slack_http("https://slack.com/api/chat.postMessage",
                                 "={{ { channel: $('Parse Slack Action').first().json.channel, "
                                 "text: '❌ Rejected by ' + $('Parse Slack Action').first().json.user "
                                 "+ ' — no order placed.' } }}"),
                      slack_cred()))

    # ── sticky notes ───────────────────────────────────────────────
    nodes.append(sticky("note1", "Note Setup", [-120, -360], 780, 280,
        "## StockPilot — setup before first run\n"
        "1. **Shopify credential** (the native *Shopify Access Token API* credential, "
        "already named `Shopify Access Token account`): Shop Subdomain "
        "`YOUR-STORE.myshopify.com`, plus the app **Access Token** and **API secret key**. "
        "The two pull nodes and the draft-order writeback all reuse it.\n"
        "   • App scopes needed: `read_products, read_orders, read_all_orders, "
        "read_inventory, write_draft_orders`\n"
        "2. **Slack credential** (Credentials → New → *Slack API*, name it `Slack API`): "
        "Access Token = your bot token `xoxb-...` (OAuth & Permissions → Install App). "
        "Needs scope `chat:write`; invite the bot to the channel.\n"
        "3. **Engine (n8n Cloud)**: Cloud can't reach localhost. Expose the engine with "
        "`cloudflared tunnel --url http://localhost:8000` (or deploy it) and paste the public "
        "`https://...` URL into the two *Engine ·* nodes (replace `YOUR-ENGINE-PUBLIC-URL`).\n"
        "4. **Slack channel**: edit it in *Build Slack Message*.", 4))

    nodes.append(sticky("note2", "Note Approval", [-120, 200], 700, 200,
        "## Approval path (Slack buttons → n8n)\n"
        "Slack calls this **Webhook** when a button is clicked. After saving + "
        "activating the workflow, copy the **Production URL** of *Slack Interactivity "
        "Webhook* and paste it into your Slack app → **Interactivity & Shortcuts → "
        "Request URL**.\n\nOn **Approve** the engine's cached plan is fetched and a "
        "Shopify **draft order** is created; on **Reject** nothing is ordered. Both "
        "post a confirmation back to Slack.", 5))

    connections = {
        "Daily Schedule 6am": {"main": [[{"node": "Shopify · Pull Orders", "type": "main", "index": 0}]]},
        "Shopify · Pull Orders": {"main": [[{"node": "Shopify · Pull Products", "type": "main", "index": 0}]]},
        "Shopify · Pull Products": {"main": [[{"node": "Build Engine Payload", "type": "main", "index": 0}]]},
        "Build Engine Payload": {"main": [[{"node": "Engine · POST /run", "type": "main", "index": 0}]]},
        "Engine · POST /run": {"main": [[{"node": "Build Slack Message", "type": "main", "index": 0}]]},
        "Build Slack Message": {"main": [[{"node": "Slack · Post Plan", "type": "main", "index": 0}]]},
        "Slack Interactivity Webhook": {"main": [[{"node": "Parse Slack Action", "type": "main", "index": 0}]]},
        "Parse Slack Action": {"main": [[{"node": "Approved?", "type": "main", "index": 0}]]},
        "Approved?": {"main": [
            [{"node": "Engine · GET /plan", "type": "main", "index": 0}],
            [{"node": "Slack · Confirm Rejected", "type": "main", "index": 0}],
        ]},
        "Engine · GET /plan": {"main": [[{"node": "Build Shopify Draft Order", "type": "main", "index": 0}]]},
        "Build Shopify Draft Order": {"main": [[{"node": "Shopify · Create Draft Order", "type": "main", "index": 0}]]},
        "Shopify · Create Draft Order": {"main": [[{"node": "Slack · Confirm Approved", "type": "main", "index": 0}]]},
    }

    workflow = {
        "name": "StockPilot",
        "nodes": nodes,
        "connections": connections,
        "active": False,
        "settings": {"executionOrder": "v1"},
        "pinData": {},
        "meta": {"instanceId": "stockpilot-template"},
        "tags": [],
    }

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "n8n", "stockpilot_workflow.json")
    with open(out, "w") as f:
        json.dump(workflow, f, indent=2)
    print(f"Wrote {out} ({len(nodes)} nodes, {len(connections)} wired)")


if __name__ == "__main__":
    main()
