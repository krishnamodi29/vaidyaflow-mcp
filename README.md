# VaidyaFlow MCP Server 🏥

**OPD Co-Pilot for High-Volume Government Hospitals**

Built for the [Agents Assemble Hackathon 2026](https://agents-assemble.devpost.com/) — Prompt Opinion Platform.

---

## The Problem

Government OPD doctors in India see 80–120 patients per shift — one every 3–4 minutes. In that time they need to recall the patient's full history, check drug safety, review labs, and document everything. Paper files get lost. Systems don't talk. Critical information falls through the cracks during shift changes.

**VaidyaFlow is the 10-second patient brief that doctors never had.**

---

## What It Does

VaidyaFlow is an MCP server exposing 4 clinical safety tools:

| Tool | What it does |
|------|-------------|
| `get_patient_brief` | Instant patient context card — conditions, meds, labs, safety flags |
| `check_prescription_safety` | Drug-condition contraindication checker with safer alternatives |
| `get_abnormal_labs` | Flags abnormal lab values with clinical interpretation |
| `generate_handoff_note` | Structured shift handoff document — no more verbal briefings |

---

## Tech Stack

- **Python + FastAPI** — MCP server
- **HAPI FHIR** — Public sandbox FHIR server (synthetic data only)
- **Prompt Opinion Platform** — A2A agent orchestration + SHARP context
- **Railway** — Deployment

---

## Local Development

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Copy env
cp .env.example .env

# 3. Run server
uvicorn main:app --reload --port 8000

# 4. Test manifest
curl http://localhost:8000/mcp

# 5. Expose to internet (for Prompt Opinion)
ngrok http 8000
# Copy the https URL
```

---

## Deploy to Railway

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login
railway login

# 3. Deploy
railway init
railway up

# 4. Get your URL
railway open
```

Your MCP URL will be: `https://your-app.railway.app/mcp`

---

## Register on Prompt Opinion

1. Go to **app.promptopinion.ai → Workspace Hub → Add MCP Server**
2. Paste your Railway URL + `/mcp`
3. Check **"Pass FHIR Context"** ✅
4. Click **Test** — you should see 4 tools appear
5. Save

---

## Architecture

```
[Doctor on Prompt Opinion]
        ↓
[VaidyaFlow A2A Agent]  ← configured on Prompt Opinion platform
        ↓ calls MCP tools via SHARP context
[VaidyaFlow MCP Server] ← this repository
        ↓
[HAPI FHIR Server]      ← synthetic patient data
```

---

## Data Safety

- Uses **synthetic data only** (HAPI FHIR public sandbox)
- No real PHI ever processed
- FHIR tokens passed via SHARP context — never stored
- Compliant with hackathon data integrity rules

---

*VaidyaFlow — because every second counts when you have 100 patients waiting.*
