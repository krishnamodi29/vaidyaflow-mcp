"""
VaidyaFlow MCP Server — OPD Co-Pilot
Agents Assemble Hackathon 2026 | Prompt Opinion Platform
Official MCP Python SDK (FastMCP) with Streamable HTTP transport
"""

import os
import httpx
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

FHIR_BASE = os.getenv("FHIR_BASE_URL", "https://hapi.fhir.org/baseR4")
PORT = int(os.getenv("PORT", 8000))

mcp = FastMCP(
    "VaidyaFlow OPD Co-Pilot",
    json_response=True,
    host="0.0.0.0",
    port=PORT,
    streamable_http_path="/mcp",
    instructions=(
        "AI-powered patient safety tools for high-volume OPD clinicians. "
        "Surfaces medication risks, flags abnormal labs, generates 10-second "
        "patient briefs, and produces intelligent shift handoff notes."
    )
)

# ── FHIR helpers ──────────────────────────────────────────────────────────────

async def fhir_get(path: str, fhir_base: str = FHIR_BASE, token: str = None) -> dict:
    headers = {"Accept": "application/fhir+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(f"{fhir_base}/{path}", headers=headers)
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return {}


async def load_patient(patient_id: str) -> dict:
    p    = await fhir_get(f"Patient/{patient_id}")
    cond = await fhir_get(f"Condition?patient={patient_id}&clinical-status=active")
    meds = await fhir_get(f"MedicationRequest?patient={patient_id}&status=active")
    obs  = await fhir_get(f"Observation?patient={patient_id}&_sort=-date&_count=20")
    alrg = await fhir_get(f"AllergyIntolerance?patient={patient_id}")
    enc  = await fhir_get(f"Encounter?patient={patient_id}&_sort=-date&_count=3")
    return {
        "patient":      p,
        "conditions":   cond.get("entry", []),
        "medications":  meds.get("entry", []),
        "observations": obs.get("entry", []),
        "allergies":    alrg.get("entry", []),
        "encounters":   enc.get("entry", []),
    }


def name_of(p: dict) -> str:
    try:
        n = p.get("name", [{}])[0]
        return ((" ".join(n.get("given", []))) + " " + n.get("family", "")).strip() or "Unknown"
    except Exception:
        return "Unknown"


def parse_conditions(entries): return [
    (e.get("resource", {}).get("code", {}).get("text") or
     e.get("resource", {}).get("code", {}).get("coding", [{}])[0].get("display", ""))
    for e in entries
    if (e.get("resource", {}).get("code", {}).get("text") or
        e.get("resource", {}).get("code", {}).get("coding", [{}])[0].get("display", ""))
]


def parse_meds(entries): return [
    (e.get("resource", {}).get("medicationCodeableConcept", {}).get("text") or
     e.get("resource", {}).get("medicationCodeableConcept", {}).get("coding", [{}])[0].get("display", ""))
    for e in entries
    if (e.get("resource", {}).get("medicationCodeableConcept", {}).get("text") or
        e.get("resource", {}).get("medicationCodeableConcept", {}).get("coding", [{}])[0].get("display", ""))
]


def parse_labs(entries):
    labs = []
    for e in entries[:12]:
        r = e.get("resource", {})
        if r.get("resourceType") != "Observation":
            continue
        nm = r.get("code", {}).get("text") or r.get("code", {}).get("coding", [{}])[0].get("display", "")
        vq = r.get("valueQuantity", {})
        val = f"{vq.get('value','')} {vq.get('unit','')}".strip()
        dt  = r.get("effectiveDateTime", "")[:10]
        interp = r.get("interpretation", [{}])
        flag = interp[0].get("coding", [{}])[0].get("code", "") if interp else ""
        if nm and val:
            labs.append({"name": nm, "value": val, "date": dt, "flag": flag})
    return labs


def parse_allergies(entries): return [
    (e.get("resource", {}).get("code", {}).get("text") or
     e.get("resource", {}).get("code", {}).get("coding", [{}])[0].get("display", ""))
    for e in entries
    if (e.get("resource", {}).get("code", {}).get("text") or
        e.get("resource", {}).get("code", {}).get("coding", [{}])[0].get("display", ""))
]


# ── Drug safety knowledge base ─────────────────────────────────────────────────

RULES = {
    "ibuprofen":     (["Chronic kidney disease","CKD","Heart failure","Peptic ulcer","Warfarin"], ["Paracetamol"]),
    "naproxen":      (["Chronic kidney disease","CKD","Heart failure","Peptic ulcer"],            ["Paracetamol"]),
    "diclofenac":    (["Chronic kidney disease","CKD","Heart failure","Peptic ulcer"],            ["Paracetamol"]),
    "aspirin":       (["Peptic ulcer","Bleeding disorder"],                                       ["Paracetamol"]),
    "metformin":     (["Chronic kidney disease","CKD","Liver failure","Heart failure"],           ["Insulin","Sitagliptin"]),
    "glibenclamide": (["Chronic kidney disease","CKD","Elderly"],                                ["Gliclazide MR"]),
    "amiodarone":    (["Thyroid disease","Liver disease","Lung disease"],                        ["Consult cardiologist"]),
    "digoxin":       (["Chronic kidney disease","CKD","Hypokalemia"],                            ["Consult cardiologist"]),
    "gentamicin":    (["Chronic kidney disease","CKD"],                                          ["Amoxicillin","Cephalosporins"]),
    "nitrofurantoin":(["Chronic kidney disease","CKD"],                                          ["Trimethoprim","Cefalexin"]),
    "atenolol":      (["Asthma","COPD","Bradycardia"],                                           ["Amlodipine","Ramipril"]),
    "ramipril":      (["Pregnancy","Hyperkalemia"],                                              ["Amlodipine"]),
}


def drug_check(drug: str, conditions: list) -> tuple:
    dl = drug.lower()
    rule = RULES.get(dl) or next((v for k, v in RULES.items() if k in dl or dl in k), None)
    if not rule:
        return [], []
    avoid, alts = rule
    warns = [f"⚠️ {drug.title()} contraindicated in {c}" for c in conditions
             for a in avoid if a.lower() in c.lower() or c.lower() in a.lower()]
    return warns, (alts if warns else [])


def safety_flags_from(conditions: list) -> list:
    flags = []
    for c in conditions:
        cl = c.lower()
        if any(k in cl for k in ["kidney","ckd","renal"]):
            flags.append("⚠️ Avoid NSAIDs, Metformin, Gentamicin, Contrast dye")
        if any(k in cl for k in ["asthma","copd"]):
            flags.append("⚠️ Avoid Beta-blockers (e.g. Atenolol)")
        if any(k in cl for k in ["peptic","ulcer"]):
            flags.append("⚠️ Avoid NSAIDs — use Paracetamol")
    return flags or ["✓ No automatic contraindication flags"]


LAB_NOTES = {
    "creatinine": "↑ Creatinine → impaired kidney function. Review renal-dose medications.",
    "potassium":  "Abnormal K+ → cardiac risk. Consider ECG.",
    "sodium":     "Abnormal Na+ → assess fluid status.",
    "glucose":    "Abnormal glucose → review diabetes management.",
    "hemoglobin": "Abnormal Hb → assess for anaemia.",
    "hba1c":      "HbA1c outside target → review diabetes plan.",
    "inr":        "Abnormal INR → review Warfarin dosing immediately.",
    "platelet":   "Abnormal platelets → bleeding/clotting risk.",
    "bilirubin":  "↑ Bilirubin → assess liver function.",
    "alt":        "↑ ALT → possible liver injury.",
}


# ── SHARP context reader ───────────────────────────────────────────────────────
# Prompt Opinion passes FHIR context via HTTP headers per SHARP spec:
#   X-FHIR-Base-URL : URL of the FHIR server (Prompt Opinion's workspace FHIR)
#   X-FHIR-Token    : OAuth bearer token for that FHIR server
#   X-Patient-Id    : ID of the currently-selected patient
#
# When the agent invokes an MCP tool, these headers come with the request,
# so we can fetch the *correct* patient's data from the *correct* FHIR server
# even though we don't pass patient_id as an argument.

from starlette.requests import Request

def get_sharp_context() -> dict:
    """Read SHARP context from the current HTTP request headers.
    
    Per Prompt Opinion's official SHARP spec, the platform passes:
      - x-fhir-server-url  : base URL of the FHIR server
      - x-fhir-access-token: bearer token for that FHIR server  
      - x-patient-id       : ID of the currently-selected patient
    """
    try:
        req: Request = mcp.get_context().request_context.request
        if req is None:
            return {}
        headers = req.headers
        return {
            "fhir_base":  headers.get("x-fhir-server-url"),
            "fhir_token": headers.get("x-fhir-access-token"),
            "patient_id": headers.get("x-patient-id"),
        }
    except Exception:
        return {}


async def load_patient_from_context(explicit_patient_id: str = "") -> dict:
    """
    Load patient data using SHARP context from request headers.
    Falls back to explicit patient_id if provided. Uses FHIR base URL from
    SHARP context (Prompt Opinion's workspace FHIR) if available, else HAPI.
    """
    ctx = get_sharp_context()
    pid = explicit_patient_id or ctx.get("patient_id")
    base = ctx.get("fhir_base") or FHIR_BASE
    token = ctx.get("fhir_token")

    if not pid:
        return {"patient": {}, "conditions": [], "medications": [], "observations": [], "allergies": [], "encounters": []}

    p    = await fhir_get(f"Patient/{pid}", base, token)
    cond = await fhir_get(f"Condition?patient={pid}", base, token)
    meds = await fhir_get(f"MedicationRequest?patient={pid}", base, token)
    obs  = await fhir_get(f"Observation?patient={pid}&_sort=-date&_count=20", base, token)
    alrg = await fhir_get(f"AllergyIntolerance?patient={pid}", base, token)
    enc  = await fhir_get(f"Encounter?patient={pid}&_sort=-date&_count=3", base, token)
    docs = await fhir_get(f"DocumentReference?patient={pid}&_sort=-date&_count=5", base, token)
    return {
        "patient":      p,
        "patient_id":   pid,
        "conditions":   cond.get("entry", []),
        "medications":  meds.get("entry", []),
        "observations": obs.get("entry", []),
        "allergies":    alrg.get("entry", []),
        "encounters":   enc.get("entry", []),
        "documents":    docs.get("entry", []),
    }


def parse_document_text(documents: list, max_docs: int = 3) -> str:
    """Extract text content from DocumentReference resources."""
    import base64
    chunks = []
    for entry in documents[:max_docs]:
        res = entry.get("resource", {})
        for content in res.get("content", []):
            attach = content.get("attachment", {})
            data = attach.get("data")
            if data:
                try:
                    text = base64.b64decode(data).decode("utf-8", errors="ignore")
                    chunks.append(text)
                except Exception:
                    pass
            # Some FHIR servers inline text directly
            if attach.get("title"):
                chunks.append(attach.get("title", ""))
    return "\n\n---\n\n".join(chunks)


# ── MCP Tools ─────────────────────────────────────────────────────────────────

@mcp.tool()
async def debug_show_context() -> str:
    """
    Diagnostic tool. Returns all HTTP headers and detected SHARP context from
    the current request. Use this to verify what Prompt Opinion is sending so
    we know the exact header names for patient ID, FHIR base URL, and token.
    """
    try:
        req: Request = mcp.get_context().request_context.request
        if req is None:
            return "No HTTP request in context (running in stdio mode?)"
        headers = dict(req.headers)
        # Redact any tokens
        redacted = {k: ("***REDACTED***" if "token" in k.lower() or "auth" in k.lower() else v)
                    for k, v in headers.items()}
        ctx = get_sharp_context()
        lines = [
            "═══════════ VAIDYAFLOW DEBUG — REQUEST CONTEXT ═══════════",
            "",
            "ALL HTTP HEADERS:",
            *[f"  {k}: {v}" for k, v in sorted(redacted.items())],
            "",
            "DETECTED SHARP CONTEXT:",
            f"  FHIR base URL : {ctx.get('fhir_base')  or '(not detected)'}",
            f"  FHIR token    : {'***present***' if ctx.get('fhir_token') else '(not detected)'}",
            f"  Patient ID    : {ctx.get('patient_id') or '(not detected)'}",
            "",
            "═══════════════════════════════════════════════════════════",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"Error reading request context: {e!r}"


@mcp.tool()
async def get_patient_brief(patient_id: str = "") -> str:
    """
    Generate a 10-second patient brief for the doctor seeing this patient.
    Automatically uses the patient from SHARP context (the currently-selected
    patient in the Prompt Opinion workspace). Reads structured FHIR resources
    AND unstructured clinical notes attached to the patient. Returns conditions,
    medications, recent labs, allergies, and safety flags in one structured card.
    Built for high-volume OPD clinics where doctors see 80-120 patients per shift.
    """
    d = await load_patient_from_context(patient_id)
    p    = d["patient"]
    cond = parse_conditions(d["conditions"])
    meds = parse_meds(d["medications"])
    labs = parse_labs(d["observations"])
    alrg = parse_allergies(d["allergies"])
    abn  = [l for l in labs if l["flag"] in ("H","L","HH","LL","A")]
    doc_text = parse_document_text(d.get("documents", []))
    last = (d["encounters"][0].get("resource",{}).get("period",{}).get("start","N/A")[:10]
            if d["encounters"] else "No visits on record")

    if not p:
        return "Unable to load patient data. No patient context available — please select a patient in the workspace."

    sections = [
        "╔══════════════════════════════════════════════════╗",
        "║           VAIDYAFLOW 10-SECOND BRIEF             ║",
        "╚══════════════════════════════════════════════════╝",
        f"PATIENT : {name_of(p)} | {p.get('gender','?').title()} | DOB {p.get('birthDate','?')}",
        f"LAST VISIT: {last}",
        "",
        f"── CONDITIONS ({len(cond)}) " + "─"*30,
        *([f"• {c}" for c in cond] or ["• None in structured FHIR resources"]),
        "",
        f"── MEDICATIONS ({len(meds)}) " + "─"*29,
        *([f"• {m}" for m in meds] or ["• None in structured FHIR resources"]),
        "",
        "── ALLERGIES " + "─"*35,
        *([f"• {a}" for a in alrg] or ["• NKDA"]),
        "",
        f"── ABNORMAL LABS ({len(abn)}) " + "─"*29,
        *([f"⚠ {l['name']}: {l['value']} ({l['date']})" for l in abn] or ["✓ All recent labs normal"]),
        "",
        "── SAFETY FLAGS " + "─"*32,
        *safety_flags_from(cond),
    ]

    if doc_text:
        sections += [
            "",
            "── CLINICAL NOTES (from uploaded documents) " + "─"*4,
            doc_text[:2000],
        ]

    sections += [
        "",
        "══════════════════════════════════════════════════",
        "VaidyaFlow | Agents Assemble 2026 | Synthetic data",
    ]
    return "\n".join(sections)


@mcp.tool()
async def check_prescription_safety(proposed_medication: str, patient_id: str = "") -> str:
    """
    Check if a proposed medication is safe for the current patient given their
    conditions, medications, and allergies. Automatically uses the patient from
    SHARP context. Returns a clear verdict with contraindication reasoning and
    safer alternatives. Prevents medication errors in busy OPD.
    """
    d    = await load_patient_from_context(patient_id)
    cond = parse_conditions(d["conditions"])
    meds = parse_meds(d["medications"])
    alrg = parse_allergies(d["allergies"])
    doc_text = parse_document_text(d.get("documents", []))

    # Also scan clinical notes for conditions/allergies mentioned in free text
    doc_lower = doc_text.lower()
    conditions_from_notes = []
    allergies_from_notes = []
    for keyword, label in [
        ("chronic kidney", "Chronic Kidney Disease (from notes)"),
        ("ckd", "Chronic Kidney Disease (from notes)"),
        ("diabetes", "Diabetes (from notes)"),
        ("hypertension", "Hypertension (from notes)"),
        ("asthma", "Asthma (from notes)"),
        ("copd", "COPD (from notes)"),
        ("peptic ulcer", "Peptic Ulcer (from notes)"),
        ("heart failure", "Heart Failure (from notes)"),
        ("warfarin", "On Warfarin (from notes)"),
    ]:
        if keyword in doc_lower and label not in conditions_from_notes:
            conditions_from_notes.append(label)
    # Pull allergy mentions
    if "allerg" in doc_lower:
        for allergen in ["penicillin", "sulfa", "ibuprofen", "aspirin", "nsaid"]:
            if allergen in doc_lower:
                allergies_from_notes.append(allergen.title())

    cond_all = cond + conditions_from_notes
    alrg_all = alrg + allergies_from_notes

    warns, alts = drug_check(proposed_medication, cond_all)
    allergy_hits = [f"🚨 ALLERGY: documented allergy to {a}" for a in alrg_all
                    if proposed_medication.lower() in a.lower() or a.lower() in proposed_medication.lower()]
    dup_hits     = [f"📋 Already on: {m}" for m in meds if proposed_medication.lower() in m.lower()]

    all_warns = allergy_hits + warns + dup_hits

    if allergy_hits:   verdict = "🚨 CONTRAINDICATED — ALLERGY"
    elif warns:        verdict = "⚠️  CONTRAINDICATED — CONDITION RISK"
    elif dup_hits:     verdict = "📋 CAUTION — POSSIBLE DUPLICATE"
    else:              verdict = "✅ SAFE TO PRESCRIBE"

    lines = [
        "PRESCRIPTION SAFETY CHECK",
        "━"*40,
        f"Drug    : {proposed_medication.title()}",
        f"Verdict : {verdict}",
        "",
        "WARNINGS:",
        *([f"  {w}" for w in all_warns] or ["  None found."]),
    ]
    if alts:
        lines += ["", f"SAFER ALTERNATIVES: {', '.join(alts)}"]
    lines += [
        "",
        f"Patient conditions : {', '.join(cond_all) or 'None'}",
        f"Patient allergies  : {', '.join(alrg_all) or 'NKDA'}",
        "",
        "VaidyaFlow | Agents Assemble 2026 | Synthetic data",
    ]
    return "\n".join(lines)


@mcp.tool()
async def get_abnormal_labs(patient_id: str = "") -> str:
    """
    Retrieve and interpret the patient's most recent lab results, flagging
    abnormal values with clinical context and action guidance. Automatically
    uses the patient from SHARP context.
    """
    d    = await load_patient_from_context(patient_id)
    labs = parse_labs(d["observations"])
    abn  = [l for l in labs if l["flag"] in ("H","L","HH","LL","A")]
    norm = [l for l in labs if l not in abn]
    total = len(labs)

    abn_lines = []
    for l in abn:
        note = next((v for k, v in LAB_NOTES.items() if k in l["name"].lower()), "Review with clinical context.")
        abn_lines.append(f"⚠ {l['name']}: {l['value']} ({l['date']})\n  → {note}")

    return "\n".join([
        "LAB RESULTS SUMMARY",
        "━"*40,
        f"Reviewed: {total}  |  Abnormal: {len(abn)}  |  Normal: {len(norm)}",
        f"Action required: {'YES ⚠️' if abn else 'NO ✓'}",
        "",
        "ABNORMAL VALUES:",
        "\n".join(abn_lines) if abn_lines else "✓ All labs within normal limits",
        "",
        "NORMAL (sample): " + (", ".join(f"{l['name']}: {l['value']}" for l in norm[:4]) or "None"),
        "",
        "VaidyaFlow | Agents Assemble 2026 | Synthetic data",
    ])


@mcp.tool()
async def generate_handoff_note(patient_id: str = "", handoff_notes: str = "") -> str:
    """
    Generate a structured clinical handoff note for shift change. Automatically
    uses the patient from SHARP context. Summarises the patient's key context,
    active issues, and priority actions for the incoming doctor. Prevents
    critical information loss during transitions.
    """
    d = await load_patient_from_context(patient_id)
    p    = d["patient"]
    name = name_of(p)
    sex  = p.get("gender", "?").title()
    dob  = p.get("birthDate", "?")
    cond = parse_conditions(d["conditions"])
    meds = parse_meds(d["medications"])
    labs = parse_labs(d["observations"])
    alrg = parse_allergies(d["allergies"])
    abn  = [l for l in labs if l["flag"] in ("H","L","HH","LL","A")]
    doc_text = parse_document_text(d.get("documents", []))

    if not p:
        return "Unable to load patient data for handoff. No patient context available."

    rx_flags = []
    for c in cond:
        cl = c.lower()
        if any(k in cl for k in ["kidney","ckd","renal"]): rx_flags.append("Avoid NSAIDs, Gentamicin, contrast dye")
        if any(k in cl for k in ["asthma","copd"]):        rx_flags.append("Avoid Beta-blockers")

    priority = (f"URGENT: Review {', '.join(l['name'] for l in abn[:3])}" if abn else "Routine follow-up only")

    return "\n".join([
        "╔══════════════════════════════════════════════════╗",
        "║         VAIDYAFLOW CLINICAL HANDOFF NOTE         ║",
        "╚══════════════════════════════════════════════════╝",
        f"PATIENT: {name} | {sex} | DOB {dob}",
        "",
        "1. ACTIVE CONDITIONS",
        *([f"   • {c}" for c in cond] or ["   • None"]),
        "",
        "2. CURRENT MEDICATIONS",
        *([f"   • {m}" for m in meds] or ["   • None"]),
        "",
        "3. ALLERGIES",
        *([f"   • {a}" for a in alrg] or ["   • NKDA"]),
        "",
        "4. ABNORMAL LABS",
        *([f"   ⚠ {l['name']}: {l['value']} ({l['date']})" for l in abn] or ["   ✓ All normal"]),
        "",
        "5. PRESCRIBING SAFETY FLAGS",
        *([f"   ! {f}" for f in rx_flags] or ["   ✓ None"]),
        "",
        "6. OUTGOING DOCTOR NOTES",
        f"   {handoff_notes or 'None provided'}",
        "",
        "7. PRIORITY ACTIONS FOR INCOMING DOCTOR",
        f"   {priority}",
        "",
        "══════════════════════════════════════════════════",
        "VaidyaFlow | Agents Assemble 2026 | Synthetic data",
    ])


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="streamable-http")

# ── ASGI app for uvicorn (Railway) ────────────────────────────────────────────
app = mcp.streamable_http_app()