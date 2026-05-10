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


# ── MCP Tools ─────────────────────────────────────────────────────────────────

@mcp.tool()
async def get_patient_brief(patient_id: str) -> str:
    """
    Generate a 10-second patient brief for a doctor seeing this patient.
    Returns active conditions, current medications, recent labs, allergy flags,
    and safety alerts in one structured card. Built for high-volume OPD clinics
    where doctors see 80-120 patients per shift.
    """
    d = await load_patient(patient_id)
    p    = d["patient"]
    cond = parse_conditions(d["conditions"])
    meds = parse_meds(d["medications"])
    labs = parse_labs(d["observations"])
    alrg = parse_allergies(d["allergies"])
    abn  = [l for l in labs if l["flag"] in ("H","L","HH","LL","A")]
    last = (d["encounters"][0].get("resource",{}).get("period",{}).get("start","N/A")[:10]
            if d["encounters"] else "No visits on record")

    return "\n".join([
        "╔══════════════════════════════════════════════════╗",
        "║           VAIDYAFLOW 10-SECOND BRIEF             ║",
        "╚══════════════════════════════════════════════════╝",
        f"PATIENT : {name_of(p)} | {p.get('gender','?').title()} | DOB {p.get('birthDate','?')}",
        f"LAST VISIT: {last}",
        "",
        f"── CONDITIONS ({len(cond)}) " + "─"*30,
        *([f"• {c}" for c in cond] or ["• None recorded"]),
        "",
        f"── MEDICATIONS ({len(meds)}) " + "─"*29,
        *([f"• {m}" for m in meds] or ["• None recorded"]),
        "",
        "── ALLERGIES " + "─"*35,
        *([f"• {a}" for a in alrg] or ["• NKDA"]),
        "",
        f"── ABNORMAL LABS ({len(abn)}) " + "─"*29,
        *([f"⚠ {l['name']}: {l['value']} ({l['date']})" for l in abn] or ["✓ All recent labs normal"]),
        "",
        "── SAFETY FLAGS " + "─"*32,
        *safety_flags_from(cond),
        "",
        "══════════════════════════════════════════════════",
        "VaidyaFlow | Agents Assemble 2026 | Synthetic data",
    ])


@mcp.tool()
async def check_prescription_safety(patient_id: str, proposed_medication: str) -> str:
    """
    Check if a proposed medication is safe for this patient given their current
    conditions, medications, and allergies. Returns a verdict with contraindication
    reasoning and safer alternatives. Prevents medication errors in busy OPD.
    """
    d    = await load_patient(patient_id)
    cond = parse_conditions(d["conditions"])
    meds = parse_meds(d["medications"])
    alrg = parse_allergies(d["allergies"])

    warns, alts = drug_check(proposed_medication, cond)
    allergy_hits = [f"🚨 ALLERGY: documented allergy to {a}" for a in alrg
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
        f"Patient conditions : {', '.join(cond) or 'None'}",
        f"Patient allergies  : {', '.join(alrg) or 'NKDA'}",
        "",
        "VaidyaFlow | Agents Assemble 2026 | Synthetic data",
    ]
    return "\n".join(lines)


@mcp.tool()
async def get_abnormal_labs(patient_id: str) -> str:
    """
    Retrieve and interpret the patient's most recent lab results, flagging
    abnormal values with clinical context and action guidance. Helps doctors
    spot critical values without manually reviewing every lab result.
    """
    d    = await load_patient(patient_id)
    labs = parse_labs(d["observations"])
    abn  = [l for l in labs if l["flag"] in ("H","L","HH","LL","A")]
    norm = [l for l in labs if l not in abn]

    abn_lines = []
    for l in abn:
        note = next((v for k, v in LAB_NOTES.items() if k in l["name"].lower()), "Review with clinical context.")
        abn_lines.append(f"⚠ {l['name']}: {l['value']} ({l['date']})\n  → {note}")

    return "\n".join([
        "LAB RESULTS SUMMARY",
        "━"*40,
        f"Reviewed: {len(labs)}  |  Abnormal: {len(abn)}  |  Normal: {len(norm)}",
        f"Action required: {'YES ⚠️' if abn else 'NO ✓'}",
        "",
        "ABNORMAL VALUES:",
        *(["\n".join(abn_lines)] or ["✓ All labs within normal limits"]),
        "",
        "NORMAL (sample): " + (", ".join(f"{l['name']}: {l['value']}" for l in norm[:4]) or "None"),
        "",
        "VaidyaFlow | Agents Assemble 2026 | Synthetic data",
    ])


@mcp.tool()
async def generate_handoff_note(patient_id: str, handoff_notes: str = "") -> str:
    """
    Generate a structured clinical handoff note for shift change. Summarises
    the patient's key context, active issues, and priority actions for the
    incoming doctor. Prevents critical information loss during transitions —
    a leading cause of preventable adverse events in hospitals.
    """
    d    = await load_patient(patient_id)
    p    = d["patient"]
    cond = parse_conditions(d["conditions"])
    meds = parse_meds(d["medications"])
    labs = parse_labs(d["observations"])
    alrg = parse_allergies(d["allergies"])
    abn  = [l for l in labs if l["flag"] in ("H","L","HH","LL","A")]

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
        f"PATIENT: {name_of(p)} | {p.get('gender','?').title()} | DOB {p.get('birthDate','?')}",
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
