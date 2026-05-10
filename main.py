"""
VaidyaFlow MCP Server
OPD Co-Pilot for high-volume government hospitals
Built for Agents Assemble Hackathon 2026 - Prompt Opinion Platform
"""

import os
import json
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="VaidyaFlow MCP Server", version="1.0.0")

FHIR_BASE = os.getenv("FHIR_BASE_URL", "https://hapi.fhir.org/baseR4")

# ─────────────────────────────────────────────
# MCP MANIFEST — tells Prompt Opinion what tools exist
# ─────────────────────────────────────────────
MCP_MANIFEST = {
    "schema_version": "1.0",
    "name": "vaidyaflow-mcp",
    "display_name": "VaidyaFlow OPD Co-Pilot",
    "description": (
        "AI-powered patient safety tools for high-volume OPD clinicians. "
        "Surfaces medication risks, flags abnormal labs, generates 10-second "
        "patient briefs, and produces intelligent shift handoff notes — "
        "built for government hospitals where doctors see 100+ patients a day."
    ),
    "version": "1.0.0",
    "tools": [
        {
            "name": "get_patient_brief",
            "description": (
                "Generate a 10-second patient brief for a doctor seeing this "
                "patient. Returns active conditions, current medications, "
                "recent labs, allergy flags, and safety alerts in one structured card."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "FHIR Patient resource ID"
                    }
                },
                "required": ["patient_id"]
            }
        },
        {
            "name": "check_prescription_safety",
            "description": (
                "Check if a proposed medication is safe for this patient given "
                "their current conditions, active medications, and allergies. "
                "Returns a safety verdict with specific contraindication reasoning "
                "and safer alternatives if available."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "FHIR Patient resource ID"
                    },
                    "proposed_medication": {
                        "type": "string",
                        "description": "Name of the medication the doctor is considering prescribing"
                    }
                },
                "required": ["patient_id", "proposed_medication"]
            }
        },
        {
            "name": "get_abnormal_labs",
            "description": (
                "Retrieve and interpret the patient's most recent lab results, "
                "flagging any abnormal values with clinical context about what "
                "they mean and what action may be needed."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "FHIR Patient resource ID"
                    }
                },
                "required": ["patient_id"]
            }
        },
        {
            "name": "generate_handoff_note",
            "description": (
                "Generate a structured clinical handoff note for shift change. "
                "Summarises the patient's key context, active issues, pending "
                "tasks, and what the incoming doctor needs to know immediately. "
                "Prevents critical information loss during shift transitions."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "FHIR Patient resource ID"
                    },
                    "handoff_notes": {
                        "type": "string",
                        "description": "Optional free-text notes from the outgoing doctor to include"
                    }
                },
                "required": ["patient_id"]
            }
        }
    ]
}


# ─────────────────────────────────────────────
# FHIR HELPERS
# ─────────────────────────────────────────────
async def fhir_get(path: str, fhir_base: str, token: str = None) -> dict:
    """Make a FHIR API call, using auth token if provided."""
    headers = {"Accept": "application/fhir+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(f"{fhir_base}/{path}", headers=headers)
        if resp.status_code == 200:
            return resp.json()
        return {}


async def get_patient_data(patient_id: str, fhir_base: str, token: str = None) -> dict:
    """Fetch all relevant FHIR resources for a patient."""
    patient       = await fhir_get(f"Patient/{patient_id}", fhir_base, token)
    conditions    = await fhir_get(f"Condition?patient={patient_id}&clinical-status=active", fhir_base, token)
    medications   = await fhir_get(f"MedicationRequest?patient={patient_id}&status=active", fhir_base, token)
    observations  = await fhir_get(f"Observation?patient={patient_id}&_sort=-date&_count=20", fhir_base, token)
    allergies     = await fhir_get(f"AllergyIntolerance?patient={patient_id}", fhir_base, token)
    encounters    = await fhir_get(f"Encounter?patient={patient_id}&_sort=-date&_count=3", fhir_base, token)

    return {
        "patient": patient,
        "conditions": conditions.get("entry", []),
        "medications": medications.get("entry", []),
        "observations": observations.get("entry", []),
        "allergies": allergies.get("entry", []),
        "encounters": encounters.get("entry", [])
    }


def extract_name(patient: dict) -> str:
    try:
        name = patient.get("name", [{}])[0]
        given = " ".join(name.get("given", []))
        family = name.get("family", "")
        return f"{given} {family}".strip()
    except Exception:
        return "Unknown Patient"


def extract_conditions(conditions: list) -> list[str]:
    result = []
    for entry in conditions:
        res = entry.get("resource", {})
        code = res.get("code", {})
        text = code.get("text") or (code.get("coding", [{}])[0].get("display", ""))
        if text:
            result.append(text)
    return result


def extract_medications(medications: list) -> list[str]:
    result = []
    for entry in medications:
        res = entry.get("resource", {})
        med = res.get("medicationCodeableConcept", {})
        name = med.get("text") or (med.get("coding", [{}])[0].get("display", ""))
        if name:
            result.append(name)
    return result


def extract_labs(observations: list) -> list[dict]:
    labs = []
    for entry in observations[:10]:
        res = entry.get("resource", {})
        if res.get("resourceType") != "Observation":
            continue
        code = res.get("code", {})
        name = code.get("text") or (code.get("coding", [{}])[0].get("display", ""))
        value_qty = res.get("valueQuantity", {})
        value = f"{value_qty.get('value', '')} {value_qty.get('unit', '')}".strip()
        date = res.get("effectiveDateTime", "")[:10] if res.get("effectiveDateTime") else ""
        interp = res.get("interpretation", [{}])
        flag = interp[0].get("coding", [{}])[0].get("code", "") if interp else ""
        if name and value:
            labs.append({"name": name, "value": value, "date": date, "flag": flag})
    return labs


def extract_allergies(allergies: list) -> list[str]:
    result = []
    for entry in allergies:
        res = entry.get("resource", {})
        code = res.get("code", {})
        name = code.get("text") or (code.get("coding", [{}])[0].get("display", ""))
        if name:
            result.append(name)
    return result


# ─────────────────────────────────────────────
# SAFETY KNOWLEDGE BASE (no external API needed)
# ─────────────────────────────────────────────
DRUG_SAFETY_RULES = {
    # NSAIDs
    "ibuprofen":    {"avoid_in": ["Chronic kidney disease", "CKD", "Heart failure", "Peptic ulcer", "Warfarin"], "alternatives": ["Paracetamol", "Acetaminophen"]},
    "naproxen":     {"avoid_in": ["Chronic kidney disease", "CKD", "Heart failure", "Peptic ulcer"], "alternatives": ["Paracetamol"]},
    "diclofenac":   {"avoid_in": ["Chronic kidney disease", "CKD", "Heart failure", "Peptic ulcer"], "alternatives": ["Paracetamol"]},
    "aspirin":      {"avoid_in": ["Peptic ulcer", "Bleeding disorder"], "alternatives": ["Paracetamol (for pain)"]},
    # Diabetes
    "metformin":    {"avoid_in": ["Chronic kidney disease", "CKD", "Liver failure", "Heart failure"], "alternatives": ["Insulin", "Sitagliptin (if GFR allows)"]},
    "glibenclamide":{"avoid_in": ["Chronic kidney disease", "CKD", "Elderly"], "alternatives": ["Gliclazide MR"]},
    # Cardiac
    "amiodarone":   {"avoid_in": ["Thyroid disease", "Liver disease", "Lung disease"], "alternatives": ["Consult cardiologist"]},
    "digoxin":      {"avoid_in": ["Chronic kidney disease", "CKD", "Hypokalemia"], "alternatives": ["Consult cardiologist"]},
    # Antibiotics
    "gentamicin":   {"avoid_in": ["Chronic kidney disease", "CKD"], "alternatives": ["Amoxicillin", "Cephalosporins"]},
    "nitrofurantoin":{"avoid_in": ["Chronic kidney disease", "CKD"], "alternatives": ["Trimethoprim", "Cefalexin"]},
    # Contrast
    "contrast":     {"avoid_in": ["Chronic kidney disease", "CKD", "Metformin"], "alternatives": ["Low-osmolar contrast with pre-hydration"]},
    # Blood pressure
    "atenolol":     {"avoid_in": ["Asthma", "COPD", "Bradycardia"], "alternatives": ["Amlodipine", "Ramipril"]},
    "ramipril":     {"avoid_in": ["Pregnancy", "Hyperkalemia", "Bilateral renal artery stenosis"], "alternatives": ["Amlodipine"]},
}

def check_drug_conditions(drug: str, conditions: list[str]) -> dict:
    drug_lower = drug.lower().strip()
    # Try exact match first, then substring
    rules = DRUG_SAFETY_RULES.get(drug_lower)
    if not rules:
        for key in DRUG_SAFETY_RULES:
            if key in drug_lower or drug_lower in key:
                rules = DRUG_SAFETY_RULES[key]
                break

    if not rules:
        return {"safe": True, "warnings": [], "alternatives": []}

    warnings = []
    for condition in conditions:
        for avoid in rules["avoid_in"]:
            if avoid.lower() in condition.lower() or condition.lower() in avoid.lower():
                warnings.append(f"⚠️ {drug.title()} is contraindicated in **{condition}**")

    return {
        "safe": len(warnings) == 0,
        "warnings": warnings,
        "alternatives": rules.get("alternatives", []) if warnings else []
    }


# ─────────────────────────────────────────────
# MCP ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/mcp")
async def mcp_manifest():
    return JSONResponse(MCP_MANIFEST)


@app.post("/mcp")
async def mcp_handler(request: Request):
    body = await request.json()

    # Extract SHARP context (patient ID + FHIR token from Prompt Opinion)
    sharp = body.get("sharp", {})
    fhir_token = sharp.get("fhir_access_token") or sharp.get("token")
    fhir_base  = sharp.get("fhir_base_url") or FHIR_BASE
    context_patient_id = sharp.get("patient_id")

    tool_name = body.get("tool")
    params    = body.get("params", {}) or body.get("input", {})

    if tool_name == "get_patient_brief":
        return await tool_get_patient_brief(params, fhir_base, fhir_token, context_patient_id)
    elif tool_name == "check_prescription_safety":
        return await tool_check_prescription_safety(params, fhir_base, fhir_token, context_patient_id)
    elif tool_name == "get_abnormal_labs":
        return await tool_get_abnormal_labs(params, fhir_base, fhir_token, context_patient_id)
    elif tool_name == "generate_handoff_note":
        return await tool_generate_handoff_note(params, fhir_base, fhir_token, context_patient_id)
    else:
        return JSONResponse({"error": f"Unknown tool: {tool_name}"}, status_code=400)


# ─────────────────────────────────────────────
# TOOL IMPLEMENTATIONS
# ─────────────────────────────────────────────
async def tool_get_patient_brief(params, fhir_base, token, context_patient_id):
    patient_id = params.get("patient_id") or context_patient_id
    if not patient_id:
        return JSONResponse({"error": "patient_id is required"}, status_code=400)

    data = await get_patient_data(patient_id, fhir_base, token)
    patient    = data["patient"]
    conditions = extract_conditions(data["conditions"])
    meds       = extract_medications(data["medications"])
    labs       = extract_labs(data["observations"])
    allergies  = extract_allergies(data["allergies"])

    name = extract_name(patient)
    dob  = patient.get("birthDate", "Unknown")
    sex  = patient.get("gender", "Unknown").title()

    # Build safety flags from known dangerous combos
    safety_flags = []
    for cond in conditions:
        if any(k in cond.lower() for k in ["kidney", "ckd", "renal"]):
            safety_flags.append("⚠️ Avoid NSAIDs, Metformin, Gentamicin, Nitrofurantoin, Contrast dye")
        if any(k in cond.lower() for k in ["asthma", "copd"]):
            safety_flags.append("⚠️ Avoid Beta-blockers (e.g. Atenolol)")
        if any(k in cond.lower() for k in ["peptic", "ulcer", "gerd"]):
            safety_flags.append("⚠️ Avoid NSAIDs — use Paracetamol instead")

    abnormal_labs = [l for l in labs if l.get("flag") in ["H", "L", "HH", "LL", "A"]]

    brief = {
        "patient_brief": {
            "name": name,
            "dob": dob,
            "sex": sex,
            "active_conditions": conditions or ["No active conditions recorded"],
            "current_medications": meds or ["No active medications recorded"],
            "allergies": allergies or ["No known allergies"],
            "recent_labs": labs[:5],
            "abnormal_lab_count": len(abnormal_labs),
            "safety_flags": safety_flags or ["✅ No automatic contraindication flags"],
            "last_encounter": data["encounters"][0].get("resource", {}).get("period", {}).get("start", "N/A")[:10]
                              if data["encounters"] else "No encounters recorded"
        },
        "summary": (
            f"Patient {name}, {sex}, DOB {dob}. "
            f"{len(conditions)} active condition(s), {len(meds)} active medication(s). "
            f"{len(abnormal_labs)} abnormal lab value(s). "
            f"{len(safety_flags)} safety flag(s) raised."
        )
    }
    return JSONResponse(brief)


async def tool_check_prescription_safety(params, fhir_base, token, context_patient_id):
    patient_id = params.get("patient_id") or context_patient_id
    proposed   = params.get("proposed_medication", "")

    if not patient_id:
        return JSONResponse({"error": "patient_id is required"}, status_code=400)
    if not proposed:
        return JSONResponse({"error": "proposed_medication is required"}, status_code=400)

    data = await get_patient_data(patient_id, fhir_base, token)
    conditions = extract_conditions(data["conditions"])
    meds       = extract_medications(data["medications"])
    allergies  = extract_allergies(data["allergies"])

    # Check drug-condition rules
    safety = check_drug_conditions(proposed, conditions)

    # Check allergy overlap
    allergy_warnings = []
    for allergy in allergies:
        if proposed.lower() in allergy.lower() or allergy.lower() in proposed.lower():
            allergy_warnings.append(f"🚨 ALLERGY ALERT: Patient has documented allergy to {allergy}")

    # Check duplicate medication
    duplicate_warnings = []
    for med in meds:
        if proposed.lower() in med.lower():
            duplicate_warnings.append(f"📋 NOTE: Patient is already on {med}")

    all_warnings = allergy_warnings + safety["warnings"] + duplicate_warnings

    verdict = "SAFE ✅" if not all_warnings else (
        "CONTRAINDICATED 🚨" if allergy_warnings else "USE WITH CAUTION ⚠️"
    )

    result = {
        "proposed_medication": proposed,
        "verdict": verdict,
        "is_safe": len(all_warnings) == 0,
        "warnings": all_warnings or ["No contraindications found for this patient's conditions"],
        "safer_alternatives": safety["alternatives"] if not safety["safe"] else [],
        "patient_conditions": conditions,
        "patient_allergies": allergies,
        "recommendation": (
            f"Do NOT prescribe {proposed}. " + " ".join(all_warnings) +
            (f" Consider: {', '.join(safety['alternatives'])}" if safety["alternatives"] else "")
            if all_warnings else
            f"{proposed} appears safe for this patient based on their current conditions and medications."
        )
    }
    return JSONResponse(result)


async def tool_get_abnormal_labs(params, fhir_base, token, context_patient_id):
    patient_id = params.get("patient_id") or context_patient_id
    if not patient_id:
        return JSONResponse({"error": "patient_id is required"}, status_code=400)

    data = await get_patient_data(patient_id, fhir_base, token)
    labs = extract_labs(data["observations"])

    abnormal = [l for l in labs if l.get("flag") in ["H", "L", "HH", "LL", "A"]]
    normal   = [l for l in labs if l not in abnormal]

    # Enrich abnormal labs with clinical interpretation
    interpretations = {
        "creatinine": "Elevated creatinine suggests impaired kidney function. Monitor renal medications.",
        "potassium":  "Abnormal potassium is a cardiac risk. Check ECG if symptomatic.",
        "sodium":     "Abnormal sodium — assess fluid status and medications.",
        "glucose":    "Abnormal glucose — assess diabetes control and current insulin/oral agents.",
        "hemoglobin": "Abnormal hemoglobin — assess for anaemia or polycythaemia.",
        "hba1c":      "HbA1c outside target — review diabetes management plan.",
        "inr":        "Abnormal INR — review Warfarin dosing immediately.",
        "platelet":   "Abnormal platelets — bleeding or clotting risk. Review medications.",
        "bilirubin":  "Elevated bilirubin — assess liver function.",
        "alt":        "Elevated ALT — possible liver injury. Review hepatotoxic medications.",
    }

    enriched = []
    for lab in abnormal:
        note = ""
        for key, interp in interpretations.items():
            if key in lab["name"].lower():
                note = interp
                break
        enriched.append({**lab, "clinical_note": note or "Review with clinical context."})

    return JSONResponse({
        "total_labs_reviewed": len(labs),
        "abnormal_count": len(enriched),
        "abnormal_labs": enriched,
        "normal_labs": normal,
        "action_required": len(enriched) > 0,
        "summary": (
            f"{len(enriched)} abnormal lab value(s) found out of {len(labs)} reviewed. "
            "Immediate review recommended." if enriched else
            f"All {len(labs)} recent lab values are within normal range."
        )
    })


async def tool_generate_handoff_note(params, fhir_base, token, context_patient_id):
    patient_id    = params.get("patient_id") or context_patient_id
    doctor_notes  = params.get("handoff_notes", "")

    if not patient_id:
        return JSONResponse({"error": "patient_id is required"}, status_code=400)

    data = await get_patient_data(patient_id, fhir_base, token)
    patient    = data["patient"]
    conditions = extract_conditions(data["conditions"])
    meds       = extract_medications(data["medications"])
    labs       = extract_labs(data["observations"])
    allergies  = extract_allergies(data["allergies"])

    name    = extract_name(patient)
    dob     = patient.get("birthDate", "Unknown")
    sex     = patient.get("gender", "Unknown").title()
    abnormal = [l for l in labs if l.get("flag") in ["H", "L", "HH", "LL", "A"]]

    # Build safety flags
    safety_flags = []
    for cond in conditions:
        if any(k in cond.lower() for k in ["kidney", "ckd", "renal"]):
            safety_flags.append("Avoid NSAIDs, Gentamicin, contrast dye")
        if any(k in cond.lower() for k in ["asthma", "copd"]):
            safety_flags.append("Avoid Beta-blockers")

    handoff = {
        "handoff_note": {
            "patient": f"{name} | {sex} | DOB: {dob}",
            "generated_at": "VaidyaFlow Auto-Handoff",
            "section_1_conditions": conditions or ["No active conditions on record"],
            "section_2_medications": meds or ["No active medications on record"],
            "section_3_allergies": allergies or ["NKDA (No Known Drug Allergies)"],
            "section_4_abnormal_labs": [
                f"{l['name']}: {l['value']} ({l['date']}) ← ABNORMAL" for l in abnormal
            ] or ["All recent labs within normal limits"],
            "section_5_safety_flags": safety_flags or ["No prescribing flags identified"],
            "section_6_doctor_notes": doctor_notes or "No additional notes from outgoing doctor.",
            "section_7_incoming_doctor_action": (
                "PRIORITY ACTIONS: " + "; ".join([
                    f"Review {l['name']} ({l['value']})" for l in abnormal[:3]
                ]) if abnormal else "No urgent actions. Routine follow-up."
            )
        },
        "formatted_text": f"""
╔══════════════════════════════════════════════════╗
║         VAIDYAFLOW CLINICAL HANDOFF NOTE         ║
╚══════════════════════════════════════════════════╝

PATIENT: {name} | {sex} | DOB: {dob}

── ACTIVE CONDITIONS ──────────────────────────────
{chr(10).join(f'• {c}' for c in (conditions or ['None recorded']))}

── CURRENT MEDICATIONS ────────────────────────────
{chr(10).join(f'• {m}' for m in (meds or ['None recorded']))}

── ALLERGIES ──────────────────────────────────────
{chr(10).join(f'• {a}' for a in (allergies or ['NKDA']))}

── ABNORMAL LABS ──────────────────────────────────
{chr(10).join(f'⚠ {l["name"]}: {l["value"]} ({l["date"]})' for l in abnormal) or '✓ All labs normal'}

── PRESCRIBING SAFETY FLAGS ───────────────────────
{chr(10).join(f'! {f}' for f in safety_flags) or '✓ No flags'}

── OUTGOING DOCTOR NOTES ──────────────────────────
{doctor_notes or 'None provided'}

── ACTION FOR INCOMING DOCTOR ─────────────────────
{('URGENT: Review ' + ', '.join([l['name'] for l in abnormal[:3]])) if abnormal else 'Routine follow-up only'}

══════════════════════════════════════════════════
Generated by VaidyaFlow | Agents Assemble 2026
══════════════════════════════════════════════════
        """.strip()
    }
    return JSONResponse(handoff)


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "VaidyaFlow MCP Server is running", "tools": 4}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "vaidyaflow-mcp"}
