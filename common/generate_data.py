"""Generate the SafeDrive Insurance practice dataset (fully synthetic, fixed seed).

Run locally:  python common/generate_data.py [output_dir]
Everything is created on disk; the setup notebook uploads it to S3.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

SEED = 42
N_POLICIES = 5000
N_CLAIMS = 20000

REGIONS = ["North", "South", "East", "West"]
MAKES = ["Toyota", "Honda", "Ford", "BMW", "Tesla", "Hyundai", "Volkswagen"]
CLAIM_TYPES = ["collision", "theft", "glass", "fire", "third_party"]
CHANNELS = ["app", "phone", "agent"]
FIRST = ["Asha", "Ravi", "Maria", "John", "Wei", "Fatima", "Carlos", "Emma", "Noah", "Priya"]
LAST = ["Sharma", "Smith", "Garcia", "Chen", "Khan", "Brown", "Silva", "Patel", "Jones", "Kim"]


# ----------------------------------------------------------------------------
# Tabular data
# ----------------------------------------------------------------------------
def make_policies(rng):
    n = N_POLICIES
    age = rng.integers(18, 80, n)
    vehicle_value = np.round(rng.lognormal(10.0, 0.45, n), -2)
    premium = np.round(300 + vehicle_value * 0.03 + (age < 25) * 400 + rng.normal(0, 60, n), 2)
    return pd.DataFrame({
        "policy_id": [f"P{i:06d}" for i in range(1, n + 1)],
        "customer_name": [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(n)],
        "age": age,
        "gender": rng.choice(["F", "M"], n),
        "region": rng.choice(REGIONS, n, p=[0.3, 0.3, 0.25, 0.15]),
        "vehicle_make": rng.choice(MAKES, n),
        "vehicle_year": rng.integers(2008, 2026, n),
        "vehicle_value": vehicle_value,
        "annual_premium": premium.clip(250),
        "tenure_months": rng.integers(1, 180, n),
        "prior_claims": rng.poisson(0.6, n),
        "credit_band": rng.choice(list("ABCD"), n, p=[0.3, 0.35, 0.25, 0.10]),
    })


def make_claims(rng, policies, n=N_CLAIMS, start="2025-01-01", days=365,
                id_start=1, amount_scale=1.0):
    pol = policies.sample(n, replace=True, random_state=int(rng.integers(1_000_000)))
    pol = pol.reset_index(drop=True)
    claim_type = rng.choice(CLAIM_TYPES, n, p=[0.5, 0.1, 0.2, 0.05, 0.15])
    base = rng.lognormal(7.6, 0.8, n) * amount_scale
    amount = np.minimum(base, pol["vehicle_value"].values * 1.1).round(2)
    delay = rng.poisson(3, n)
    hour = rng.integers(0, 24, n)
    police = rng.choice(["Y", "N"], n, p=[0.6, 0.4])
    witnesses = rng.poisson(0.8, n)

    # Hidden fraud signal: large claim vs vehicle value, no police report,
    # late reporting, night-time incident, new customer, many prior claims.
    ratio = amount / pol["vehicle_value"].values
    z = (-6.2 + 5.0 * ratio + 1.6 * (police == "N") + 0.22 * delay
         + 1.2 * ((hour < 5) | (hour > 22)) + 1.5 * (pol["tenure_months"].values < 6)
         + 0.6 * pol["prior_claims"].values + 1.0 * (claim_type == "theft")
         - 0.7 * witnesses)
    is_fraud = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    dates = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, days, n), unit="D")
    return pd.DataFrame({
        "claim_id": [f"C{i:07d}" for i in range(id_start, id_start + n)],
        "policy_id": pol["policy_id"].values,
        "claim_date": dates.strftime("%Y-%m-%d"),
        "report_delay_days": delay,
        "claim_type": claim_type,
        "claim_amount": amount,
        "incident_hour": hour,
        "police_report": police,
        "witnesses": witnesses,
        "num_vehicles": rng.integers(1, 4, n),
        "channel": rng.choice(CHANNELS, n, p=[0.5, 0.3, 0.2]),
        "is_fraud": is_fraud,
    })


def make_claims_v2(rng, policies):
    """A later month of claims with deliberate quality problems and drift."""
    df = make_claims(rng, policies, n=3000, start="2026-01-01", days=31,
                     id_start=N_CLAIMS + 1, amount_scale=1.4)       # drift
    df.loc[df.sample(frac=0.06, random_state=1).index, "claim_amount"] = np.nan
    df.loc[df.sample(frac=0.04, random_state=2).index, "police_report"] = None
    df.loc[df.sample(frac=0.03, random_state=3).index, "incident_hour"] = np.nan
    df.loc[df.sample(15, random_state=4).index, "claim_amount"] = 9_999_999.0   # outliers
    df.loc[df.sample(10, random_state=5).index, "report_delay_days"] = -5       # invalid
    df.loc[df.sample(20, random_state=6).index, "claim_type"] = "COLLISION "    # bad category
    dupes = df.sample(120, random_state=7)                                      # duplicates
    return pd.concat([df, dupes], ignore_index=True).sample(frac=1, random_state=8)


def make_telematics(rng, policies, n=5000):
    ts = pd.Timestamp("2026-01-15 08:00:00") + pd.to_timedelta(np.sort(rng.integers(0, 3600, n)), unit="s")
    speed = rng.normal(62, 22, n).clip(0, 190).round(1)
    return pd.DataFrame({
        "event_id": [f"E{i:07d}" for i in range(n)],
        "policy_id": rng.choice(policies["policy_id"].values[:200], n),
        "ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "speed_kmh": speed,
        "accel_g": rng.normal(0, 0.25, n).round(3),
        "harsh_brake": (rng.random(n) < 0.05 + (speed > 110) * 0.15).astype(int),
        "lat": (12.9 + rng.normal(0, 0.05, n)).round(5),
        "lon": (77.6 + rng.normal(0, 0.05, n)).round(5),
    })


# ----------------------------------------------------------------------------
# Text data
# ----------------------------------------------------------------------------
POLICY_DOCS = {
    "01_coverage_overview.md": ("coverage", """# SafeDrive Motor Policy - Coverage Overview

## Comprehensive cover
SafeDrive Comprehensive covers accidental damage to your vehicle, theft, fire, glass damage and third-party liability. The maximum payout for damage to your own vehicle is the insured declared value (IDV) shown in your policy schedule.

## Third-party only cover
Third-party only policies cover injury to other people and damage to their property up to USD 1,000,000 per incident. Damage to your own vehicle is not covered.

## Glass cover
Windscreen and window glass repair is covered with a reduced deductible of USD 50. Glass claims do not affect your no-claim bonus. A maximum of two glass claims is allowed per policy year.

## Roadside assistance
All comprehensive policies include 24x7 roadside assistance: towing up to 50 km, battery jump start, flat tyre support and emergency fuel delivery (up to 5 litres, fuel cost charged).

## Rental car benefit
If your vehicle is in an approved repair shop for more than 3 days after a covered claim, SafeDrive pays for a rental car up to USD 40 per day for a maximum of 14 days.
"""),
    "02_exclusions.md": ("exclusions", """# SafeDrive Motor Policy - Exclusions

SafeDrive does not pay claims in the following situations.

## Driver related exclusions
- The driver did not hold a valid driving licence at the time of the incident.
- The driver was under the influence of alcohol or drugs.
- The vehicle was used for racing, speed testing or any illegal activity.

## Usage related exclusions
- A private vehicle was used for commercial ride-sharing or delivery without the Commercial Use add-on.
- Damage that happened outside the covered territory.

## Wear and tear
Normal wear and tear, mechanical or electrical breakdown, and tyre damage (unless caused by an accident) are not covered.

## Late reporting
Claims reported more than 30 days after the incident may be rejected unless the policyholder shows a valid reason for the delay. Theft claims must be reported to the police within 24 hours.

## Flood and natural events
Engine damage caused by driving through flood water (hydrostatic lock) is covered only if the Engine Protect add-on was purchased.
"""),
    "03_claims_process.md": ("claims", """# SafeDrive Motor Policy - Claims Process

## Step 1: Report the claim
Report the incident through the SafeDrive mobile app, by phone on 1-800-555-0100, or through your agent. You will receive a claim ID that starts with the letter C.

## Step 2: Submit documents
Upload the claim form, photos of the damage, a copy of the driving licence and the vehicle registration. A police report is mandatory for theft, third-party injury and any claim above USD 5,000.

## Step 3: Inspection
An adjuster inspects the vehicle within 2 business days. For claims below USD 1,000 a photo-based digital inspection is used and no physical visit is needed.

## Step 4: Decision and payment
Simple claims are decided within 7 business days after all documents are received. Complex claims or claims referred to the Special Investigation Unit can take up to 30 days. Approved amounts are paid to the repair shop (cashless) or reimbursed to your bank account within 3 business days.

## Deductibles
The standard deductible is USD 250 per claim. Drivers under 25 years pay an additional young-driver deductible of USD 200.

## Appeals
If your claim is rejected you can appeal within 60 days by writing to appeals@safedrive.example.
"""),
    "04_fraud_policy.md": ("fraud", """# SafeDrive - Fraud Prevention Policy

## What counts as insurance fraud
Fraud includes staging an accident, inflating repair costs, claiming for damage that existed before the policy started, and reporting a vehicle as stolen when it is not.

## How claims are screened
Every claim is scored by an automated fraud detection model. Claims with a score above 0.80 are sent to the Special Investigation Unit (SIU). Claims with a score between 0.50 and 0.80 receive an additional document check by a senior adjuster. The model score alone is never used to reject a claim; a human investigator makes the final decision.

## Common red flags
- A large claim shortly after the policy start date.
- No police report and no independent witnesses.
- The incident was reported many days after it happened.
- The claimed amount is close to the full value of the vehicle.

## Consequences
Confirmed fraud leads to rejection of the claim, cancellation of the policy and may be reported to law enforcement.

## Fairness
SafeDrive monitors the fraud model for bias across age groups, gender and region every quarter.
"""),
    "05_premium_and_discounts.md": ("pricing", """# SafeDrive Motor Policy - Premiums and Discounts

## How the premium is calculated
The annual premium depends on the vehicle value, vehicle age, driver age, region and claim history.

## No-claim bonus (NCB)
You earn a discount for every claim-free year: 20% after 1 year, 25% after 2 years, 35% after 3 years, 45% after 4 years and 50% after 5 or more years. Making a claim (other than a glass claim) resets the bonus to zero unless the NCB Protect add-on was purchased.

## Telematics discount
Customers who share driving data through the SafeDrive telematics device can earn up to 15% discount for safe driving. Harsh braking events and speeds above 120 km/h reduce the safe driving score.

## Payment options
Premiums can be paid annually or in 12 monthly instalments. Monthly payment adds a 4% service fee.

## Cancellation
You can cancel within 15 days of purchase for a full refund if no claim was made. After 15 days the refund is pro-rated and a USD 25 administration fee applies.
"""),
}

# Golden questions for evaluating the RAG assistant (question, answer, source document).
QA_EVAL = [
    ("What is the standard deductible per claim?", "The standard deductible is USD 250 per claim.", "03_claims_process.md"),
    ("How much extra deductible does a driver under 25 pay?", "Drivers under 25 pay an additional young-driver deductible of USD 200.", "03_claims_process.md"),
    ("How many glass claims are allowed per policy year?", "A maximum of two glass claims is allowed per policy year.", "01_coverage_overview.md"),
    ("Does a glass claim affect my no-claim bonus?", "No. Glass claims do not affect the no-claim bonus.", "01_coverage_overview.md"),
    ("What is the rental car benefit?", "Up to USD 40 per day for a maximum of 14 days if the vehicle is in an approved repair shop for more than 3 days.", "01_coverage_overview.md"),
    ("Is damage covered if the driver was drunk?", "No. Claims are not paid if the driver was under the influence of alcohol or drugs.", "02_exclusions.md"),
    ("Can I use my private car for ride-sharing?", "Only with the Commercial Use add-on; otherwise claims are excluded.", "02_exclusions.md"),
    ("How soon must a theft be reported to the police?", "Theft claims must be reported to the police within 24 hours.", "02_exclusions.md"),
    ("When is a police report mandatory?", "For theft, third-party injury and any claim above USD 5,000.", "03_claims_process.md"),
    ("How long does a simple claim decision take?", "Simple claims are decided within 7 business days after all documents are received.", "03_claims_process.md"),
    ("What fraud score sends a claim to the Special Investigation Unit?", "Claims with a fraud score above 0.80 are sent to the SIU.", "04_fraud_policy.md"),
    ("Can the fraud model reject a claim on its own?", "No. A human investigator makes the final decision.", "04_fraud_policy.md"),
    ("What is the no-claim bonus after 3 claim-free years?", "35% discount.", "05_premium_and_discounts.md"),
    ("What is the maximum telematics discount?", "Up to 15% discount for safe driving.", "05_premium_and_discounts.md"),
    ("What fee applies to monthly premium payments?", "Monthly payment adds a 4% service fee.", "05_premium_and_discounts.md"),
    ("Is flood damage to the engine covered?", "Only if the Engine Protect add-on was purchased.", "02_exclusions.md"),
]

NOTE_TEMPLATES = [
    "Spoke with {name} on {phone}. Customer says the car was hit while parked near the office. Policy {policy}. Photos look consistent with the description.",
    "{name} ({email}) reported theft of the vehicle. No police report yet. Story changed twice during the call. Recommend SIU review for policy {policy}.",
    "Windscreen crack reported by {name}. Simple glass claim, approved for cashless repair. Contact number {phone}.",
    "Customer {name} was very upset about the delay. Called from {phone}. Rear bumper damage, estimate looks reasonable. Policy {policy}.",
    "Third-party collision at a junction. {name} admits fault. Other driver's details collected. SSN on file 123-45-{ssn}. Policy {policy}.",
    "Fire damage to engine bay. {name} purchased the policy three weeks ago. Repair estimate is close to vehicle value. Email {email}.",
]


def make_notes(rng, claims, policies, n=400):
    rows = []
    names = policies.set_index("policy_id")["customer_name"]
    for _, c in claims.sample(n, random_state=11).iterrows():
        name = names[c["policy_id"]]
        note = rng.choice(NOTE_TEMPLATES).format(
            name=name, policy=c["policy_id"],
            phone=f"+1-555-{rng.integers(100, 999)}-{rng.integers(1000, 9999)}",
            email=name.lower().replace(" ", ".") + "@example.com",
            ssn=rng.integers(1000, 9999))
        rows.append({"claim_id": c["claim_id"], "claim_type": c["claim_type"],
                     "is_fraud": int(c["is_fraud"]), "note": note})
    return rows


def make_finetune(rng):
    """Prompt/completion pairs for FM customization, with a few bad records to find."""
    rows = [{"prompt": q, "completion": a} for q, a, _ in QA_EVAL]
    rows += [
        {"prompt": "How do I report a claim?", "completion": "Use the SafeDrive mobile app, call 1-800-555-0100, or contact your agent."},
        {"prompt": "What does third-party only cover include?", "completion": "Injury to other people and damage to their property up to USD 1,000,000 per incident."},
        {"prompt": "How long do I have to appeal a rejected claim?", "completion": "You can appeal within 60 days."},
        {"prompt": "What is included in roadside assistance?", "completion": "Towing up to 50 km, battery jump start, flat tyre support and emergency fuel delivery."},
    ]
    bad = [
        {"prompt": "What is the deductible?", "completion": ""},                                   # empty answer
        {"prompt": "", "completion": "The deductible is USD 250."},                                # empty prompt
        {"prompt": rows[0]["prompt"], "completion": rows[0]["completion"]},                        # duplicate
        {"prompt": "Who is the customer on policy P000123?", "completion": "John Smith, phone +1-555-321-9876, SSN 123-45-6789."},  # PII
        {"prompt": "Is my claim approved?", "completion": "You are an idiot for asking, read the policy."},  # toxic
        {"prompt": "Explain coverage.", "completion": "word " * 3000},                             # too long
        {"question": "Wrong keys?", "answer": "This record uses the wrong field names."},          # schema
    ]
    return rows + bad


# ----------------------------------------------------------------------------
# Images (simple drawings so that no download is needed)
# ----------------------------------------------------------------------------
def make_images(rng, out_dir, claims):
    from PIL import Image, ImageDraw

    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    colors = {"red": (200, 30, 30), "blue": (40, 80, 200), "white": (235, 235, 235), "black": (30, 30, 30)}
    for i in range(12):
        color = rng.choice(list(colors))
        damage = rng.choice(["dent", "scratch", "broken_glass", "none"])
        img = Image.new("RGB", (640, 400), (150, 190, 230))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 300, 640, 400], fill=(90, 90, 90))                      # road
        d.rounded_rectangle([110, 190, 530, 300], radius=25, fill=colors[color])  # body
        d.polygon([(200, 190), (250, 120), (400, 120), (460, 190)], fill=colors[color])
        d.polygon([(215, 185), (255, 130), (320, 130), (320, 185)], fill=(200, 230, 250))
        d.polygon([(330, 185), (330, 130), (395, 130), (440, 185)], fill=(200, 230, 250))
        for x in (190, 450):
            d.ellipse([x - 40, 260, x + 40, 340], fill=(20, 20, 20))
            d.ellipse([x - 18, 282, x + 18, 318], fill=(160, 160, 160))
        if damage == "dent":
            d.ellipse([260, 215, 340, 275], fill=(60, 60, 60))
        elif damage == "scratch":
            for k in range(4):
                d.line([240, 215 + 12 * k, 420, 235 + 12 * k], fill=(250, 250, 250), width=3)
        elif damage == "broken_glass":
            for k in range(6):
                d.line([270, 160, 220 + 20 * k, 130 + 9 * k], fill=(30, 30, 30), width=2)
        name = f"damage_{i:02d}.png"
        img.save(os.path.join(out_dir, name))
        manifest.append({"file": name, "vehicle_color": color, "damage_type": damage,
                         "claim_id": claims["claim_id"].iloc[i]})

    # A scanned-style claim form for Amazon Textract.
    c = claims.iloc[0]
    form = Image.new("RGB", (900, 600), "white")
    d = ImageDraw.Draw(form)
    lines = ["SAFEDRIVE INSURANCE - MOTOR CLAIM FORM", "",
             f"Claim ID: {c['claim_id']}", f"Policy Number: {c['policy_id']}",
             f"Date of Incident: {c['claim_date']}", f"Claim Type: {c['claim_type']}",
             f"Claimed Amount (USD): {c['claim_amount']}",
             f"Police Report Filed: {c['police_report']}",
             "Claimant Name: John Smith", "Phone: +1-555-321-9876", "",
             "Description: Vehicle was hit from behind at a traffic signal."]
    for k, line in enumerate(lines):
        d.text((40, 30 + 42 * k), line, fill="black", font_size=24)
    form.save(os.path.join(out_dir, "claim_form.png"))
    with open(os.path.join(out_dir, "manifest.jsonl"), "w") as f:
        f.writelines(json.dumps(m) + "\n" for m in manifest)


def write_jsonl(path, rows):
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def generate(out_dir="data"):
    """Create every data file under out_dir and return the folder path."""
    rng = np.random.default_rng(SEED)
    os.makedirs(os.path.join(out_dir, "policy_docs"), exist_ok=True)

    policies = make_policies(rng)
    claims = make_claims(rng, policies)
    policies.to_csv(os.path.join(out_dir, "policies.csv"), index=False)
    claims.to_csv(os.path.join(out_dir, "claims.csv"), index=False)
    make_claims_v2(rng, policies).to_parquet(os.path.join(out_dir, "claims_v2.parquet"), index=False)
    make_telematics(rng, policies).to_json(
        os.path.join(out_dir, "telematics_stream.jsonl"), orient="records", lines=True)

    for name, (category, text) in POLICY_DOCS.items():
        with open(os.path.join(out_dir, "policy_docs", name), "w") as f:
            f.write(text)
        # Sidecar metadata file in the format Bedrock Knowledge Bases expects.
        with open(os.path.join(out_dir, "policy_docs", name + ".metadata.json"), "w") as f:
            json.dump({"metadataAttributes": {"category": category, "product": "motor", "year": 2026}}, f)

    write_jsonl(os.path.join(out_dir, "adjuster_notes.jsonl"), make_notes(rng, claims, policies))
    write_jsonl(os.path.join(out_dir, "finetune_qa.jsonl"), make_finetune(rng))
    write_jsonl(os.path.join(out_dir, "qa_eval.jsonl"),
                [{"question": q, "answer": a, "source": s} for q, a, s in QA_EVAL])
    make_images(rng, os.path.join(out_dir, "images"), claims)
    return out_dir


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data"
    generate(target)
    for root, _, files in os.walk(target):
        for name in sorted(files):
            print(os.path.join(root, name))
