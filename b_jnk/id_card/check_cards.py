#!/usr/bin/env python3
"""Check card_rules.detect on made-up pages (no real patient data).

    python b_jnk/id_card/check_cards.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from card_rules import detect  # noqa: E402

HEADER = "Patient: Doe Jane DOB: 01/02/1960\n"

# (name, text, expected card type or None for "not a card")
PAGES = [
    ("state id front", """NEW MEXICO IDENTIFICATION CARD
Id # 100000001 Date of Birth 01/01/1950 ISSUED 03/27/2025 EXPIRES 03/26/2033
DOE, JANE 123 MAIN ST ALBUQUERQUE, NM 87111
WEIGHT 120 SEX F CLASS I HEIGHT 5'01" EYES BRO ENDORSEMENTS NONE DONOR RESTRICTIONS NONE""", "state_id"),
    ("state id back", """CLASS: I-ID For identification purposes only RESTRICTIONS: NONE ENDORSEMENTS: NONE
STATE OF NEW MEXICO IDENTIFICATION CARD NO. 100000001
This card is provided for the purpose of establishing that the bearer described on the card
was not the holder of a New Mexico driver's license as of the date of issuance of this card.
This identification card is not a license. ISSUED FOR IDENTIFICATION PURPOSES ONLY.""", "state_id"),
    ("licence front, scanned page", HEADER + """ARIZONA The Grand Canyon State DRIVER LICENSE
4d DLN D00000001 4b EXP 05/21/2029 3 DOB 06/30/1950 4a ISS 05/21/2024 DOE JANE
8 1 MAIN ST MESA AZ 85204 15 SEX F 16 HGT 5'-09" 18 EYES GRN 17 WGT 180 lb
9 CLASS D 9a ENDORSEMENTS NONE 12 RESTRICTIONS B DONOR VETERAN 5 DD 1192030000000
Document: ID Page 1 of 2""", "drivers_license"),
    ("licence back, read sideways", HEADER + """CLASS D - Operator 9a ENDORSEMENTS NONE 12 RESTRICTIONS
B - Corrective lenses You Must Report a Change of Address Within 10 Days
Document: ID Page 2 of 2""", "drivers_license_or_id"),
    ("insurance front", """UnitedHealthcare UCard DOE, JANE Member ID 900000000-00
AARP Medicare Advantage from UHC NM-0004 (PPO) With Dental
Group Number: 90000 H0000-000-000 Payer ID: LIFE1
ixBIN RxPCN RxGRP 610097 9999 COS Optum Care New Mexico PCP $0 Specialist $25
MedicareRx Prescription Drug Coverage""", "insurance"),
    ("insurance back", """Benefit Award Card #: 6102 0000 0000 0000 CSN: 397 Exp: 08/28 Printed: 07-16-2025
For Members: myAARPMedicare.com 1-866-000-0000, TTY 711
Earned rewards expire 1 mo. after plan terminates
Providers: optumproportal.com 1-800-000-0000 Provider Authorization: 1-800-000-0000
Dental Providers: uhcdental.com 1-877-000-0000 For Pharmacists: 1-877-000-0000
Med Claims: P.O Box 30539, Salt Lake City, UT 84130 Rx Claims: OptumRx P.O. Box 650287, Dallas, TX
Medicare limiting charges apply.""", "insurance"),
    ("insurance back with PASSPORT logo", HEADER + """Benefit Award Card #: 6102 0000 0000 0000 CSN: 033 Exp: 02/29
For Members: myAARPMedicare.com 1-877-000-0000, TTY 711 PASSPORT
Providers: optumproportal.com 1-877-000-0000 Provider Authorization: 1-877-000-0000
For Pharmacists: 1-877-000-0000 Med Claims: P.O. Box 30539 Rx Claims: OptumRx P.O. Box 650287
Document: UHC Page 2 of 2""", "insurance"),
    ("medicare card", """MEDICARE HEALTH INSURANCE Name/Nombre JANE DOE Medicare Number/Número de Medicare
1EG4-TE5-MK72 Entitled to/Con derecho a HOSPITAL (PART A) 03-01-2016 MEDICAL (PART B) 03-01-2016""", "medicare"),
    ("ssn card", """SOCIAL SECURITY 000-00-0000 THIS NUMBER HAS BEEN ESTABLISHED FOR JANE DOE""", "ssn_card"),

    # Not cards.
    ("registration form", """PATIENT REGISTRATION Please complete all fields
Primary insurance: Aetna Member ID ________ Group # ________ Insured's name ________
Relationship to patient ____ Employer ________ Emergency contact ________
Patient signature ____________ Date signed ______""", None),
    ("insurance letter", """Dear Jane Doe, UnitedHealthcare has received your request. Member ID 900000000
Group Number 90000. Claim number 12345. This is not a bill. Sincerely, Member Services""", None),
    ("vitals", """Vital signs: Height 5'6" Weight 160 lb BP 120/80 Pulse 72. Class II obesity.
Diet restrictions: none. Assessment: stable. Plan: follow up in 3 months.""", None),
    ("consent", """CONSENT FOR TREATMENT I hereby authorize the clinic to provide care.
I acknowledge that my insurance (Humana, member ID on file) may not cover all services.
Patient signature ________ Date ______""", None),
    ("med list mentioning copay", """Medication list: metformin 500 mg BID, lisinopril 10 mg daily.
Patient states copay for brand inhaler is too high. Diagnosis: type 2 diabetes.""", None),
    ("fax cover", """FAX COVER SHEET To: Clinic From: UnitedHealthcare Fax number 555-0100 Pages: 3
Re: Member ID 900000000 prior authorization. Confidential.""", None),
    ("records request audit sheet", """Audit Fulfillment Patient DOB: Member ID. Member HIC Requestor Account
Project ID Provider: Records Requested Project Due Date Dates of Service: 01/01/2023-12/31/2024
Please only send records for the specific measure(s) listed.""", None),
    ("plain clinic letter", """We saw your patient today for follow up. Her driver's license was renewed and she may
drive. Exam normal. Assessment: resolved concussion.""", None),
]


def main() -> int:
    bad = 0
    for name, text, want in PAGES:
        r = detect(text)
        got = r.card_type if r.is_card else None
        ok = got == want
        bad += not ok
        why = " | ".join(r.matched) if r.is_card else (" | ".join(r.blocked_by) or "no card words")
        print(f"{'ok  ' if ok else 'FAIL'} {name:36s} want {want or 'no card':22s} got {got or 'no card':22s} "
              f"{r.confidence:6s} {why[:120]}")
    print(f"\n{len(PAGES) - bad}/{len(PAGES)} right")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
