# Dataset Landing Checklist

**Purpose:** The system design makes specific assumptions about column names,
value casing, and file paths in the hackathon dataset. Until the real
`dataset/` files are present, these assumptions are *unverified*. The moment
the dataset lands — and BEFORE any batch run or code is trusted — work through
this checklist. Each item is a known silent-failure or schema-mismatch risk.

A wrong column name = `KeyError` on row 1 = empty `output.csv` = 0/30 technical
execution. A wrong path prefix = every image unreadable = near-zero accuracy
with no crash. These are the exact failure profiles that scored 3/30 last cycle.

---

## 1. `claims.csv` — input header names (HARD GATE)

Expected columns, exact snake_case, this order:

- [ ] `user_id`
- [ ] `image_paths`
- [ ] `user_claim`   ← **historically dangerous**: earlier drafts used `claim_transcript`. If the real header differs, fix `ClaimRow` + validator + `safe_defaults` before running.
- [ ] `claim_object`

Verify: `head -1 dataset/claims.csv` matches the four names above exactly.

## 2. `claim_object` value casing

- [ ] Values are lowercase `car` / `laptop` / `package` (not `Car`, `CAR`, `Laptop`).
- Validator already lowercases defensively, but confirm — if values are
  title-case, the `Literal` type in `ClaimRow`/`ClaimOutput` still expects
  lowercase, so the defensive `.lower()` must run *before* model construction.

## 3. `image_paths` prefix (SILENT-FAILURE GATE — see SYSTEM_DESIGN §4.2)

- [ ] Confirm CSV stores paths as `images/test/...` / `images/sample/...`
      WITHOUT the `dataset/` prefix.
- [ ] `resolve_image_path()` locates ≥1 real sample file at startup.
- [ ] Stage 0 logs image-resolution hit rate; aborts if 0% resolve.
- [ ] Image ID extraction: `Path(p).stem` → `img_1` (claim-local, not a path).

## 4. `sample_claims.csv` — output header names (HARD GATE)

The 14 expected/ground-truth output columns, exact order:

- [ ] `user_id`, `image_paths`, `user_claim`, `claim_object`,
      `evidence_standard_met`, `evidence_standard_met_reason`, `risk_flags`,
      `issue_type`, `object_part`, `claim_status`,
      `claim_status_justification`, `supporting_image_ids`, `valid_image`,
      `severity`
- [ ] Our `output.csv` header is byte-for-byte identical to this (assert at write time).
- [ ] Booleans (`evidence_standard_met`, `valid_image`) serialize lowercase
      `true`/`false` — matching the ground-truth file, NOT Python `True`/`False`.

## 5. `user_history.csv` — column names → `UserHistory` model

Expected (confirm each against the real header):

- [ ] `user_id`
- [ ] `past_claim_count`
- [ ] `accept_claim`
- [ ] `manual_review_claim`
- [ ] `rejected_claim`
- [ ] `last_90_days_claim_count`
- [ ] `history_flags`     (e.g. `none` or `user_history_risk;manual_review_required`)
- [ ] `history_summary`   (free text — must be injection-screened, §8/§24.1)

If names differ, update `UserHistory` in `code/pipeline/models.py`.

## 6. `evidence_requirements.csv` — column names → `EvidenceRequirement`

Expected (confirm against real header):

- [ ] `requirement_id`
- [ ] `claim_object`      (`all` / `car` / `laptop` / `package`)
- [ ] `applies_to`        (e.g. `dent or scratch`, `multi-image rows`)
- [ ] `minimum_image_evidence`
- [ ] Confirm the actual `requirement_id` values (e.g. `REQ_GENERAL_OBJECT_PART`,
      `REQ_GENERAL_MULTI_IMAGE`, `REQ_REVIEW_TRUST`) match what
      `select_requirements()` hardcodes. Update the selection logic if not.

## 7. Ground-truth value vocabularies (calibrate, don't assume)

From the 20 sample rows, confirm the *actual* sets used:

- [ ] `risk_flags` vocabulary — confirm the 11 flags we sanction match what
      appears in ground truth; no out-of-vocab flag, no missing flag.
- [ ] `issue_type` vocabulary — confirm all values (incl. `crushed_packaging`,
      `glass_shatter`, etc.) are covered by our `Literal` enum.
- [ ] `severity` set is exactly `none/low/medium/high/unknown`.
- [ ] `claim_status` set is exactly `supported/contradicted/not_enough_information`.
- [ ] `supporting_image_ids` format: semicolon-separated IDs or `none`.
- [ ] Re-confirm the §11.1 calibration table (all 20 rows) against the real file —
      especially the independence cases (user_008: `valid_image=false` +
      `evidence_standard_met=true`).

---

## How to use this

1. Dataset lands → run `head -1` on each CSV, tick the header gates (1, 4, 5, 6).
2. Run the path-resolution sanity check (3).
3. Diff the §11.1 calibration table against the real sample rows (7).
4. Only then trust a full batch run.

Any unticked HARD GATE box = do not run the batch yet.
