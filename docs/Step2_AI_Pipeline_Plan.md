# Step 2 — AI-Assisted Water Conflict Chronology: Implementation Plan

**Scope:** Build, validate, and deploy a human-in-the-loop AI pipeline that identifies and classifies water-related conflict events for the Water Conflict Chronology (WCC).

**Timeline:** Months 2–12
**Budget:** $15,000 (revised from $10,000 to include live pipeline deployment and the human-moderation workflow)

---

## 1. Current Starting Point

A substantial portion of the core technical work is already built and publicly available, which de-risks the project and accelerates delivery:

- **A trained, published classification model** that sorts events into the three WCC categories — Trigger, Casualty, and Weapon — at ~82% accuracy. It is lightweight and fast enough to screen large volumes of text cheaply.
- **Curated, versioned training datasets** drawn from WCC historical records and ACLED conflict data, plus synthetic "hard negatives" (peaceful water news) to reduce false positives.
- **Reproducible training and evaluation infrastructure** for retraining and tracking model performance over time.

The design is deliberately frugal: rather than relying on a large language model, the task is handled by a small, specialized classifier that is cheap and fast enough to screen text at scale on ordinary hardware. The model is published as **open weights on Hugging Face**, so it remains transparent, auditable, and reusable by the wider research community at no licensing cost.

In workplan terms, data curation and model development (M3–4 and M4–7) are largely complete. The remaining work turns the model into an operational, expert-supervised system.

---

## 2. Remaining Work

**2.1 Data & model refinement (M3–7).** Strengthen weaker categories (notably the Weapon label) through targeted data augmentation, add non-English sources, and retrain to agreed performance thresholds. Add calibrated confidence scores to each prediction to drive review routing.
*Checkpoint C2a — Training Dataset Ready: end of Month 5.*

**2.2 Live ingestion pipeline (M5–8).** Build the layer that pulls open-source text streams and feeds candidate events to the classifier. The primary source will be **GDELT**, using its Global Knowledge Graph themes to pre-filter for water-relevant material before classification — substantially reducing the volume of text that needs to be parsed. GDELT is not entirely free but is queried via Google Cloud at very low cost, and the classifier itself is inexpensive to run since it is a small model rather than a large language model — so running costs stay minimal even at scale. ACLED and other sources will be considered as complementary inputs.

**2.3 Human review & moderation workflow (M5–8).** Build a review queue where every AI-flagged event is confirmed, rejected, or reclassified by an expert before entering the database. High-confidence events are fast-tracked; borderline ones are escalated, with defined turnaround standards and reliability checks.

**2.4 Beta testing & validation (M7–10).** Test the model prospectively against a live period and retrospectively against historical events, benchmarking accuracy against expert coding and the pre-AI baseline, and quantifying staff-time savings.
*Checkpoint C2b — Beta Model Validated: end of Month 9.*

**2.5 Live deployment & historical backfill (M9–12).** Deploy the full pipeline (ingestion → classification → review → database) for ongoing, semi-autonomous WCC updates. GDELT's coverage reaches back roughly the last 10 years, so the pipeline can be run retrospectively over that period to augment the historical chronology and fill coverage gaps, not just capture new events going forward; for events older than that we would draw on the existing data sources already assembled rather than GDELT. Finalize a report on performance, limitations, and the review protocol.
*Checkpoint C3 — WCC Updated & Pipeline Live: end of Month 12.*

---

## 3. Budget — $15,000

| Workstream | Est. |
|---|---:|
| Data & model refinement (2.1) | $4,500 |
| Live pipeline & moderation workflow (2.2–2.3) | $6,500 |
| Validation, deployment & documentation (2.4–2.5) | $4,000 |
| **Total** | **$15,000** |

*Excludes data and compute costs (GDELT queries via Google Cloud, model hosting), which are low but should be budgeted as a small separate line item.*

---

## 4. Technology Readiness Level

Currently ~TRL 4 (model validated against held-out historical data). By project end, the pipeline reaches **TRL 6–7** — demonstrated and operating in the live WCC environment with human-in-the-loop oversight (TRL 7 contingent on the Month 9–12 live deployment proceeding as planned).

---

## 5. Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Classification accuracy insufficient | Conservative deployment thresholds; human review catches errors; active learning on weak categories |
| Data source access cost/terms change | Primary source (GDELT) is open and low-cost via Google Cloud; complement with ACLED/others; theme pre-filtering keeps query and compute costs minimal |
| Specialized expertise retention | Fully documented, modular pipeline; cross-train ≥2 team members |
