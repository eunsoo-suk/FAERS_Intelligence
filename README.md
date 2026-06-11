# FAERS Intelligence Platform

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://faers-intelligence.streamlit.app)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**🔗 Live demo:** **[faers-intelligence.streamlit.app](https://faers-intelligence.streamlit.app)**

> First load takes 2–5 minutes — the app downloads a 1.6 GB SQLite database
> from Hugging Face on cold start. Subsequent interactions are instant.

![Dashboard overview](figures/dashboard_overview.png)

---

## What this is

An end-to-end pharmacovigilance platform that turns raw FDA adverse-event
reports into **ranked, severity-calibrated, citable drug-safety signals**.
Built on **1.4 million real FAERS reports** spanning two decades, validated
against the OMOP negative-control standard and a clinical gold standard.

Post-marketing drug safety today relies on hand-curated signal review by
regulators — slow, opinion-heavy, and biased by signal volume. This project
asks: can a reproducible, statistically grounded, LLM-augmented system
produce signals a clinical reviewer can actually trust?

---

## The pipeline

Five phases, each phase's output feeding the next.

### Phase 1 — ETL
- Ingests quarterly FAERS ASCII files (2004 Q3 → 2024 Q3) into one unified
  SQLite database.
- Conservative dedup: latest `caseversion` + demographic fingerprint when
  age is known.
- Drug-name normalization to a single RxNorm-style ingredient.
- Modern FAERS kept separate from pre-2012 legacy AERS so the two coding
  eras never silently merge.
- **Output:** `faers.db` (1.58 GB SQLite, **1,409,287 reports**).

### Phase 2 — Signal detection
For every (drug, event) pair, builds a 2×2 contingency table and asks
whether the event shows up disproportionately.
- **PRR / ROR** with 95% CI (Haldane–Anscombe +0.5 correction for zero cells).
- **Yates χ²** with Benjamini–Hochberg FDR to control false positives
  across thousands of tests.
- **Empirical-Bayes Gamma-Poisson shrinkage (EBGM / EB05)** — DuMouchel's
  single-component approximation of the FDA's MGPS, pulling unstable
  small-count ratios toward the baseline.
- **Output:** **14,750 detected signals** across 18 target drugs.

### Phase 3 — Dual-severity calibration
Severity is derived **two independent ways** that capture different
information — and their disagreement is itself the finding.
- **Empirical severity (FAERS-GT)** — computed at the **PT level** (not
  drug-PT) from real outcome codes, which removes the bias of drugs used in
  already-sick populations.
- **Clinical PT severity (LLM)** — zero-shot classification of the MedDRA
  term itself by Claude Sonnet, grounded by anchor examples.
- Cross-validated with Cohen's κ, quadratic-weighted κ, ±1-tier agreement,
  and binary FATAL∪SEVERE accuracy.
- **On a clinical gold standard, LLM severity outperforms the FAERS-outcome
  GT (+38 points).** The LLM recovers *treatable emergencies* — agranulocytosis,
  major haemorrhage — that the outcome-based GT under-rates because the
  patients survive.

### Phase 4 — RAG-grounded Q&A
- All signals plus per-drug summaries embedded in ChromaDB with rich
  metadata (drug, event, severity, EBGM, n_cases, death_rate).
- **Per-drug split retrieval** — divides top-k evenly across the drugs in
  a multi-drug query so sparse drugs aren't drowned out by louder ones.
- Natural-language queries parsed into structured metadata filters
  **before** vector similarity.
- Answers forced to flag small-n (n<10) and patient-population effects.
- **Drug-precision: 0% (naive v5) → 100% (v6)** on a 17-query evaluation set.

### Phase 5 — Streamlit dashboard
Live exposure of every output: signal explorer with interactive volcano
plot, drug-level safety profiles with LLM synthesis, drug-vs-drug
comparison, dual-severity analysis, validation against positive and
negative controls, and natural-language Q&A.

---

## Results

### Signal explorer

![Signal Explorer with volcano plot](figures/dashboard_signal_explorer.png)

5,589 statistically significant (drug, event) pairs after filtering at
EB05 ≥ 2.0 and n ≥ 3. Color encodes empirical severity — the cluster of
red and orange dots at the high-EBGM end is exactly where a regulatory
reviewer should look first.

### Drug-level synthesis

![Drug Profile — amlodipine](figures/dashboard_drug_profile.png)

LLM-generated safety assessments grounded in the detected signals, with
structured sections (overall profile, most concerning signals, expected
vs. novel, class comparison, risk–benefit, recommendations) and links to
representative cases.

### Validation against OMOP negative controls

The hard pharmacovigilance standard: pre-validated unrelated drug-event
pairs that *shouldn't* be flagged.

![Validation page — 93% sensitivity, 89.7% specificity](figures/dashboard_validation.png)

| Metric | Result | What it measures |
|---|---|---|
| **Sensitivity** | **93%** | Established positive-control signals correctly detected |
| **Specificity (OMOP)** | **89.7%** | Validated unrelated pairs correctly rejected (FP rate 10.3%) |
| **LLM clinical accuracy** | **77%** | LLM severity classification vs clinical gold standard |
| **+38 pts** | LLM vs FAERS-GT | LLM recovers treatable emergencies the outcome-based GT misses |

The residual false positives (omeprazole/elderly, etc.) reflect age and
comorbidity confounding — an *intrinsic* limit of disproportionality
methods, not a code defect.

---

## Tech stack

- **Python 3.11** — pandas, numpy, scipy, scikit-learn
- **SQLite** for the unified relational store (1.58 GB)
- **ChromaDB** + sentence-transformers for vector retrieval
- **Anthropic Claude Sonnet 4** for severity classification and RAG synthesis
- **Streamlit** + Plotly for the interactive dashboard
- **Hugging Face Datasets** hosts the 1.58 GB database, downloaded on first
  cold start via `huggingface_hub`
- **Streamlit Community Cloud** hosts the live app

---

## Repository layout

```
notebooks/
  phase1_etl.ipynb                # quarterly FAERS → faers.db
  phase2_signal_detection.ipynb   # PRR/ROR/χ²/EBGM
  phase3_severity_calibration.ipynb  # dual-severity (FAERS-GT × LLM)
  phase4_rag_qa.ipynb             # ChromaDB + per-drug split retrieval
  phase5_dashboard.ipynb          # Streamlit prototype
app/
  app.py                          # production Streamlit app
  requirements.txt
results/                          # per-drug summaries, signal lists, LLM outputs (~5 MB)
figures/                          # dashboard + notebook artifacts
```

Heavy artifacts (`faers.db`, `chromadb/`) are not tracked in Git — they
are regeneratable from the notebooks, or downloaded from
[the Hugging Face dataset](https://huggingface.co/datasets/eunsoosuk/faers-intelligence-data)
on first launch.

---

## Running locally

```bash
git clone https://github.com/eunsoo-suk/FAERS_Intelligence.git
cd FAERS_Intelligence

pip install -r app/requirements.txt
streamlit run app/app.py
```

On first run, `app.py` calls `bootstrap_data()` which downloads
`faers.db` (1.58 GB) and `chromadb.tar.gz` (50 MB) from the public
Hugging Face dataset into `data/`. Subsequent runs are instant.

To regenerate everything from scratch, run the notebooks in
`notebooks/` in order (Phase 1 → Phase 5).

---

## Data provenance

- **FAERS quarterly ASCII files** — public, U.S. FDA
  ([fis.fda.gov](https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html)).
- **MedDRA preferred terms** — used as supplied by FDA; no MedDRA license is
  required to consume FAERS event labels in this read-only manner.
- **OMOP negative-control pairs** — derived from the OHDSI / EU-ADR reference set.

---

## Limitations of FAERS

- **Voluntary reporting** — under-reporting common; severe outcomes
  over-represented.
- **No denominator** — without prescription/exposure counts, true
  incidence is unknowable.
- **Reporting bias** — Weber effect (launch spikes), media-driven spikes,
  country differences.
- **Association ≠ causation** — a drug appearing in a report does not
  mean it caused the event.
- **Confounding by indication** — sicker patients on certain drugs →
  background mortality inflates empirical severity.

---

## Methods

- DuMouchel WH. *Bayesian data mining in large frequency tables, with an
  application to the FDA spontaneous reporting system.* The American
  Statistician, 1999.
- Bate A, Evans SJW. *Quantitative signal detection using spontaneous ADR
  reporting.* Pharmacoepidemiology & Drug Safety, 2009.
- Fusaroli M et al. *The DiAna dictionary…* Drug Safety, 2024.
- van Puijenbroek EP et al. *A comparison of measures of disproportionality…*
  Pharmacoepidemiology & Drug Safety, 2002.

---

## License

MIT — see [`LICENSE`](LICENSE).
