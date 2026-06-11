# FAERS Intelligence Platform

A reproducible pharmacovigilance pipeline that ingests the U.S. FDA Adverse Event
Reporting System (FAERS), detects drug-event signals with empirical-Bayes shrinkage,
calibrates severity using both real outcomes and an LLM, and exposes everything through
a Streamlit dashboard with retrieval-augmented Q&A.

## Pipeline

1. **Phase 1 — ETL.** Quarterly FAERS ASCII files are ingested into a single SQLite
   database (`faers.db`). Reports are deduplicated conservatively (latest caseversion +
   demographic fingerprint when age is known), drug names normalized to a single
   RxNorm-style ingredient, and modern FAERS kept separate from pre-2012 legacy AERS.

2. **Phase 2 — Signal detection.** For each (drug, event) pair we build a 2x2 table and
   compute PRR/ROR with 95% CI (Haldane-Anscombe +0.5 for zero cells), Yates chi-square
   with Benjamini-Hochberg FDR, and empirical-Bayes Gamma-Poisson shrinkage (EBGM/EB05) —
   DuMouchel's single-component approximation of the FDA's MGPS algorithm.

3. **Phase 3 — Severity calibration.** Two independent severity sources: ground-truth at
   the PT level (computed across all drugs to remove patient-population bias) and an LLM
   zero-shot classification. Cross-validated with Cohen's kappa, quadratic-weighted
   kappa, and adjacent agreement, then merged conservatively (max-rank). Validated
   against a clinical gold standard on established positive controls.

4. **Phase 4 — RAG.** All signals plus per-drug summaries are embedded in ChromaDB.
   Natural-language queries are parsed into structured metadata filters before vector
   similarity is applied. Per-drug split retrieval guarantees coverage even for sparse
   drugs. UNCLASSIFIED signals are hidden by default.

5. **Phase 5 — Dashboard.** A Streamlit app that surfaces every output from phases 1-4:
   signal explorer, drug profiles, drug comparison, calibration metrics, validation
   against positive/negative controls, and natural-language Q&A.

## Repository layout

```
FAERS_Intelligence/
├── notebooks/
│   ├── phase1_etl.ipynb
│   ├── phase2_signal_detection.ipynb
│   ├── phase3_severity_calibration.ipynb
│   ├── phase4_rag_qa.ipynb
│   └── phase5_dashboard.ipynb
├── app/
│   ├── app.py
│   ├── requirements.txt
│   └── .streamlit/
│       ├── config.toml
│       └── secrets.toml.example
└── data/
    ├── db/faers.db              # produced by Phase 1
    ├── chromadb/                # produced by Phase 4
    └── results/                 # produced by Phases 2-4
        ├── meta_v6.json
        ├── severity_calibrated_v6.csv
        ├── gold_standard_validation_v6.csv
        ├── case_summaries_v6.csv
        ├── drug_interpretations_v6.json
        ├── rag_qa_v6.json
        ├── rag_v5_vs_v6_comparison.csv
        └── rag_retrieval_eval_v6.csv
```

## Running the pipeline (Colab)

Phases 1-4 are designed to run on Google Colab with the project on Google Drive at
`/content/drive/MyDrive/FAERS_Intelligence`. Set `ANTHROPIC_API_KEY` as a Colab Secret
before running Phase 3 or Phase 4 — never paste keys into the notebook. Run phases in
order; each phase's output is the next phase's input.

Approximate LLM cost: ~1,500 calls for Phase 3 severity classification, ~90 calls for
case summaries (3 per drug), ~30 calls for drug interpretations (1 per drug), plus
on-demand calls from Phase 4 / Phase 5 Q&A.

## Running the dashboard locally

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

The dashboard auto-detects the project directory in this order:

1. `FAERS_BASE` environment variable
2. `/content/drive/MyDrive/FAERS_Intelligence` (Colab default)
3. `~/FAERS_Intelligence`
4. The current working directory

Provide the Anthropic API key via the in-app password field, the environment variable
`ANTHROPIC_API_KEY`, or `.streamlit/secrets.toml`. The Q&A page requires it; every other
page works without it.

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub.
2. At https://share.streamlit.io, connect the repository and set the entry point to
   `app/app.py`.
3. Under Settings → Secrets, add `ANTHROPIC_API_KEY = "sk-ant-..."`.
4. Deploy. The data files committed under `data/` are picked up automatically because
   the path-detection logic falls back to the current working directory.

If `data/` exceeds GitHub's 100 MB-per-file limit, switch to Git LFS or host the data
externally and fetch on first startup.

## Notes

- The dashboard is read-only — none of its loaders write to the FAERS database.
- `UNCLASSIFIED` is preserved in the data for transparency but hidden from charts and
  RAG queries by default. The Drug Profile page shows a Coverage metric that exposes
  what fraction of signals were classifiable.
- Negative controls (metformin, lisinopril, etc.) are tracked as an OMOP-style
  reference set and should have a low signal rate. The Validation page surfaces this.
