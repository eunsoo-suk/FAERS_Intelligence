# FAERS Intelligence — v6 Dashboard
# Single-file Streamlit app exposing the full v6 pipeline output.

import os, json, sqlite3, hashlib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2 as _chi2_dist

# =============== Setup ===============

st.set_page_config(
    page_title="FAERS Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

DRUG_CONTEXT = {
    "ciprofloxacin":  ("fluoroquinolone antibiotic",       "bacterial infections"),
    "isotretinoin":   ("retinoid",                          "severe acne"),
    "clozapine":      ("atypical antipsychotic",            "treatment-resistant schizophrenia"),
    "warfarin":       ("vitamin K antagonist",              "anticoagulation"),
    "dolutegravir":   ("integrase inhibitor (GSK)",         "HIV"),
    "cabotegravir":   ("long-acting integrase inhibitor",   "HIV PrEP/treatment"),
    "mepolizumab":    ("anti-IL-5 monoclonal antibody",     "severe eosinophilic asthma"),
    "semaglutide":    ("GLP-1 receptor agonist",            "type 2 diabetes, obesity"),
    "tirzepatide":    ("GLP-1/GIP dual agonist",            "type 2 diabetes, obesity"),
    "lecanemab":      ("anti-amyloid mAb",                  "early Alzheimer disease"),
    "pembrolizumab":  ("anti-PD-1 checkpoint inhibitor",    "various cancers"),
    "metformin":      ("biguanide",                          "type 2 diabetes"),
    "lisinopril":     ("ACE inhibitor",                     "hypertension"),
    "omeprazole":     ("proton pump inhibitor",             "GERD"),
    "amoxicillin":    ("penicillin antibiotic",             "bacterial infections"),
    "amlodipine":     ("calcium channel blocker",           "hypertension"),
    "rofecoxib":      ("COX-2 inhibitor (withdrawn 2004)",  "pain"),
    "rosiglitazone":  ("thiazolidinedione (restricted)",    "type 2 diabetes"),
}

GROUP_COLOR = {"validation":"#e74c3c","gsk":"#2ecc71",
               "trending":"#3498db","negative_control":"#95a5a6","meta":"#7f8c8d"}
SEV_COLOR   = {"FATAL":"#c0392b","SEVERE":"#e67e22",
               "MODERATE":"#f1c40f","MILD":"#27ae60","UNCLASSIFIED":"#7f8c8d"}
SEV_ORDER   = ["FATAL","SEVERE","MODERATE","MILD"]

# ============================================================
#   DESIGN LAYER  (inject once, near top of app after st.set_page_config)
# ============================================================
def inject_css():
    st.markdown("""
    <style>
      /* ---- typography & base ---- */
      html, body, [class*="css"] { font-family: 'Inter','Segoe UI',system-ui,sans-serif; }
      .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1300px; }

      /* ---- headings ---- */
      h1 { font-weight: 750; letter-spacing:-0.02em; }
      h2, h3 { font-weight: 650; letter-spacing:-0.01em; }

      /* ---- KPI metric cards ---- */
      div[data-testid="stMetric"] {
        background: linear-gradient(180deg,#ffffff 0%,#f7f9fc 100%);
        border: 1px solid #e6e9ef;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(16,24,40,.04), 0 1px 3px rgba(16,24,40,.06);
      }
      div[data-testid="stMetric"]:hover { border-color:#cfd6e4; transition:.15s; }
      div[data-testid="stMetricLabel"] p { font-size:.82rem; color:#667085; font-weight:600; }
      div[data-testid="stMetricValue"]  { font-size:1.9rem; font-weight:750; color:#101828; }

      /* ---- dataframe polish ---- */
      div[data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; }

      /* ---- divider tighten ---- */
      hr { margin: 1.2rem 0; border-color:#eef1f6; }

      /* ---- sidebar ---- */
      section[data-testid="stSidebar"] { background:#0f1729; }
      section[data-testid="stSidebar"] * { color:#e6e9ef !important; }
      section[data-testid="stSidebar"] a { color:#7cc4ff !important; }

      /* ---- info / warning boxes softer ---- */
      div[data-testid="stAlert"] { border-radius:10px; }
    </style>
    """, unsafe_allow_html=True)


# ---- Plotly common theme (call on every figure) ----
def _style_fig(fig, height=None):
    fig.update_layout(
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#344054"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=8, r=8, t=28, b=8),
        title_font=dict(size=15, color="#101828"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        colorway=["#2980b9","#27ae60","#e67e22","#c0392b","#8e44ad","#16a085"],
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef1f6", zeroline=False,
                     linecolor="#d0d5dd", ticks="outside", tickcolor="#d0d5dd")
    fig.update_yaxes(showgrid=True, gridcolor="#eef1f6", zeroline=False,
                     linecolor="#d0d5dd")
    if height: fig.update_layout(height=height)
    return fig

inject_css()


# =============== Path detection ===============
def detect_base():
    for c in ["/content/drive/MyDrive/FAERS_Intelligence",
              os.path.expanduser("~/FAERS_Intelligence"),
              os.path.abspath(".")]:
        if os.path.isfile(os.path.join(c,"data","db","faers.db")):
            return c
    return os.path.abspath(".")

BASE = os.environ.get("FAERS_BASE", detect_base())
DB_PATH     = os.path.join(BASE, "data", "db", "faers.db")
CHROMA_DIR  = os.path.join(BASE, "data", "chromadb")
RESULTS_DIR = os.path.join(BASE, "results")

# =============== Loaders (cached) ===============
@st.cache_data(show_spinner=False)
def load_meta():
    if not os.path.exists(DB_PATH): return {}
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT key,value FROM meta", conn)
        return dict(zip(df.key, df.value))
    finally:
        conn.close()

@st.cache_data(show_spinner="Loading signals…")
def load_signals():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM signal_results", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    # v6.2 stored as prod_ai/group ; standardize
    if "prod_ai" in df.columns and "target_drug" not in df.columns:
        df = df.rename(columns={"prod_ai":"target_drug","group":"target_group"})
    return df

@st.cache_data(show_spinner=False)
def load_calib():
    p = os.path.join(RESULTS_DIR, "severity_calibrated_v6.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_case_summaries():
    p = os.path.join(RESULTS_DIR, "case_summaries_v6.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_interpretations():
    p = os.path.join(RESULTS_DIR, "drug_interpretations_v6.json")
    return json.load(open(p)) if os.path.exists(p) else {}

@st.cache_data(show_spinner=False)
def load_rag_qa_cached():
    p = os.path.join(RESULTS_DIR, "rag_qa_v6.json")
    return json.load(open(p)) if os.path.exists(p) else []

@st.cache_data(show_spinner=False)
def load_v5_v6():
    p = os.path.join(RESULTS_DIR, "rag_v5_vs_v6_comparison.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_retrieval_eval():
    p = os.path.join(RESULTS_DIR, "rag_retrieval_eval_v6.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

# =============== Live RAG (optional) ===============
@st.cache_resource(show_spinner="Connecting to ChromaDB…")
def get_chroma_collection():
    if not os.path.isdir(CHROMA_DIR): return None
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        return client.get_collection("faers_signals_v6")
    except Exception:
        return None

def live_retrieve(query, filt, k=8, collection=None):
    if collection is None: return None
    drugs = filt.get("drugs") or []
    sev = filt.get("severity"); grp = filt.get("group")
    def _aux():
        a=[]
        if sev: a.append({"final_severity":{"$eq":sev}})
        if grp: a.append({"group":{"$eq":grp}})
        return a
    def _q(where, kk):
        try:    return collection.query(query_texts=[query], n_results=kk, where=where)
        except: return collection.query(query_texts=[query], n_results=kk)
    if len(drugs) <= 1:
        conds = _aux()
        if len(drugs)==1: conds.append({"drug":{"$eq":drugs[0]}})
        where = conds[0] if len(conds)==1 else ({"$and":conds} if conds else None)
        return _q(where, k)
    k_each = max(3, k//len(drugs))
    docs, metas, ids = [], [], []
    for d in drugs:
        conds = [{"drug":{"$eq":d}}] + _aux()
        where = conds[0] if len(conds)==1 else {"$and":conds}
        r = _q(where, k_each)
        if r["documents"][0]:
            docs += r["documents"][0]; metas += r["metadatas"][0]; ids += r["ids"][0]
    return {"documents":[docs], "metadatas":[metas], "ids":[ids]}

# =============== Load ===============
meta   = load_meta()
sig    = load_signals()
calib  = load_calib()
cases  = load_case_summaries()
interp = load_interpretations()
qa_cache = load_rag_qa_cached()
v5v6   = load_v5_v6()
ev     = load_retrieval_eval()

if sig.empty:
    st.error(f"signal_results not found at {DB_PATH}. Run Phase 2 first.")
    st.stop()

# Merge dual-severity from calib if available
if not calib.empty and "pt_severity" in calib.columns:
    work = sig.merge(calib[["target_drug","pt","final_severity","pt_severity",
                            "gt_severity","death_rate","serious_rate","n_cases","median_age"]],
                     on=["target_drug","pt"], how="left", suffixes=("","_c"))
else:
    work = sig.copy()
    for c in ["final_severity","pt_severity","gt_severity","death_rate",
              "serious_rate","n_cases","median_age"]:
        if c not in work.columns: work[c] = np.nan

work["final_severity"] = work["final_severity"].fillna("UNCLASSIFIED")
work["target_group"]   = work.get("target_group","").fillna("")

# =============== Sidebar ===============
st.sidebar.markdown("## FAERS Intelligence")

PAGES = [" Overview",
         " Signal Explorer",
         " Drug Profile",
         " Drug Comparison",
         " Dual-Severity",
         " Q&A (RAG)",
         " Validation",
         " About"]
page = st.sidebar.radio("Navigation", PAGES, index=0)

n_total = int(meta.get("n_reports_total","0"))

st.sidebar.markdown("---")
st.sidebar.markdown("**⚠️ FAERS Caveats**")
st.sidebar.markdown(
    "- Reporting bias\n"
    "- No denominator\n"
    "- Association ≠ causation\n"
    "- Voluntary reports"
)

# ============================================================
#                     PAGE: Overview
# ============================================================
def page_overview():
    st.title("FAERS Intelligence Platform")
    st.markdown(
        "End-to-end pharmacovigilance pipeline: **quarterly FAERS ASCII → "
        "EB-shrunk signal detection → dual-severity calibration → RAG-grounded Q&A**."
    )

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Reports analyzed", f"{n_total:,}")
    c2.metric("Drugs", f"{sig['target_drug'].nunique()}")
    c3.metric("Signals detected", f"{len(sig):,}")
    c4.metric("FATAL signals",
              f"{int((work['final_severity']=='FATAL').sum()):,}")
    c5.metric("SEVERE signals",
              f"{int((work['final_severity']=='SEVERE').sum()):,}")

    st.divider()

    # ---- Top signals by EB05 (table, full width) ----
    st.subheader("Top signals by EB05")
    top = (work.sort_values("eb05", ascending=False)
                .head(15)[["target_drug","pt","a","ebgm","eb05","prr",
                           "final_severity","target_group"]]
                .rename(columns={"a":"n_reports"}))
    st.dataframe(top, use_container_width=True, hide_index=True)

    st.divider()

    # ---- Severity distribution + Signal rate by group, side by side ----
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Severity distribution")
        sev_all = work["final_severity"].value_counts().reindex(
            SEV_ORDER + ["UNCLASSIFIED"]).fillna(0)
        n_labeled = int(sev_all.reindex(SEV_ORDER).sum())
        n_unclass = int(sev_all.get("UNCLASSIFIED", 0))
        n_signals = n_labeled + n_unclass
        coverage = (n_labeled / n_signals * 100) if n_signals else 0
        # 도넛은 labeled signals 만 표시
        sev_counts = sev_all.reindex(SEV_ORDER)
        fig = go.Figure(go.Pie(
            labels=sev_counts.index, values=sev_counts.values,
            marker_colors=[SEV_COLOR.get(s,"#888") for s in sev_counts.index],
            hole=.5, sort=False))
        fig.update_layout(height=340, margin=dict(l=0,r=0,t=10,b=0),
                          legend=dict(orientation="v", x=1.0, y=0.5))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"**Coverage:** {n_labeled:,} of {n_signals:,} signals classified "
                   f"({coverage:.1f}%). {n_unclass:,} unclassified (hidden).")

    with col_r:
        st.subheader("Signal rate by drug group")
        grp = (work.groupby("target_group")
                   .agg(pairs=("a","size"),
                        signal_rate=("is_signal","mean"))
                   .reset_index())
        grp["signal_rate_%"] = (grp["signal_rate"]*100).round(1)

        fig = px.bar(grp, x="target_group", y="signal_rate_%",
                     labels={"signal_rate_%":"signal rate (%)","target_group":""},
                     height=340)
        fig.update_traces(
            marker_color=[GROUP_COLOR.get(g, "#888") for g in grp["target_group"]]
        )
        fig = _style_fig(fig)
        fig.update_layout(showlegend=False, title="",
                          margin=dict(l=60, r=20, t=10, b=50))

        st.plotly_chart(fig, use_container_width=True)
        st.caption("Negative-control group should be lowest.")


# ============================================================
#                     PAGE: Signal Explorer
# ============================================================
def page_explorer():
    st.title(" Signal Explorer")
    st.markdown("Filter and inspect every detected drug-event pair.")

    c1,c2,c3 = st.columns(3)
    with c1:
        drugs_sel = st.multiselect("Drug",
            sorted(work["target_drug"].dropna().unique()),
            default=[])
    with c2:
        sev_sel = st.multiselect("Empirical severity",
            SEV_ORDER, default=[])
    with c3:
        grp_sel = st.multiselect("Drug group",
            sorted(work["target_group"].dropna().unique()),
            default=[])

    c4,c5,c6 = st.columns(3)
    with c4:
        # EB05 슬라이더: long-tail 이라 q99 또는 20 으로 cap → 세밀한 조절 가능
        eb05_cap = float(min(work["eb05"].max(),
                             work["eb05"].quantile(0.99),
                             20.0))
        min_eb05 = st.slider("Min EB05", 0.0, eb05_cap, 2.0, step=0.1)
    with c5:
        min_a = st.slider("Min reports (a)", 3, 200, 3)
    with c6:
        # 테이블용 top N (볼케이노는 필터된 전체를 사용)
        n_total = max(len(work), 100)
        max_rows = st.slider("Table: top N rows", 100,
                             int(min(n_total, 10000)),
                             int(min(n_total, 2000)),
                             step=100)

    df = work.copy()
    if drugs_sel: df = df[df["target_drug"].isin(drugs_sel)]
    if sev_sel:   df = df[df["final_severity"].isin(sev_sel)]
    if grp_sel:   df = df[df["target_group"].isin(grp_sel)]
    df = df[(df["eb05"] >= min_eb05) & (df["a"] >= min_a)]

    st.caption(f"{len(df):,} pairs after filters")
    if df.empty:
        st.info("No pairs match current filters.")
        return

    # Volcano: 필터된 데이터 전체를 사용 (top N 으로 자르지 않음)
    st.subheader("Volcano: EBGM vs −log10(p)")
    plot = df.copy()
    plot["log2_ebgm"] = np.log2(plot["ebgm"].clip(.05, 2000))
    # chi2 → nlp, 매우 큰 chi2 에서 logsf underflow 방지
    chi2_vals = pd.to_numeric(plot["chi2"], errors="coerce").clip(lower=0, upper=1e12)
    plot["nlp"] = -_chi2_dist.logsf(chi2_vals, 1) / np.log(10)
    plot = plot[np.isfinite(plot["nlp"])]

    fig = px.scatter(plot, x="log2_ebgm", y="nlp",
                     color="final_severity", color_discrete_map=SEV_COLOR,
                     category_orders={"final_severity":
                         ["UNCLASSIFIED","MILD","MODERATE","SEVERE","FATAL"]},
                     hover_data=["target_drug","pt","a","ebgm","eb05","prr"],
                     labels={"nlp":"−log10(p)", "log2_ebgm":"log2(EBGM)"},
                     height=420)
    # UNCLASSIFIED trace 를 옅게 처리 → FATAL/SEVERE 가 시각적으로 두드러짐
    for tr in fig.data:
        if tr.name == "UNCLASSIFIED":
            tr.marker.opacity = 0.25
    fig = _style_fig(fig)
    fig.update_layout(margin=dict(l=60, r=20, t=10, b=40), title="")
    fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="grey", line_width=1)
    fig.add_vline(x=1, line_dash="dash", line_color="grey", line_width=1)
    st.plotly_chart(fig, use_container_width=True)

    # Table: EB05 상위 N 개
    st.subheader("Filtered signals")
    show = (df.sort_values("eb05", ascending=False)
              .head(max_rows)
              [["target_drug","pt","a","ebgm","eb05","prr","ror",
                "final_severity","pt_severity","n_cases","death_rate","target_group"]]
              .copy())
    show.columns = ["drug","event","n","EBGM","EB05","PRR","ROR",
                    "empirical","clinical PT","cases","death rate","group"]
    st.dataframe(show, use_container_width=True, hide_index=True)

# ============================================================
#                     PAGE: Drug Profile
# ============================================================
SAFETY_SECTIONS = [
    ("overall_safety_profile",  "Overall Safety Profile"),
    ("most_concerning_signals", "Most Concerning Signals"),
    ("expected_vs_novel",       "Expected vs Novel Signals"),
    ("class_comparison",        "Class Comparison"),
    ("risk_benefit",            "Risk–Benefit Assessment"),
    ("recommendations",         "Recommendations"),
]
CASE_SECTIONS = [
    ("signal_overview",       "Signal overview"),
    ("statistical_evidence",  "Statistical evidence"),
    ("case_outcomes",         "Case outcomes"),
    ("mechanism",             "Plausible mechanism"),
    ("class_comparison",      "Class comparison"),
    ("recommendation",        "Recommendation"),
]

def _data_driven_assessment(dwork, drug, dclass, indi):
    n_total  = len(dwork)
    n_fatal  = int((dwork["final_severity"]=="FATAL").sum())
    n_severe = int((dwork["final_severity"]=="SEVERE").sum())
    pct_fs   = (n_fatal+n_severe)/n_total*100 if n_total else 0
    top3 = dwork.sort_values("eb05", ascending=False).head(3)
    top3_txt = "; ".join(
        f"{r['pt']} (EBGM {r['ebgm']:.1f}, n={int(r['a'])}, {r['final_severity']})"
        for _, r in top3.iterrows()) or "no signals detected"
    return {
        "overall_safety_profile":
            f"{drug} ({dclass}, indicated for {indi}) shows {n_total:,} drug-event "
            f"signals, of which {n_fatal+n_severe:,} ({pct_fs:.1f}%) are FATAL/SEVERE.",
        "most_concerning_signals": f"Top three signals by EB05: {top3_txt}.",
        "expected_vs_novel":
            "Expected-vs-novel classification requires domain review; compare top signals "
            "against the drug's labeled adverse events.",
        "class_comparison":
            f"Class-level comparison for {dclass} is not available in the automated summary; "
            "review against established class warnings.",
        "risk_benefit":
            f"With {pct_fs:.1f}% FATAL/SEVERE rate, monitoring intensity should reflect "
            "the severity distribution shown above.",
        "recommendations":
            "Routine pharmacovigilance monitoring; targeted review of the highest-EB05 "
            "events if novel for the drug class.",
    }

def _render_case_card(signal_row, llm_row):
    r = signal_row
    dr = r.get("death_rate")
    dr_txt = f"{float(dr)*100:.1f}%" if pd.notna(dr) else "—"
    g1, g2, g3 = st.columns(3)
    g1.markdown(f"**Reports (a)**\n\n{int(r['a']):,}")
    g1.markdown(f"**EBGM**\n\n{r['ebgm']:.2f}")
    g2.markdown(f"**EB05**\n\n{r['eb05']:.2f}")
    g2.markdown(f"**PRR**\n\n{r['prr']:.2f}")
    g3.markdown(f"**Cases analyzed**\n\n{int(r.get('n_cases',0) or 0):,}")
    g3.markdown(f"**Death rate**\n\n{dr_txt}")

    has_struct = (llm_row is not None and
                  any(isinstance(llm_row.get(k), str) and llm_row.get(k).strip()
                      for k,_ in CASE_SECTIONS))
    if has_struct:
        st.markdown("---")
        for key, label in CASE_SECTIONS:
            v = llm_row.get(key)
            if isinstance(v, str) and v.strip():
                st.markdown(f"**{label}.** {v}")
    elif llm_row is not None and isinstance(llm_row.get("summary"), str) and llm_row["summary"].strip():
        st.markdown("---")
        st.markdown(llm_row["summary"])

def page_drug_profile():
    st.title("Drug Profile")
    drug = st.selectbox("Select drug",
        sorted(work["target_drug"].dropna().unique()))
    if not drug: return

    dclass, indi = DRUG_CONTEXT.get(drug, ("Unknown","Unknown"))
    dwork = work[work["target_drug"]==drug]
    group = dwork["target_group"].iloc[0] if not dwork.empty else "—"

    # Header
    st.markdown(f"### {drug}")
    st.caption(f"{dclass}  •  {indi}  •  Group: **{group}**")

    n_total    = len(dwork)
    n_fatal    = int((dwork["final_severity"]=="FATAL").sum())
    n_severe   = int((dwork["final_severity"]=="SEVERE").sum())
    n_unclass  = int((dwork["final_severity"]=="UNCLASSIFIED").sum())
    n_labeled  = n_total - n_unclass
    pct_fs     = (n_fatal+n_severe)/n_total*100 if n_total else 0
    coverage   = (n_labeled / n_total * 100) if n_total else 0

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Total signals",  f"{n_total:,}")
    k2.metric("FATAL",          f"{n_fatal:,}")
    k3.metric("SEVERE",         f"{n_severe:,}")
    k4.metric("Fatal/Severe %", f"{pct_fs:.1f}%")
    k5.metric("Coverage",       f"{coverage:.1f}%",
              help=f"{n_labeled:,} of {n_total:,} signals classified; {n_unclass:,} unclassified")
    st.divider()

    # Safety Profile (강제 6섹션)
    st.subheader("Safety Profile Assessment")
    src_int = interp.get(drug)
    if isinstance(src_int, dict) and any((src_int.get(k) or "").strip() for k,_ in SAFETY_SECTIONS):
        sections = src_int
        st.caption("LLM-generated assessment")
    else:
        sections = _data_driven_assessment(dwork, drug, dclass, indi)
        st.caption("Auto-generated assessment from signal data (no LLM available)")
    for key, title in SAFETY_SECTIONS:
        txt = (sections.get(key) or "").strip()
        if txt:
            st.markdown(f"**{title}**  \n{txt}")
    st.divider()

    # Signal Overview
    st.subheader("Signal Overview")
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("**Top 15 signals by EB05**")
        top = (dwork.sort_values("eb05", ascending=False).head(15)
               [["pt","a","ebgm","eb05","prr","final_severity",
                 "pt_severity","n_cases","death_rate"]]).copy()
        top.columns = ["event","n","EBGM","EB05","PRR",
                       "empirical","clinical","cases","death rate"]
        st.dataframe(top, use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Severity distribution**")
        st.caption(f"Showing {n_labeled:,} labeled signals "
                   f"({n_unclass:,} unclassified hidden)")
        sc = (dwork["final_severity"].value_counts()
              .reindex(SEV_ORDER).fillna(0))
        fig = go.Figure(go.Bar(
            x=sc.values, y=sc.index, orientation="h",
            marker_color=[SEV_COLOR.get(s,"#888") for s in sc.index],
            text=[int(v) for v in sc.values], textposition="outside"))
        fig = _style_fig(fig)
        fig.update_layout(height=320, title="",
                          margin=dict(l=80, r=40, t=10, b=30), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    st.divider()

    # Representative Cases (항상 top 3)
    st.subheader("Representative Cases")
    st.caption("Top 3 adverse events by EB05")
    top3 = dwork.sort_values("eb05", ascending=False).head(3)
    for _, row in top3.iterrows():
        pt, sev = row["pt"], row["final_severity"]
        c = pd.DataFrame()
        if not cases.empty and {"drug","event"}.issubset(cases.columns):
            c = cases[(cases["drug"].str.lower()==drug.lower()) & (cases["event"]==pt)]
        with st.expander(f"{pt}  •  {sev}", expanded=False):
            _render_case_card(row, c.iloc[0] if not c.empty else None)

# ============================================================
#                     PAGE: Drug Comparison
# ============================================================
def page_comparison():
    st.title(" Drug Comparison")
    drugs = st.multiselect("Compare drugs (2-4 recommended)",
        sorted(work["target_drug"].dropna().unique()),
        default=["semaglutide","metformin"])
    if len(drugs) < 2:
        st.info("Select at least 2 drugs.")
        return

    sub = work[work["target_drug"].isin(drugs)]
    summary = (sub.groupby("target_drug").agg(
                  signals=("a","size"),
                  fatal=("final_severity", lambda s:(s=="FATAL").sum()),
                  severe=("final_severity", lambda s:(s=="SEVERE").sum()),
                  median_ebgm=("ebgm","median"),
                  top_ebgm=("ebgm","max"),
                  reports=("a","sum"),
              ).reindex(drugs).reset_index())
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("EBGM distribution per drug")
    fig = px.box(sub, x="target_drug", y="ebgm", points="all",
                 color="target_drug", log_y=True, height=380)
    fig = _style_fig(fig)
    fig.update_layout(showlegend=False, title="",
                      margin=dict(l=60, r=20, t=10, b=40))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Forest plot — top signals with EB05–EB95 interval")
    st.caption("Each drug's strongest signals shown with their 90% credibility "
               "interval (EB05 to EB95). Non-overlapping intervals = robust differences.")
    fr_rows=[]
    for d in drugs:
        dd=(sub[sub["target_drug"]==d].sort_values("eb05",ascending=False).head(6))
        for _,r in dd.iterrows():
            fr_rows.append(dict(drug=d, label=f"{d[:10]} · {str(r['pt'])[:24]}",
                                ebgm=float(r["ebgm"]),
                                lo=float(r.get("eb05",np.nan)),
                                hi=float(r.get("eb95",r.get("ebgm",np.nan))),
                                sev=str(r.get("final_severity","")) ))
    fr=pd.DataFrame(fr_rows)
    if not fr.empty:
        fr=fr.sort_values(["drug","ebgm"])
        ff=go.Figure()
        for d in drugs:
            sd=fr[fr["drug"]==d]
            ff.add_trace(go.Scatter(
                x=sd["ebgm"], y=sd["label"], mode="markers",
                marker=dict(size=9, color=GROUP_COLOR.get(
                    work[work.target_drug==d]["target_group"].iloc[0] if len(work[work.target_drug==d]) else "meta","#2980b9")),
                error_x=dict(type="data", symmetric=False,
                             array=(sd["hi"]-sd["ebgm"]).clip(lower=0),
                             arrayminus=(sd["ebgm"]-sd["lo"]).clip(lower=0),
                             thickness=1.4, width=4, color="#98a2b3"),
                name=d, hovertemplate="%{y}<br>EBGM=%{x:.1f}<extra></extra>"))
        ff.add_vline(x=1.0, line_dash="dot", line_color="#c0392b",
                     annotation_text="EBGM=1 (no signal)", annotation_position="top")
        ff=_style_fig(ff, height=max(360, 26*len(fr)))
        ff.update_layout(title="",
                         xaxis_type="log", xaxis_title="EBGM (log scale)",
                         yaxis_title=None,
                         margin=dict(l=240, r=20, t=30, b=40),
                         legend=dict(orientation="h", y=1.04))
        st.plotly_chart(ff, use_container_width=True)

    st.subheader("Shared adverse events (top by sum-EBGM across selection)")
    shared = (sub.pivot_table(index="pt", columns="target_drug",
                              values="ebgm", aggfunc="max")
                  .dropna())
    if not shared.empty:
        shared["sum"] = shared.sum(axis=1)
        st.dataframe(shared.sort_values("sum",ascending=False).head(15).drop(columns="sum"),
                     use_container_width=True)
    else:
        st.info("No adverse events are signals across all selected drugs.")

    st.subheader("Per-drug top 5 signals")
    cols = st.columns(len(drugs))
    for col,d in zip(cols,drugs):
        with col:
            st.markdown(f"**{d}**")
            t = (sub[sub["target_drug"]==d]
                 .sort_values("eb05",ascending=False).head(5)
                 [["pt","ebgm","final_severity"]])
            t.columns=["event","EBGM","sev"]
            st.dataframe(t, use_container_width=True, hide_index=True)

# ============================================================
#                     PAGE: Dual-Severity Analysis
# ============================================================
def page_dual_severity():
    st.title(" Dual-Severity Analysis")
    st.markdown(
        "Finding: **two complementary severity signals** measure different things.\n"
        "- **Clinical PT severity** (`pt_severity`): intrinsic risk of the MedDRA term (LLM-classified, grounded by anchors).\n"
        "- **Empirical outcome severity** (`final_severity`): observed FAERS mortality/seriousness.\n\n"
        "Their disagreement is itself the finding — neither is wrong; they capture different aspects."
    )

    if calib.empty or "pt_severity" not in calib.columns:
        st.warning("severity_calibrated_v6.csv missing pt_severity. Run Phase 3 v6.2 first.")
        return

    cdf = calib.dropna(subset=["pt_severity","final_severity"]).copy()
    cdf = cdf[cdf["final_severity"].isin(SEV_ORDER) & cdf["pt_severity"].isin(SEV_ORDER)]
    if cdf.empty:
        st.info("No comparable rows.")
        return

    # Confusion matrix
    cm = pd.crosstab(cdf["pt_severity"], cdf["final_severity"]).reindex(
        index=SEV_ORDER, columns=SEV_ORDER, fill_value=0)
    c1,c2 = st.columns([1.2,1])
    with c1:
        st.subheader("Disagreement matrix")
        fig = px.imshow(cm.values,
                        x=cm.columns, y=cm.index,
                        text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Empirical (final)", y="Clinical PT", color="count"))
        fig.update_layout(height=380, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Summary")
        diag = np.trace(cm.values)
        adj = sum(cm.values[i,j] for i in range(4) for j in range(4) if abs(i-j)<=1)
        total = int(cm.values.sum())
        st.metric("Pairs evaluated", f"{total:,}")
        st.metric("Exact agreement", f"{diag/total:.1%}")
        st.metric("Adjacent (±1) agreement", f"{adj/total:.1%}")
        # weighted kappa (quadratic)
        try:
            from sklearn.metrics import cohen_kappa_score
            k  = cohen_kappa_score(cdf["pt_severity"], cdf["final_severity"], labels=SEV_ORDER)
            wk = cohen_kappa_score(cdf["pt_severity"], cdf["final_severity"], labels=SEV_ORDER, weights="quadratic")
            st.metric("Cohen's κ", f"{k:.3f}")
            st.metric("Quadratic-weighted κ", f"{wk:.3f}")
        except Exception:
            pass

    st.divider()
    st.subheader("Disagreement examples (clinical vs empirical mismatch)")
    mismatch = cdf[cdf["pt_severity"]!=cdf["final_severity"]].copy()
    mismatch["gap"] = mismatch.apply(
        lambda r: SEV_ORDER.index(r["pt_severity"]) - SEV_ORDER.index(r["final_severity"]),
        axis=1)
    # Biggest empirical > clinical (i.e., FAERS-level severity inflated vs PT risk)
    top_emp = mismatch[mismatch["gap"]>0].sort_values("n_cases", ascending=False).head(8)
    top_clin = mismatch[mismatch["gap"]<0].sort_values("n_cases", ascending=False).head(8)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**Empirical >> Clinical** (patient-population effect candidates)")
        st.dataframe(top_emp[["target_drug","pt","pt_severity","final_severity",
                              "n_cases","death_rate"]],
                     use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Clinical >> Empirical** (well-managed despite serious PT)")
        st.dataframe(top_clin[["target_drug","pt","pt_severity","final_severity",
                              "n_cases","death_rate"]],
                     use_container_width=True, hide_index=True)

# ============================================================
#                     PAGE: Q&A (RAG)
# ============================================================
def page_qa():
    st.title(" Q&A — grounded RAG")
    st.markdown(
        "Each answer is grounded **only** in retrieved FAERS signals, with citations. "
        "Filters are auto-extracted from the question."
    )

    tab1, tab2 = st.tabs(["Cached demos", " Live (requires API)"])

    with tab1:
        if not qa_cache:
            st.info("No cached Q&A. Run Phase 4 v6 to generate `rag_qa_v6.json`.")
        else:
            st.caption(f"{len(qa_cache)} demo answers from Phase 4 v6.2")
            for q in qa_cache:
                with st.expander(f" {q['query']}"):
                    cols = st.columns([1,2])
                    with cols[0]:
                        st.markdown("**Filters extracted**")
                        st.json(q.get("filters",{}), expanded=False)
                        st.metric("Retrieved", q.get("retrieved_n",0))
                        st.metric("Citations",  len(q.get("citations",[])))
                        st.metric("Fallback?",  "Yes" if q.get("retrieval_fallback") else "No")
                    with cols[1]:
                        st.markdown(q["answer"])

    with tab2:
        st.markdown("Live answers call Anthropic + ChromaDB. Provide API key below.")
        api_key = st.text_input("ANTHROPIC_API_KEY", type="password",
                                value=os.environ.get("ANTHROPIC_API_KEY",""))
        query = st.text_input("Question",
            placeholder="e.g. What are the strongest fatal signals for semaglutide?")
        go = st.button("Ask", type="primary")
        if go and query:
            if not api_key:
                st.error("API key required.")
                return
            col = get_chroma_collection()
            if col is None:
                st.error(f"ChromaDB at {CHROMA_DIR} not available.")
                return
            try:
                from anthropic import Anthropic
                client = Anthropic(api_key=api_key)
            except Exception as e:
                st.error(f"Anthropic SDK unavailable: {e}")
                return

            DRUGS = sorted(work["target_drug"].dropna().unique())
            parser_sys = (
                "Extract structured filters from a pharmacovigilance question. "
                "Return ONLY a JSON object: "
                "drugs: list (subset of allowed); severity: FATAL|SEVERE|MODERATE|MILD|null; "
                "group: validation|gsk|trending|negative_control|null; top_k: int 1-20; "
                "intent: one of risk_profile, severity_focus, comparison, mechanism, signal_list, general. "
                "Allowed drugs: " + ", ".join(DRUGS)
            )
            with st.spinner("Parsing question…"):
                try:
                    r = client.messages.create(
                        model="claude-sonnet-4-20250514", max_tokens=300,
                        system=parser_sys,
                        messages=[{"role":"user","content":f"Question: {query}\nReturn JSON only."}])
                    txt = r.content[0].text.strip()
                    if txt.startswith("```"):
                        txt = txt.split("\n",1)[1].rsplit("```",1)[0]
                    filt = json.loads(txt)
                    filt["drugs"] = [d for d in (filt.get("drugs") or []) if d in DRUGS]
                    if filt.get("severity") not in {None,"FATAL","SEVERE","MODERATE","MILD"}:
                        filt["severity"]=None
                    filt.setdefault("top_k",5)
                    if filt.get("intent")=="comparison" and len(filt["drugs"])>=2:
                        filt["top_k"] = max(filt["top_k"], 5*len(filt["drugs"]))
                except Exception as e:
                    st.warning(f"Parser failed; falling back. {e}")
                    filt = {"drugs":[], "top_k":5, "intent":"general"}

            st.markdown("**Filters**"); st.json(filt, expanded=False)
            with st.spinner("Retrieving from ChromaDB…"):
                ret = live_retrieve(query, filt, k=filt.get("top_k",5), collection=col)
            metas = ret["metadatas"][0]; ids = ret["ids"][0]; docs = ret["documents"][0]
            if not docs:
                st.warning("No documents retrieved.")
                return
            st.caption(f"Retrieved {len(docs)} docs — drugs: "
                       + ", ".join(sorted(set(m.get('drug','') for m in metas))))

            # pop-effect flag
            BENIGN = {"TASTE DISORDER","DYSGEUSIA","HYPOAESTHESIA","PARAESTHESIA","MIGRAINE",
                      "HEADACHE","NAUSEA","FATIGUE","DIZZINESS","DRY MOUTH","ARTHRALGIA",
                      "PRURITUS","RASH","DRUG INEFFECTIVE","OFF LABEL USE",
                      "PRODUCT DOSE OMISSION ISSUE","BACK PAIN","COUGH","CONSTIPATION",
                      "DIARRHOEA","ABDOMINAL PAIN","ASTHENIA","INSOMNIA","ALOPECIA",
                      "DRY SKIN","EYE IRRITATION","VOMITING","DECREASED APPETITE"}
            def _pop(m):
                try:
                    dr=float(m.get("death_rate",0) or 0); n=int(m.get("n_cases",0) or 0)
                    return (dr>=0.30 and n>=10 and any(b in str(m.get("event","")).upper() for b in BENIGN))
                except: return False
            pop_flags = [_pop(m) for m in metas]

            ctx = "\n\n".join(
                f"[{ids[i]}] (drug={m.get('drug')}, event={m.get('event')}, "
                f"severity={m.get('final_severity')}, EBGM={m.get('ebgm'):.1f}, "
                f"n_cases={m.get('n_cases')}, death_rate={float(m.get('death_rate',0)):.2f}, "
                f"small_n={int(int(m.get('n_cases',0))<10)}, pop_effect={int(pop_flags[i])})\n{docs[i]}"
                for i,m in enumerate(metas))

            answer_sys = (
                "You are a clinical pharmacovigilance assistant grounded ONLY in CONTEXT. "
                "Cite each claim with [sig_…]. Surface BOTH empirical severity (final_severity) "
                "and clinical PT risk; flag disagreement. When n<10 add '(small-n, unstable)'. "
                "When pop_effect=1 add '(likely patient-population effect — PT not typically fatal)'. "
                "For comparisons, structure side-by-side per drug. End with a one-line FAERS caveat."
            )
            with st.spinner("Synthesizing answer…"):
                r = client.messages.create(
                    model="claude-sonnet-4-20250514", max_tokens=1100,
                    system=answer_sys,
                    messages=[{"role":"user","content":f"QUESTION:\n{query}\n\nCONTEXT:\n{ctx}"}])
                ans = r.content[0].text
            st.markdown(ans)
            if sum(pop_flags):
                st.caption(f"{sum(pop_flags)} retrieved row(s) flagged as likely patient-population effect.")

            with st.expander("Retrieved documents (raw)"):
                st.dataframe(pd.DataFrame(metas), use_container_width=True)

# ============================================================
#                     PAGE: Validation
# ============================================================
def page_validation():
    st.title("Validation & Trust")
    st.markdown("How far can you trust these signals? Three independent checks: "
                "**sensitivity** (do we catch known signals?), **specificity** "
                "(do we reject non-signals?), and **clinical accuracy** (is the "
                "severity right vs. a clinical gold standard?).")

    # ---------- helpers (self-contained; recompute from data on the fly) ----------
    SEV_RANK = {"MILD":0,"MODERATE":1,"SEVERE":2,"FATAL":3}

    # ===== 1. POSITIVE CONTROLS — sensitivity =====
    KNOWN = {
        "ciprofloxacin":["TENDON RUPTURE","TENDON DISORDER","TENDONITIS","ARTHRALGIA"],
        "isotretinoin":["DEPRESSION","CHEILITIS","DRY SKIN"],
        "clozapine":["AGRANULOCYTOSIS","NEUTROPENIA","MYOCARDITIS"],
        "warfarin":["HAEMORRHAGE","INTERNATIONAL NORMALISED RATIO INCREASED","GASTROINTESTINAL HAEMORRHAGE"],
        "rofecoxib":["MYOCARDIAL INFARCTION","CEREBROVASCULAR ACCIDENT"],
        "rosiglitazone":["CARDIAC FAILURE CONGESTIVE","MYOCARDIAL INFARCTION","OEDEMA PERIPHERAL"],
    }
    prows=[]
    for d,events in KNOWN.items():
        for e in events:
            sub = work[(work["target_drug"]==d) &
                       (work["pt"].str.upper()==e)]
            if len(sub)==0:
                sub = work[(work["target_drug"]==d) &
                           (work["pt"].str.contains(e, case=False, na=False))]
            if len(sub):
                r=sub.sort_values("eb05",ascending=False).iloc[0]
                prows.append(dict(drug=d, event=e, a=int(r["a"]),
                                  EB05=round(r["eb05"],1),
                                  detected=bool(r.get("is_signal",False))))
    pdf=pd.DataFrame(prows)
    sens_hit=int(pdf["detected"].sum()); sens_n=len(pdf)
    sens_pct=sens_hit/max(sens_n,1)*100

    # ===== 2. NEGATIVE CONTROLS (OMOP pair-level) — specificity =====
    NEG_PAIRS = {
        "metformin":["TINNITUS","CATARACT","CONJUNCTIVITIS","ONYCHOMYCOSIS","NASAL CONGESTION","NEPHROLITHIASIS"],
        "lisinopril":["TINNITUS","CATARACT","NASAL CONGESTION","NEPHROLITHIASIS"],
        "omeprazole":["TINNITUS","CATARACT","CONJUNCTIVITIS","ONYCHOMYCOSIS","TONSILLITIS","NASAL CONGESTION","NEPHROLITHIASIS","PLANTAR FASCIITIS"],
        "amlodipine":["TINNITUS","CATARACT","NASAL CONGESTION","NEPHROLITHIASIS","PLANTAR FASCIITIS"],
        "amoxicillin":["TINNITUS","CATARACT","CONJUNCTIVITIS","TONSILLITIS","NASAL CONGESTION","NEPHROLITHIASIS"],
    }
    nrows=[]
    for d,pts in NEG_PAIRS.items():
        for pt in pts:
            hit=work[(work["target_drug"]==d)&(work["pt"]==pt)]
            if len(hit):
                r=hit.iloc[0]
                nrows.append(dict(drug=d, pt=pt, EB05=round(float(r["eb05"]),2),
                                  is_signal=bool(r.get("is_signal",False))))
    ndf=pd.DataFrame(nrows)
    fp=int(ndf["is_signal"].sum()); neg_n=len(ndf)
    spec_pct=(1-fp/max(neg_n,1))*100

    # ===== 3. GOLD STANDARD — clinical accuracy =====
    gp = os.path.join(RESULTS_DIR,"gold_standard_validation_v6.csv")
    gold = pd.read_csv(gp) if os.path.exists(gp) else pd.DataFrame()
    def _acc(col):
        if gold.empty or col not in gold: return np.nan
        v=gold[gold[col].notna()]
        return np.mean([a==b for a,b in zip(v["gold"],v[col])])*100 if len(v) else np.nan
    llm_acc=_acc("llm"); gt_acc=_acc("gt_sev")

    # ===================== KPI ROW =====================
    k1,k2,k3,k4 = st.columns(4)
    with k1:
        st.metric("Sensitivity", f"{sens_pct:.0f}%",
                  help=f"{sens_hit}/{sens_n} established positive-control signals detected")
    with k2:
        st.metric("Specificity (OMOP)", f"{spec_pct:.1f}%",
                  help=f"{neg_n-fp}/{neg_n} validated unrelated drug-event pairs correctly rejected")
    with k3:
        st.metric("LLM clinical accuracy", f"{llm_acc:.0f}%" if pd.notna(llm_acc) else "—",
                  delta=f"+{llm_acc-gt_acc:.0f} pts vs FAERS-GT" if pd.notna(llm_acc) and pd.notna(gt_acc) else None,
                  help="Exact-match severity vs clinical gold standard on established signals")
    with k4:
        st.metric("Signals analysed", f"{int(work['is_signal'].sum()):,}")

    st.divider()

    # ===================== SENSITIVITY + SPECIFICITY side by side =====================
    cL, cR = st.columns(2)

    with cL:
        st.subheader("Sensitivity — known signals caught")
        if not pdf.empty:
            pv=pdf.copy()
            pv["status"]=np.where(pv["detected"],"detected","missed")
            fig=px.bar(pv.sort_values("EB05"), x="EB05", y="event",
                       orientation="h", color="status",
                       color_discrete_map={"detected":"#27ae60","missed":"#c0392b"},
                       hover_data=["drug","a"], height=420)
            fig=_style_fig(fig)
            fig.update_layout(title="",
                              yaxis_title=None, legend_title=None,
                              margin=dict(l=200, r=20, t=60, b=40),
                              legend=dict(orientation="h",
                                          yanchor="bottom", y=1.02,
                                          xanchor="right",  x=1.0))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"**{sens_hit}/{sens_n}** established signals detected. "
                       "Bars = EB05 (signal strength); green = correctly flagged.")

    with cR:
        st.subheader("Specificity — OMOP negative controls")
        # Gauge
        g=go.Figure(go.Indicator(
            mode="gauge+number", value=spec_pct,
            number={"suffix":"%","font":{"size":40}},
            gauge={"axis":{"range":[0,100]},
                   "bar":{"color":"#27ae60"},
                   "steps":[{"range":[0,70],"color":"#fdecea"},
                            {"range":[70,90],"color":"#fef5e7"},
                            {"range":[90,100],"color":"#eafaf1"}],
                   "threshold":{"line":{"color":"#c0392b","width":3},
                                "thickness":0.8,"value":95}}))
        g.update_layout(height=240, margin=dict(l=20,r=20,t=10,b=0))
        st.plotly_chart(g, use_container_width=True)
        st.caption(f"**{fp}/{neg_n}** validated unrelated pairs falsely flagged "
                   f"(FP rate {fp/max(neg_n,1)*100:.1f}%). Uses OMOP-style pair-level "
                   "controls — the rigorous PV standard, not drug-level rates.")
        if fp>0:
            with st.expander(f"View {fp} residual false positives (confounding)"):
                st.dataframe(ndf[ndf["is_signal"]], use_container_width=True, hide_index=True)
                st.caption("These reflect age/comorbidity confounding (e.g. omeprazole "
                           "in elderly) — an intrinsic limit of disproportionality, "
                           "not a code defect.")

    st.divider()

    # ===================== GOLD STANDARD — clinical accuracy =====================
    st.subheader("Clinical accuracy — LLM vs FAERS-derived GT")
    if not gold.empty:
        cG1, cG2 = st.columns([1, 1.4])
        with cG1:
            acc_df = pd.DataFrame({
                "source":["FAERS-GT","LLM","Calibrated"],
                "accuracy":[_acc("gt_sev"), _acc("llm"), _acc("final")]})
            color_map = {"FAERS-GT":"#95a5a6","LLM":"#2980b9","Calibrated":"#8e44ad"}
            fig = px.bar(acc_df, x="source", y="accuracy",
                         text="accuracy", height=320)
            fig.update_traces(
                marker_color=[color_map[s] for s in acc_df["source"]],
                texttemplate="%{text:.0f}%", textposition="outside")
            fig = _style_fig(fig)
            fig.update_layout(showlegend=False, title="",
                              yaxis_title="Exact accuracy (%)",
                              xaxis_title=None, yaxis_range=[0,110],
                              margin=dict(l=60, r=20, t=30, b=40))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Accuracy against a clinical gold standard on established signals. "
                       "The LLM **outperforms** the FAERS-outcome GT.")
        with cG2:
            disp=gold.copy()
            disp["match"]=np.where(disp["gold"]==disp["llm"],"✅",
                          np.where(disp.apply(lambda r:abs(SEV_RANK.get(r["gold"],0)-SEV_RANK.get(r["llm"],0))<=1,axis=1),"~","❌"))
            show=disp[["drug","pt","gold","llm","gt_sev","match"]].rename(
                columns={"gold":"GOLD","llm":"LLM","gt_sev":"FAERS-GT"})
            st.dataframe(show, use_container_width=True, hide_index=True, height=320)
        st.info("**Why this matters:** FAERS-GT is death-rate driven, so it under-rates "
                "*treatable* emergencies (agranulocytosis, major haemorrhage). The LLM "
                "recovers these from clinical knowledge — the low LLM-vs-GT agreement "
                "(κ≈0.18) is **complementarity, not error**.")
    else:
        st.warning("gold_standard_validation_v6.csv not found — run Phase 3 §5b first.")

    st.divider()

    # ===================== Retrieval eval =====================
    if not ev.empty:
        st.subheader("RAG retrieval precision")
        keep=[c for c in ["qtype","query","drug_precision","severity_precision",
                          "pt_in_top5","all_drugs_covered"] if c in ev.columns]
        st.dataframe(ev[keep], use_container_width=True, hide_index=True)
        st.caption("Metadata-based precision (not keyword matching) on 17 evaluation queries.")


def page_about():
    st.title("About the FAERS Intelligence Pipeline")
    st.markdown(
        "**FAERS** = FDA Adverse Event Reporting System. Voluntary post-marketing reports of "
        "drug-related adverse events. This dashboard is the end of a 5-phase pipeline."
    )

    st.subheader("Pipeline")
    st.markdown(
        "1. **Phase 1 (ETL)** — Quarterly FAERS ASCII files are ingested into one unified "
        "`faers.db`. Reports are deduplicated conservatively (latest caseversion + demographic "
        "fingerprint when age is known), drug names normalized to a single RxNorm-style "
        "ingredient, and modern FAERS kept separate from pre-2012 legacy AERS so the two "
        "coding eras never silently merge.\n"
        "2. **Phase 2 (Signal detection)** — For each (drug, event) pair we build a 2×2 table "
        "and ask whether the event shows up disproportionately. We compute PRR/ROR with 95% CI "
        "(Haldane-Anscombe +0.5 to handle zero cells), Yates χ² with Benjamini-Hochberg FDR to "
        "control false positives across thousands of tests, and **empirical-Bayes Gamma-Poisson "
        "shrinkage** (EBGM/EB05) that pulls unstable small-count ratios toward the baseline — "
        "DuMouchel's single-component approximation of the FDA's MGPS.\n"
        "3. **Phase 3 (LLM × Ground-truth)** — Severity is derived two independent ways: "
        "ground-truth from FAERS outcome codes computed at the **PT level** (not drug-PT, which "
        "removes the bias of drugs used in already-sick populations), plus an LLM zero-shot "
        "classification over every detected signal. The two are cross-validated with Cohen's κ, "
        "**quadratic-weighted κ**, ±1-tier agreement, and binary FATAL∪SEVERE accuracy. On a "
        "clinical gold standard, LLM severity actually outperforms FAERS-outcome GT — their "
        "disagreement reflects complementary information, not error.\n"
        "4. **Phase 4 (RAG)** — All signals plus per-drug summaries are embedded in ChromaDB. "
        "**Per-drug split retrieval** divides k evenly across the drugs in the query so sparse "
        "drugs aren't starved by louder ones, and natural-language queries are parsed into "
        "structured metadata filters (drug, severity, group) *before* vector similarity. Every "
        "answer is forced to flag small-n and the patient-population-effect. Drug-precision rose "
        "from 0% (v5 naive search) to 100% (v6).\n"
        "5. **Phase 5 (Dashboard)** — This Streamlit app surfaces every output from Phases 1–4: "
        "signal explorer, drug profiles, drug-vs-drug comparison, calibration metrics, "
        "validation against positive/negative controls, and natural-language Q&A."
    )

    st.subheader("Data provenance")
    prov = meta.get("provenance","[]")
    try:
        prov_list = json.loads(prov)
        st.json(prov_list, expanded=False)
    except Exception:
        st.code(prov)

    st.subheader("Limitations of FAERS")
    st.markdown(
        "- **Voluntary reporting** — under-reporting common; severe outcomes over-represented.\n"
        "- **No denominator** — without prescription/exposure counts, true incidence is "
        "unknowable.\n"
        "- **Reporting bias** — Weber effect (spike around launch), media-driven spikes, "
        "country differences.\n"
        "- **No causality** — drug appears in a report doesn't mean drug caused the event.\n"
        "- **Confounding by indication** — sicker patients on certain drugs → background "
        "mortality inflates empirical severity.\n"
        "- **Duplicate cases** — even after careful dedup, some over-/under-merging is possible."
    )

    st.subheader("Methods")
    st.markdown(
        "- DuMouchel WH. _Bayesian data mining in large frequency tables, with an application "
        "to the FDA spontaneous reporting system._ The American Statistician, 1999.\n"
        "- Bate A, Evans SJW. _Quantitative signal detection using spontaneous ADR reporting._ "
        "Pharmacoepidemiology & Drug Safety, 2009.\n"
        "- Fusaroli M et al. _The DiAna dictionary…_ Drug Safety, 2024.\n"
        "- van Puijenbroek EP et al. _A comparison of measures of disproportionality…_ 2002."
    )

# ============================================================
#                     Router
# ============================================================
ROUTES = {
    " Overview":           page_overview,
    " Signal Explorer":     page_explorer,
    " Drug Profile":        page_drug_profile,
    " Drug Comparison":      page_comparison,
    " Dual-Severity":       page_dual_severity,
    " Q&A (RAG)":           page_qa,
    " Validation":          page_validation,
    " About":               page_about,
}
ROUTES[page]()
