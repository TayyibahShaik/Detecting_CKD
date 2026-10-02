"""CKD Detection - Streamlit dashboard.
One file: trains your models from datset.csv and serves the UI.
Run:  streamlit run app.py   (put datset.csv next to this file, or upload it in the page)
"""
import os
import json
import urllib.request
from urllib.parse import quote
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="CKD Detect", page_icon="🩺", layout="wide")

CSV_NAME = "datset.csv"
GITHUB_REPO = "TayyibahShaik/Detecting_CKD"   # dataset is also fetched from here if not found locally
# Same features as your notebook, minus 'id' (it leaks the label because the file is sorted by class)
FEATURES = ["age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "htn",
            "bu", "dm", "sc", "cad", "sod", "appet", "pot", "pe", "hemo", "ane"]
MAPS = {
    "rbc": {"normal": 0.0, "abnormal": 1.0}, "pc": {"normal": 0.0, "abnormal": 1.0},
    "pcc": {"notpresent": 0.0, "present": 1.0}, "ba": {"notpresent": 0.0, "present": 1.0},
    "htn": {"no": 0.0, "yes": 1.0}, "dm": {"no": 0.0, "yes": 1.0},
    "cad": {"no": 0.0, "yes": 1.0}, "appet": {"good": 0.0, "poor": 1.0},
    "pe": {"no": 0.0, "yes": 1.0}, "ane": {"no": 0.0, "yes": 1.0},
}
CKD = 0  # ckd -> 0, notckd -> 1 (same as your LabelEncoder)

# Form text -> number used by the models
YN = {"no": 0.0, "yes": 1.0}
ENC = {
    "rbc": {"normal": 0.0, "abnormal": 1.0}, "pc": {"normal": 0.0, "abnormal": 1.0},
    "pcc": {"not present": 0.0, "present": 1.0}, "ba": {"not present": 0.0, "present": 1.0},
    "htn": YN, "dm": YN, "cad": YN, "pe": YN, "ane": YN,
    "appet": {"good": 0.0, "poor": 1.0},
}
# Example patients for one-click demo
HEALTHY = dict(age=40, bp=70, sg=1.020, al=0, su=0, rbc="normal", pc="normal", pcc="not present",
               ba="not present", bgr=100, bu=25.0, sc=0.9, sod=140.0, pot=4.3, hemo=15.0,
               htn="no", dm="no", cad="no", appet="good", pe="no", ane="no")
SICK = dict(age=62, bp=90, sg=1.010, al=3, su=1, rbc="abnormal", pc="abnormal", pcc="present",
            ba="not present", bgr=160, bu=80.0, sc=4.5, sod=128.0, pot=5.2, hemo=9.0,
            htn="yes", dm="yes", cad="no", appet="poor", pe="yes", ane="yes")
# Typical adult reference ranges (vary by lab) for the indicator cards
RANGES = {"sc": ("Serum creatinine", "mg/dL", 0.6, 1.3), "bu": ("Blood urea", "mg/dL", 15, 40),
          "hemo": ("Hemoglobin", "g/dL", 12, 17.5), "sod": ("Sodium", "mEq/L", 135, 145),
          "pot": ("Potassium", "mEq/L", 3.5, 5.0), "bgr": ("Random glucose", "mg/dL", 70, 140)}


# ------------------------------ data & models ------------------------------
def clean(df):
    df = df.copy()
    for c in [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]:
        df[c] = df[c].astype(str).str.strip().replace({"?": np.nan, "nan": np.nan})
    for col, m in MAPS.items():
        df[col] = df[col].map(m)
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    y = df["classification"].map({"ckd": 0, "notckd": 1})
    ok = y.notna()
    return df.loc[ok, FEATURES], y[ok].astype(int)


def make_models():
    imp = lambda: KNNImputer(n_neighbors=2)
    sc = lambda: StandardScaler()
    return {
        "Random Forest": Pipeline([("i", imp()), ("m", RandomForestClassifier(n_estimators=300, random_state=42))]),
        "Logistic Regression": Pipeline([("i", imp()), ("s", sc()), ("m", LogisticRegression(max_iter=1000))]),
        "SVM": Pipeline([("i", imp()), ("s", sc()), ("m", SVC(probability=True, random_state=42))]),
        "KNN": Pipeline([("i", imp()), ("s", sc()), ("m", KNeighborsClassifier())]),
        "Naive Bayes": Pipeline([("i", imp()), ("m", GaussianNB())]),
        "Voting Classifier": Pipeline([("i", imp()), ("m", VotingClassifier(
            [("LR", Pipeline([("s", sc()), ("m", LogisticRegression(max_iter=1000))])),
             ("RF", RandomForestClassifier(random_state=42)),
             ("DT", DecisionTreeClassifier(random_state=42))], voting="soft"))]),
    }


@st.cache_resource(show_spinner="Training models (first run only)...")
def train(csv_bytes_or_path):
    df = pd.read_csv(csv_bytes_or_path)
    X, y = clean(df)
    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models, acc = make_models(), {}
    for name, m in models.items():
        m.fit(xtr, ytr)
        acc[name] = round(m.score(xte, yte) * 100, 1)
    return models, acc


# ------------------------------ UI helpers ------------------------------
def build_row(vals):
    return {f: (ENC[f][vals[f]] if f in ENC else float(vals[f])) for f in FEATURES}


def status(v, lo, hi):
    return ("Low", "low") if v < lo else ("High", "high") if v > hi else ("Normal", "ok")


def indicators_html(vals):
    cards = ""
    for k, (name, unit, lo, hi) in RANGES.items():
        v = float(vals[k])
        label, cls = status(v, lo, hi)
        cards += (f'<div class="ind"><div class="ind-name">{name}</div>'
                  f'<div class="ind-val">{v:g} <span>{unit}</span></div>'
                  f'<span class="chip {cls}">{label}</span>'
                  f'<div class="ind-ref">Typical: {lo:g} to {hi:g}</div></div>')
    return f'<div class="ind-grid">{cards}</div>'


def result_html(is_ckd, p):
    if is_ckd:
        cls, icon, title, sub = "bad", "⚠️", "CKD likely detected", "The model sees a pattern consistent with chronic kidney disease."
    else:
        cls, icon, title, sub = "good", "✅", "CKD not detected", "The model sees no strong pattern of chronic kidney disease."
    pos = min(max(p, 0.0), 1.0) * 100
    return (f'<div class="result {cls}"><div class="r-title">{icon} {title}</div><div class="r-sub">{sub}</div></div>'
            f'<div class="meter-wrap"><div class="meter-label"><span>Low risk</span><b>{p:.1%} probability of CKD</b><span>High risk</span></div>'
            f'<div class="meter"><div class="dot" style="left:{pos}%"></div></div></div>')


CSS = """
<style>
.block-container{padding-top:1.6rem;max-width:1300px}
.hero{background:linear-gradient(135deg,#0f766e 0%,#2563eb 100%);padding:28px 34px;border-radius:20px;color:#fff;margin-bottom:22px;box-shadow:0 8px 24px rgba(37,99,235,.25)}
.hero h1{margin:0;font-size:2.1rem;color:#fff;font-weight:800;letter-spacing:-.5px}
.hero p{margin:8px 0 14px;opacity:.92;font-size:1.05rem}
.pill{display:inline-block;background:rgba(255,255,255,.18);padding:5px 14px;border-radius:99px;margin-right:8px;font-size:.85rem;font-weight:600}
.sec{font-weight:700;font-size:1.05rem;margin:0 0 6px}
.result{padding:22px 26px;border-radius:18px;color:#fff;margin-bottom:14px}
.result.bad{background:linear-gradient(135deg,#dc2626,#f97316);box-shadow:0 6px 18px rgba(220,38,38,.3)}
.result.good{background:linear-gradient(135deg,#059669,#10b981);box-shadow:0 6px 18px rgba(5,150,105,.3)}
.r-title{font-size:1.6rem;font-weight:800}.r-sub{opacity:.95;margin-top:4px}
.meter-wrap{margin:6px 0 18px}
.meter-label{display:flex;justify-content:space-between;font-size:.85rem;margin-bottom:10px;opacity:.9}
.meter{height:14px;border-radius:99px;background:linear-gradient(90deg,#10b981,#facc15,#ef4444);position:relative}
.dot{position:absolute;top:-5px;width:24px;height:24px;border-radius:50%;background:#fff;border:4px solid #1f2937;transform:translateX(-50%);box-shadow:0 2px 6px rgba(0,0,0,.35)}
.ind-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.ind{background:rgba(127,127,127,.08);border:1px solid rgba(127,127,127,.25);border-radius:14px;padding:14px}
.ind-name{font-size:.82rem;opacity:.75}.ind-val{font-size:1.4rem;font-weight:800;margin:2px 0 6px}
.ind-val span{font-size:.75rem;font-weight:500;opacity:.7}.ind-ref{font-size:.75rem;opacity:.65;margin-top:6px}
.chip{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.78rem;font-weight:700}
.chip.ok{background:rgba(16,185,129,.18);color:#10b981}.chip.high{background:rgba(239,68,68,.18);color:#ef4444}.chip.low{background:rgba(245,158,11,.2);color:#f59e0b}
.empty{border:2px dashed rgba(127,127,127,.35);border-radius:18px;padding:40px 24px;text-align:center;opacity:.85}
.empty .big{font-size:2.6rem}
</style>
"""

def find_csv():
    """Find the dataset automatically: by name first, then any CSV in the repo that has the CKD columns."""
    here = Path(__file__).resolve().parent
    bases = [here, Path.cwd(), here.parent]
    for base in bases:
        for name in (CSV_NAME, "dataset.csv", "kidney_disease.csv", "ckd.csv"):
            if (base / name).is_file():
                return base / name
    seen = set()
    for base in bases:
        for p in sorted(base.rglob("*.csv")):
            if p in seen or ".git" in p.parts or "site-packages" in p.parts:
                continue
            seen.add(p)
            try:
                cols = {c.strip().lower() for c in pd.read_csv(p, nrows=1).columns}
            except Exception:
                continue
            if {"classification", "hemo", "sc"} <= cols:
                return p
    return None


def local_csv_report():
    """Names + first columns of every CSV the app can see (shown only if nothing is found)."""
    here = Path(__file__).resolve().parent
    out = []
    for p in sorted(set(here.rglob("*.csv")) | set(Path.cwd().rglob("*.csv")))[:10]:
        try:
            out.append(f"{p.name}: {', '.join(list(pd.read_csv(p, nrows=1).columns)[:8])}")
        except Exception:
            out.append(f"{p.name}: (could not read)")
    return out


@st.cache_data(show_spinner="Fetching dataset from GitHub...", ttl=3600)
def fetch_github_csv():
    """Find a CKD-looking CSV in the GitHub repo and return its raw URL (or None)."""
    for branch in ("main", "master"):
        try:
            api = f"https://api.github.com/repos/{GITHUB_REPO}/git/trees/{branch}?recursive=1"
            with urllib.request.urlopen(api, timeout=15) as r:
                tree = json.load(r)["tree"]
        except Exception:
            continue
        for item in tree:
            if item["path"].lower().endswith(".csv"):
                raw = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{branch}/{quote(item['path'])}"
                try:
                    cols = {c.strip().lower() for c in pd.read_csv(raw, nrows=1).columns}
                except Exception:
                    continue
                if {"classification", "hemo", "sc"} <= cols:
                    return raw
    return None


# ===== UI =====
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="hero"><h1>🩺 CKD Detect</h1>'
    '<p>AI-assisted screening for Chronic Kidney Disease. Enter patient details and get an instant prediction.</p>'
    '<span class="pill">21 clinical features</span><span class="pill">6 ML models</span><span class="pill">Instant results</span></div>',
    unsafe_allow_html=True)

found = find_csv()
if found is not None:
    src, data_label = str(found), found.name
else:
    url = fetch_github_csv()
    if url is not None:
        src, data_label = url, url.rsplit("/", 1)[-1] + " (from GitHub)"
    else:
        st.warning("Could not find the CKD dataset in the repository. Upload the CSV once to continue.")
        with st.expander("Why wasn't it found?"):
            st.write("The app looks for a CSV with the columns classification, hemo and sc. CSV files it can see:")
            st.code("\n".join(local_csv_report()) or "(none)")
        up = st.file_uploader("Upload the CKD dataset (CSV)", type="csv")
        if up is None:
            st.stop()
        src, data_label = up, "uploaded file"
models, acc = train(src)

# default values + example loaders
for f, v in HEALTHY.items():
    st.session_state.setdefault(f"in_{f}", v)


def load(example):
    for f, v in example.items():
        st.session_state[f"in_{f}"] = v


with st.sidebar:
    st.markdown("### ⚙️ Settings")
    choice = st.selectbox("Prediction model", list(models), format_func=lambda n: f"{n}  ({acc[n]}%)")
    st.markdown("### 📊 Test accuracy (%)")
    st.bar_chart(pd.Series(acc, name="accuracy"), horizontal=True, height=260)
    st.caption(f"Dataset: {data_label}")
    st.caption("Accuracy on a held-out 20% test split. The 'id' column is excluded to avoid label leakage.")
    st.info("Educational demo only. Not a medical diagnosis. Consult a clinician.")

left, right = st.columns([1.25, 1], gap="large")

with left:
    b1, b2, _ = st.columns([1, 1, 1.2])
    b1.button("🟢 Load healthy example", on_click=load, args=(HEALTHY,), use_container_width=True)
    b2.button("🔴 Load CKD example", on_click=load, args=(SICK,), use_container_width=True)

    with st.form("patient"):
        with st.container(border=True):
            st.markdown('<div class="sec">👤 Patient details</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.number_input("Age (years)", 1, 120, key="in_age")
            c2.number_input("Blood pressure (mm/Hg)", 40, 220, key="in_bp")
            c3.selectbox("Specific gravity", [1.005, 1.010, 1.015, 1.020, 1.025], key="in_sg")
            c1, c2, c3 = st.columns(3)
            c1.selectbox("Albumin (0-5)", [0, 1, 2, 3, 4, 5], key="in_al")
            c2.selectbox("Sugar (0-5)", [0, 1, 2, 3, 4, 5], key="in_su")
            c3.number_input("Random glucose (mg/dL)", 20, 600, key="in_bgr")
        with st.container(border=True):
            st.markdown('<div class="sec">🧪 Blood tests</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.number_input("Blood urea (mg/dL)", 1.0, 400.0, step=1.0, key="in_bu")
            c2.number_input("Serum creatinine (mg/dL)", 0.1, 80.0, step=0.1, format="%.1f", key="in_sc")
            c3.number_input("Hemoglobin (g/dL)", 3.0, 20.0, step=0.1, format="%.1f", key="in_hemo")
            c1, c2, _ = st.columns(3)
            c1.number_input("Sodium (mEq/L)", 100.0, 170.0, step=1.0, key="in_sod")
            c2.number_input("Potassium (mEq/L)", 2.0, 50.0, step=0.1, format="%.1f", key="in_pot")
        with st.container(border=True):
            st.markdown('<div class="sec">🔬 Urine tests</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.selectbox("Red blood cells", ["normal", "abnormal"], key="in_rbc")
            c2.selectbox("Pus cell", ["normal", "abnormal"], key="in_pc")
            c3.selectbox("Pus clumps", ["not present", "present"], key="in_pcc")
            c4.selectbox("Bacteria", ["not present", "present"], key="in_ba")
        with st.container(border=True):
            st.markdown('<div class="sec">📋 Medical history</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.selectbox("Hypertension", ["no", "yes"], key="in_htn")
            c2.selectbox("Diabetes mellitus", ["no", "yes"], key="in_dm")
            c3.selectbox("Coronary artery disease", ["no", "yes"], key="in_cad")
            c1, c2, c3 = st.columns(3)
            c1.selectbox("Appetite", ["good", "poor"], key="in_appet")
            c2.selectbox("Pedal edema", ["no", "yes"], key="in_pe")
            c3.selectbox("Anemia", ["no", "yes"], key="in_ane")
        go = st.form_submit_button("🔍 Predict", type="primary", use_container_width=True)

    if go:
        st.session_state["last"] = {f: st.session_state[f"in_{f}"] for f in FEATURES}

with right:
    if "last" not in st.session_state:
        st.markdown('<div class="empty"><div class="big">🧬</div><h3>Your result will appear here</h3>'
                    '<p>Fill in the form (or load an example) and press <b>Predict</b>.</p></div>',
                    unsafe_allow_html=True)
    else:
        vals = st.session_state["last"]
        X = pd.DataFrame([build_row(vals)])[FEATURES]
        m = models[choice]
        pred = int(m.predict(X)[0])
        p = float(m.predict_proba(X)[0][list(m.classes_).index(CKD)])
        st.markdown(result_html(pred == CKD, p), unsafe_allow_html=True)
        st.caption(f"Model: {choice} · test accuracy {acc[choice]}%")

        t1, t2, t3 = st.tabs(["🧪 Key indicators", "🤖 Compare models", "📈 What matters"])
        with t1:
            st.markdown(indicators_html(vals), unsafe_allow_html=True)
            st.caption("Typical adult reference ranges; they vary by lab and patient.")
        with t2:
            rows = []
            for n, mm in models.items():
                pp = float(mm.predict_proba(X)[0][list(mm.classes_).index(CKD)]) * 100
                rows.append({"Model": n, "Verdict": "CKD" if int(mm.predict(X)[0]) == CKD else "No CKD",
                             "CKD probability": pp, "Test accuracy %": acc[n]})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                         column_config={"CKD probability": st.column_config.ProgressColumn(
                             "CKD probability", min_value=0, max_value=100, format="%.0f%%")})
        with t3:
            imp = models["Random Forest"].named_steps["m"].feature_importances_
            st.bar_chart(pd.Series(imp, index=FEATURES).sort_values().tail(10), horizontal=True)
            st.caption("Top 10 features by Random Forest importance.")

st.caption("⚕️ This tool is for education and demonstration only and does not replace professional medical advice.")
