import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, json, sqlite3, hashlib, uuid, io, base64, warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime as dt, timedelta
from fpdf import FPDF
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Poverty | AI Classification", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")

# ─── THEME CSS ────────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
*{font-family:'Inter',sans-serif;box-sizing:border-box;}
html,body,.stApp{background:#070d1a!important;color:#dde3f0;}
/* Sidebar */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0e1525 0%,#060c17 100%)!important;border-right:1px solid rgba(99,102,241,.18)!important;width:310px!important;}
[data-testid="stSidebarContent"]{padding:24px 16px;}
.stRadio>label{display:none;}
.stRadio>div{display:flex;flex-direction:column;gap:4px;}
.stRadio>div>label{padding:12px 16px!important;border-radius:10px!important;color:#94a3b8!important;font-size:14px!important;font-weight:600!important;cursor:pointer!important;transition:all .25s ease!important;background:transparent!important;border:1px solid transparent!important;display:flex!important;align-items:center!important;}
.stRadio>div>label:hover{background:rgba(255,255,255,0.03)!important;color:#e2e8f0!important;transform:translateX(4px)!important;}
.stRadio>div>label[data-checked="true"],.stRadio>div>label[aria-checked="true"]{background:linear-gradient(90deg,rgba(99,102,241,.15) 0%,transparent 100%)!important;color:white!important;border-left:3px solid #6366f1!important;border-radius:0 10px 10px 0!important;border-top:1px solid transparent!important;border-right:1px solid transparent!important;border-bottom:1px solid transparent!important;}
/* Cards */
.card{background:linear-gradient(135deg,rgba(14,21,37,.9),rgba(10,15,28,.95));border:1px solid rgba(255,255,255,.06);border-radius:16px;padding:24px;margin:8px 0;transition:border-color .2s,box-shadow .2s;}
.card:hover{border-color:rgba(99,102,241,.3);box-shadow:0 0 20px rgba(99,102,241,.1);}
.card-sm{background:linear-gradient(135deg,rgba(14,21,37,.9),rgba(10,15,28,.95));border:1px solid rgba(255,255,255,.06);border-radius:12px;padding:16px;}
/* KPI Tiles */
.kpi{background:linear-gradient(135deg,rgba(14,21,37,.98),rgba(9,14,26,.98));border:1px solid rgba(99,102,241,.2);border-radius:14px;padding:20px 16px;text-align:center;transition:all .25s;position:relative;overflow:hidden;}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#6366f1,#8b5cf6,#06b6d4);}
.kpi:hover{transform:translateY(-3px);box-shadow:0 12px 30px rgba(99,102,241,.2);border-color:rgba(99,102,241,.4);}
.kpi-icon{font-size:26px;margin-bottom:8px;}
.kpi-val{font-size:30px;font-weight:800;background:linear-gradient(135deg,#6366f1,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.1;}
.kpi-lbl{font-size:11px;color:#64748b;letter-spacing:1.2px;text-transform:uppercase;margin-top:4px;}
/* Badges */
.badge{display:inline-flex;align-items:center;gap:6px;padding:5px 14px;border-radius:100px;font-weight:700;font-size:13px;}
.badge-extreme{background:rgba(239,68,68,.12);color:#f87171;border:1px solid rgba(239,68,68,.35);}
.badge-moderate{background:rgba(245,158,11,.12);color:#fbbf24;border:1px solid rgba(245,158,11,.35);}
.badge-low{background:rgba(16,185,129,.12);color:#34d399;border:1px solid rgba(16,185,129,.35);}
/* Recommendation cards */
.rec{display:flex;gap:12px;align-items:flex-start;background:rgba(99,102,241,.06);border:1px solid rgba(99,102,241,.12);border-radius:10px;padding:12px 15px;margin:5px 0;transition:all .2s;}
.rec:hover{background:rgba(99,102,241,.1);border-color:rgba(99,102,241,.25);}
.rec-num{background:linear-gradient(135deg,#6366f1,#8b5cf6);color:white;border-radius:50%;width:24px;height:24px;min-width:24px;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;margin-top:1px;}
.rec-text{font-size:13.5px;color:#c4cdd8;line-height:1.5;}
/* Buttons */
.stButton>button{background:linear-gradient(135deg,#6366f1,#4f46e5)!important;color:white!important;border:none!important;border-radius:10px!important;padding:11px 28px!important;font-weight:600!important;font-size:14px!important;letter-spacing:.3px!important;transition:all .3s!important;}
.stButton>button:hover{background:linear-gradient(135deg,#818cf8,#6366f1)!important;transform:translateY(-2px)!important;box-shadow:0 8px 25px rgba(99,102,241,.45)!important;}
/* Inputs */
.stSelectbox>div>div,.stTextInput>div>div>input,.stNumberInput>div>div>input{background:#0e1525!important;border:1px solid rgba(99,102,241,.2)!important;color:#e2e8f0!important;border-radius:9px!important;font-size:13.5px!important;}
.stSelectbox>div>div:focus-within,.stTextInput>div>div>input:focus{border-color:rgba(99,102,241,.55)!important;box-shadow:0 0 0 3px rgba(99,102,241,.12)!important;}
.stTextInput label,.stSelectbox label,.stNumberInput label{color:#8b95a8!important;font-size:12px!important;font-weight:500!important;letter-spacing:.3px!important;}
/* Tabs */
.stTabs [data-baseweb="tab-list"]{background:rgba(9,14,26,.8)!important;border-radius:12px!important;padding:5px!important;border:1px solid rgba(255,255,255,.06)!important;}
.stTabs [data-baseweb="tab"]{color:#64748b!important;border-radius:9px!important;padding:9px 20px!important;font-weight:600!important;font-size:13.5px!important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#6366f1,#4f46e5)!important;color:white!important;}
/* Data tables */
[data-testid="stDataFrame"]{border:1px solid rgba(99,102,241,.12)!important;border-radius:12px!important;overflow:hidden!important;}
/* Section titles */
.sec-title{font-size:20px;font-weight:700;color:white;margin:0 0 2px 0;}
.sec-sub{font-size:13px;color:#64748b;margin:0 0 20px 0;}
.page-header{background:linear-gradient(135deg,rgba(99,102,241,.08),rgba(139,92,246,.04));border:1px solid rgba(99,102,241,.12);border-radius:14px;padding:20px 24px;margin-bottom:20px;}
/* Login */
.login-wrap{max-width:430px;margin:50px auto;}
.login-logo{font-size:42px;font-weight:900;background:linear-gradient(135deg,#6366f1,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;display:block;margin-bottom:4px;}
.login-tag{text-align:center;color:#64748b;font-size:13px;margin-bottom:28px;}
.sidebar-logo{font-size:18px;font-weight:800;background:linear-gradient(135deg,#6366f1,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;display:block;margin-bottom:16px;padding-bottom:14px;border-bottom:1px solid rgba(99,102,241,.15);}
.user-pill{background:rgba(99,102,241,.1);border:1px solid rgba(99,102,241,.2);border-radius:10px;padding:10px 12px;font-size:12.5px;color:#c4cdd8;margin:0 0 14px 0;display:flex;align-items:center;gap:8px;}
.stat-row{display:flex;gap:8px;margin:6px 0;}
.stat-chip{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);border-radius:8px;padding:8px 12px;flex:1;text-align:center;}
.stat-chip-val{font-size:18px;font-weight:700;color:white;}
.stat-chip-lbl{font-size:10px;color:#64748b;letter-spacing:.8px;text-transform:uppercase;}
div[data-testid="stForm"]{background:transparent!important;border:none!important;padding:0!important;}
.stProgress>div>div>div{background:linear-gradient(90deg,#6366f1,#06b6d4)!important;border-radius:100px!important;}
hr{border-color:rgba(255,255,255,.06)!important;}
</style>""", unsafe_allow_html=True)

# ─── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE_DIR, "Poverty_app.db")
CHART_DIR = os.path.join(BASE_DIR, "outputs", "charts", "eda")
UPLOAD_DIR= os.path.join(BASE_DIR, "uploads")
PDF_DIR   = os.path.join(BASE_DIR, "outputs", "pdfs")
for d in [UPLOAD_DIR, PDF_DIR, CHART_DIR, os.path.join(UPLOAD_DIR,"profiles")]:
    os.makedirs(d, exist_ok=True)

LABELS = ["Extreme Poverty","Moderate Poverty","Low Poverty"]
COLORS = ["#f87171","#fbbf24","#34d399"]
BADGE_CSS = ["badge-extreme","badge-moderate","badge-low"]
ICONS  = ["🔴","🟡","🟢"]
BADGE_ICONS=["💔","⚠️","✅"]

RECOMMENDATIONS = {
    0: {"label":"Extreme Poverty","focus":["Emergency Relief","Basic Needs","Social Protection"],
        "icon":"💔","color":"#f87171",
        "recs":["Apply immediately for government social safety net programs — cash transfers and food assistance.",
                "Enroll in community-based microfinance or cooperative savings groups to build financial resilience.",
                "Access free vocational training in high-demand local trades: tailoring, carpentry, agriculture.",
                "Connect with NGOs providing emergency housing support, clean water, or sanitation infrastructure.",
                "Prioritize primary healthcare registration to access subsidized medical services.",
                "Explore school feeding programs and fee waivers to maintain children's education continuity.",
                "Apply for utility subsidy programs (electricity, water) to reduce monthly living costs.",
                "Join agricultural cooperatives or input subsidy schemes if farming is the primary livelihood.",
                "Utilize community health workers for maternal/child health and nutrition counseling.",
                "Document and apply for all eligible poverty alleviation grants, disability support, or pensions."]},
    1: {"label":"Moderate Poverty","focus":["Skill Development","Asset Building","Income Diversification"],
        "icon":"⚠️","color":"#fbbf24",
        "recs":["Invest in certified vocational or digital literacy training for higher-paying formal employment.",
                "Explore micro-entrepreneurship grants or low-interest loans to grow a small business.",
                "Diversify household income by combining formal employment with gig work or seasonal agriculture.",
                "Enroll in government health insurance schemes to prevent financial shocks from medical costs.",
                "Upgrade housing incrementally using low-cost durable materials (interlocking bricks, metal roofing).",
                "Access adult education or high-school equivalency programs to boost long-term employability.",
                "Open a dedicated savings account with auto-transfers — target 3 to 6 months of expenses.",
                "Use mobile banking and digital wallets to reduce transaction fees and track household spending.",
                "Participate in local farming co-ops to increase agricultural yield and improve market access.",
                "Apply for subsidized housing loans or rent-to-own schemes to transition to permanent housing."]},
    2: {"label":"Low Poverty","focus":["Wealth Accumulation","Higher Education","Financial Planning"],
        "icon":"✅","color":"#34d399",
        "recs":["Build a comprehensive household budget with automated savings targeting 20% or more of income.",
                "Invest in mutual funds, treasury bills, or real estate to build long-term wealth.",
                "Enroll children in quality secondary and tertiary education and pursue scholarship opportunities.",
                "Upgrade digital infrastructure: reliable internet and computers for remote work or e-commerce.",
                "Purchase comprehensive health and life insurance to protect household assets and stability.",
                "Pursue professional certifications for household members to access senior or corporate roles.",
                "Diversify investments across multiple asset classes — stocks, bonds, property — to reduce risk.",
                "Implement energy-efficient upgrades such as solar panels and water recycling systems.",
                "Start or join investment clubs to pool capital for larger business ventures or property deals.",
                "Create a formal estate plan with a will, trust, and power of attorney for generational wealth."]}
}

# ─── PLOTLY LAYOUT DEFAULTS ───────────────────────────────────────────────────
PTHEME = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
              font=dict(color="#8b95a8", family="Inter"), 
              margin=dict(t=48, b=20, l=20, r=20),
              legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c4cdd8")),
              xaxis=dict(gridcolor="rgba(255,255,255,.05)", linecolor="rgba(255,255,255,.06)", tickfont=dict(color="#64748b")),
              yaxis=dict(gridcolor="rgba(255,255,255,.05)", linecolor="rgba(255,255,255,.06)", tickfont=dict(color="#64748b")))

# ─── DATABASE ──────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=20)
    conn.execute("PRAGMA journal_mode=WAL")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY,username TEXT UNIQUE,
        password_hash TEXT,email TEXT,role TEXT DEFAULT 'user',profile_pic_path TEXT,created_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions(id TEXT PRIMARY KEY,user_id TEXT,username TEXT,
        timestamp TEXT,class_name TEXT,confidence REAL,features_json TEXT,recommendations_json TEXT,dataset_name TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS admin_logs(id TEXT PRIMARY KEY,admin_id TEXT,action TEXT,details TEXT,timestamp TEXT)''')
    if not c.execute("SELECT 1 FROM users WHERE username='admin'").fetchone():
        c.execute("INSERT INTO users VALUES(?,?,?,?,?,?,?)",
            (str(uuid.uuid4()),"admin",hashlib.sha256(b"admin123").hexdigest(),"admin@Poverty.ug","admin",None,dt.now().isoformat()))
    conn.commit(); conn.close()

def db_q(sql,params=(),fetch='all'):
    conn=sqlite3.connect(DB_PATH,check_same_thread=False,timeout=20); conn.execute("PRAGMA journal_mode=WAL")
    c=conn.cursor(); c.execute(sql,params)
    r=c.fetchall() if fetch=='all' else c.fetchone(); conn.commit(); conn.close(); return r

def db_x(sql,params=()):
    conn=sqlite3.connect(DB_PATH,check_same_thread=False,timeout=20); conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(sql,params); conn.commit(); conn.close()

def hp(pw): return hashlib.sha256(pw.encode()).hexdigest()

def login_user(u,p):
    r=db_q("SELECT id,username,email,role,profile_pic_path FROM users WHERE username=? AND password_hash=?",(u,hp(p)),fetch='one')
    return {"id":r[0],"username":r[1],"email":r[2],"role":r[3],"profile_pic":r[4]} if r else None

def register_user(u,p,e,pic):
    if db_q("SELECT 1 FROM users WHERE username=?",(u,),fetch='one'): return False,"Username already taken."
    pp=None
    if pic:
        pp=os.path.join(UPLOAD_DIR,"profiles",f"{uuid.uuid4().hex}.{pic.name.split('.')[-1]}")
        open(pp,"wb").write(pic.read())
    db_x("INSERT INTO users VALUES(?,?,?,?,?,?,?)",(str(uuid.uuid4()),u,hp(p),e,"user",pp,dt.now().isoformat()))
    return True,"Account created! Please sign in."

def log_action(aid,action,details):
    db_x("INSERT INTO admin_logs VALUES(?,?,?,?,?)",(str(uuid.uuid4()),aid,action,details,dt.now().isoformat()))

# ─── ML PIPELINE ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    try:
        # Prefer app_model if available (trained on synthetic data matching app inputs)
        model_path = os.path.join(BASE_DIR, "app_model.pkl") if os.path.exists(os.path.join(BASE_DIR, "app_model.pkl")) else os.path.join(BASE_DIR, "champion_model.pkl")
        m=joblib.load(model_path)
        s=joblib.load(os.path.join(BASE_DIR,"data_scaler.pkl"))
        f=json.load(open(os.path.join(BASE_DIR,"feature_list.json")))
        return m,s,f
    except Exception as e:
        st.error(f"❌ ML pipeline failed to load: {e}"); return None,None,None

HPLINE=30612  # UGX annual poverty line

def build_features(age_s, edu_s, gender, marital, hh_size, income_s, region, urban):
    age_m={"18-25":21.5,"26-35":30.5,"36-45":40.5,"46-55":50.5,"56-65":60.5,"66+":70}
    edu_g={"No formal":0,"Primary":1,"Secondary":2,"Tertiary":3}
    edu_r={"No formal":1,"Primary":4,"Secondary":7,"Tertiary":11}
    inc_m={"<50k":25000,"50k-100k":75000,"100k-200k":150000,"200k-500k":350000,"500k-1M":750000,">1M":1250000}
    age=age_m.get(age_s,30.5); eg=edu_g.get(edu_s,0); er=edu_r.get(edu_s,1)
    inc_a=inc_m.get(income_s,75000)*12; ct=inc_a*0.85; nf=inc_a*0.35
    lp=ct/max(1,hh_size); cp=lp/12; ns=nf/max(1,ct); wr=cp/max(1,HPLINE)
    sx=1 if gender=="Female" else 0
    ma={"Married":1,"Consensual":2,"Divorced":3,"Widowed":4,"Single":5}.get(marital,1)
    ag=0 if age<30 else (1 if age<50 else 2)
    hg=0 if hh_size<=2 else (1 if hh_size<=5 else (2 if hh_size<=8 else 3))
    return [age,sx,eg,ma,hh_size,int(urban),region,cp,lp,ns,wr,ag,hg,er]

def _normalize_probabilities(proba):
    proba = np.asarray(proba, dtype=float)
    proba = np.clip(proba, 0.0, None)
    total = proba.sum()
    if total == 0:
        return np.ones_like(proba) / len(proba)
    return proba / total

def _softmax(logits, temperature=2.5):
    z = np.asarray(logits, dtype=float) / temperature
    z = z - z.max()
    e = np.exp(z)
    total = e.sum()
    if total == 0:
        return np.ones_like(e) / len(e)
    return e / total

def _is_coarse_distribution(proba):
    """Detect isotonic-calibrated outputs stuck at 0%, 50%, or 100%."""
    pct = np.round(np.asarray(proba, dtype=float) * 100, 1)
    return set(pct.tolist()).issubset({0.0, 50.0, 100.0})

def get_class_probabilities(model, X):
    """Return granular class probabilities that always sum to 1."""
    base = model.estimator if hasattr(model, "estimator") else model
    proba = None
    if hasattr(base, "predict_proba"):
        proba = _normalize_probabilities(base.predict_proba(X)[0])
    if proba is None or _is_coarse_distribution(proba):
        if hasattr(base, "decision_function"):
            proba = _softmax(base.decision_function(X)[0])
        elif hasattr(model, "predict_proba"):
            proba = _normalize_probabilities(model.predict_proba(X)[0])
    if proba is None:
        proba = np.ones(3) / 3
    return _normalize_probabilities(proba)

def distribute_class_probabilities(proba, pred_idx):
    """Top class keeps its share; the remainder is split across the other two."""
    proba = _normalize_probabilities(proba)
    pred_idx = int(pred_idx)
    winner_share = float(proba[pred_idx])
    others = [i for i in range(len(proba)) if i != pred_idx]
    remainder = max(0.0, 1.0 - winner_share)
    distributed = np.zeros(len(proba))
    distributed[pred_idx] = winner_share
    other_weights = proba[others]
    if remainder > 0:
        if other_weights.sum() == 0:
            distributed[others[0]] = remainder / 2
            distributed[others[1]] = remainder / 2
        else:
            other_weights = other_weights / other_weights.sum()
            for i, idx in enumerate(others):
                distributed[idx] = remainder * other_weights[i]
    return _normalize_probabilities(distributed)

def run_predict(feats, model, scaler, feature_cols):
    df = pd.DataFrame([feats], columns=feature_cols)
    X = scaler.transform(df)
    raw_proba = get_class_probabilities(model, X)
    pred = int(np.argmax(raw_proba))
    proba = distribute_class_probabilities(raw_proba, pred)
    return pred, proba.tolist()

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def encode_img(path):
    if path and os.path.exists(path):
        return f"data:image/png;base64,{base64.b64encode(open(path,'rb').read()).decode()}"
    return ""

def save_pred(user,cls,conf,feat_d,recs,src):
    db_x("INSERT INTO predictions (id, user_id, username, timestamp, class_name, confidence, features_json, recommendations_json, dataset_name) VALUES(?,?,?,?,?,?,?,?,?)",
         (str(uuid.uuid4()),user["id"],user["username"],dt.now().isoformat(),
          cls,conf,json.dumps(feat_d),json.dumps(recs),src))

def make_pdf(data):
    pdf=FPDF(); pdf.add_page(); pdf.set_margins(14,14,14); pdf.set_auto_page_break(True,12)
    # Header
    pdf.set_fill_color(99,102,241); pdf.rect(0,0,220,28,style='F')
    pdf.set_font("Helvetica","B",17); pdf.set_text_color(255,255,255)
    pdf.cell(0,12,"",ln=True); pdf.cell(0,8,"  Poverty - Prediction Report",ln=True)
    pdf.set_text_color(0,0,0); pdf.ln(6)
    pdf.set_font("Helvetica","",10)
    pdf.cell(0,5,f"Report ID : {uuid.uuid4().hex[:10].upper()}",ln=True)
    pdf.cell(0,5,f"Generated : {dt.now().strftime('%Y-%m-%d %H:%M:%S')}",ln=True)
    pdf.cell(0,5,f"Analyst   : {data['username']}",ln=True); pdf.ln(4)
    # Result box
    cls_colors={"Extreme Poverty":(239,68,68),"Moderate Poverty":(245,158,11),"Low Poverty":(16,185,129)}
    r,g,b=cls_colors.get(data.get("name", data.get("class_name", "Unknown")),(99,102,241))
    pdf.set_fill_color(r,g,b); pdf.set_text_color(255,255,255); pdf.set_font("Helvetica","B",13)
    pdf.cell(0,10,f"  Classification: {data.get('name', data.get('class_name', 'Unknown'))}   Confidence: {data.get('conf', data.get('confidence', 0.0)):.1%}",ln=True,fill=True)
    pdf.set_text_color(0,0,0); pdf.ln(4)
    # Recommendations
    pdf.set_font("Helvetica","B",11); pdf.cell(0,7,"AI Recommendations:",ln=True)
    pdf.set_font("Helvetica","",9)
    for i,ri in enumerate(data['recs'],1):
        clean_r = str(ri).replace('\t',' ').replace('\n',' ').replace('\xa0',' ').replace('—','-').replace('–','-').replace('’',"'").replace('“','"').replace('”','"').strip()
        pdf.set_x(14); pdf.multi_cell(0,5,f"  {i}. {clean_r}")
    pdf.ln(4)
    # Input profile
    pdf.set_font("Helvetica","B",11); pdf.cell(0,7,"Household Input Profile:",ln=True)
    pdf.set_font("Helvetica","",9)
    for k,v in data['inputs'].items():
        clean_v = str(v).replace('\t',' ').replace('\n',' ').replace('\xa0',' ').replace('—','-').replace('–','-').replace('’',"'").replace('“','"').replace('”','"').strip()
        pdf.set_x(14); pdf.multi_cell(0,5,f"  * {k}: {clean_v}")
    out=io.BytesIO(); pdf.output(out); out.seek(0); return out


# ─── LOGIN PAGE ─────────────────────────────────────────────────────────────────
def page_login():
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("""<style>
        div[data-testid="stTabs"] {
            background: linear-gradient(135deg,rgba(14,21,37,.95),rgba(6,12,23,.98));
            border: 1px solid rgba(99,102,241,.18);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.05);
            margin-bottom: 50px;
        }
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(9,14,26,.8) !important;
            border-radius: 12px !important;
            padding: 5px !important;
            margin-bottom: 20px !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            gap: 10px !important;
            border: 1px solid rgba(255,255,255,.06) !important;
        }
        .stTabs [data-baseweb="tab"] {
            flex-grow: 0 !important;
            flex-basis: auto !important;
            width: 120px !important;
            justify-content: center !important;
            border: none !important;
            border-bottom: none !important;
            border-bottom-color: transparent !important;
            outline: none !important;
            box-shadow: none !important;
            text-align: center !important;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            border: none !important;
            border-bottom: none !important;
        }
        .stTabs [data-baseweb="tab-border"] {
            display: none !important;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            display: none !important;
            height: 0px !important;
            background-color: transparent !important;
            border-bottom: none !important;
        }
        .login-logo { margin-top: 30px; }
        </style>""", unsafe_allow_html=True)
        st.markdown('<div class="login-wrap">', unsafe_allow_html=True)
        st.markdown('<div style="text-align:center"><span class="login-logo">Poverty</span></div>', unsafe_allow_html=True)
        st.markdown('<p class="login-tag">Uganda Household Poverty Intelligence System</p>', unsafe_allow_html=True)
        t1,t2 = st.tabs(["  Sign In  ","   Register  "])
        with t1:
            with st.form("lf"):
                u=st.text_input("Username",placeholder="Enter your username")
                p=st.text_input("Password",type="password",placeholder="Enter your password")
                if st.form_submit_button("Sign In",use_container_width=True):
                    usr=login_user(u,p)
                    if usr: st.session_state.update({"li":True,"usr":usr}); st.rerun()
                    else: st.error("Invalid credentials. Try again.")
        with t2:
            with st.form("rf"):
                u=st.text_input("Username",placeholder="Choose a username")
                p=st.text_input("Password",type="password",placeholder="Choose a strong password")
                e=st.text_input("Email",placeholder="your@email.com")
                pic=st.file_uploader("Profile Photo (optional)",type=["png","jpg","jpeg"])
                if st.form_submit_button("Create Account",use_container_width=True):
                    if u and p:
                        ok,msg=register_user(u,p,e,pic)
                        st.success(msg) if ok else st.error(msg)
                    else: st.error("Username and password are required.")
        st.markdown("</div>", unsafe_allow_html=True)

# ─── USER SIDEBAR ──────────────────────────────────────────────────────────────
def user_sidebar(usr):
    with st.sidebar:
        st.markdown('<span class="sidebar-logo">🌍 Poverty</span>', unsafe_allow_html=True)
        if usr.get("profile_pic") and os.path.exists(usr["profile_pic"]):
            st.image(usr["profile_pic"], width=72)
        st.markdown(f'<div class="user-pill">👤 <span><b>{usr["username"]}</b><br><span style="color:#6366f1;font-size:10px;text-transform:uppercase">Researcher</span></span></div>', unsafe_allow_html=True)
        total=db_q("SELECT COUNT(*) FROM predictions WHERE user_id=?",(usr["id"],),fetch='one')[0]
        st.markdown(f'<div class="stat-row"><div class="stat-chip"><div class="stat-chip-val">{total}</div><div class="stat-chip-lbl">Predictions</div></div></div>', unsafe_allow_html=True)
        st.markdown("---")
        menu=st.radio("nav",["🔮 Predict","📂 Bulk Upload"," Data & EDA"," History","👤 Profile"],label_visibility="collapsed")
        st.markdown("---")
        if st.button("🚪 Logout",use_container_width=True):
            st.session_state.clear(); st.rerun()
    return menu

# ─── SINGLE PREDICTION ─────────────────────────────────────────────────────────
def page_predict(usr):
    st.markdown('<div class="page-header"><p class="sec-title">🔮 Household Poverty Prediction</p><p class="sec-sub">Fill the household profile below to generate an AI-powered poverty classification.</p></div>', unsafe_allow_html=True)
    with st.form("pf", clear_on_submit=False):
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("**👤 Head of Household**")
            q_age  = st.selectbox("Age range",["18-25","26-35","36-45","46-55","56-65","66+"])
            q_gen  = st.selectbox("Gender",["Male","Female"])
            q_mar  = st.selectbox("Marital status",["Married","Single","Divorced","Widowed","Consensual"])
            q_edu  = st.selectbox("Education level",["No formal","Primary","Secondary","Tertiary"])
            q_emp  = st.selectbox("Employment status",["Employed","Self-employed","Unemployed","Retired","Other"])
        with c2:
            st.markdown("**🏠 Household Profile**")
            q_hhsz = st.number_input("Household size",1,30,4)
            q_chi  = st.number_input("Number of children",0,20,2)
            q_dep  = st.number_input("Dependents (non-earning)",0,20,1)
            q_inc  = st.selectbox("Monthly income (UGX)",["<50k","50k-100k","100k-200k","200k-500k","500k-1M",">1M"])
            q_src  = st.selectbox("Primary income source",["Farming","Business","Salary","Remittances","Other"])
        with c3:
            st.markdown("**🏗️ Infrastructure**")
            q_house= st.selectbox("House type",["Permanent","Semi-permanent","Temporary","Informal"])
            q_wall = st.selectbox("Wall material",["Brick/Stone","Iron Sheets","Mud/Wattle","Wood/Thatch","Other"])
            q_light= st.selectbox("Lighting source",["Electricity","Solar","Paraffin","Firewood","Battery","None"])
            q_water= st.selectbox("Water source",["Piped","Borehole","Well","Rainwater","River","Vendor"])
            q_toil = st.selectbox("Toilet type",["Flush","Improved Pit","Unimproved Pit","Shared","None"])
            q_urb  = st.selectbox("Settlement type",["Urban","Rural"])
            q_reg  = st.selectbox("Region",["Central (1)","Eastern (2)","Northern (3)","Western (4)"])
        st.markdown("---")
        cout, _ = st.columns(2)
        submitted = cout.form_submit_button(" Run AI Classification",use_container_width=True)

    if submitted:
        model,scaler,features=load_pipeline()
        if model is None: return
        rc={"Central (1)":1,"Eastern (2)":2,"Northern (3)":3,"Western (4)":4}.get(q_reg,3)
        uv=1 if q_urb=="Urban" else 0
        fv=build_features(q_age,q_edu,q_gen,q_mar,q_hhsz,q_inc,rc,uv)
        with st.spinner("Running inference…"):
            pred,proba=run_predict(fv,model,scaler,features)
        recs=RECOMMENDATIONS[pred]["recs"]
        inputs_d={"Age":q_age,"Gender":q_gen,"Marital":q_mar,"Education":q_edu,"Employment":q_emp,
                  "Household Size":q_hhsz,"Children":q_chi,"Dependents":q_dep,
                  "Monthly Income":q_inc,"Income Source":q_src,"House":q_house,
                  "Walls":q_wall,"Lighting":q_light,"Water":q_water,"Toilet":q_toil,
                  "Settlement":q_urb,"Region":q_reg}
        save_pred(usr,LABELS[pred],float(max(proba)),inputs_d,recs,"Single Form")
        st.session_state["lp"]={"cls":pred,"name":LABELS[pred],"conf":float(max(proba)),
                                 "proba":proba,"recs":recs,"inputs":inputs_d,"username":usr["username"]}

    if "lp" in st.session_state:
        p=st.session_state["lp"]; cls=p["cls"]; conf=p["conf"]; proba=p["proba"]
        ic=RECOMMENDATIONS[cls]["icon"]; col=RECOMMENDATIONS[cls]["color"]
        # Result banner
        st.markdown(f"""<div class="card" style="border-color:{col}33;border-left:4px solid {col};">
          <div style="display:flex;align-items:center;gap:14px;">
            <span style="font-size:40px">{ic}</span>
            <div>
              <p style="color:#64748b;font-size:12px;letter-spacing:1px;text-transform:uppercase;margin:0">CLASSIFICATION RESULT</p>
              <p style="font-size:26px;font-weight:800;color:white;margin:4px 0"><span style="color:{col}">{p['name']}</span></p>
              <p style="color:#8b95a8;font-size:13px;margin:0">Model confidence: <b style="color:{col}">{conf:.1%}</b> · Focus areas: {' · '.join(RECOMMENDATIONS[cls]['focus'])}</p>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Gauge row
        g1,g2,g3=st.columns(3)
        for col_w,(lbl,prob,clr) in zip([g1,g2,g3],zip(LABELS,proba,COLORS)):
            fig=go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(prob*100,1),
                delta={"reference":33.3,"valueformat":".1f","suffix":"% vs base"},
                number={"suffix":"%","font":{"size":32,"color":clr}},
                title={"text":lbl,"font":{"size":12,"color":"#8b95a8"}},
                gauge={"axis":{"range":[0,100],"tickcolor":"#334155","tickwidth":1,"dtick":25},
                       "bar":{"color":clr,"thickness":0.25},
                       "bgcolor":"rgba(255,255,255,0.04)","borderwidth":0,
                       "steps":[{"range":[0,33],"color":"rgba(255,255,255,.03)"},
                                 {"range":[33,66],"color":"rgba(255,255,255,.05)"},
                                 {"range":[66,100],"color":"rgba(255,255,255,.07)"}],
                       "threshold":{"line":{"color":clr,"width":2},"thickness":0.75,"value":prob*100}}))
            fig.update_layout(height=220,**PTHEME); col_w.plotly_chart(fig,use_container_width=True)

        # Probability radar + bar side by side
        r1,r2=st.columns(2)
        fig_radar=go.Figure(go.Scatterpolar(
            r=[p*100 for p in proba]+[proba[0]*100],
            theta=LABELS+[LABELS[0]], fill='toself',
            fillcolor=f"rgba{tuple(int(COLORS[cls].lstrip('#')[i:i+2],16) for i in (0,2,4))+(0.15,)}",
            line=dict(color=COLORS[cls],width=2)))
        fig_radar.update_layout(title="Probability Radar",polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True,range=[0,100],gridcolor="rgba(255,255,255,.06)",tickfont=dict(color="#64748b",size=9)),
            angularaxis=dict(gridcolor="rgba(255,255,255,.06)",tickfont=dict(color="#c4cdd8",size=11))),
            **PTHEME,height=280); r1.plotly_chart(fig_radar,use_container_width=True)

        fig_hbar=go.Figure(go.Bar(
            x=[v*100 for v in proba], y=LABELS, orientation='h',
            marker=dict(color=COLORS,
                        line=dict(color=COLORS,width=1),
                        opacity=0.85),
            text=[f"{v*100:.1f}%" for v in proba],textposition="outside",
            textfont=dict(color="#c4cdd8",size=12)))
        fig_hbar.update_layout(title="Class Probability Breakdown",xaxis_title="Probability (%)",height=280,**PTHEME)
        fig_hbar.update_xaxes(range=[0,110],gridcolor="rgba(255,255,255,.05)")
        r2.plotly_chart(fig_hbar,use_container_width=True)

        # Input profile summary heatmap-style
        with st.expander("📋 Input Profile Summary",expanded=False):
            hm_keys=list(p["inputs"].keys()); hm_vals=list(p["inputs"].values())
            cols2=st.columns(4)
            for i,(k,v) in enumerate(p["inputs"].items()):
                cols2[i%4].markdown(f'<div class="card-sm" style="margin:4px 0"><div style="color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:.8px">{k}</div><div style="color:white;font-weight:600;font-size:14px;margin-top:3px">{v}</div></div>', unsafe_allow_html=True)

        # Recommendations
        st.markdown(f'<div class="card"><p class="sec-title" style="font-size:17px">💡 AI-Tailored Recommendations — {p["name"]}</p><p class="sec-sub" style="margin-bottom:14px">Based on the classification, here are 10 targeted recommendations for intervention and improvement.</p>', unsafe_allow_html=True)
        for i,rec in enumerate(p["recs"],1):
            st.markdown(f'<div class="rec"><div class="rec-num">{i}</div><div class="rec-text">{rec}</div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Downloads
        st.markdown("#### 📥 Export Results")
        d1, d2, _, _ = st.columns(4)
        pdf_b=make_pdf(p)
        d1.download_button("📄 PDF Report",data=pdf_b,file_name=f"Poverty_{dt.now().strftime('%Y%m%d_%H%M%S')}.pdf",mime="application/pdf",use_container_width=True)
        df_dl=pd.DataFrame([{**p["inputs"],"Class":p["name"],"Confidence":f"{conf:.2%}"}])
        d2.download_button("📊 CSV Export",data=df_dl.to_csv(index=False).encode(),file_name=f"Poverty_{dt.now().strftime('%Y%m%d_%H%M%S')}.csv",mime="text/csv",use_container_width=True)


# ─── BULK UPLOAD ────────────────────────────────────────────────────────────────
def page_bulk(usr):
    st.markdown('<div class="page-header"><p class="sec-title">📂 Bulk CSV Prediction</p><p class="sec-sub">Upload a CSV of households for batch poverty classification with full analytics.</p></div>', unsafe_allow_html=True)
    tpl=pd.DataFrame([{"Age":"26-35","Gender":"Male","Marital":"Married","Education":"Primary","HouseholdSize":5,"MonthlyIncome":"<50k","Region":"Northern (3)","Urban":"Rural"}])
    st.download_button("📥 Download CSV Template",data=tpl.to_csv(index=False).encode(),file_name="Poverty_template.csv",mime="text/csv")
    up=st.file_uploader("Upload household CSV",type=["csv"])
    if not up: return
    df=pd.read_csv(up)
    df=df.drop(columns=["Class", "Confidence", "Extreme%", "Moderate%", "Low%"], errors='ignore')
    
    if not any(k in df.columns for k in ["Age", "Education", "Region", "Household Size", "HouseholdSize"]):
        st.error("❌ **Invalid CSV Format Detected:** The uploaded dataset does not contain expected header columns (e.g., 'Age', 'Education', 'MonthlyIncome'). The AI has halted to prevent returning 100% identical fallback confidences. Please align your file with the Download CSV Template.")
        return
        
    st.markdown(f"**✅ {len(df)} rows loaded** — Preview:")
    st.dataframe(df.head(5),use_container_width=True)
    if not st.button("⚡ Run Batch Inference",use_container_width=True): return
    model,scaler,features=load_pipeline()
    if model is None: return
    results=[]; bar=st.progress(0,"Classifying households…")
    for i,row in df.iterrows():
        bar.progress((i+1)/len(df),f"Processing {i+1} / {len(df)}")
        try:
            r_reg = row.get("Region", "Northern (3)")
            rc={"Central (1)":1,"Eastern (2)":2,"Northern (3)":3,"Western (4)":4}.get(str(r_reg),3)
            r_urb = row.get("Urban", row.get("Settlement", "Rural"))
            ur=1 if str(r_urb)=="Urban" else 0
            fv=build_features(str(row.get("Age","26-35")),str(row.get("Education","No formal")),
                              str(row.get("Gender","Male")),str(row.get("Marital","Married")),
                              int(row.get("HouseholdSize", row.get("Household Size", 4))),
                              str(row.get("MonthlyIncome", row.get("Monthly Income", "<50k"))),rc,ur)
            pred,proba=run_predict(fv,model,scaler,features)
            results.append({"Class":LABELS[pred],"Confidence":f"{max(proba)*100:.1f}%","Extreme%":f"{proba[0]*100:.1f}","Moderate%":f"{proba[1]*100:.1f}","Low%":f"{proba[2]*100:.1f}"})
        except: results.append({"Class":"Error","Confidence":"0%","Extreme%":"0","Moderate%":"0","Low%":"0"})
    bar.empty()
    df_out=pd.concat([df.reset_index(drop=True),pd.DataFrame(results)],axis=1)
    st.markdown("#### 📜 Batch Classification Output")
    st.dataframe(df_out, height=350, use_container_width=True)
    dist=df_out["Class"].value_counts()
    # KPI row
    k1,k2,k3,k4=st.columns(4)
    k1.markdown(f'<div class="kpi"><div class="kpi-icon">🏠</div><div class="kpi-val">{len(df_out)}</div><div class="kpi-lbl">Total Households</div></div>',unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi"><div class="kpi-icon">💔</div><div class="kpi-val" style="background:linear-gradient(135deg,#f87171,#ef4444);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{dist.get("Extreme Poverty",0)}</div><div class="kpi-lbl">Extreme Poverty</div></div>',unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi"><div class="kpi-icon">⚠️</div><div class="kpi-val" style="background:linear-gradient(135deg,#fbbf24,#f59e0b);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{dist.get("Moderate Poverty",0)}</div><div class="kpi-lbl">Moderate Poverty</div></div>',unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi"><div class="kpi-icon">✅</div><div class="kpi-val" style="background:linear-gradient(135deg,#34d399,#10b981);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{dist.get("Low Poverty",0)}</div><div class="kpi-lbl">Low Poverty</div></div>',unsafe_allow_html=True)
    # Charts
    c1,c2=st.columns(2)
    fig_pie=go.Figure(go.Pie(labels=dist.index.tolist(),values=dist.values.tolist(),
        hole=0.55,marker=dict(colors=[{"Extreme Poverty":"#f87171","Moderate Poverty":"#fbbf24","Low Poverty":"#34d399"}.get(l,"#6366f1") for l in dist.index],
        line=dict(color="#070d1a",width=2)),textfont=dict(color="white",size=12)))
    fig_pie.add_annotation(text=f"<b>{len(df_out)}</b><br>Total",x=0.5,y=0.5,showarrow=False,font=dict(color="white",size=14))
    fig_pie.update_layout(title="Batch Class Distribution",showlegend=True,height=300,**PTHEME)
    c1.plotly_chart(fig_pie,use_container_width=True)
    # Stacked region bar
    if "Region" in df_out.columns:
        grp=df_out.groupby(["Region","Class"]).size().reset_index(name="n")
        fig_stk=px.bar(grp,x="Region",y="n",color="Class",color_discrete_map={"Extreme Poverty":"#f87171","Moderate Poverty":"#fbbf24","Low Poverty":"#34d399"},
            title="Poverty Distribution by Region",barmode="stack",template="plotly_dark")
        fig_stk.update_layout(height=300,**PTHEME); c2.plotly_chart(fig_stk,use_container_width=True)
    st.dataframe(df_out,use_container_width=True)
    st.download_button("📥 Download Results",data=df_out.to_csv(index=False).encode(),file_name=f"batch_results_{dt.now().strftime('%Y%m%d_%H%M%S')}.csv",mime="text/csv",use_container_width=True)
    # save to DB
    for _,row in df_out.iterrows():
        try: db_x("INSERT INTO predictions (id, user_id, username, timestamp, class_name, confidence, features_json, recommendations_json, dataset_name) VALUES(?,?,?,?,?,?,?,?,?)",(str(uuid.uuid4()),usr["id"],usr["username"],dt.now().isoformat(),row["Class"],float(str(row["Confidence"]).replace("%",""))/100,"{}","[]",up.name))
        except: pass

# ─── EDA & ANALYTICS ───────────────────────────────────────────────────────────
def page_eda():
    st.markdown('<div class="page-header"><p class="sec-title">📊 Data Analytics & Model Insights</p><p class="sec-sub">Visualizations from Uganda UNPS 2019-20 training data and model evaluation reports.</p></div>', unsafe_allow_html=True)
    # Live Model Performance Cards
    pf_path=os.path.join(BASE_DIR,"model_performance.json")
    if os.path.exists(pf_path):
        pf=json.load(open(pf_path)); m=pf.get("metrics",{})
        k1,k2,k3,k4,k5=st.columns(5)
        for col,icon,lbl,val in [(k1,"🏆","Champion",pf.get("model","—")),(k2,"🎯","Accuracy",f"{m.get('accuracy',0):.2%}"),(k3,"📐","F1-Macro",f"{m.get('f1',0):.2%}"),(k4,"🔬","Precision",f"{m.get('precision',0):.2%}"),(k5,"🔄","Recall",f"{m.get('recall',0):.2%}")]:
            col.markdown(f'<div class="kpi"><div class="kpi-icon">{icon}</div><div class="kpi-val" style="font-size:18px">{val}</div><div class="kpi-lbl">{lbl}</div></div>',unsafe_allow_html=True)
        # All-models animated bar chart
        models=list(pf.get("all_models",{}).keys()); f1s=[pf["all_models"][m]["f1"] for m in models]
        accs=[pf["all_models"][m]["accuracy"] for m in models]
        fig_mc=go.Figure()
        fig_mc.add_trace(go.Bar(name="F1-Macro",x=models,y=f1s,marker_color="#6366f1",marker_line=dict(color="#818cf8",width=1),text=[f"{v:.3f}" for v in f1s],textposition="outside",textfont=dict(color="#c4cdd8")))
        fig_mc.add_trace(go.Bar(name="Accuracy",x=models,y=accs,marker_color="#06b6d4",marker_line=dict(color="#22d3ee",width=1),text=[f"{v:.3f}" for v in accs],textposition="outside",textfont=dict(color="#c4cdd8")))
        fig_mc.update_layout(title="All Models — Performance Comparison",barmode="group",height=340,**PTHEME)
        fig_mc.update_yaxes(range=[0.88,1.01])
        st.plotly_chart(fig_mc,use_container_width=True)
        # Class distribution treemap
        cd=pf.get("class_distribution",{})
        if cd:
            fig_tree=go.Figure(go.Treemap(
                labels=list(cd.keys()), parents=[""]*len(cd), values=list(cd.values()),
                marker=dict(colors=["#f87171","#fbbf24","#34d399"][:len(cd)]),
                textfont=dict(color="white",size=14),
                texttemplate="<b>%{label}</b><br>%{value} households<br>%{percentRoot:.1%}"))
            fig_tree.update_layout(title="Training Data — Class Distribution Treemap",height=280,**PTHEME)
            st.plotly_chart(fig_tree,use_container_width=True)
    st.markdown("---")
    # Feature importance
    fi_path=os.path.join(BASE_DIR,"shap_feature_impact.json")
    if os.path.exists(fi_path):
        fi=json.load(open(fi_path))
        fi_df=pd.DataFrame([{"Feature":k,"Importance":v["mean_abs_impact"]} for k,v in fi.items()]).sort_values("Importance",ascending=True)
        fig_fi=go.Figure(go.Bar(x=fi_df["Importance"],y=fi_df["Feature"],orientation="h",
            marker=dict(color=fi_df["Importance"],colorscale=[[0,"#1e2d45"],[0.5,"#6366f1"],[1,"#06b6d4"]],showscale=False,line=dict(width=0)),
            text=[f"{v:.4f}" for v in fi_df["Importance"]],textposition="outside",textfont=dict(color="#c4cdd8",size=11)))
        fig_fi.update_layout(title="Feature Importance (SHAP mean |impact|)",height=380,xaxis_title="Mean Absolute Impact",**PTHEME)
        st.plotly_chart(fig_fi,use_container_width=True)
    # Static EDA image gallery
    st.markdown("### 📸 Training Data Visualizations")
    charts=[("01_class_distribution.png","Class Distribution"),("02_welfare_distribution.png","Welfare Distribution"),
            ("03_household_size.png","Household Size"),("04_education_vs_poverty.png","Education vs Poverty"),
            ("05_geographic_poverty.png","Geographic Distribution"),("06_age_distribution.png","Age Distribution"),
            ("07_correlation_heatmap.png","Correlation Heatmap"),("08_missing_data.png","Missing Data Map"),
            ("09_welfare_ratio_boxplot.png","Welfare Ratio Boxplot"),("10_consumption_pc.png","Consumption per Capita"),
            ("11_confusion_matrix.png","Confusion Matrix"),("12_model_comparison.png","Model Comparison"),
            ("13_feature_importance.png","Feature Importance")]
    for i in range(0,len(charts),2):
        cols=st.columns(2)
        for j,ci in enumerate(charts[i:i+2]):
            fname,title=ci; path=os.path.join(CHART_DIR,fname)
            cols[j].markdown(f"**{title}**")
            if os.path.exists(path): cols[j].image(path,use_container_width=True)
            else: cols[j].info(f"Chart not found: {fname}")


# ─── HISTORY ────────────────────────────────────────────────────────────────────
def page_history(usr):
    st.markdown('<div class="page-header"><p class="sec-title">📜 Prediction History</p><p class="sec-sub">Your complete activity log with class trends and analytics.</p></div>', unsafe_allow_html=True)
    rows=db_q("SELECT class_name,confidence,timestamp,dataset_name FROM predictions WHERE user_id=? ORDER BY timestamp DESC LIMIT 200",(usr["id"],))
    if not rows: st.info("No predictions yet. Run your first prediction to see history here."); return
    df=pd.DataFrame(rows,columns=["Class","Confidence","Timestamp","Source"])
    df["Date"]=df["Timestamp"].apply(lambda x:str(x)[:10])
    df["ConfPct"]=(df["Confidence"]*100).round(1)
    # KPIs
    dist=df["Class"].value_counts()
    k1,k2,k3,k4=st.columns(4)
    k1.markdown(f'<div class="kpi"><div class="kpi-icon">🏠</div><div class="kpi-val">{len(df)}</div><div class="kpi-lbl">Total Predictions</div></div>',unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi"><div class="kpi-icon">💔</div><div class="kpi-val" style="background:linear-gradient(135deg,#f87171,#ef4444);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{dist.get("Extreme Poverty",0)}</div><div class="kpi-lbl">Extreme Poverty</div></div>',unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi"><div class="kpi-icon">⚠️</div><div class="kpi-val" style="background:linear-gradient(135deg,#fbbf24,#f59e0b);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{dist.get("Moderate Poverty",0)}</div><div class="kpi-lbl">Moderate Poverty</div></div>',unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi"><div class="kpi-icon">✅</div><div class="kpi-val" style="background:linear-gradient(135deg,#34d399,#10b981);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{dist.get("Low Poverty",0)}</div><div class="kpi-lbl">Low Poverty</div></div>',unsafe_allow_html=True)
    # Charts
    c1,c2=st.columns(2)
    fig_do=go.Figure(go.Pie(labels=dist.index.tolist(),values=dist.values.tolist(),hole=0.6,
        marker=dict(colors=[{"Extreme Poverty":"#f87171","Moderate Poverty":"#fbbf24","Low Poverty":"#34d399"}.get(l,"#6366f1") for l in dist.index],line=dict(color="#070d1a",width=2)),textfont=dict(color="white")))
    top_cls=dist.idxmax(); fig_do.add_annotation(text=f"<b>{top_cls.split()[0]}</b><br>dominant",x=0.5,y=0.5,showarrow=False,font=dict(color="white",size=11))
    fig_do.update_layout(title="My Class Distribution",height=300,**PTHEME); c1.plotly_chart(fig_do,use_container_width=True)
    daily=df.groupby("Date").size().reset_index(name="Count")
    fig_line=go.Figure(go.Scatter(x=daily["Date"],y=daily["Count"],mode="lines+markers",
        fill="tozeroy",fillcolor="rgba(99,102,241,0.1)",line=dict(color="#6366f1",width=2),
        marker=dict(color="#818cf8",size=6)))
    fig_line.update_layout(title="Activity Over Time",height=300,xaxis_title="Date",yaxis_title="Predictions",**PTHEME)
    c2.plotly_chart(fig_line,use_container_width=True)
    # Confidence scatter
    fig_sc=go.Figure(go.Scatter(x=df["Date"],y=df["ConfPct"],mode="markers",
        marker=dict(color=[{"Extreme Poverty":"#f87171","Moderate Poverty":"#fbbf24","Low Poverty":"#34d399"}.get(c,"#6366f1") for c in df["Class"]],
                    size=9,opacity=0.8,line=dict(color="rgba(0,0,0,.3)",width=1)),
        hovertemplate="<b>%{y:.1f}%</b> confidence<br>Date: %{x}"))
    fig_sc.update_layout(title="Confidence Scores Over Time (coloured by class)",height=260,yaxis_title="Confidence %",xaxis_title="Date",**PTHEME)
    st.plotly_chart(fig_sc,use_container_width=True)
    # Table with filters
    st.markdown("---")
    fc1,fc2=st.columns([2,2])
    cls_filter=fc1.multiselect("Filter by class",LABELS,default=LABELS)
    src_filter=fc2.text_input("Filter by source",placeholder="e.g. Single Form")
    dff=df[df["Class"].isin(cls_filter)]
    if src_filter: dff=dff[dff["Source"].str.contains(src_filter,case=False,na=False)]
    st.dataframe(dff[["Date","Class","ConfPct","Source"]].rename(columns={"ConfPct":"Confidence %"}),use_container_width=True)
    st.download_button("📥 Export History",dff.to_csv(index=False).encode(),file_name="my_history.csv",mime="text/csv")

# ─── PROFILE ────────────────────────────────────────────────────────────────────
def page_profile(usr):
    st.markdown('<div class="page-header"><p class="sec-title">👤 User Profile</p><p class="sec-sub">Your account overview and usage statistics.</p></div>', unsafe_allow_html=True)
    c1,c2=st.columns([1,2])
    with c1:
        if usr.get("profile_pic") and os.path.exists(usr["profile_pic"]):
            st.image(usr["profile_pic"],width=130)
        else:
            st.markdown(f'<div style="width:90px;height:90px;border-radius:50%;background:linear-gradient(135deg,#6366f1,#06b6d4);display:flex;align-items:center;justify-content:center;font-size:36px;color:white;font-weight:700">{usr["username"][0].upper()}</div>',unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="card"><b style="font-size:20px;color:white">{usr["username"]}</b><br><span style="color:#64748b">{usr.get("email","—")}</span><br><br>', unsafe_allow_html=True)
        total=db_q("SELECT COUNT(*) FROM predictions WHERE user_id=?",(usr["id"],),fetch='one')[0]
        cls_rows=db_q("SELECT class_name,COUNT(*) FROM predictions WHERE user_id=? GROUP BY class_name",(usr["id"],))
        k1,k2=st.columns(2)
        k1.markdown(f'<div class="kpi"><div class="kpi-val">{total}</div><div class="kpi-lbl">Total Predictions</div></div>',unsafe_allow_html=True)
        top=db_q("SELECT class_name FROM predictions WHERE user_id=? GROUP BY class_name ORDER BY COUNT(*) DESC LIMIT 1",(usr["id"],),fetch='one')
        k2.markdown(f'<div class="kpi"><div class="kpi-val" style="font-size:14px">{top[0][:8] if top else "—"}</div><div class="kpi-lbl">Most Predicted</div></div>',unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)
        if cls_rows:
            fig=go.Figure(go.Bar(x=[r[0] for r in cls_rows],y=[r[1] for r in cls_rows],
                marker_color=["#f87171","#fbbf24","#34d399"][:len(cls_rows)],
                text=[r[1] for r in cls_rows],textposition="outside",textfont=dict(color="#c4cdd8")))
            fig.update_layout(title="My Predictions by Class",height=240,yaxis_title="Count",**PTHEME)
            st.plotly_chart(fig,use_container_width=True)
            
    cout, _ = st.columns(2)
    if cout.button("🚪 Sign Out",use_container_width=True):
        st.session_state.clear(); st.rerun()

# ─── ADMIN DASHBOARD ───────────────────────────────────────────────────────────
def admin_sidebar(usr):
    with st.sidebar:
        st.markdown(f'''
        <div style="background: linear-gradient(180deg, rgba(14,21,37,0.95) 0%, rgba(10,15,28,0.98) 100%); 
                    border: 1px solid rgba(99,102,241,0.2); 
                    border-radius: 16px; 
                    padding: 24px 16px; 
                    text-align: center; 
                    margin-bottom: 24px;
                    box-shadow: 0 10px 25px rgba(0,0,0,0.4);">
            <div style="width: 58px; height: 58px; 
                        background: linear-gradient(135deg, #6366f1, #06b6d4); 
                        border-radius: 50%; 
                        margin: 0 auto 14px auto; 
                        display: flex; align-items: center; justify-content: center; 
                        font-size: 24px; color: white; font-weight: 800;
                        box-shadow: 0 8px 16px rgba(99,102,241,0.3);
                        border: 2px solid rgba(255,255,255,0.1);">
                {usr["username"][0].upper()}
            </div>
            <h3 style="margin: 0; color: white; font-size: 16px; font-weight: 700; letter-spacing: 0.5px;">{usr["username"]}</h3>
            <div style="display: inline-block; margin-top: 8px; background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.3); padding: 4px 12px; border-radius: 20px;">
                <span style="color: #818cf8; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; font-weight: 700;">System Admin</span>
            </div>
        </div>
        <div style="font-size: 11px; font-weight: 700; color: #64748b; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 14px; margin-left: 8px;">Command Center</div>
        ''', unsafe_allow_html=True)
        
        options = [
            "❖ System Overview",
            "⌗ User Management",
            "∿ Telemetry & Monitor",
            "⎔ Model Registry",
            "▤ Audit Trail"
        ]
        menu = st.radio("nav", options, label_visibility="collapsed")
        
        st.markdown("---")
        if st.button("⏏ Sign Out", use_container_width=True): 
            st.session_state.clear()
            st.rerun()
    return menu

def admin_overview():
    st.markdown('<div class="page-header"><p class="sec-title"> System Overview</p><p class="sec-sub">Real-time system statistics and activity trends.</p></div>',unsafe_allow_html=True)
    tu=db_q("SELECT COUNT(*) FROM users WHERE role='user'",fetch='one')[0]
    tp=db_q("SELECT COUNT(*) FROM predictions",fetch='one')[0]
    ex=db_q("SELECT COUNT(*) FROM predictions WHERE class_name='Extreme Poverty'",fetch='one')[0]
    mo=db_q("SELECT COUNT(*) FROM predictions WHERE class_name='Moderate Poverty'",fetch='one')[0]
    lo=db_q("SELECT COUNT(*) FROM predictions WHERE class_name='Low Poverty'",fetch='one')[0]
    k1,k2,k3,k4,k5=st.columns(5)
    for col,icon,lbl,val,grd in [(k1,"👥","Active Users",tu,"#6366f1,#8b5cf6"),(k2,"🏠","All Predictions",tp,"#6366f1,#06b6d4"),(k3,"💔","Extreme Poverty",ex,"#f87171,#ef4444"),(k4,"⚠️","Moderate Poverty",mo,"#fbbf24,#f59e0b"),(k5,"✅","Low Poverty",lo,"#34d399,#10b981")]:
        col.markdown(f'<div class="kpi"><div class="kpi-icon">{icon}</div><div class="kpi-val" style="background:linear-gradient(135deg,{grd});-webkit-background-clip:text;-webkit-text-fill-color:transparent">{val}</div><div class="kpi-lbl">{lbl}</div></div>',unsafe_allow_html=True)
    if tp>0:
        # Donut + area
        c1,c2=st.columns(2)
        fig_d=go.Figure(go.Pie(labels=LABELS,values=[ex,mo,lo],hole=0.6,
            marker=dict(colors=COLORS,line=dict(color="#070d1a",width=2)),textfont=dict(color="white")))
        fig_d.add_annotation(text=f"<b>{tp}</b><br>Total",x=0.5,y=0.5,showarrow=False,font=dict(color="white",size=14))
        fig_d.update_layout(title="All-Time Class Distribution",height=320,**PTHEME)
        c1.plotly_chart(fig_d,use_container_width=True)
        daily=db_q("SELECT substr(timestamp,1,10) d,COUNT(*) FROM predictions GROUP BY d ORDER BY d")
        if daily:
            ddf=pd.DataFrame(daily,columns=["Date","Count"])
            fig_a=go.Figure(go.Scatter(x=ddf["Date"],y=ddf["Count"],fill="tozeroy",mode="lines+markers",
                fillcolor="rgba(99,102,241,.12)",line=dict(color="#6366f1",width=2.5),marker=dict(color="#818cf8",size=7)))
            fig_a.update_layout(title="Daily Prediction Activity",height=320,xaxis_title="Date",yaxis_title="Predictions",**PTHEME)
            c2.plotly_chart(fig_a,use_container_width=True)
        # user activity bar
        user_data=db_q("SELECT username,COUNT(*) FROM predictions GROUP BY username ORDER BY COUNT(*) DESC LIMIT 8")
        if user_data:
            udf=pd.DataFrame(user_data,columns=["User","Predictions"])
            fig_u=go.Figure(go.Bar(x=udf["User"],y=udf["Predictions"],
                marker=dict(color=udf["Predictions"],colorscale=[[0,"#1e2d45"],[1,"#6366f1"]],showscale=False,line=dict(color="#818cf8",width=1)),
                text=udf["Predictions"],textposition="outside",textfont=dict(color="#c4cdd8")))
            fig_u.update_layout(title="Top Active Researchers",height=300,**PTHEME)
            st.plotly_chart(fig_u,use_container_width=True)

def admin_users(adm):
    st.markdown('<div class="page-header"><p class="sec-title">👥 User Management</p><p class="sec-sub">Manage researcher accounts and access control.</p></div>',unsafe_allow_html=True)
    rows=db_q("SELECT id,username,email,role,created_at FROM users ORDER BY created_at DESC")
    dfu=pd.DataFrame(rows,columns=["ID","Username","Email","Role","Created"])
    dfu["Created"]=dfu["Created"].apply(lambda x:str(x)[:10])
    search=st.text_input("🔍 Search by username or email",placeholder="Start typing…")
    df_show=dfu[dfu["Username"].str.contains(search,case=False,na=False)|dfu["Email"].str.contains(search,case=False,na=False)] if search else dfu
    st.dataframe(df_show[["Username","Email","Role","Created"]],use_container_width=True)
    # Predictions per user
    up=db_q("SELECT username,COUNT(*) FROM predictions GROUP BY username ORDER BY COUNT(*) DESC")
    if up:
        udf=pd.DataFrame(up,columns=["Username","Count"])
        fig=go.Figure(go.Bar(x=udf["Username"],y=udf["Count"],marker_color="#6366f1",text=udf["Count"],textposition="outside",textfont=dict(color="#c4cdd8")))
        fig.update_layout(title="Predictions per User",height=260,**PTHEME); st.plotly_chart(fig,use_container_width=True)
    st.markdown("---")
    non_admin=[r for r in rows if r[3]!="admin"]
    if non_admin:
        choices=[f"{r[1]} — {r[2]}" for r in non_admin]
        sel=st.selectbox("Select user to delete",choices)
        if st.button("🗑️ Delete User",type="primary"):
            uname=sel.split(" — ")[0]
            db_x("DELETE FROM users WHERE username=?",(uname,)); db_x("DELETE FROM predictions WHERE username=?",(uname,))
            log_action(adm["id"],"DELETE_USER",f"Deleted: {uname}"); st.success(f"User {uname} deleted."); st.rerun()

def admin_monitor():
    st.markdown('<div class="page-header"><p class="sec-title">📈 Predictions Monitor</p><p class="sec-sub">System-wide prediction analytics and confidence tracking.</p></div>',unsafe_allow_html=True)
    rows=db_q("SELECT username,class_name,confidence,timestamp,dataset_name FROM predictions ORDER BY timestamp DESC LIMIT 500")
    if not rows: st.info("No predictions yet."); return
    df=pd.DataFrame(rows,columns=["User","Class","Confidence","Timestamp","Source"])
    df["Date"]=df["Timestamp"].apply(lambda x:str(x)[:10]); df["ConfPct"]=df["Confidence"]*100
    c1,c2,c3=st.columns(3)
    dist=df["Class"].value_counts()
    fig1=go.Figure(go.Pie(labels=dist.index.tolist(),values=dist.values.tolist(),hole=0.55,
        marker=dict(colors=[{"Extreme Poverty":"#f87171","Moderate Poverty":"#fbbf24","Low Poverty":"#34d399"}.get(l,"#6366f1") for l in dist.index],line=dict(color="#070d1a",width=2)),textfont=dict(color="white")))
    fig1.update_layout(title="Class Distribution",height=290,showlegend=False,**PTHEME); c1.plotly_chart(fig1,use_container_width=True)
    fig2=go.Figure(go.Histogram(x=df["ConfPct"],nbinsx=20,marker_color="#6366f1",marker_line=dict(color="#818cf8",width=1),opacity=0.85))
    fig2.update_layout(title="Confidence Distribution",height=290,xaxis_title="Confidence (%)",**PTHEME); c2.plotly_chart(fig2,use_container_width=True)
    daily_cls=df.groupby(["Date","Class"]).size().reset_index(name="n")
    fig3=px.area(daily_cls,x="Date",y="n",color="Class",color_discrete_map={"Extreme Poverty":"#f87171","Moderate Poverty":"#fbbf24","Low Poverty":"#34d399"},template="plotly_dark",title="Daily by Class")
    fig3.update_layout(height=290,**PTHEME); c3.plotly_chart(fig3,use_container_width=True)
    # Waterfall confidence over time
    fig_w=go.Figure(go.Scatter(x=df.sort_values("Timestamp")["Timestamp"],y=df.sort_values("Timestamp")["ConfPct"],
        mode="lines",fill="tozeroy",fillcolor="rgba(6,182,212,.08)",line=dict(color="#06b6d4",width=2)))
    fig_w.update_layout(title="Prediction Confidence Over Time",height=250,yaxis_title="Confidence (%)",**PTHEME)
    st.plotly_chart(fig_w,use_container_width=True)
    st.dataframe(df[["Date","User","Class","ConfPct","Source"]].rename(columns={"ConfPct":"Confidence %"}),use_container_width=True)

def admin_ml():
    st.markdown('<div class="page-header"><p class="sec-title">🧠 ML Model Registry</p><p class="sec-sub">Champion model metrics, performance benchmarks, and artifact verification.</p></div>',unsafe_allow_html=True)
    pf=json.load(open(os.path.join(BASE_DIR,"model_performance.json"))) if os.path.exists(os.path.join(BASE_DIR,"model_performance.json")) else {}
    if pf:
        m=pf.get("metrics",{}); champ=pf.get("model","—")
        k1,k2,k3,k4=st.columns(4)
        for col,icon,lbl,val in [(k1,"🎯","Accuracy",f"{m.get('accuracy',0):.2%}"),(k2,"📐","F1-Macro",f"{m.get('f1',0):.2%}"),(k3,"🔬","Precision",f"{m.get('precision',0):.2%}"),(k4,"🔄","Recall",f"{m.get('recall',0):.2%}")]:
            col.markdown(f'<div class="kpi"><div class="kpi-icon">{icon}</div><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div></div>',unsafe_allow_html=True)
        st.success(f"✅ Champion: **{champ}** | Trained: {str(pf.get('timestamp',''))[:10]} | Samples: {pf.get('total_samples',0):,}")
        # Spider chart of all models
        models=list(pf.get("all_models",{}).items())
        if models:
            cats=["Accuracy","F1","Precision","Recall"]; fig_sp=go.Figure()
            for nm,mv in models:
                vals=[mv["accuracy"],mv["f1"],mv["precision"],mv["recall"]]
                fig_sp.add_trace(go.Scatterpolar(r=vals+[vals[0]],theta=cats+[cats[0]],fill='toself',name=nm,opacity=0.7))
            fig_sp.update_layout(title="Model Comparison — Radar",polar=dict(bgcolor="rgba(0,0,0,0)",radialaxis=dict(range=[0.85,1],gridcolor="rgba(255,255,255,.06)"),angularaxis=dict(gridcolor="rgba(255,255,255,.06)")),height=380,**PTHEME)
            st.plotly_chart(fig_sp,use_container_width=True)
            rows=[{"Model":n,"Accuracy":f"{v['accuracy']:.4f}","F1":f"{v['f1']:.4f}","Precision":f"{v['precision']:.4f}","Recall":f"{v['recall']:.4f}","Champion":"✅" if n==champ else ""} for n,v in pf.get("all_models",{}).items()]
            st.dataframe(pd.DataFrame(rows),use_container_width=True)
    st.markdown("---")
    # SHAP importance
    fi_path=os.path.join(BASE_DIR,"shap_feature_impact.json")
    if os.path.exists(fi_path):
        fi=json.load(open(fi_path)); fi_df=pd.DataFrame([{"Feature":k,"Impact":v["mean_abs_impact"]} for k,v in fi.items()]).sort_values("Impact",ascending=True)
        fig_fi=go.Figure(go.Bar(x=fi_df["Impact"],y=fi_df["Feature"],orientation="h",
            marker=dict(color=fi_df["Impact"],colorscale=[[0,"#1e2d45"],[0.5,"#6366f1"],[1,"#06b6d4"]],showscale=True),
            text=[f"{v:.4f}" for v in fi_df["Impact"]],textposition="outside",textfont=dict(color="#c4cdd8")))
        fig_fi.update_layout(title="SHAP Feature Impact",height=400,**PTHEME); st.plotly_chart(fig_fi,use_container_width=True)
    # Artifact health
    st.markdown("#### Artifact Health Check")
    for art in ["champion_model.pkl","data_scaler.pkl","preprocessing_imputer.pkl","feature_list.json","model_performance.json","inference_config.json","training_sample.csv"]:
        path=os.path.join(BASE_DIR,art); sz=os.path.getsize(path) if os.path.exists(path) else 0
        icon="✅" if os.path.exists(path) else "❌"; sz_str=f"{sz/1024:.1f} KB" if sz>0 else "missing"
        st.markdown(f"**{icon} {art}** — {sz_str}")

def admin_logs():
    st.markdown('<div class="page-header"><p class="sec-title">📋 Audit Trail</p><p class="sec-sub">All administrator actions logged with timestamps.</p></div>',unsafe_allow_html=True)
    rows=db_q("SELECT admin_id,action,details,timestamp FROM admin_logs ORDER BY timestamp DESC LIMIT 300")
    if not rows: st.info("No admin actions logged yet."); return
    df=pd.DataFrame(rows,columns=["Admin","Action","Details","Timestamp"])
    df["Timestamp"]=df["Timestamp"].apply(lambda x:str(x)[:19])
    action_dist=df["Action"].value_counts()
    fig=go.Figure(go.Bar(x=action_dist.index.tolist(),y=action_dist.values.tolist(),marker_color="#6366f1",text=action_dist.values.tolist(),textposition="outside"))
    fig.update_layout(title="Admin Actions Distribution",height=250,**PTHEME); st.plotly_chart(fig,use_container_width=True)
    st.dataframe(df,use_container_width=True)

# ─── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    init_db()
    if "li" not in st.session_state: st.session_state.li=False
    if "usr" not in st.session_state: st.session_state.usr=None
    if not st.session_state.li: page_login(); return
    usr=st.session_state.usr
    if usr["role"]=="admin":
        menu=admin_sidebar(usr)
        if menu=="❖ System Overview": admin_overview()
        elif menu=="⌗ User Management": admin_users(usr)
        elif menu=="∿ Telemetry & Monitor": admin_monitor()
        elif menu=="⎔ Model Registry": admin_ml()
        elif menu=="▤ Audit Trail": admin_logs()
    else:
        menu=user_sidebar(usr)
        if menu=="🔮 Predict": page_predict(usr)
        elif menu=="📂 Bulk Upload": page_bulk(usr)
        elif menu==" Data & EDA": page_eda()
        elif menu==" History": page_history(usr)
        elif menu=="👤 Profile": page_profile(usr)

if __name__=="__main__":
    main()
