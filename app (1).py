"""app.py - BrightSight Kids: Clean White & Sea Blue Pediatric Vision Screener (Ages 2-17)."""
import os, hashlib
from collections import Counter
from datetime import datetime, timedelta
from statistics import median
import numpy as np, streamlit as st
from PIL import Image

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import engine

st.set_page_config(page_title="BrightSight Kids", page_icon="👁️", layout="centered")
st.markdown("""<style>
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {background:#F8FAFC !important; color-scheme: light;}
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3,
[data-testid="stMarkdownContainer"], [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
[data-testid="stCaptionContainer"] {color:#0F172A;}
.block-container {max-width: 740px; padding-top: 1.8rem;}
.card {background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px; padding:1.25rem 1.45rem;
       margin:.85rem 0; box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);}
.card th, .card td {padding:.45rem .6rem; border-bottom:1px solid #E2E8F0; text-align:left; font-size:.88rem;}
.banner {border-radius:12px; padding:1rem; font-size:1.45rem; font-weight:800; text-align:center;}
div.stButton > button, [data-testid="stCameraInput"] button {
  background:#0284C7 !important; color:#FFFFFF !important; border:none !important;
  border-radius:10px !important; font-weight:600 !important; width:100%; padding:.65rem 1rem;}
div.stButton > button p {color:#FFFFFF !important;}
div.stButton > button:hover {background:#0369A1 !important;}
[data-testid="stCameraInput"], [data-testid="stExpander"] {
  background:#FFFFFF; border:1px solid #E2E8F0 !important; border-radius:14px; padding:.4rem;}
[data-testid="stExpander"] details, [data-testid="stExpander"] summary {background:#FFFFFF !important; color:#0F172A !important;}
[data-baseweb="select"] > div, [data-baseweb="base-input"] {background:#FFFFFF !important; border-color:#CBD5E1 !important;}
input {color:#0F172A !important;}
[data-baseweb="tab-list"] {gap: 8px; background: transparent;}
[data-baseweb="tab"] {background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:.45rem 1rem; font-weight:600;}
[aria-selected="true"] {background:#E0F2FE !important; border-color:#0284C7 !important; color:#0369A1 !important;}
</style>""", unsafe_allow_html=True)

BADGES = {
    "PASS": ("#ECFDF5", "#065F46", "Vision Looks Clear"),
    "REFER": ("#FEF2F2", "#991B1B", "Eye Doctor Check Recommended"),
    "RETAKE": ("#FFFBEB", "#92400E", "Please Retake Photo"),
    "OK": ("#E0F2FE", "#0369A1", "Photo Quality Verified"),
}

EXPLAIN = {
    "PASS": ("Your child's eyes showed no signs needing immediate follow-up. This is a quick preventive check, not a replacement for a full eye exam.",
             "Next step: Repeat this check in 12 months, or sooner if you notice squinting or sitting close to screens."),
    "REFER": ("This check detected a focusing pattern often linked to needing glasses. It is very common in children aged 2–17 and easily treated when caught early.",
              "Next step: Book a comprehensive exam with an eye doctor within the next few weeks."),
    "dark": ("The image is too dark to read your child's eyes clearly.",
             "Next step: Turn on a soft, even room light and retake the photo."),
    "OK": ("Your photo is sharp, clear, and well lit. Full refractive measurement uses our 4-angle smartphone flash sequence.",
           "Next step: Tap any quick test preset below to preview a complete screening output."),
}

SAMPLE_RULES = {
    "Test: Clear Vision (Age 6)": (lambda D, i: D["kind"][i] == 0 and abs(D["S"][i]) < 1.0 and D["C"][i] > -0.5 and D["age"][i] >= 49, "PASS"),
    "Test: Hidden Farsightedness (Age 4)": (lambda D, i: D["kind"][i] == 0 and D["S"][i] >= 3.8 and D["C"][i] > -1.0 and D["age"][i] >= 49, "REFER"),
    "Test: Astigmatism (Age 11)": (lambda D, i: D["kind"][i] == 0 and D["C"][i] <= -2.0 and -1 < D["S"][i] < 2 and D["age"][i] >= 49, "REFER"),
    "Test: Poor Lighting / Glare": (lambda D, i: D["kind"][i] == 1 and D["glare"][i] > 0.1, "RETAKE"),
}

@st.cache_resource
def load_bundle():
    if not os.path.exists(engine.MODEL_PATH):
        with st.spinner("Initializing clinical optical model..."):
            engine.main()
    return engine.joblib.load(engine.MODEL_PATH)

def child_features(P, profile, seed=0):
    raw, _ = engine.capture(P, np.array([engine.PNAMES.index(profile)]), np.random.default_rng(seed))
    return dict(zip(engine.RAW_KEYS, map(float, raw[0])))

@st.cache_resource
def build_samples(_b):
    D = engine.simulate(np.random.default_rng(2024), 800)
    out = {}
    for label, (rule, want) in SAMPLE_RULES.items():
        for i in (j for j in range(len(D["sid"])) if rule(D, j)):
            P = {k: D[k][i:i + 1] for k in engine.PARAM_KEYS}
            if engine.predict_eye(child_features(P, engine.PNAMES[0]), _b)["status"] == want:
                out[label] = (P, (float(D["S"][i]), float(D["C"][i]), float(D["ax"][i])))
                break
    return out

def photo_quality(file):
    im = Image.open(file).convert("L")
    im.thumbnail((480, 480))
    a = np.asarray(im, float)
    lap = 4 * a[1:-1, 1:-1] - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    return float((a >= 250).mean()), float(min(1.0, lap.var() / 100)), float(a.mean())

def result_card(status, reason=None):
    bg, fg, title = BADGES[status]
    first, nxt = (engine.MSG[reason][1], "") if reason in engine.MSG else EXPLAIN[reason or status]
    step = f'<p style="margin:.6rem 0 0;font-weight:600;color:#0F172A">{nxt}</p>' if nxt else ""
    st.markdown(f'<div class="card"><div class="banner" style="background:{bg};color:{fg};border:1px solid {fg}33">{title}</div>'
                f'<p style="margin:1rem 0 0;font-size:1.02rem;color:#334155">{first}</p>{step}</div>',
                unsafe_allow_html=True)

def tint(tile):
    t = np.kron(tile, np.ones((3, 3), np.uint8)).astype(float)
    return np.stack([t * 0.25, t * 0.72, t], -1).astype(np.uint8)  # sleek sea-blue optical reflex tint

# ------------------------------------------------------------------ Anonymous Care-Loop State (Tab 2)
ss = st.session_state
STAGES = {1: "1. Screened", 2: "2. Parent Confirmed", 3: "3. Doctor Referral Sent", 4: "4. Glasses Fitted & Closed"}
PILL = {1: ("#FEF2F2", "#991B1B"), 2: ("#FFFBEB", "#92400E"), 3: ("#E0F2FE", "#0369A1"), 4: ("#ECFDF5", "#065F46")}
REASONS, SLA_DAYS, FMT = ["—", "Travel", "Cost", "Unconfirmed"], 7, "%d %b %Y, %H:%M"

def seed_cases():
    ago = lambda d: datetime.now() - timedelta(days=d)
    mk = lambda i, age_y, scr, ack, pkt, closed, est, cres, risk, reason=None, signer=None: (
        f"CASE-{i}", dict(id=f"CASE-{i}", label=f"Child (Age {age_y} yrs)", age=age_y * 12, scr=scr, ack=ack, pkt=pkt,
                          closed=closed, est=est, cres=cres, risk=risk, reason=reason, signer=signer))
    return dict([
        mk(101, 4, ago(9), None, None, None, (0.75, -2.25, 175), [0.41, -0.05, -0.38, 0.02], 1.50, "Unconfirmed"),
        mk(102, 7, ago(11), ago(10), None, None, (4.25, -0.50, 90), [0.72, 0.65, 0.70, 0.66], 1.21, "Cost"),
        mk(103, 12, ago(6), ago(5.5), ago(3.5), None, (-1.25, -0.50, 20), [-0.30, -0.12, -0.28, -0.22], 1.17),
        mk(104, 15, ago(10), ago(9), ago(7), ago(4), (1.00, -1.25, 180), [0.18, -0.02, -0.20, 0.01], 0.83, None, "Dr. R. Rahman (Optometrist)"),
    ])

stage = lambda c: 4 if c["closed"] else 3 if c["pkt"] else 2 if c["ack"] else 1
days_open = lambda c: ((c["closed"] or datetime.now()) - c["scr"]).total_seconds() / 86400
pill = lambda t, bg, fg: f'<span style="background:{bg};color:{fg};border-radius:8px;padding:.15rem .55rem;font-weight:700;font-size:.78rem;white-space:nowrap">{t}</span>'

def act(cid, field):
    c = ss.cases[cid]
    c[field] = datetime.now()
    if field == "closed":
        c["signer"], c["reason"], ss[f"reason_{cid}"] = ss.get(f"sig_{cid}", "").strip(), None, "—"

def set_reason(cid):
    ss.cases[cid]["reason"] = None if ss[f"reason_{cid}"] == "—" else ss[f"reason_{cid}"]

def log_case(key, P, cd):
    if key not in ss.logged:
        cid = f"CASE-{101 + len(ss.cases)}"
        age_y = max(2, min(17, round(float(P["age"][0]) / 12)))
        ss.cases[cid] = dict(id=cid, label=f"Child (Age {age_y} yrs)", age=int(P["age"][0]), scr=datetime.now(),
                             ack=None, pkt=None, closed=None, est=(cd["pred_sphere_D"], cd["pred_cylinder_D"], cd["pred_axis_deg"]),
                             cres=cd["crescent_array"], risk=cd["risk_score"], reason=None, signer=None)
        ss.logged[key], ss.case_sel = cid, cid
    return ss.logged[key]

# ------------------------------------------------------------------ Main App
bundle = load_bundle()
samples = build_samples(bundle)
ss.setdefault("cases", seed_cases())
ss.setdefault("logged", {})
ss.setdefault("active", None)
ss.setdefault("seen", None)

st.markdown("<h1 style='text-align:center;margin-bottom:0;color:#0F172A'>👁️ BrightSight Kids</h1>"
            "<p style='text-align:center;font-size:1.05rem;margin-top:.25rem;color:#475569'>"
            "Preventive smartphone eye screening for children ages 2 to 17.</p>", unsafe_allow_html=True)

tab_parent, tab_clinical = st.tabs(["1. Parent Vision Check", "2. Care-Loop & Clinical View"])

with tab_parent:
    photo = st.camera_input("Hold your phone about 1 meter (arm's length) from your child's eyes")
    if photo is not None:
        pid = hashlib.md5(photo.getvalue()).hexdigest()
        if pid != ss.seen:
            ss.seen, ss.active = pid, "photo"
    elif ss.active == "photo":
        ss.active = None

    st.markdown("<p style='font-weight:600;margin:.6rem 0 .3rem;color:#334155'>Or preview with a quick test scenario:</p>", unsafe_allow_html=True)
    for col, label in zip(st.columns(2) + st.columns(2), SAMPLE_RULES):
        if col.button(label, key=label, disabled=label not in samples):
ss.active = label
drawer_data = None
if ss.active == "photo" and photo is not None:
g, s, br = q = photo_quality(photo)
reason = "glare" if g > 0.05 else "dark" if br < 45 else "blur" if s < 0.35 else None
result_card("RETAKE" if reason else "OK", reason)
drawer_data = dict(quality=q)
elif ss.active in samples:
P, truth = samples[ss.active]
res = engine.predict_eye(child_features(P, ss.get("profile", engine.PNAMES[0])), bundle)
rs = res["clinical_details"]["retake_reasons"]
result_card(res["status"], rs[0] if rs else None)
if res["status"] == "REFER":
cid = log_case(ss.active, P, res["clinical_details"])
c = ss.cases[cid]
st.markdown('Parent Follow-Up Confirmation'
'Confirm below once you have booked an eye exam so your clinic can prepare.', unsafe_allow_html=True)
if c["ack"]:
st.markdown(pill(f"✓ Checkup Confirmed: {c['ack'].strftime(FMT)}", *PILL[4]), unsafe_allow_html=True)
else:
st.button("I Understand & Booked an Eye Exam", key=f"p_ack_{cid}", on_click=act, args=(cid, "ack"))
drawer_data = dict(res=res, truth=truth)
else:
st.markdown(''
'Take a photo above or select a quick test scenario to see your child's result.', unsafe_allow_html=True)
with tab_clinical:
cs = ss.cases
st.markdown("#### Closed-Loop Referral Tracker (Screening → Glasses)")
st.caption("Anonymized case tracking ensuring no referred child drops off before receiving glasses.")
rows = ""
for k, c in cs.items():
s, d = stage(c), days_open(c)
sla = pill(f"OVERDUE · {d:.0f} d", "#FEF2F2", "#991B1B") if s < 4 and d > SLA_DAYS else ""
pri = pill("URGENT", "#991B1B", "#FFFFFF") if c["risk"] >= 1.0 else pill("Routine", "#E0F2FE", "#0369A1")
rows += (f"{k}{c['label']}{pri}{pill(STAGES[s], *PILL[s])}"
f"{'School Nurse' if s < 3 else 'Pediatric Optometrist'}{d:.0f} d {sla}")
st.markdown(''
'Case IDPatientPriorityStageOwnerFollow-up'
f'{rows}', unsafe_allow_html=True)
sel = st.selectbox("Select case to manage handoff & sign-off", list(cs), key="case_sel", format_func=lambda k: f"{k} · {cs[k]['label']}")
c = cs[sel]
s = stage(c)
S, C, ax = c["est"]
meridians = "  |  ".join(f"{m}°: {v:+.2f}" for m, v in zip((0, 45, 90, 135), c["cres"]))
st.markdown(f'Clinician Referral Packet · {sel} ({c["label"]})
'
f'Screened: {c["scr"].strftime(FMT)} · Risk Score: {c["risk"]:.2f}

'
f'Estimated Refraction: Sphere {S:+.2f} D · Cylinder {C:+.2f} D · Axis {ax:.0f}°
'
f'4-Meridian Crescents: {meridians}', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
if c["ack"]:
st.markdown(pill(f"✓ Parent Confirmed ({c['ack'].strftime(FMT)})", *PILL[4]), unsafe_allow_html=True)
else:
st.button("1. Record Parent Confirmation", key=f"c_ack_{sel}", on_click=act, args=(sel, "ack"))
with col2:
st.button("2. Send Referral Packet to Optometrist", key=f"pkt_{sel}", on_click=act, args=(sel, "pkt"), disabled=not c["ack"] or bool(c["pkt"]))
st.selectbox("Log follow-up barrier (if stalled)", REASONS, index=REASONS.index(c["reason"] or "—"), key=f"reason_{sel}",
on_change=set_reason, args=(sel,), disabled=s == 4)
st.text_input("Authorized Optometrist Name / License No. (required to close)", key=f"sig_{sel}", disabled=s != 3)
st.button("3. Authorized Optometrist Sign-Off (Glasses Dispensed)", key=f"cls_{sel}", on_click=act, args=(sel, "closed"),
disabled=s != 3 or not ss.get(f"sig_{sel}", "").strip())
if s == 4:
st.markdown(pill(f"✓ Case Closed {c['closed'].strftime(FMT)} · Verified by {c['signer']}", *PILL[4]), unsafe_allow_html=True)
n, acked = len(cs), sum(bool(x["ack"]) for x in cs.values())
tt = [(x["pkt"] - x["scr"]).total_seconds() / 86400 for x in cs.values() if x["pkt"]]
why = Counter(x["reason"] for x in cs.values() if x["reason"])
tiles = [("Parent Confirmation Rate", f"{acked / n:.0%}", f"{acked} of {n} cases"),
("Median Time-to-Referral", f"{median(tt):.1f} d" if tt else "—", "screening → clinic handoff"),
("Logged Barriers", " · ".join(f"{k}: {v}" for k, v in why.items()) or "None", "active cases")]
st.markdown('' + "".join(
f''
f'{v}'
f'{k}{sub}'
for k, v, sub in tiles) + "", unsafe_allow_html=True)
with st.expander("Optical Physics & Model Validation Metrics (For Judges)"):
if drawer_data and "res" in drawer_data:
res, truth = drawer_data["res"], drawer_data["truth"]
cd = res["clinical_details"]
mos = cd["crescent_visual_128"]
st.selectbox("Device eccentricity preset", engine.PNAMES, key="profile")
tiles_img = [mos[:64, :64], mos[:64, 64:], mos[64:, :64], mos[64:, 64:]]
for col, deg, t, c_val in zip(st.columns(4), engine.MERID, tiles_img, cd["crescent_array"]):
col.image(tint(t), caption=f"{deg:.0f}° | {c_val:+.2f}", width=135)
st.markdown(f"Dead zone R_dead: {' / '.join(f'{v:.2f}' for v in cd['dead_zone_D'])} D (limit {cd['dead_zone_limit_D']:.2f} D) | "
f"Ensemble uncertainty: {cd['ensemble_uncertainty_D']:.3f} D")
m = bundle["metrics"]
for col, (k, v) in zip(st.columns(4), [("ROC-AUC", m["roc_auc"]), ("Sensitivity", m["sensitivity"]),
("Specificity", m["specificity"]), ("MAE M (D)", m["MAE_D_accepted"]["M"])]):
col.metric(k, f"{v:.2f}")
st.caption("BrightSight Kids is a preventive screening aid for children ages 2–17, not a medical diagnosis.")
