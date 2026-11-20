"""app.py - BrightSight Kids: a calm, parent-friendly front-end for engine.py (ages 2-17).
Run:  pip install streamlit numpy scikit-learn joblib pillow   then   streamlit run app.py"""
import os, hashlib
import numpy as np, streamlit as st
from PIL import Image

os.chdir(os.path.dirname(os.path.abspath(__file__)))   # engine.py + screener_model.joblib live next to this file
import engine

st.set_page_config(page_title="BrightSight Kids", page_icon="💧", layout="centered")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
:root {--sea:#0E8FB0; --sea-dark:#0A6B86; --aqua:#E6F6FA; --line:#D3ECF3; --ink:#12313B; --mute:#5B7C88;}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {background:#FFFFFF !important; color-scheme: light;}
html, body, .stApp, .stApp * {font-family:'Nunito', sans-serif;}
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp span,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"] {color:var(--ink);}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {display:none !important;}
.block-container {max-width: 600px; padding-top: 2.2rem; padding-bottom: 3rem;}

.brand {text-align:center; margin-bottom:1.6rem;}
.brand .logo {display:inline-flex; width:64px; height:64px; border-radius:20px; align-items:center; justify-content:center;
  background:linear-gradient(135deg,#7FD6EA,#0E8FB0); font-size:2rem; box-shadow:0 8px 20px #0E8FB033;}
.brand h1 {font-size:2rem; font-weight:800; margin:.7rem 0 .1rem; letter-spacing:-.5px;}
.brand p {color:var(--mute) !important; margin:0; font-size:1.05rem;}

.step {display:flex; align-items:center; gap:.7rem; margin:1.8rem 0 .7rem; font-weight:800; font-size:1.15rem;}
.step .n {width:30px; height:30px; border-radius:50%; background:var(--sea); color:#fff; display:flex; align-items:center;
  justify-content:center; font-size:.95rem;}
.step .n {color:#fff !important;}

.card {background:#fff; border:1px solid var(--line); border-radius:22px; padding:1.3rem 1.4rem; margin:.7rem 0;
  box-shadow:0 6px 24px #0E8FB014;}
.tip {background:var(--aqua); border-radius:16px; padding:.8rem 1rem; font-size:.95rem; margin:.6rem 0;}

.result {border-radius:24px; padding:1.8rem 1.5rem; text-align:center; margin:.8rem 0; border:1px solid var(--line);
  box-shadow:0 8px 28px #0E8FB018;}
.result .icon {font-size:2.6rem; line-height:1;}
.result h2 {font-size:1.6rem; font-weight:800; margin:.6rem 0 .5rem;}
.result p {font-size:1.05rem; margin:.3rem 0;}
.result .next {margin-top:1rem; padding:.8rem 1rem; border-radius:14px; background:#ffffffcc; font-weight:700;}

div.stButton > button, [data-testid="stCameraInput"] button {background:var(--sea) !important; color:#fff !important; border:none !important;
  border-radius:14px !important; font-weight:700 !important; padding:.65rem 1rem; box-shadow:0 4px 12px #0E8FB033;}
div.stButton > button p {color:#fff !important;}
div.stButton > button:hover, [data-testid="stCameraInput"] button:hover {background:var(--sea-dark) !important;}
[data-testid="stCameraInput"] {border:1px solid var(--line); border-radius:22px; padding:.5rem; background:#fff;}
[data-testid="stCameraInput"] img, [data-testid="stCameraInput"] video {border-radius:16px;}
[data-testid="stExpander"] {border:1px solid var(--line) !important; border-radius:16px; background:#fff;}
[data-testid="stExpander"] details, [data-testid="stExpander"] summary {background:#fff !important;}
[data-baseweb="select"] > div {background:#fff !important; border-color:var(--line) !important; border-radius:14px !important;}
[data-testid="stSlider"] [role="slider"] {background:var(--sea) !important;}
.fine {text-align:center; color:var(--mute) !important; font-size:.85rem; margin-top:2rem;}
</style>""", unsafe_allow_html=True)

# status/reason -> (background, accent, icon, title, message, next step)
LOOK = {"PASS":   ("#E8F8F1", "#1B7F5C", "✅", "Vision looks clear",
                   "We did not see anything that needs follow-up right now.",
                   "Repeat this check in about a year, or sooner if your child squints or sits very close to screens."),
        "REFER":  ("#E6F2FB", "#0A6B86", "🩺", "An eye exam is a good idea",
                   "We noticed a pattern that is often linked to needing glasses. This is very common and easy to treat, especially when found early.",
                   "Please book a full eye exam with an eye doctor in the next few weeks."),
        "RETAKE": ("#FFF6E0", "#9A6A00", "🔄", "Let's try that again", "", ""),
        "OK":     ("#E6F6FA", "#0A6B86", "📷", "Photo looks good",
                   "Your photo is clear and well lit.",
                   "A full reading needs the 4-flash scan in the finished phone app.")}
DARK = ("The photo is too dark to see your child's eyes clearly.", "Turn on a soft, even room light and retake the photo.")

def age_tip(months):
    if months < 48:  return "Little ones move a lot. Hold a favourite toy beside the phone so they look straight at it."
    if months < 132: return "Ask your child to look at the phone and keep very still, like a game of statues."
    return "Ask your child to look straight at the camera and keep their eyes wide open."

@st.cache_resource
def load_bundle():
    if not os.path.exists(engine.MODEL_PATH):
        with st.spinner("Getting things ready for the first time (about a minute)..."):
            engine.main()
    return engine.joblib.load(engine.MODEL_PATH)

# Demo children: only used for the optional preview at the bottom of the page
SAMPLE_RULES = {
    "Clear vision": (lambda D, i: D["kind"][i] == 0 and abs(D["S"][i]) < 1.0 and D["C"][i] > -0.5 and D["age"][i] >= 49, "PASS"),
    "Needs a check": (lambda D, i: D["kind"][i] == 0 and D["C"][i] <= -2.0 and -1 < D["S"][i] < 2 and D["age"][i] >= 49, "REFER"),
    "Poor lighting": (lambda D, i: D["kind"][i] == 1 and D["glare"][i] > 0.1, "RETAKE")}

def child_features(P, seed=0):
    raw, _ = engine.capture(P, np.array([0]), np.random.default_rng(seed))
    return dict(zip(engine.RAW_KEYS, map(float, raw[0])))

@st.cache_resource
def build_samples(_b):
    D = engine.simulate(np.random.default_rng(2024), 800); out = {}
    for label, (rule, want) in SAMPLE_RULES.items():
        for i in (j for j in range(len(D["sid"])) if rule(D, j)):
            P = {k: D[k][i:i + 1] for k in engine.PARAM_KEYS}
            if engine.predict_eye(child_features(P), _b)["status"] == want:
                out[label] = P; break
    return out

def photo_quality(file):  # (clipped-highlight fraction, sharpness 0-1, mean brightness 0-255)
    im = Image.open(file).convert("L"); im.thumbnail((480, 480)); a = np.asarray(im, float)
    lap = 4 * a[1:-1, 1:-1] - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    return float((a >= 250).mean()), float(min(1.0, lap.var() / 100)), float(a.mean())

def result_card(status, reason=None):
    bg, fg, icon, title, msg, nxt = LOOK[status]
    if status == "RETAKE":
        if reason in engine.MSG: title, msg = engine.MSG[reason]
        elif reason == "dark": title, msg = "The photo is too dark", DARK[0]
        nxt = DARK[1] if reason == "dark" else ""
        if reason in engine.MSG: title, msg, nxt = engine.MSG[reason][0], engine.MSG[reason][1], ""
    step = f'<div class="next" style="color:{fg}">{nxt}</div>' if nxt else ""
    st.markdown(f'<div class="result" style="background:{bg}"><div class="icon">{icon}</div>'
                f'<h2 style="color:{fg} !important">{title}</h2><p>{msg}</p>{step}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ page
ss = st.session_state
bundle = load_bundle(); samples = build_samples(bundle)
ss.setdefault("demo", None)

st.markdown('<div class="brand"><div class="logo">👁️</div><h1>BrightSight Kids</h1>'
            '<p>A gentle, quick eye check for children aged 2 to 17</p></div>', unsafe_allow_html=True)

st.markdown('<div class="step"><span class="n">1</span>How old is your child?</div>', unsafe_allow_html=True)
years = st.select_slider("Age", options=list(range(2, 18)), value=6, format_func=lambda y: f"{y} years", label_visibility="collapsed")
st.markdown(f'<div class="tip">💡 {age_tip(years * 12)}</div>', unsafe_allow_html=True)

st.markdown('<div class="step"><span class="n">2</span>Take a photo</div>', unsafe_allow_html=True)
st.caption("Hold the phone at arm's length (about 1 metre) in a softly lit room, away from windows.")
photo = st.camera_input("Photo of your child's eyes", label_visibility="collapsed")

st.markdown('<div class="step"><span class="n">3</span>Your result</div>', unsafe_allow_html=True)
if photo is not None:
    g, s, br = photo_quality(photo)
    reason = "glare" if g > 0.05 else "dark" if br < 45 else "blur" if s < 0.35 else None
    result_card("RETAKE" if reason else "OK", reason)
elif ss.demo in samples:
    res = engine.predict_eye(child_features(samples[ss.demo]), bundle)
    rs = res["clinical_details"]["retake_reasons"]
    result_card(res["status"], rs[0] if rs else None)
    st.caption("This is an example result, not a real child.")
else:
    st.markdown('<div class="card" style="text-align:center;color:#5B7C88">'
                'Your result will appear here right after you take the photo.</div>', unsafe_allow_html=True)

with st.expander("Preview example results"):
    for col, label in zip(st.columns(len(SAMPLE_RULES)), SAMPLE_RULES):
        if col.button(label, key=label, disabled=label not in samples):
            ss.demo = label; st.rerun()

st.markdown('<p class="fine">BrightSight Kids is a screening aid, not a diagnosis. It does not replace an eye exam.</p>', unsafe_allow_html=True)
