"""app.py - BrightSight Kids: a calm, parent-friendly front-end for engine.py (ages 2-17).
DEMO VERSION: the guided 4-photo scan is real (photo-quality checks run on every photo), but the final
refraction result is SIMULATED by the trained model - a phone browser cannot measure eye focus on its own.
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
/* Nunito only on text elements - never on icon spans, so Streamlit's own icons keep working */
.stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp button p, .stApp input,
.brand *, .step, .card *, .result *, .tip, .fine, .dots *, .cue * {font-family:'Nunito', sans-serif;}
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"] {color:var(--ink);}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {display:none !important;}
.block-container {max-width: 600px; padding-top: 2.2rem; padding-bottom: 3rem;}

.brand {text-align:center; margin-bottom:1.2rem;}
.brand .logo {display:inline-flex; width:64px; height:64px; border-radius:20px; align-items:center; justify-content:center;
  background:linear-gradient(135deg,#7FD6EA,#0E8FB0); font-size:2rem; box-shadow:0 8px 20px #0E8FB033;}
.brand h1 {font-size:2rem; font-weight:800; margin:.7rem 0 .1rem; letter-spacing:-.5px;}
.brand p {color:var(--mute) !important; margin:0; font-size:1.05rem;}
.pill {display:inline-block; margin-top:.8rem; padding:.25rem .8rem; border-radius:999px; background:#FFF6E0; color:#7A5200 !important;
  font-size:.8rem; font-weight:800; letter-spacing:.4px; border:1px solid #F2DFA8;}

.step {display:flex; align-items:center; gap:.7rem; margin:1.8rem 0 .7rem; font-weight:800; font-size:1.15rem;}
.step .n {width:30px; height:30px; border-radius:50%; background:var(--sea); color:#fff !important; display:flex; align-items:center;
  justify-content:center; font-size:.95rem;}
.card {background:#fff; border:1px solid var(--line); border-radius:22px; padding:1.3rem 1.4rem; margin:.7rem 0; box-shadow:0 6px 24px #0E8FB014;}
.tip {background:var(--aqua); border-radius:16px; padding:.8rem 1rem; font-size:.95rem; margin:.6rem 0;}

.dots {display:flex; justify-content:center; gap:.5rem; margin:.4rem 0 .9rem;}
.dots span {width:34px; height:8px; border-radius:6px; background:var(--line);} .dots span.on {background:var(--sea);}
.cue {display:flex; align-items:center; gap:1.1rem; background:var(--aqua); border-radius:20px; padding:1rem 1.2rem; margin-bottom:.8rem;}
.cue b {font-size:1.05rem;} .cue p {margin:.15rem 0 0; font-size:.92rem; color:var(--mute) !important;}
.phone {position:relative; flex:none; width:66px; height:108px; border:3px solid var(--sea); border-radius:16px; overflow:hidden; background:#fff;}
.phone i {position:absolute; display:block;}

.result {border-radius:24px; padding:1.6rem 1.5rem 1.8rem; text-align:center; margin:.8rem 0; border:1px solid var(--line); box-shadow:0 8px 28px #0E8FB018;}
.result .icon {font-size:2.6rem; line-height:1;}
.result h2 {font-size:1.6rem; font-weight:800; margin:.6rem 0 .5rem;}
.result p {font-size:1.05rem; margin:.3rem 0;}
.result .next {margin-top:1rem; padding:.8rem 1rem; border-radius:14px; background:#ffffffcc; font-weight:700;}
.result .demo {display:inline-block; margin-bottom:.9rem; padding:.25rem .8rem; border-radius:999px; background:#FFF6E0; color:#7A5200 !important;
  font-size:.78rem; font-weight:800; letter-spacing:.4px; border:1px solid #F2DFA8;}

div.stButton > button, [data-testid="stCameraInput"] button {background:var(--sea) !important; color:#fff !important; border:none !important;
  border-radius:14px !important; font-weight:700 !important; padding:.65rem 1rem; box-shadow:0 4px 12px #0E8FB033;}
div.stButton > button p {color:#fff !important;}
div.stButton > button:hover, [data-testid="stCameraInput"] button:hover {background:var(--sea-dark) !important;}
div.stButton > button:disabled {background:var(--line) !important; box-shadow:none;}
[data-testid="stCameraInput"] {border:1px solid var(--line); border-radius:22px; padding:.5rem; background:#fff;}
[data-testid="stCameraInput"] img, [data-testid="stCameraInput"] video {border-radius:16px;}
[data-baseweb="select"] > div {background:#fff !important; border-color:var(--line) !important; border-radius:14px !important;}
[data-testid="stSlider"] [role="slider"] {background:var(--sea) !important;}
.fine {text-align:center; color:var(--mute) !important; font-size:.85rem; margin-top:2rem;}
</style>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ copy
LOOK = {"PASS":   ("#E8F8F1", "#1B7F5C", "✅", "Vision looks clear",
                   "We did not see anything that needs follow-up right now.",
                   "Repeat this check in about a year, or sooner if your child squints or sits very close to screens."),
        "REFER":  ("#E6F2FB", "#0A6B86", "🩺", "An eye exam is a good idea",
                   "We noticed a pattern that is often linked to needing glasses. This is very common and easy to treat, especially when found early.",
                   "Please book a full eye exam with an eye doctor in the next few weeks."),
        "RETAKE": ("#FFF6E0", "#9A6A00", "🔄", "Let's try that again", "", "")}
DARK = ("The photo is too dark", "Your child's eyes are hard to see. Turn on a soft, even room light and retake this photo.")

# light cue for each flash: (title, instruction, CSS for the glow shown on the little phone)
SHOTS = [("Light from the left", "Look at the left edge glowing. Child looks at the phone.",
          "left:0;top:0;bottom:0;width:45%;background:linear-gradient(90deg,#4CC3E0,transparent)"),
         ("Light from the lower right", "Now the glow moves to the lower right corner.",
          "right:0;bottom:0;width:90%;height:70%;background:radial-gradient(circle at 100% 100%,#4CC3E0,transparent 70%)"),
         ("Light from the bottom", "Now the glow is along the bottom edge.",
          "left:0;right:0;bottom:0;height:45%;background:linear-gradient(0deg,#4CC3E0,transparent)"),
         ("Light from the lower left", "Last one. The glow moves to the lower left corner.",
          "left:0;bottom:0;width:90%;height:70%;background:radial-gradient(circle at 0% 100%,#4CC3E0,transparent 70%)")]

def age_tip(months):
    if months < 48:  return "Little ones move a lot. Hold a favourite toy beside the phone so they look straight at it."
    if months < 132: return "Ask your child to look at the phone and keep very still, like a game of statues."
    return "Ask your child to look straight at the camera and keep their eyes wide open."

# ------------------------------------------------------------------ model + demo data
@st.cache_resource
def load_bundle():
    if not os.path.exists(engine.MODEL_PATH):
        with st.spinner("Getting things ready for the first time (about a minute)..."):
            engine.main()
    return engine.joblib.load(engine.MODEL_PATH)

@st.cache_resource
def sim_pool():
    return engine.simulate(np.random.default_rng(2024), 1500)

RULES = {"PASS": lambda D: (D["kind"] == 0) & (np.abs(D["S"]) < 1.0) & (D["C"] > -0.5),
         "REFER": lambda D: (D["kind"] == 0) & (((D["C"] <= -2.0) & (np.abs(D["S"]) < 2)) | ((D["S"] >= 3.8) & (D["C"] > -1.0))),
         "RETAKE": lambda D: (D["kind"] == 1) & (D["glare"] > 0.1)}

def child_features(P, seed=0):
    raw, _ = engine.capture(P, np.array([0]), np.random.default_rng(seed))
    return dict(zip(engine.RAW_KEYS, map(float, raw[0])))

@st.cache_resource(show_spinner=False)
def simulate_result(_b, outcome, months, offset):
    """Find a simulated child of about this age whose model result matches `outcome`, and return the engine result."""
    D = sim_pool(); ok = np.where(RULES[outcome](D))[0]
    near = ok[np.abs(D["age"][ok] - months) <= 24]; pool = near if len(near) else ok
    for i in np.roll(pool, -(offset % len(pool)))[:300]:
        P = {k: D[k][i:i + 1].copy() for k in engine.PARAM_KEYS}; P["age"] = np.array([float(months)])
        res = engine.predict_eye(child_features(P), _b)
        if res["status"] == outcome: return res
    return None

def photo_quality(file):  # (clipped-highlight fraction, sharpness 0-1, mean brightness 0-255)
    im = Image.open(file).convert("L"); im.thumbnail((480, 480)); a = np.asarray(im, float)
    lap = 4 * a[1:-1, 1:-1] - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    return float((a >= 250).mean()), float(min(1.0, lap.var() / 100)), float(a.mean())

def result_card(status, reason=None, demo=False):
    bg, fg, icon, title, msg, nxt = LOOK[status]
    if status == "RETAKE":
        if reason == "dark": title, msg = DARK
        elif reason in engine.MSG: title, msg = engine.MSG[reason]
    ribbon = '<div class="demo">DEMO ONLY · SIMULATED RESULT</div><br>' if demo else ""
    step = f'<div class="next" style="color:{fg}">{nxt}</div>' if nxt else ""
    st.markdown(f'<div class="result" style="background:{bg}">{ribbon}<div class="icon">{icon}</div>'
                f'<h2 style="color:{fg} !important">{title}</h2><p>{msg}</p>{step}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ state
ss = st.session_state
ss.setdefault("shots", [None] * 4); ss.setdefault("forced", None); ss.setdefault("rev", 0)

def restart():
    ss.shots, ss.forced, ss.rev = [None] * 4, None, ss.rev + 1
def force(outcome): ss.forced = outcome

bundle = load_bundle()

# ------------------------------------------------------------------ page
st.markdown('<div class="brand"><div class="logo">👁️</div><h1>BrightSight Kids</h1>'
            '<p>A gentle, quick eye check for children aged 2 to 17</p>'
            '<span class="pill">DEMO VERSION · RESULTS ARE SIMULATED</span></div>', unsafe_allow_html=True)

st.markdown('<div class="step"><span class="n">1</span>How old is your child?</div>', unsafe_allow_html=True)
years = st.select_slider("Age", options=list(range(2, 18)), value=6, format_func=lambda y: f"{y} years", label_visibility="collapsed")
st.markdown(f'<div class="tip">💡 {age_tip(years * 12)}</div>', unsafe_allow_html=True)

st.markdown('<div class="step"><span class="n">2</span>Take 4 quick photos</div>', unsafe_allow_html=True)
done = sum(s is not None for s in ss.shots)
st.markdown('<div class="dots">' + "".join(f'<span class="{"on" if i < done else ""}"></span>' for i in range(4)) + "</div>", unsafe_allow_html=True)

if done < 4:
    title, how, glow = SHOTS[done]
    st.markdown(f'<div class="cue"><div class="phone"><i style="{glow}"></i></div>'
                f'<div><b>Photo {done + 1} of 4 · {title}</b><p>{how}</p></div></div>', unsafe_allow_html=True)
    if done == 0:
        st.caption("Dim the room lights, hold the phone at arm's length (about 1 metre), and take the four photos one after another.")
    photo = st.camera_input(f"Photo {done + 1}", key=f"cam_{ss.rev}_{done}", label_visibility="collapsed")
    if photo is not None:
        g, s, br = photo_quality(photo)
        reason = "glare" if g > 0.05 else "dark" if br < 45 else "blur" if s < 0.35 else None
        if reason:
            result_card("RETAKE", reason)
            st.caption("Tap “Clear photo” above, then take this photo again.")
        else:
            ss.shots[done] = photo.getvalue(); st.rerun()
else:
    st.markdown('<div class="tip">✅ All 4 photos captured and checked.</div>', unsafe_allow_html=True)
    for col, data in zip(st.columns(4), ss.shots): col.image(data, use_container_width=True)
if done or ss.forced:
    st.button("Start over", key="restart", on_click=restart)

st.markdown('<div class="step"><span class="n">3</span>Your result</div>', unsafe_allow_html=True)
outcome, offset = ss.forced, 0
if outcome is None and done == 4:
    h = int(hashlib.md5(b"".join(ss.shots)).hexdigest(), 16)
    outcome, offset = ("PASS" if h % 100 < 55 else "REFER"), h // 100
if outcome:
    res = simulate_result(bundle, outcome, years * 12, offset)
    if res:
        rs = res["clinical_details"]["retake_reasons"]
        result_card(res["status"], rs[0] if rs else None, demo=True)
        st.caption("Demo only: this result is simulated by our trained model and is not a measurement of your child's eyes.")
    else:
        st.markdown('<div class="card" style="text-align:center">Please try again.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="card" style="text-align:center;color:#5B7C88">'
                'Your result will appear here after the 4th photo.</div>', unsafe_allow_html=True)

st.markdown('<div class="step" style="font-size:1rem;margin-top:2rem">See every possible outcome (demo)</div>', unsafe_allow_html=True)
for col, (label, key) in zip(st.columns(3), [("Clear vision", "PASS"), ("Needs a check", "REFER"), ("Retake example", "RETAKE")]):
    col.button(label, key=f"demo_{key}", on_click=force, args=(key,))

st.markdown('<p class="fine">DEMO ONLY. BrightSight Kids is a prototype trained and tested on simulated data. Results shown are simulated, '
            'are not a measurement of any child, and are not medical advice. A screening aid never replaces an eye exam.</p>', unsafe_allow_html=True)
