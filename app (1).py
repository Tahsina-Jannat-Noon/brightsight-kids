"""app.py - BrightSight Kids: consumer-friendly Streamlit front-end for engine.py.
Run:  pip install streamlit numpy scikit-learn joblib   then   streamlit run app.py"""
import os, hashlib
from collections import Counter
from datetime import datetime, timedelta
from statistics import median
import numpy as np, streamlit as st
from PIL import Image

os.chdir(os.path.dirname(os.path.abspath(__file__)))   # engine.py + screener_model.joblib live next to this file
import engine

st.set_page_config(page_title="BrightSight Kids", page_icon="👁️", layout="centered")
st.markdown("""<style>
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {background:#FFF2CC !important; color-scheme: light;}
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricLabel"], .stApp [data-testid="stCaptionContainer"] {color:#2C2410;}
.block-container {max-width: 760px; padding-top: 2rem;}  .banner {border-radius:14px; padding:.95rem 1.1rem; font-size:1.6rem; font-weight:800; text-align:center;}
.card {background:#FFF9E6; border:1px solid #E6D29C; border-radius:18px; padding:1.2rem 1.4rem; margin:.8rem 0;}
.card th, .card td {padding:.35rem .5rem; border-bottom:1px solid #E6D29C; text-align:left; vertical-align:top;}
div.stButton > button, [data-testid="stCameraInput"] button {background:#D99B26 !important; color:#2C2410 !important; border:none !important;
  border-radius:12px !important; font-weight:700 !important; width:100%; padding:.7rem 1rem;}  div.stButton > button p {color:#2C2410 !important;}
div.stButton > button:hover {filter:brightness(.94);}
[data-testid="stCameraInput"], [data-testid="stExpander"] {background:#FFF9E6; border:1px solid #E6D29C !important; border-radius:16px; padding:.4rem;}
[data-testid="stExpander"] details, [data-testid="stExpander"] summary {background:#FFF9E6 !important; color:#2C2410 !important;}
[data-baseweb="select"] > div, [data-baseweb="base-input"] {background:#FFF9E6 !important; border-color:#E6D29C !important;}  input {color:#2C2410 !important;}
</style>""", unsafe_allow_html=True)

# status -> (badge bg, badge text, big friendly title)
BADGES = {"PASS": ("#E3F2E6", "#1E5631", "Vision Looks Clear"),
          "REFER": ("#FCE4E1", "#9E2A2B", "Eye Doctor Check Recommended"),
          "RETAKE": ("#FDE69A", "#7A5200", "Please Retake Photo"),
          "OK": ("#FFF2CC", "#2C2410", "Photo Looks Good")}
EXPLAIN = {  # (two plain-English sentences, next step); retake reasons reuse engine.MSG wording
    "PASS": ("Your child's eyes showed no signs that need follow-up today. This is a quick check, not a full eye exam.",
             "Next step: repeat this check in about a year, or sooner if your child squints or sits very close to screens."),
    "REFER": ("The check found a pattern that is often linked to needing glasses. That is very common and easy to treat, especially when found early.",
              "Next step: book a full eye exam with an eye doctor in the next few weeks."),
    "dark": ("The photo is too dark to see your child's eyes clearly. A bit more light will fix that.",
             "Next step: turn on a soft, even room light and retake the photo."),
    "OK": ("Your photo is sharp and well lit. A full reading needs the 4-flash scan in the finished phone app.",
           "Next step: tap one of the sample children below to see a complete result.")}
SAMPLE_RULES = {  # label -> (selector on simulated eye, expected status)
    "Try Sample: Healthy Child": (lambda D, i: D["kind"][i] == 0 and abs(D["S"][i]) < 1.0 and D["C"][i] > -0.5 and D["age"][i] >= 49, "PASS"),
    "Try Sample: Hidden Farsightedness": (lambda D, i: D["kind"][i] == 0 and D["S"][i] >= 3.8 and D["C"][i] > -1.0 and D["age"][i] >= 49, "REFER"),
    "Try Sample: Astigmatism": (lambda D, i: D["kind"][i] == 0 and D["C"][i] <= -2.0 and -1 < D["S"][i] < 2 and D["age"][i] >= 49, "REFER"),
    "Try Sample: Bad Room Lighting / Glare": (lambda D, i: D["kind"][i] == 1 and D["glare"][i] > 0.1, "RETAKE")}

@st.cache_resource
def load_bundle():
    if not os.path.exists(engine.MODEL_PATH):
        with st.spinner("First run: training the screening model (about a minute)..."):
            engine.main()
    return engine.joblib.load(engine.MODEL_PATH)

def child_features(P, profile, seed=0):  # re-capture one simulated child on the chosen device profile
    raw, _ = engine.capture(P, np.array([engine.PNAMES.index(profile)]), np.random.default_rng(seed))
    return dict(zip(engine.RAW_KEYS, map(float, raw[0])))

@st.cache_resource
def build_samples(_b):  # deterministic demo children; each gives its intended outcome on the default device
    D = engine.simulate(np.random.default_rng(2024), 800); out = {}
    for label, (rule, want) in SAMPLE_RULES.items():
        for i in (j for j in range(len(D["sid"])) if rule(D, j)):
            P = {k: D[k][i:i + 1] for k in engine.PARAM_KEYS}
            if engine.predict_eye(child_features(P, engine.PNAMES[0]), _b)["status"] == want:
                out[label] = (P, (float(D["S"][i]), float(D["C"][i]), float(D["ax"][i])))
                break
    return out

def photo_quality(file):  # real pixel checks: (clipped-highlight fraction, sharpness 0-1, mean brightness 0-255)
    im = Image.open(file).convert("L"); im.thumbnail((480, 480)); a = np.asarray(im, float)
    lap = 4 * a[1:-1, 1:-1] - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    return float((a >= 250).mean()), float(min(1.0, lap.var() / 100)), float(a.mean())

def result_card(status, reason=None, note=""):
    bg, fg, title = BADGES[status]
    first, nxt = (engine.MSG[reason][1], "") if reason in engine.MSG else EXPLAIN[reason or status]
    step = f'<p style="margin:0;font-weight:700">{nxt}</p>' if nxt else ""
    st.markdown(f'<div class="card"><div class="banner" style="background:{bg};color:{fg};border:1px solid {fg}33">{title}</div>'
                f'<p style="margin:.9rem 0 .4rem;font-size:1.05rem">{first}</p>{step}{note}</div>',
                unsafe_allow_html=True)

def tint(tile):                                   # grayscale tile -> upscaled warm red-reflex look
    t = np.kron(tile, np.ones((3, 3), np.uint8)).astype(float)
    return np.stack([t, t * .45, t * .3], -1).astype(np.uint8)

def doctor_drawer(b, res=None, truth=None, quality=None):
    with st.expander("For Doctors & Judges: Optical Physics & AI Metrics"):
        if quality:
            g, s, br = quality
            st.markdown(f"**Live photo checks:** clipped highlights {g:.1%} | sharpness {s:.2f} | brightness {br:.0f}/255. "
                        "Refraction needs the 4-meridian flash sequence, which a plain browser photo cannot provide.")
        if res:
            cd = res["clinical_details"]; mos = cd["crescent_visual_128"]
            st.selectbox("Device eccentricity preset (re-captures the same child on another phone geometry)", engine.PNAMES, key="profile")
            tiles = [mos[:64, :64], mos[:64, 64:], mos[64:, :64], mos[64:, 64:]]
            st.markdown("**4-meridian crescent simulation** (bright crescent = defocus beyond the dead zone)")
            for col, deg, t, c in zip(st.columns(4), engine.MERID, tiles, cd["crescent_array"]):
                col.image(tint(t), caption=f"{deg:.0f}° | {c:+.2f}", width=140)
            S, C, ax = cd["pred_sphere_D"], cd["pred_cylinder_D"], cd["pred_axis_deg"]
            M, J0, J45 = (float(v[0]) for v in engine.power_vectors(*(np.array([x]) for x in (S, C, ax)))) if S is not None else (0, 0, 0)
            est = ("| Estimate | withheld: capture rejected by the uncertainty gate |\n|---|---|" if S is None else
                   f"| Sphere / Cylinder / Axis | {S:+.2f} D / {C:+.2f} D / {ax:.0f}° |\n|---|---|\n| Thibos M / J0 / J45 | {M:+.2f} / {J0:+.2f} / {J45:+.2f} D |")
            truth_row = f"\n| Simulated ground truth (S / C / axis) | {truth[0]:+.2f} / {truth[1]:+.2f} / {truth[2]:.0f}° |" if truth else ""
            st.markdown(est + truth_row +
                        f"\n| Device eccentricity e (0/45/90/135°) | {' / '.join(f'{v:.2f}' for v in engine.PROFILES[cd['device_profile']])} mm |"
                        f"\n| Ensemble uncertainty | {cd['ensemble_uncertainty_D']:.3f} D (abstain above {b['gate_thr']:.3f} D) |"
                        f"\n| Dead zone R_dead per meridian | {' / '.join(f'{v:.2f}' for v in cd['dead_zone_D'])} D (limit {cd['dead_zone_limit_D']:.2f} D) |"
                        f"\n| Confidence | {cd['confidence']:.2f} |"
                        f"\n| Risk score (refer at ≥ {b['refer_t']:.2f}) | {cd['risk_score']:.2f} |"
                        f"\n| Gate reasons | {', '.join(cd['retake_reasons']) or 'none'} |")
        m = b["metrics"]; st.markdown(f"**Held-out test set** ({m['n_eyes_test']} simulated eyes, group-split by child)")
        for col, (k, v) in zip(st.columns(4), [("ROC-AUC", m["roc_auc"]), ("Sensitivity", m["sensitivity"]),
                                               ("Specificity", m["specificity"]), ("MAE of M (D)", m["MAE_D_accepted"]["M"])]):
            col.metric(k, f"{v:.2f}")
        if res:
            pp = m["per_profile"][cd["device_profile"]]
            st.caption(f"{cd['device_profile']} only: {pp['coverage_accepted']:.0%} accepted | ROC-AUC {pp['roc_auc']:.2f} | "
                       f"sensitivity {pp['sensitivity']:.2f} | specificity {pp['specificity']:.2f}")
        st.caption(f"MAE J0 {m['MAE_D_accepted']['J0']:.2f} D, J45 {m['MAE_D_accepted']['J45']:.2f} D | "
                   f"{m['coverage_accepted']:.0%} of captures accepted | {m['retake_rate_corrupted']:.0%} of corrupted captures sent to retake.")

# ------------------------------------------------------------------ Care-Loop Tracker (Screening -> Glasses)
ss = st.session_state
STAGES = {1: "1. Screened", 2: "2. Parent Acknowledged", 3: "3. Optometrist Consult", 4: "4. Glasses Fitted & Closed"}
PILL = {1: ("#FCE4E1", "#9E2A2B"), 2: ("#FDE69A", "#7A5200"), 3: ("#FFF2CC", "#2C2410"), 4: ("#E3F2E6", "#1E5631")}
REASONS, SLA_DAYS, FMT = ["—", "Travel", "Cost", "Unconfirmed"], 7, "%d %b %Y, %H:%M"
PARENT = {  # calm message, extra line when risk is high, urgent badge, routine badge, acknowledgement label
    "English": ("Your child's eye check found a pattern that is often linked to needing glasses. This is common and very treatable, especially when found early, so there is no need to worry. Please book a full eye exam with an eye doctor within 7 days.",
                "Because early treatment matters most at this age, please do not delay the visit.", "URGENT PRIORITY", "Routine follow-up", "Parent acknowledged"),
    "বাংলা": ("আপনার সন্তানের চোখ পরীক্ষায় এমন একটি লক্ষণ পাওয়া গেছে, যা প্রায়ই চশমার প্রয়োজনের সঙ্গে সম্পর্কিত। এটি খুবই সাধারণ এবং আগে ধরা পড়লে সহজেই সমাধান করা যায়, তাই দুশ্চিন্তার কিছু নেই। অনুগ্রহ করে ৭ দিনের মধ্যে একজন চক্ষু চিকিৎসকের কাছে পূর্ণাঙ্গ চোখ পরীক্ষার সময় নিন।",
              "এই বয়সে দ্রুত চিকিৎসা সবচেয়ে কার্যকর, তাই অনুগ্রহ করে দেরি করবেন না।", "জরুরি অগ্রাধিকার", "নিয়মিত ফলো-আপ", "অভিভাবক নিশ্চিত করেছেন")}

def seed_cases():  # four fictional demo children, one per loop stage (ages in months, risk = referral-threshold ratio)
    ago = lambda d: datetime.now() - timedelta(days=d)
    mk = lambda i, child, age, scr, ack, pkt, closed, est, cres, risk, reason=None, signer=None: (
        f"BSK-{i}", dict(id=f"BSK-{i}", child=child, age=age, scr=scr, ack=ack, pkt=pkt, closed=closed, est=est, cres=cres, risk=risk, reason=reason, signer=signer))
    return dict([mk(1001, "Ayesha R., 7 y", 84, ago(9), None, None, None, (0.75, -2.25, 175), [0.41, -0.05, -0.38, 0.02], 1.50, "Unconfirmed"),
                 mk(1002, "Rafi H., 5 y", 60, ago(11), ago(10), None, None, (4.25, -0.50, 90), [0.72, 0.65, 0.70, 0.66], 1.21, "Cost"),
                 mk(1003, "Nusrat J., 9 y", 108, ago(6), ago(5.5), ago(3.5), None, (-1.25, -0.50, 20), [-0.30, -0.12, -0.28, -0.22], 1.17),
                 mk(1004, "Tanvir A., 6 y", 72, ago(10), ago(9), ago(7), ago(4), (1.00, -1.25, 180), [0.18, -0.02, -0.20, 0.01], 0.83, None, "Dr. Demo Optometrist")])

stage = lambda c: 4 if c["closed"] else 3 if c["pkt"] else 2 if c["ack"] else 1
days_open = lambda c: ((c["closed"] or datetime.now()) - c["scr"]).total_seconds() / 86400
pill = lambda t, bg, fg: f'<span style="background:{bg};color:{fg};border-radius:10px;padding:.1rem .55rem;font-weight:700;font-size:.8rem;white-space:nowrap">{t}</span>'

def act(cid, field):  # button callbacks run before the page re-renders, so table and metrics are never stale
    c = ss.cases[cid]; c[field] = datetime.now()
    if field == "closed":
        c["signer"], c["reason"], ss[f"reason_{cid}"] = ss.get(f"sig_{cid}", "").strip(), None, "—"

def set_reason(cid): ss.cases[cid]["reason"] = None if ss[f"reason_{cid}"] == "—" else ss[f"reason_{cid}"]

def log_case(key, P, cd):  # a new REFER capture from the top of the app joins the tracker once and is auto-selected
    if key not in ss.logged:
        cid = f"BSK-{1001 + len(ss.cases)}"
        ss.cases[cid] = dict(id=cid, child=f"Sample: {key.replace('Try Sample: ', '')}", age=int(P["age"][0]), scr=datetime.now(), ack=None, pkt=None,
                             closed=None, est=(cd["pred_sphere_D"], cd["pred_cylinder_D"], cd["pred_axis_deg"]), cres=cd["crescent_array"],
                             risk=cd["risk_score"], reason=None, signer=None)
        ss.logged[key], ss.case_sel = cid, cid
    return ss.logged[key]

def tracker():
    cs = ss.cases
    st.markdown("### Care-Loop Tracker (Screening → Glasses)")
    st.caption("Pre-loaded cases are fictional demo data. New REFER results from above are added automatically.")
    rows = ""                                                                       # 2. visibility, ownership, SLA
    for k, c in cs.items():
        s, d = stage(c), days_open(c)
        sla = pill(f"OVERDUE · {d:.0f} d", "#FDE69A", "#7A5200") if s < 4 and d > SLA_DAYS else ""
        pri = pill("URGENT", "#9E2A2B", "#FFF9E6") if c["risk"] >= 1.0 else pill("Routine", "#FFF2CC", "#2C2410")
        rows += (f"<tr><td>{k}</td><td>{c['child']}</td><td>{pri}</td><td>{pill(STAGES[s], *PILL[s])}</td>"
                 f"<td>{'School Nurse' if s < 3 else 'Pediatric Optometrist'}</td><td>{d:.0f} d {sla}</td></tr>")
    st.markdown('<div class="card" style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:.9rem"><tr><th>Case ID</th>'
                f'<th>Child</th><th>Priority</th><th>Status</th><th>Owner</th><th>Follow-up</th></tr>{rows}</table></div>', unsafe_allow_html=True)
    sel = st.selectbox("Open a case", list(cs), key="case_sel", format_func=lambda k: f"{k} · {cs[k]['child']}")
    c = cs[sel]; s = stage(c); hi = c["risk"] >= 1.0; S, C, ax = c["est"]
    calm, urgent, b_urg, b_rou, ack_lbl = PARENT[st.radio("Language / ভাষা", list(PARENT), horizontal=True, key="lang")]   # 1. parent card
    badge = pill(b_urg, "#9E2A2B", "#FFF9E6") if hi else pill(b_rou, "#FFF2CC", "#2C2410")
    st.markdown(f'<div class="card">{badge}<p style="font-size:1.05rem;margin:.7rem 0 0">{calm}{" " + urgent if hi else ""}</p></div>', unsafe_allow_html=True)
    if c["ack"]: st.markdown(pill(f"✓ {ack_lbl}: {c['ack'].strftime(FMT)}", *PILL[4]), unsafe_allow_html=True)
    else: st.button("Parent Confirmed & Booked Checkup", key=f"ack_{sel}", on_click=act, args=(sel, "ack"))
    st.selectbox("Log drop-off reason (if the family has stalled)", REASONS, index=REASONS.index(c["reason"] or "—"), key=f"reason_{sel}",
                 on_change=set_reason, args=(sel,), disabled=s == 4)
    st.markdown("**Clinician handoff & human-governed closure**")                  # 3. packet + sign-off
    meridians = " &nbsp; ".join(f"{m}° {v:+.2f}" for m, v in zip((0, 45, 90, 135), c["cres"]))
    st.markdown(f'<div class="card"><b>Optometrist Referral Packet · {sel}</b><br>{c["child"]} · screened {c["scr"].strftime(FMT)} · risk score {c["risk"]:.2f}<br>'
                f'Estimate: sphere {S:+.2f} D, cylinder {C:+.2f} D, axis {ax:.0f}°<br>4-meridian crescent summary: {meridians}<br>'
                '<span style="font-size:.85rem">AI screening estimate only - not a diagnosis; a full clinical exam is required.</span></div>', unsafe_allow_html=True)
    iso = lambda t: t.isoformat(timespec="minutes") if t else None
    with st.expander("View packet as JSON"):
        st.json(dict(case_id=sel, child=c["child"], age_months=c["age"], priority="URGENT" if hi else "ROUTINE", risk_score=round(c["risk"], 2),
                     estimate=dict(sphere_D=S, cylinder_D=C, axis_deg=ax), crescent_summary=dict(zip(["0", "45", "90", "135"], c["cres"])),
                     screened_at=iso(c["scr"]), parent_acknowledged_at=iso(c["ack"]), packet_sent_at=iso(c["pkt"])))
    st.button("Send Referral Packet to Optometrist", key=f"pkt_{sel}", on_click=act, args=(sel, "pkt"), disabled=not c["ack"] or bool(c["pkt"]))
    st.caption("Waiting for the parent to confirm before the packet can be sent." if not c["ack"] else f"Packet sent: {iso(c['pkt']) or 'not yet'}.")
    st.text_input("Optometrist name / license no. (required for sign-off)", key=f"sig_{sel}", disabled=s != 3)
    st.button("Authorized Optometrist Sign-Off (Glasses Dispensed)", key=f"cls_{sel}", on_click=act, args=(sel, "closed"),
              disabled=s != 3 or not ss.get(f"sig_{sel}", "").strip())
    if s == 4: st.markdown(pill(f"✓ Closed {c['closed'].strftime(FMT)} · signed off by {c['signer']}", *PILL[4]), unsafe_allow_html=True)
    n, acked = len(cs), sum(bool(x["ack"]) for x in cs.values())                   # 4. loop-closure impact metrics
    tt = [(x["pkt"] - x["scr"]).total_seconds() / 86400 for x in cs.values() if x["pkt"]]
    why = Counter(x["reason"] for x in cs.values() if x["reason"])
    tiles = [("Parent Acknowledgement Rate", f"{acked / n:.0%}", f"{acked} of {n} families"),
             ("Median Time-to-Consult", f"{median(tt):.1f} days" if tt else "—", "screening → referral to optometrist"),
             ("Logged Drop-Off Reasons", " · ".join(f"{k} {v}" for k, v in why.items()) or "None logged", "open cases")]
    st.markdown('<div style="display:flex;gap:.6rem;flex-wrap:wrap">' + "".join(
        f'<div class="card" style="flex:1;min-width:170px;text-align:center;margin:.2rem 0"><div style="font-size:1.3rem;font-weight:800">{v}</div>'
        f'<div style="font-weight:700">{k}</div><div style="font-size:.8rem">{sub}</div></div>' for k, v, sub in tiles) + "</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ page
bundle = load_bundle(); samples = build_samples(bundle)
ss.setdefault("cases", seed_cases()); ss.setdefault("logged", {})
ss.setdefault("active", None); ss.setdefault("seen", None)
st.markdown("<h1 style='text-align:center;margin-bottom:0'>👁️ BrightSight Kids</h1>"
            "<p style='text-align:center;font-size:1.1rem;margin-top:.2rem'>Quick, painless smartphone eye check for children.</p>",
            unsafe_allow_html=True)
photo = st.camera_input("Take a photo of your child's eyes at arm's length (1 meter)")
if photo is not None:
    pid = hashlib.md5(photo.getvalue()).hexdigest()
    if pid != ss.seen:
        ss.seen, ss.active = pid, "photo"
elif ss.active == "photo":
    ss.active = None
st.markdown("**No camera handy? Tap a sample child:**")
for col, label in zip(st.columns(2) + st.columns(2), SAMPLE_RULES):
    if col.button(label, key=label, disabled=label not in samples):
        ss.active = label

drawer = None
if ss.active == "photo" and photo is not None:
    g, s, br = q = photo_quality(photo)
    reason = "glare" if g > 0.05 else "dark" if br < 45 else "blur" if s < 0.35 else None
    result_card("RETAKE" if reason else "OK", reason)
    drawer = dict(quality=q)
elif ss.active in samples:
    P, truth = samples[ss.active]; res = engine.predict_eye(child_features(P, ss.get("profile", engine.PNAMES[0])), bundle)
    rs = res["clinical_details"]["retake_reasons"]
    result_card(res["status"], rs[0] if rs else None, '<p style="margin:.7rem 0 0;font-size:.85rem">Sample result from a simulated child.</p>')
    if res["status"] == "REFER":
        st.caption(f"Added to the Care-Loop Tracker below as {log_case(ss.active, P, res['clinical_details'])}.")
    drawer = dict(res=res, truth=truth)
else:
    st.markdown('<div class="card">Take a photo or tap a sample above. Your result will appear here in a moment.</div>', unsafe_allow_html=True)
tracker()
if drawer:
    doctor_drawer(bundle, **drawer)
st.caption("BrightSight Kids is a screening aid, not a diagnosis. This prototype is trained and tested on simulated data only.")
