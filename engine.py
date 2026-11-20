#!/usr/bin/env python3
"""engine.py - Pediatric smartphone eccentric-photorefraction screener.
Physics simulator -> leakage-free ensemble -> uncertainty/abstain gate -> predict_eye() API.
Run `python engine.py` to simulate, train, evaluate and save screener_model.joblib.
Requires: numpy, scikit-learn, joblib."""
import json, warnings, functools
import numpy as np, joblib
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.metrics import roc_auc_score, confusion_matrix
warnings.filterwarnings("ignore")

SEED, N_SUBJ, N_MEMB = 7, 2500, 6
MERID = np.array([0., 45., 90., 135.])        # OLED/flash meridians (deg)
PROFILES = {"iPhone Front OLED Profile": np.array([4.0, 5.66, 4.0, 5.66]),    # source eccentricity e per meridian (mm);
            "Standard Rear Flash Profile": np.array([6.0, 8.49, 6.0, 8.49])}  # illustrative values - calibrate per device
PNAMES, PECC = list(PROFILES), np.array(list(PROFILES.values()))
D_MIN, D_MAX = 0.85, 1.15                     # accepted working distance (m) = range the model was trained on
PARAM_KEYS = ["S", "C", "ax", "A", "d", "Dp", "dt", "age", "kind", "glare", "blur"]   # one eye + one capture
CRES_KEYS = ["c0", "c45", "c90", "c135"]      # signed crescent fraction of pupil, per meridian
RAW_KEYS = CRES_KEYS + ["pupil_mm", "distance_m", "latency_ms", "age_months", "glare_frac", "sharpness", "profile_id"]
PUPIL_MIN, GLARE_MAX, TARGET_SENS = 3.6, 0.02, 0.80  # hard optical rules; sensitivity target for REFER cut-off
NOMINAL_D, DP_BEST, FLAT = 1.0, 6.5, 0.08     # nominal distance (m); pupil a dim room can reach (mm); |crescent| counted as flat
MODEL_PATH = "screener_model.joblib"

# ---------------------------------------------------------------- optics
def power_vectors(S, C, ax):                  # Thibos M, J0, J45 (minus-cylinder notation)
    a = np.deg2rad(ax)
    return S + C / 2, -(C / 2) * np.cos(2 * a), -(C / 2) * np.sin(2 * a)

def directional_R(M, J0, J45, phi):           # R(phi) = M + J0 cos2phi + J45 sin2phi
    p = np.deg2rad(phi)
    return M[:, None] + J0[:, None] * np.cos(2 * p) + J45[:, None] * np.sin(2 * p)

def decode(mu):                               # (M,J0,J45) -> sphere, cylinder (<=0), axis (0-180)
    amp = np.hypot(mu[..., 1], mu[..., 2])
    return mu[..., 0] + amp, -2 * amp, (0.5 * np.rad2deg(np.arctan2(mu[..., 2], mu[..., 1]))) % 180

def thresholds(age):                          # age-banded (hyperopia, myopia, astigmatism) cut-offs [D]
    pick = lambda x, y, z: np.where(age < 31, x, np.where(age < 49, y, z))
    return pick(4.5, 4.0, 3.5), pick(3.5, 3.0, 1.5), pick(2.0, 2.0, 1.5)

def risk_score(S, C, age):
    """Max over criteria of (value / threshold); >=1 means amblyopia-risk refractive error.
    Simplified AAPOS-style cut-offs - verify vs current guidelines."""
    hy, my, asg = thresholds(age)
    return np.maximum.reduce([S / hy, -(S + C) / my, -C / asg])

def ecc_rows(raw):                            # per-row source eccentricities (mm) of the capture's device profile
    return PECC[raw[:, 10].astype(int)]

def dead_zone(raw):                           # R_dead = e/(d*Dp) per meridian, and smallest referral cut-off [D]
    _, my, asg = thresholds(raw[:, 7])
    return ecc_rows(raw) / (raw[:, 5] * raw[:, 4])[:, None], np.minimum(my, asg)

def hard_flags(raw):
    """Rule-based capture checks (vectorised over rows of raw) -> {reason: bool array}, in priority order."""
    dead, lim = dead_zone(raw); mean, e = dead.mean(1), ecc_rows(raw).mean(1)
    wide = mean >= lim                                         # blind zone wider than the smallest referral cut-off
    fixable = e / (NOMINAL_D * raw[:, 4]) < lim       # ...and a capture at the nominal distance would be fine
    flat = ((np.abs(raw[:, :4]) <= FLAT).all(1) & ~wide & (2 * mean >= lim)   # all 4 crescents inside the dead zone,
            & (mean > e / (NOMINAL_D * DP_BEST)))                    # and a dimmer room could narrow it
    return {"glare": raw[:, 8] > GLARE_MAX, "small_pupil": raw[:, 4] < PUPIL_MIN,
            "bad_distance": (raw[:, 5] < D_MIN) | (raw[:, 5] > D_MAX),      # outside the 0.85-1.15 m window
            "too_close": wide & fixable, "dead_zone": wide & ~fixable, "no_signal": flat}

def accepted(raw, u, thr):                    # True where no hard rule fires and the ensemble is confident
    return ~np.any(list(hard_flags(raw).values()), axis=0) & (u <= thr)

def featurize(raw):                           # raw (n,11) in RAW_KEYS order -> + dead zone per meridian
    dead = ecc_rows(raw) / (raw[:, 5:6] * raw[:, 4:5])    # R_dead = e / (d * Dp)  [D]
    c = raw[:, :4]                              # meridian contrasts ~ J0 / J45 / mean signal
    return np.hstack([raw, dead, c[:, :1] - c[:, 2:3], c[:, 1:2] - c[:, 3:4], c.mean(1, keepdims=True)])

# ---------------------------------------------------------------- simulator
def capture(P, prof, rng):
    """Photorefraction core: eye/capture parameters P + device profile ids -> raw feature rows and R_dead."""
    M, J0, J45 = power_vectors(P["S"], P["C"], P["ax"]); d, Dp, blur = P["d"], P["Dp"], P["blur"]
    Dfc = directional_R(M, J0, J45, MERID) - P["A"][:, None] + (1 / d)[:, None]    # net defocus seen by the camera
    dead = PECC[prof] / (d * Dp)[:, None]                                          # R_dead = e/(d*Dp) per meridian
    f = np.sign(Dfc) * np.clip(1 - dead / np.maximum(np.abs(Dfc), 1e-6), 0, 1)     # inside dead zone -> 0
    f = f * (1 - 0.45 * blur[:, None]) + rng.normal(0, 1, f.shape) * (0.025 + 0.12 * blur[:, None])
    f = np.clip(np.where((P["kind"] == 1)[:, None], f + rng.normal(0, 0.4, f.shape), f), -1, 1)
    sharp = np.clip(1 - 0.8 * blur + rng.normal(0, 0.03, len(d)), 0, 1)
    return np.column_stack([f, Dp, d, P["dt"], P["age"], P["glare"], sharp, prof]), dead

def simulate(rng, n=N_SUBJ):
    sid, eye, m = np.repeat(np.arange(n), 2), np.tile([0, 1], n), 2 * n
    age_s = rng.uniform(12, 144, n)
    base = np.where(rng.random(n) < 0.15, rng.normal(0, 2.8, n),
                    rng.normal(0.9 + 1.2 * np.exp(-age_s / 60), 1.0, n))
    aniso = np.where(rng.random(n) < 0.10, rng.normal(0, 1.2, n), 0.0)
    S = np.clip(base[sid] + rng.normal(0, 0.35, m) + aniso[sid] * eye, -6, 6)
    C = np.clip(-np.minimum(4, rng.gamma(1.3, 0.55, n))[sid] + rng.normal(0, 0.2, m), -4, 0)
    u = rng.random(n)
    ax_od = np.where(u < .45, rng.normal(180, 10, n), np.where(u < .75, rng.normal(90, 10, n), rng.uniform(0, 180, n))) % 180
    ax = np.where(eye == 0, ax_od[sid], (180 - ax_od[sid] + rng.normal(0, 8, m)) % 180)  # mirror-symmetric OS
    age, d = age_s[sid], np.clip(rng.normal(1.0, 0.04, m), D_MIN, D_MAX)
    Dp = np.clip(rng.normal(5.6 + 0.3 * np.exp(-age_s / 60), 1.0, n)[sid] + rng.normal(0, 0.15, m), 3.0, 7.5)
    dt = np.clip(np.exp(rng.normal(np.log(200), 0.45, m)), 50, 450); M, J0, J45 = power_vectors(S, C, ax)
    A_max = np.clip(np.abs(rng.normal(0.5, 0.3, m)) * (1.3 - age / 240) + 0.1 * np.maximum(M, 0), 0, 2.0)
    A = A_max * (1 - np.exp(-dt / 120))     # residual accommodation: stronger if young/hyperopic, builds with latency
    kind = np.where(rng.random(m) < 0.15, rng.integers(1, 4, m), 0)   # 15% corrupted: 1 glare, 2 small pupil, 3 blur
    Dp = np.where(kind == 2, rng.uniform(3.0, 3.6, m), Dp)
    glare = np.where(kind == 1, rng.uniform(0.04, 0.30, m), rng.uniform(0, 0.006, m))
    blur = np.where(kind == 3, rng.uniform(0.4, 1.0, m), rng.uniform(0, 0.1, m))
    P = dict(S=S, C=C, ax=ax, A=A, d=d, Dp=Dp, dt=dt, age=age, kind=kind, glare=glare, blur=blur)
    raw, dead = capture(P, rng.integers(0, len(PNAMES), m), rng)     # each eye captured on a random device profile
    return dict(sid=sid, eye=eye, Y=np.column_stack([M, J0, J45]), raw=raw, f=raw[:, :4], dead=dead, **P)

def render(f, dp, glare, blur, rng, chunk=250):
    """128x128 uint8 mosaic: four 64x64 pupil tiles (0,45 / 90,135 deg) showing crescents, glint, glare, blur."""
    ax = np.arange(-32, 32, dtype=np.float32) + 0.5
    X, Y = np.meshgrid(ax, ax); rad = np.hypot(X, Y)
    ph = np.deg2rad(MERID).astype(np.float32)
    PROJ = X[None] * np.cos(ph)[:, None, None] + Y[None] * np.sin(ph)[:, None, None]
    sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    out = np.empty((len(f), 128, 128), np.uint8)
    for s in range(0, len(f), chunk):
        sl = slice(s, s + chunk)
        fc = f[sl].astype(np.float32)[:, :, None, None]
        r = (dp[sl] / 7.5 * 28).astype(np.float32)[:, None, None, None]
        g = glare[sl].astype(np.float32)[:, None, None, None]
        n = len(fc)
        cres = sig((np.where(fc >= 0, 1, -1) * PROJ[None] - r * (1 - 2 * np.abs(fc))) / 0.9)
        img = 0.08 + sig((r - rad) / 0.9) * (0.22 + 0.6 * cres) + 0.7 * np.exp(-rad ** 2 / 2)  # iris+reflex+glint
        gx, gy = (rng.uniform(-.4, .4, (n, 1, 1, 1)) * r).astype(np.float32), (rng.uniform(-.4, .4, (n, 1, 1, 1)) * r).astype(np.float32)
        blob = sig((r * np.sqrt(g) - np.hypot(X - gx, Y - gy)) / 0.8) * (g > GLARE_MAX)
        img = np.maximum(img, blob) + rng.normal(0, 0.015, img.shape).astype(np.float32)
        b = blur[sl] > 0.4                                         # horizontal motion smear
        if b.any():
            img[b] = np.mean([np.roll(img[b], k, axis=3) for k in range(-3, 4)], axis=0)
        mos = np.concatenate([np.concatenate([img[:, 0], img[:, 1]], 2), np.concatenate([img[:, 2], img[:, 3]], 2)], 1)
        out[sl] = (np.clip(mos, 0, 1) * 255).astype(np.uint8)
    return out

# ---------------------------------------------------------------- ensemble + gate
def fit_ensemble(X, Y, rng):                    # bagged HGB: N_MEMB bootstrap members per target
    return [[HGB(max_iter=200, learning_rate=0.06, max_depth=5, min_samples_leaf=30, l2_regularization=1.0, random_state=int(rng.integers(1e9)))
             .fit(X[i], Y[i, j]) for i in [rng.integers(0, len(X), len(X)) for _ in range(N_MEMB)]]
            for j in range(3)]

def ens_predict(models, X):
    P = np.array([[mm.predict(X) for mm in row] for row in models])   # (3, members, n)
    return P.mean(1).T, P.std(1).T

def uncertainty(sd):
    return np.sqrt((sd ** 2).sum(-1))

def choose_refer_t(models, thr, D, mask):       # highest cut-off on val that still reaches TARGET_SENS
    mu, sd = ens_predict(models, featurize(D["raw"][mask])); S, C, _ = decode(mu)
    acc = accepted(D["raw"][mask], uncertainty(sd), thr)
    y = (risk_score(D["S"][mask], D["C"][mask], D["age"][mask]) >= 1.0)[acc]; ps = risk_score(S, C, D["age"][mask])[acc]
    return next((float(t) for t in np.arange(0.95, 0.2, -0.05) if (ps[y] >= t).mean() >= TARGET_SENS), 0.2)

def clf_metrics(y, ps, acc, refer_t):
    tn, fp, fn, tp = confusion_matrix(y[acc], ps[acc] >= refer_t, labels=[False, True]).ravel()
    return {"coverage_accepted": round(float(acc.mean()), 4), "sensitivity": round(float(tp / (tp + fn)), 4),
            "specificity": round(float(tn / (tn + fp)), 4), "roc_auc": round(float(roc_auc_score(y[acc], ps[acc])), 4)}

def evaluate(models, thr, D, mask, refer_t):
    raw = D["raw"][mask]; mu, sd = ens_predict(models, featurize(raw)); u = uncertainty(sd)
    S, C, _ = decode(mu); age, kind, Yt = D["age"][mask], D["kind"][mask], D["Y"][mask]
    acc, fl = accepted(raw, u, thr), hard_flags(raw)
    dz = fl["too_close"] | fl["dead_zone"] | fl["no_signal"]                 # dead-zone rules only
    y = risk_score(D["S"][mask], D["C"][mask], age) >= 1.0; ps = risk_score(S, C, age); prof = raw[:, 10].astype(int)
    mae = lambda k: {n: round(float(np.abs(mu[k, i] - Yt[k, i]).mean()), 4) for i, n in enumerate(["M", "J0", "J45"])}
    return {"n_eyes_test": int(len(u)), "refer_cutoff": round(refer_t, 2), "referral_prevalence": round(float(y.mean()), 4),
            "retake_rate_clean": round(float((~acc[kind == 0]).mean()), 4),
            "retake_rate_dead_zone_rules_clean": round(float(dz[kind == 0].mean()), 4),
            "retake_rate_corrupted": round(float((~acc[kind > 0]).mean()), 4),
            "MAE_D_accepted": mae(acc), "MAE_D_all_no_gate": mae(np.ones(len(u), bool)),
            "MAE_sphere_D_accepted": round(float(np.abs(S[acc] - D["S"][mask][acc]).mean()), 4),
            "MAE_cylinder_D_accepted": round(float(np.abs(C[acc] - D["C"][mask][acc]).mean()), 4),
            **clf_metrics(y, ps, acc, refer_t),
            "per_profile": {n: clf_metrics(y[prof == i], ps[prof == i], acc[prof == i], refer_t) for i, n in enumerate(PNAMES)}}

# ---------------------------------------------------------------- inference API
MSG = {
    "pass": ("Vision check looks good", "Your child's scan did not show a pattern that needs follow-up right now. This is a quick screening, not an eye exam - see an eye doctor sooner if your child squints or sits very close to screens."),
    "refer": ("Please book an eye exam", "The scan found a pattern that is sometimes linked to needing glasses. This does not mean anything is seriously wrong - please book a full eye exam in the next few weeks."),
    "glare": ("Too much glare - let's try again", "A bright reflection got in the way. Move away from windows and bright lights, then take the picture again."),
    "small_pupil": ("The room is too bright - let's try again", "The child's pupils are too small for a clear reading. Dim the lights, let the child look at a far wall for 30 seconds, then retake."),
    "bad_distance": ("Hold the phone about 1 meter away", "The phone is too close or too far for a reliable reading. Hold it about 1 meter (arm's length) from your child's face, then take the picture again."),
    "too_close": ("Move the phone back a little", "The phone is too close for this reading to be precise. Hold it at arm's length, about 1 meter from your child's face, and retake."),
    "dead_zone": ("Let's make the room a bit dimmer", "The child's pupils are too small for the reading to be precise. Dim the room lights, let the child look at a far wall for 30 seconds, then retake."),
    "no_signal": ("We need a clearer reading", "The picture did not show enough detail to tell whether glasses might help. Dim the lights slightly so the pupils open wider, keep the phone about 1 meter away, and retake. Please do not treat this as a pass."),
    "blur": ("The picture was blurry - let's try again", "The child or phone moved. Ask the child to keep still, rest your elbows on a table, and retake the picture."),
    "uncertain": ("We couldn't get a clear reading", "The scan was not reliable enough to give an answer. Make sure the room is evenly lit, the phone is about 1 meter away and both eyes are open, then retake. Please do not treat this as a pass.")}

@functools.cache
def _bundle():
    return joblib.load(MODEL_PATH)

def predict_eye(features_dict, bundle=None):
    """features_dict keys: c0,c45,c90,c135,pupil_mm,distance_m,latency_ms,age_months,glare_frac,sharpness,
    plus optional device_profile (a key of PROFILES; default iPhone Front OLED)."""
    b, F = bundle or _bundle(), dict(features_dict)
    if F.get("device_profile", PNAMES[0]) not in PROFILES: raise ValueError(f"device_profile must be one of {PNAMES}")
    F.setdefault("profile_id", PNAMES.index(F.get("device_profile", PNAMES[0])))
    miss = [k for k in RAW_KEYS if k not in F]
    if miss:
        raise ValueError(f"missing feature keys: {miss}")
    raw = np.array([[float(F[k]) for k in RAW_KEYS]])
    mu, sd = ens_predict(b["models"], featurize(raw)); mu, u = mu[0], float(uncertainty(sd[0]))
    flags, (dead, lim) = hard_flags(raw), dead_zone(raw)
    reasons = [k for k, v in flags.items() if v[0]]            # glare, pupil, dead-zone rules (priority order)
    if u > b["gate_thr"]: reasons.append("blur" if F["sharpness"] < 0.6 else "uncertain")
    S, C, ax = (float(v) for v in decode(mu))
    score = float(risk_score(S, C, F["age_months"]))
    status = "RETAKE" if reasons else ("REFER" if score >= b["refer_t"] else "PASS")
    title, msg = MSG[reasons[0] if reasons else status.lower()]
    cres = np.array([F[k] for k in CRES_KEYS], float)
    visual = render(cres[None], np.array([F["pupil_mm"]]), np.array([F["glare_frac"]]),
                    np.array([1.0 - F["sharpness"]]), np.random.default_rng(0))[0]
    return {"status": status, "simple_title": title, "simple_message": msg,
            "clinical_details": {
                "pred_sphere_D": None if status == "RETAKE" else round(S, 2),
                "pred_cylinder_D": None if status == "RETAKE" else round(C, 2),
                "pred_axis_deg": None if status == "RETAKE" else round(ax, 1),
                "confidence": 0.0 if any(r in flags for r in reasons) else round(1 / (1 + (u / b["gate_thr"]) ** 2), 3),
                "ensemble_uncertainty_D": round(u, 3), "dead_zone_D": dead[0].round(2).tolist(),
                "dead_zone_limit_D": round(float(lim[0]), 2), "risk_score": round(score, 3),
                "retake_reasons": reasons, "device_profile": PNAMES[int(F["profile_id"])], "meridians_deg": MERID.tolist(),
                "crescent_array": cres.round(3).tolist(),     # signed crescent fraction per meridian
                "crescent_visual_128": visual}}               # uint8 128x128 mosaic (0/45 top, 90/135 bottom)

# ---------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(SEED)
    D = simulate(rng)
    imgs = render(D["f"], D["Dp"], D["glare"], D["blur"], rng)
    print(f"{len(D['sid'])} eyes / {N_SUBJ} subjects | images {imgs.shape} | corrupted {np.mean(D['kind'] > 0):.1%}")
    perm = rng.permutation(N_SUBJ); a, b_ = int(.6 * N_SUBJ), int(.8 * N_SUBJ); role = np.empty(N_SUBJ, int)
    role[perm[:a]], role[perm[a:b_]], role[perm[b_:]] = 0, 1, 2
    tr, va, te = (role[D["sid"]] == k for k in (0, 1, 2))
    for p, q in ((tr, va), (tr, te), (va, te)):              # group split: no subject (L/R eye) shared
        assert not np.intersect1d(D["sid"][p], D["sid"][q]).size
    models = fit_ensemble(featurize(D["raw"][tr]), D["Y"][tr], rng)
    sd_v = ens_predict(models, featurize(D["raw"][va]))[1]
    thr = float(np.quantile(uncertainty(sd_v)[D["kind"][va] == 0], 0.90))  # abstain above 90th pct of clean-val uncertainty
    refer_t = choose_refer_t(models, thr, D, va)
    metrics = evaluate(models, thr, D, te, refer_t)
    bundle = {"models": models, "gate_thr": thr, "refer_t": refer_t, "meridians": MERID, "profiles": PROFILES, "raw_keys": RAW_KEYS, "metrics": metrics}
    joblib.dump(bundle, MODEL_PATH); json.dump(metrics, open("test_metrics.json", "w"), indent=2)
    np.savez_compressed("sample_crescents.npz", images=imgs[:256], features=D["raw"][:256], kind=D["kind"][:256])
    print(f"gate threshold = {thr:.3f} D\n" + json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
