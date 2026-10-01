"""Scoring. Two paths: trained model (models/model.joblib) or a transparent, rule-based DEMO scorer."""
import os, math
import numpy as np
from .features import ALL, GROUPS, vector

MODEL_PATH = os.environ.get("PG_MODEL", os.path.join(os.path.dirname(__file__), "../../models/model.joblib"))
ANOM_PATH = os.environ.get("PG_ANOMALY", os.path.join(os.path.dirname(__file__), "../../models/anomaly.joblib"))
LIMITS = ("Probabilistic risk estimate from observable signals, not proof of identity or intent. "
          "Humans can post regularly, use templates or write generic bios; AI-assisted writing is hard to tell apart from human writing.")

def ramp(x, lo, hi):
    """Linear 0-100 map; works for lo>hi (inverse)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return float(np.clip((x - lo) / (hi - lo), 0, 1) * 100)

# (feature, value at risk 0, value at risk 100, human-readable signal)
RULES = {
    "linguistic": [("txt_sentlen_cv", .8, .2, "Very uniform sentence lengths"),
                   ("txt_repeat_trigram", 0, .15, "Frequently repeated 3-word phrases"),
                   ("txt_len_cv", .8, .1, "Posts are almost identical in length")],
    "content_similarity": [("sim_mean", .1, .6, "High semantic similarity between posts"),
                           ("sim_near_dup_frac", 0, .2, "Many near-duplicate post pairs"),
                           ("sim_template_frac", 0, .4, "Template-like post structure")],
    "behavioural": [("beh_interval_cv", 1.5, .1, "Highly regular posting intervals"),
                    ("beh_top_minute_frac", .1, .5, "Posts cluster on the same minute of the hour"),
                    ("beh_hour_entropy", 3.5, 1.5, "Activity concentrated in few hours"),
                    ("beh_burst_frac", 0, .5, "Large share of burst posting"),
                    ("beh_per_day", 5, 60, "Very high posting volume")],
    "profile": [("prof_username_digit_frac", .2, .6, "Digit-heavy username (weak signal)"),
                ("prof_completeness", .8, .2, "Sparse profile (weak signal)"),
                ("prof_posts_per_age_day", 5, 60, "Very high posts per day of account age")],
    "network": [("net_repeat_frac", .2, .8, "Repeated interactions with the same accounts"),
                ("net_sync_frac", .05, .5, "Synchronised interaction timing"),
                ("net_reciprocity", .3, .9, "Unusually high reciprocity")],
}
WEIGHTS = {"linguistic": .15, "content_similarity": .2, "behavioural": .3, "profile": .1, "network": .25}

def anomaly_score(vec):
    """Isolation Forest fitted on a human reference set (see train.py). None if no reference exists."""
    if not os.path.exists(ANOM_PATH):
        return None
    import joblib
    m = joblib.load(ANOM_PATH)
    x = m["imputer"].transform(vec.reshape(1, -1))
    raw = -m["iso"].score_samples(x)[0]  # higher = more anomalous
    lo, hi = m["range"]
    return float(np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1) * 100)

def demo_score(feats, n_posts):
    comp, signals = {}, []
    for c, rules in RULES.items():
        parts = [(ramp(feats[f], lo, hi), msg) for f, lo, hi, msg in rules]
        parts = [(s, m) for s, m in parts if s is not None]
        comp[c] = float(np.mean([s for s, _ in parts])) if parts else None
        signals += [(s, m) for s, m in parts if s >= 60]
    comp["anomaly"] = anomaly_score(vector(feats))
    avail = {c: v for c, v in comp.items() if v is not None and c in WEIGHTS}
    wsum = sum(WEIGHTS[c] for c in avail)
    risk = sum(WEIGHTS[c] * v for c, v in avail.items()) / wsum if wsum else None
    coverage = wsum / sum(WEIGHTS.values())
    spread = float(np.std(list(avail.values()))) if len(avail) > 1 else 50.0
    conf = float(np.clip(coverage * (1 - spread / 60) * min(1, n_posts / 30), 0, 1))
    if risk is None or n_posts < 5 or len(avail) < 2 or conf < .25:
        cls, reason = "INCONCLUSIVE", "Available evidence is insufficient to reliably classify this account."
    else:
        cls = "HUMAN_LIKE" if risk < 35 else "AI_ASSISTED" if risk < 60 else "SYNTHETIC_AUTOMATED"
        reason = None
    return dict(classification=cls, risk_score=None if risk is None else round(risk),
                confidence=round(conf, 2), component_scores={k: None if v is None else round(v) for k, v in comp.items()},
                top_signals=[m for _, m in sorted(signals, reverse=True)[:5]], note=reason,
                status="heuristic_demo")

_MODEL = None
def load_model():
    global _MODEL
    if _MODEL is None and os.path.exists(MODEL_PATH):
        import joblib
        _MODEL = joblib.load(MODEL_PATH)
    return _MODEL

def model_score(feats, n_posts):
    M = load_model()
    cols = [ALL.index(c) for c in M["columns"]]
    x = vector(feats)[cols].reshape(1, -1)
    p = M["pipeline"].predict_proba(x)[0]
    classes = list(M["pipeline"].classes_)
    prob = dict(zip(classes, map(float, p)))
    risk = 100 * (0.5 * prob.get("ai_assisted", 0) + prob.get("synthetic", 0))
    conf = max(prob.values())
    # model-agnostic occlusion attribution: replace one feature by its training median, watch risk change
    def r(v):
        q = dict(zip(classes, M["pipeline"].predict_proba(v)[0]))
        return 100 * (0.5 * q.get("ai_assisted", 0) + q.get("synthetic", 0))
    contrib = []
    for j, name in zip(cols, M["columns"]):
        if math.isnan(x[0, cols.index(j)]):
            continue
        y = x.copy(); y[0, cols.index(j)] = M["medians"][name]
        contrib.append((risk - r(y), name))
    contrib.sort(key=lambda t: -abs(t[0]))
    cls = max(prob, key=prob.get)
    if conf < .5 or n_posts < 5:
        cls = "INCONCLUSIVE"
    comp = {g: None for g in ["linguistic", "behavioural", "profile", "network", "content_similarity", "anomaly"]}
    return dict(classification=cls.upper() if cls != "INCONCLUSIVE" else cls, risk_score=round(risk),
                confidence=round(conf, 2), class_probabilities=prob, component_scores=comp,
                top_signals=[f"{n} ({d:+.1f} risk pts)" for d, n in contrib[:5]],
                note="Per-feature attribution via occlusion against training medians.", status="model_prediction")

def analyze(feats, n_posts):
    return model_score(feats, n_posts) if load_model() else demo_score(feats, n_posts)
