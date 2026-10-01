"""Feature extraction. Every value is computed from submitted data; missing signals -> NaN."""
import math, re
from collections import Counter
from datetime import datetime, timezone
import numpy as np
from .schemas import Account

NAN = float("nan")
EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
_ST = None

def _dt(s):
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:
        return None

def embed(texts):
    """Transformer embeddings if sentence-transformers is installed, else TF-IDF (documented fallback)."""
    global _ST
    try:
        if _ST is None:
            from sentence_transformers import SentenceTransformer
            _ST = SentenceTransformer("all-MiniLM-L6-v2")
        return _ST.encode(texts, normalize_embeddings=True), "sentence-transformers/all-MiniLM-L6-v2"
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        m = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(texts)
        return m.toarray(), "tfidf-fallback"

def text_features(texts):
    texts = [t for t in texts if t and t.strip()]
    if len(texts) < 3:
        return {}, "none"
    joined = " ".join(texts).lower()
    toks = re.findall(r"\w+", joined)
    sents = [len(re.findall(r"\w+", s)) for t in texts for s in re.split(r"[.!?]+", t) if s.strip()]
    chars = Counter(joined)
    n = sum(chars.values())
    ent = -sum(c / n * math.log2(c / n) for c in chars.values())
    tri = Counter(zip(toks, toks[1:], toks[2:]))
    rep = sum(c for c in tri.values() if c > 1) / max(1, sum(tri.values()))
    E, backend = embed(texts)
    S = E @ E.T
    iu = np.triu_indices(len(texts), 1)
    sims = S[iu]
    lens = np.array([len(t) for t in texts], float)
    return {
        "txt_ttr": len(set(toks)) / max(1, len(toks)),
        "txt_sentlen_cv": float(np.std(sents) / (np.mean(sents) + 1e-9)) if sents else NAN,
        "txt_punct_rate": sum(ch in ".,;:!?-" for ch in joined) / max(1, len(joined)),
        "txt_emoji_rate": len(EMOJI.findall(joined)) / max(1, len(texts)),
        "txt_char_entropy": ent,
        "txt_repeat_trigram": rep,
        "txt_len_cv": float(np.std(lens) / (np.mean(lens) + 1e-9)),
        "sim_mean": float(sims.mean()),
        "sim_max": float(sims.max()),
        "sim_near_dup_frac": float((sims > 0.9).mean()),
        "sim_template_frac": float((sims > 0.6).mean()),
    }, backend

def behaviour_features(posts):
    ts = sorted(d for d in (_dt(p.timestamp) for p in posts if p.timestamp) if d)
    if len(ts) < 5:
        return {}
    iv = np.diff([t.timestamp() for t in ts])
    hrs = np.array([t.hour for t in ts])
    hist = np.bincount(hrs, minlength=24) / len(hrs)
    h_ent = -sum(p * math.log2(p) for p in hist if p > 0)
    span_days = max((ts[-1] - ts[0]).total_seconds() / 86400, 1e-3)
    mins = Counter(t.minute for t in ts)
    half = len(iv) // 2
    m1, m2 = np.median(iv[:half]) + 1e-9, np.median(iv[half:]) + 1e-9
    return {
        "beh_posts": float(len(ts)),
        "beh_per_day": len(ts) / span_days,
        "beh_interval_cv": float(np.std(iv) / (np.mean(iv) + 1e-9)),
        "beh_hour_entropy": h_ent,
        "beh_night_frac": float(((hrs < 6)).mean()),
        "beh_burst_frac": float((iv < 60).mean()),
        "beh_top_minute_frac": mins.most_common(1)[0][1] / len(ts),
        "beh_shift_ratio": float(abs(math.log(m2 / m1))),
    }

def profile_features(a: Account):
    f = {}
    if a.followers is not None and a.following is not None:
        f["prof_log_ff_ratio"] = math.log((a.followers + 1) / (a.following + 1))
    if a.created_at and _dt(a.created_at):
        age = (datetime.now(timezone.utc) - _dt(a.created_at)).days
        f["prof_age_days"] = float(max(age, 0))
        if f["prof_age_days"] > 0 and a.posts:
            f["prof_posts_per_age_day"] = len(a.posts) / f["prof_age_days"]
    f["prof_completeness"] = sum([bool(a.bio), bool(a.username), a.created_at is not None,
                                  a.followers is not None, a.following is not None]) / 5
    if a.username:
        f["prof_username_digit_frac"] = sum(c.isdigit() for c in a.username) / len(a.username)
    f["prof_bio_len"] = float(len(a.bio))
    return f

def network_features(a: Account):
    import networkx as nx
    if len(a.interactions) < 5:
        return {}
    G = nx.DiGraph()
    pairs = Counter((i.source, i.target) for i in a.interactions)
    for (s, t), c in pairs.items():
        G.add_edge(s, t, w=c)
    U = G.to_undirected()
    ts = sorted(d.timestamp() for d in (_dt(i.timestamp) for i in a.interactions if i.timestamp) if d)
    sync = float((np.diff(ts) < 5).mean()) if len(ts) > 4 else NAN
    return {
        "net_reciprocity": nx.reciprocity(G) or 0.0,
        "net_clustering": nx.average_clustering(U),
        "net_repeat_frac": 1 - len(pairs) / len(a.interactions),
        "net_partners": float(U.number_of_nodes() - 1),
        "net_components": float(nx.number_connected_components(U)),
        "net_sync_frac": sync,
    }

GROUPS = {
    "text": ["txt_ttr", "txt_sentlen_cv", "txt_punct_rate", "txt_emoji_rate", "txt_char_entropy",
             "txt_repeat_trigram", "txt_len_cv"],
    "similarity": ["sim_mean", "sim_max", "sim_near_dup_frac", "sim_template_frac"],
    "behaviour": ["beh_posts", "beh_per_day", "beh_interval_cv", "beh_hour_entropy", "beh_night_frac",
                  "beh_burst_frac", "beh_top_minute_frac", "beh_shift_ratio"],
    "profile": ["prof_log_ff_ratio", "prof_age_days", "prof_posts_per_age_day", "prof_completeness",
                "prof_username_digit_frac", "prof_bio_len"],
    "network": ["net_reciprocity", "net_clustering", "net_repeat_frac", "net_partners",
                "net_components", "net_sync_frac"],
}
ALL = [k for g in GROUPS.values() for k in g]

def extract(a: Account):
    tf, backend = text_features([p.text for p in a.posts])
    f = {**tf, **behaviour_features(a.posts), **profile_features(a), **network_features(a)}
    return {k: f.get(k, NAN) for k in ALL}, backend

def vector(feats):
    return np.array([feats[k] for k in ALL], float)
