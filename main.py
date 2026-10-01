import os
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .schemas import Account
from .features import extract, ALL, GROUPS
from .scoring import analyze as run, load_model, LIMITS
from . import research

app = FastAPI(title="PersonaGuard", version="0.1.0")
FRONT = os.path.join(os.path.dirname(__file__), "../../frontend")

def _full(a: Account):
    feats, backend = extract(a)
    r = run(feats, len(a.posts))
    r["limitations"] = LIMITS
    r["data_available"] = {g: any(feats[c] == feats[c] for c in cols) for g, cols in GROUPS.items()}
    if not r["data_available"]["network"]:
        r["note_network"] = "Network analysis disabled: fewer than 5 interactions supplied. Score uses remaining signals."
    r["embedding_backend"] = backend
    return r, feats

# Nothing submitted is stored; every request is processed in memory and discarded.
@app.post("/analyze")
def analyze(a: Account): return _full(a)[0]
@app.post("/predict")
def predict(a: Account):
    r = _full(a)[0]
    return {k: r[k] for k in ("classification", "risk_score", "confidence", "status")}
@app.post("/features")
def features(a: Account): return {k: (None if v != v else v) for k, v in _full(a)[1].items()}
@app.post("/explain")
def explain(a: Account):
    r = _full(a)[0]
    return {"evidence": r["top_signals"], "model_inference": {k: r[k] for k in ("classification", "risk_score", "status")},
            "uncertainty": {"confidence": r["confidence"], "limitations": LIMITS, "note": r.get("note")}}
@app.get("/health")
def health(): return {"ok": True}
@app.get("/model-info")
def info():
    M = load_model()
    if not M:
        return {"mode": "heuristic_demo", "detail": "No trained model found in models/model.joblib. Scores come from transparent threshold rules and have NOT been validated.",
                "features": ALL}
    return {"mode": "trained", "model": M["model"], "columns": M["columns"], "held_out_metrics": M["metrics"]}

@app.post("/research/ablation")
def ablation(accounts: List[Account], model: str = "logreg"):
    try:
        X, y, g = research.build_xy(accounts)
        return research.ablation(X, y, g, model)
    except ValueError as e:
        raise HTTPException(422, str(e))
@app.post("/research/compare")
def compare(accounts: List[Account]):
    try:
        X, y, g = research.build_xy(accounts)
        return {n: research.fit_eval(X, y, g, n, tune=False)[1] for n in research.MODELS}
    except ValueError as e:
        raise HTTPException(422, str(e))

app.mount("/static", StaticFiles(directory=FRONT), name="static")
@app.get("/")
def home(): return FileResponse(os.path.join(FRONT, "index.html"))
