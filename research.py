"""Training, evaluation, ablation. Splits by account so no account appears in both train and test."""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, roc_auc_score,
                             average_precision_score, confusion_matrix, brier_score_loss)
from sklearn.preprocessing import label_binarize
from .features import ALL, GROUPS, extract, vector

MODELS = {
    "logreg": (LogisticRegression(max_iter=2000, class_weight="balanced"), {"clf__C": [.1, 1, 10]}),
    "random_forest": (RandomForestClassifier(class_weight="balanced", random_state=0), {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 8]}),
    "grad_boost": (GradientBoostingClassifier(random_state=0), {"clf__n_estimators": [100, 200], "clf__max_depth": [2, 3]}),
    "mlp": (MLPClassifier(max_iter=800, random_state=0), {"clf__hidden_layer_sizes": [(32,), (64, 32)]}),
}
ABLATION = {
    "text only": ["text", "similarity"], "behaviour only": ["behaviour"], "profile only": ["profile"],
    "network only": ["network"], "text+behaviour": ["text", "similarity", "behaviour"],
    "text+behaviour+profile": ["text", "similarity", "behaviour", "profile"],
    "all features": list(GROUPS),
}

def build_xy(accounts):
    X, y, g = [], [], []
    for a in accounts:
        if not a.label:
            raise ValueError(f"account '{a.username}' has no label")
        X.append(vector(extract(a)[0])); y.append(a.label); g.append(a.username or str(len(g)))
    return np.array(X), np.array(y), np.array(g)

def cols_for(groups):
    return [c for k in groups for c in GROUPS[k]]

def pipe(name):
    clf, grid = MODELS[name]
    return Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                     ("sc", StandardScaler()), ("clf", clf)]), grid

def evaluate(y_true, proba, classes):
    pred = np.array(classes)[proba.argmax(1)]
    p, r, f, s = precision_recall_fscore_support(y_true, pred, labels=classes, zero_division=0)
    out = {"accuracy": accuracy_score(y_true, pred),
           "per_class": {c: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])} for i, c in enumerate(classes)},
           "macro_f1": float(np.mean(f)), "confusion_matrix": confusion_matrix(y_true, pred, labels=classes).tolist(),
           "classes": list(classes)}
    try:
        Y = label_binarize(y_true, classes=classes)
        if Y.shape[1] == 1: Y = np.hstack([1 - Y, Y])
        out["roc_auc"] = float(roc_auc_score(Y, proba, average="macro", multi_class="ovr")) if len(set(y_true)) > 1 else None
        out["pr_auc"] = float(average_precision_score(Y, proba, average="macro"))
        out["brier"] = float(np.mean([brier_score_loss(Y[:, i], proba[:, i]) for i in range(Y.shape[1])]))
    except Exception:
        out["roc_auc"] = out["pr_auc"] = out["brier"] = None
    conf, ok = proba.max(1), pred == y_true
    bins = np.linspace(0, 1, 6)
    out["reliability"] = [{"conf": float(conf[m].mean()), "acc": float(ok[m].mean()), "n": int(m.sum())}
                          for lo, hi in zip(bins[:-1], bins[1:]) if (m := (conf >= lo) & (conf <= hi)).sum()]
    return out

def fit_eval(X, y, groups, model="logreg", columns=None, tune=True, seed=0):
    idx = [ALL.index(c) for c in (columns or ALL)]
    Xs = X[:, idx]
    tr, te = next(GroupShuffleSplit(1, test_size=.25, random_state=seed).split(Xs, y, groups))
    if len(set(groups[tr])) < 6 or len(set(y[tr])) < 2:
        raise ValueError("Need >= 6 training accounts and >= 2 classes.")
    p, grid = pipe(model)
    if tune:
        p = GridSearchCV(p, grid, cv=GroupKFold(min(3, len(set(groups[tr])))), scoring="f1_macro",
                         error_score="raise").fit(Xs[tr], y[tr], groups=groups[tr]).best_estimator_
    else:
        p.fit(Xs[tr], y[tr])
    try:  # probability calibration (needs >= 3 samples per class)
        if min(np.unique(y[tr], return_counts=True)[1]) >= 3:
            p = CalibratedClassifierCV(p, cv=3, method="sigmoid").fit(Xs[tr], y[tr])
    except Exception:
        pass
    classes = list(p.classes_)
    return p, evaluate(y[te], p.predict_proba(Xs[te]), classes), idx, tr

def ablation(X, y, groups, model="logreg"):
    rows = []
    for name, gs in ABLATION.items():
        try:
            _, m, _, _ = fit_eval(X, y, groups, model, cols_for(gs), tune=False)
            rows.append({"setup": name, "accuracy": m["accuracy"], "precision_macro": float(np.mean([v["precision"] for v in m["per_class"].values()])),
                         "recall_macro": float(np.mean([v["recall"] for v in m["per_class"].values()])),
                         "f1_macro": m["macro_f1"], "roc_auc": m["roc_auc"]})
        except Exception as e:
            rows.append({"setup": name, "error": str(e)})
    return rows

def train_and_save(accounts, model, path, anomaly_path=None):
    import joblib
    X, y, g = build_xy(accounts)
    p, metrics, idx, tr = fit_eval(X, y, g, model)
    joblib.dump({"pipeline": p, "columns": [ALL[i] for i in idx],
                 "medians": {c: float(np.nanmedian(X[:, i])) if not np.all(np.isnan(X[:, i])) else 0.0 for i, c in enumerate(ALL)},
                 "metrics": metrics, "model": model}, path)
    if anomaly_path:  # Isolation Forest on human-labelled reference accounts only
        H = X[y == "human"]
        if len(H) >= 10:
            imp = SimpleImputer(strategy="median").fit(H)
            iso = IsolationForest(random_state=0).fit(imp.transform(H))
            s = -iso.score_samples(imp.transform(H))
            joblib.dump({"imputer": imp, "iso": iso, "range": (float(s.min()), float(s.max()))}, anomaly_path)
    return metrics
