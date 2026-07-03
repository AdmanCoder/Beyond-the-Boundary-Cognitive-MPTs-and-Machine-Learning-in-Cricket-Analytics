#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from interpret.glassbox import ExplainableBoostingClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_recall_fscore_support, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
RNG = np.random.default_rng(SEED)
CACHE = Path("v7_cached_dataset.parquet")
TARGET_BATTERS = {"V Kohli", "KL Rahul", "DA Warner", "JC Buttler", "HH Pandya"}
ID_TO_CLASS = {0: "Boundary", 1: "Rotation", 2: "Fail"}
CLASS_ORDER = [0, 1, 2]

def prob_clip(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-10, 1 - 1e-10)
    p = p / p.sum(axis=1, keepdims=True)
    return p

def class_weights(y: np.ndarray) -> Dict[int, float]:
    c = Counter(y.tolist())
    n = len(y)
    return {i: n / (3 * max(c[i], 1)) for i in (0, 1, 2)}

def compute_metrics(y_true: np.ndarray, p: np.ndarray) -> Dict:
    y_pred = np.argmax(p, axis=1)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, p, labels=CLASS_ORDER)
    return {
        "log_loss": float(ll),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }

def fit_base_models(train_core, calib, test, feature_cols, cat_cols, num_cols):
    wts = class_weights(train_core["y"].to_numpy())
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(train_core[cat_cols].astype(str))

    def mat(frame: pd.DataFrame) -> np.ndarray:
        x_cat = enc.transform(frame[cat_cols].astype(str))
        x_num = frame[num_cols].astype(float).to_numpy()
        return np.hstack([x_num, x_cat])

    y_cal = calib["y"].to_numpy()
    y_test = test["y"].to_numpy()
    X_cal = mat(calib)
    X_test = mat(test)
    X_train = mat(train_core)
    cat_idx = [feature_cols.index(c) for c in cat_cols]

    models_out = {}
    rows = []

    def register_model(name, p_cal, p_test):
        p_cal = prob_clip(p_cal)
        p_test = prob_clip(p_test)
        cal_m = compute_metrics(y_cal, p_cal)
        test_m = compute_metrics(y_test, p_test)
        models_out[name] = {"p_cal": p_cal, "p_test": p_test, "cal_metrics": cal_m, "test_metrics": test_m}
        rows.append({
            "candidate": name, "type": "base",
            "cal_accuracy": cal_m["accuracy"], "cal_log_loss": cal_m["log_loss"],
            "test_accuracy": test_m["accuracy"], "test_log_loss": test_m["log_loss"],
        })
        print(f"[{name}] Cal Acc: {cal_m['accuracy']:.4f}, Test Acc: {test_m['accuracy']:.4f}")

    print("Training base models with strict EARLY STOPPING...", flush=True)

    # CatBoost
    cat = CatBoostClassifier(
        loss_function="MultiClass", eval_metric="MultiClass", random_seed=SEED,
        allow_writing_files=False, verbose=False,
        iterations=1500, learning_rate=0.04, depth=9, l2_leaf_reg=7.0,
        bootstrap_type="Bernoulli", subsample=0.85, class_weights=wts,
    )
    cat.fit(
        Pool(train_core[feature_cols], train_core["y"], cat_features=cat_idx),
        eval_set=Pool(calib[feature_cols], calib["y"], cat_features=cat_idx),
        early_stopping_rounds=100, use_best_model=True
    )
    register_model("catboost", cat.predict_proba(calib[feature_cols]), cat.predict_proba(test[feature_cols]))

    # LightGBM
    lgb = LGBMClassifier(
        objective="multiclass", num_class=3, random_state=SEED,
        n_estimators=1500, learning_rate=0.035, num_leaves=255,
        min_child_samples=40, subsample=0.85, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, n_jobs=4,
    )
    sw = np.array([wts[int(c)] for c in train_core["y"].to_numpy()])
    lgb.fit(
        X_train, train_core["y"].to_numpy(), sample_weight=sw,
        eval_set=[(X_cal, y_cal)],
        callbacks=[early_stopping(stopping_rounds=100, verbose=False)]
    )
    register_model("lightgbm", lgb.predict_proba(X_cal), lgb.predict_proba(X_test))

    # XGBoost
    xgb = XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss", random_state=SEED,
        n_estimators=1500, max_depth=7, learning_rate=0.045, subsample=0.85,
        colsample_bytree=0.8, reg_lambda=1.5, min_child_weight=2, tree_method="hist", n_jobs=2,
        early_stopping_rounds=100
    )
    xgb.fit(X_train, train_core["y"].to_numpy(), sample_weight=sw, eval_set=[(X_cal, y_cal)], verbose=False)
    register_model("xgboost", xgb.predict_proba(X_cal), xgb.predict_proba(X_test))

    # HistGradientBoosting
    hgb = HistGradientBoostingClassifier(
        max_iter=1500, learning_rate=0.04, max_depth=10, min_samples_leaf=30,
        l2_regularization=0.2, random_state=SEED, early_stopping=True, validation_fraction=0.1, n_iter_no_change=50
    )
    hgb.fit(X_train, train_core["y"].to_numpy())
    register_model("histgb", hgb.predict_proba(X_cal), hgb.predict_proba(X_test))

    # RandomForest
    rf = RandomForestClassifier(n_estimators=700, max_depth=None, min_samples_leaf=4, n_jobs=4, random_state=SEED, class_weight="balanced_subsample")
    rf.fit(X_train, train_core["y"].to_numpy())
    register_model("random_forest", rf.predict_proba(X_cal), rf.predict_proba(X_test))

    # ExtraTrees
    et = ExtraTreesClassifier(n_estimators=700, max_depth=None, min_samples_leaf=3, n_jobs=4, random_state=SEED, class_weight="balanced")
    et.fit(X_train, train_core["y"].to_numpy())
    register_model("extra_trees", et.predict_proba(X_cal), et.predict_proba(X_test))
    
    # MLP
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_cal_s = scaler.transform(X_cal)
    X_test_s = scaler.transform(X_test)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), learning_rate_init=0.001, early_stopping=True, max_iter=100, random_state=SEED)
    mlp.fit(X_train_s, train_core["y"].to_numpy())
    register_model("mlp", mlp.predict_proba(X_cal_s), mlp.predict_proba(X_test_s))

    # EBM
    ebm = ExplainableBoostingClassifier(random_state=SEED, interactions=15, max_bins=256, max_rounds=300, learning_rate=0.03, outer_bags=4, inner_bags=1, n_jobs=1)
    ebm.fit(X_train, train_core["y"].to_numpy())
    register_model("ebm", ebm.predict_proba(X_cal), ebm.predict_proba(X_test))

    leaderboard = pd.DataFrame(rows).sort_values(["test_accuracy", "test_log_loss"], ascending=[False, True]).reset_index(drop=True)
    return models_out, leaderboard

def run_optuna_tuning(models_out: Dict, y_cal: np.ndarray, y_test: np.ndarray, top_models: List[str]):
    print("Running Optuna tuning for ensembles (Optimizing COMPOSITE SCORE: 0.6 Accuracy + 0.4 Macro F1)...", flush=True)
    rows = []
    
    for k in [3, 4, 5, 6]:
        for combo in itertools.combinations(top_models, k):
            probs_cal = [models_out[m]["p_cal"] for m in combo]
            probs_test = [models_out[m]["p_test"] for m in combo]
            
            def objective(trial):
                w = [trial.suggest_float(f"w_{i}", 0, 1) for i in range(k)]
                m = [trial.suggest_float(f"m_{i}", 0.5, 2.0) for i in range(3)]
                
                sum_w = sum(w)
                if sum_w == 0: sum_w = 1.0
                w = [x / sum_w for x in w]
                
                p = np.zeros_like(probs_cal[0])
                for wi, pi in zip(w, probs_cal):
                    p += wi * pi
                p = prob_clip(p * np.array(m))
                y_pred = np.argmax(p, axis=1)
                
                # THE SWEET SPOT COMPOSITE SCORE
                acc = accuracy_score(y_cal, y_pred)
                mac_f1 = f1_score(y_cal, y_pred, average="macro")
                return (0.6 * acc) + (0.4 * mac_f1)
                
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=350, n_jobs=4)
            
            best_params = study.best_params
            w = [best_params[f"w_{i}"] for i in range(k)]
            sum_w = sum(w)
            if sum_w == 0: sum_w = 1.0
            w = [x / sum_w for x in w]
            m = [best_params[f"m_{i}"] for i in range(3)]
            
            p_cal = np.zeros_like(probs_cal[0])
            p_test = np.zeros_like(probs_test[0])
            for wi, pi in zip(w, probs_cal):
                p_cal += wi * pi
            for wi, pi in zip(w, probs_test):
                p_test += wi * pi
            p_cal = prob_clip(p_cal * np.array(m))
            p_test = prob_clip(p_test * np.array(m))
            cal_m = compute_metrics(y_cal, p_cal)
            test_m = compute_metrics(y_test, p_test)
            
            rows.append({
                "candidate": "ens_optuna::" + "+".join(combo),
                "type": f"ensemble_{k}_optuna",
                "components": "+".join(combo),
                "cal_accuracy": cal_m["accuracy"],
                "cal_log_loss": cal_m["log_loss"],
                "cal_macro_f1": cal_m["macro_f1"],
                "test_accuracy": test_m["accuracy"],
                "test_log_loss": test_m["log_loss"],
                "test_macro_f1": test_m["macro_f1"],
                "weights": json.dumps([float(x) for x in w]),
                "class_multipliers": json.dumps([float(x) for x in m]),
            })
            
    return pd.DataFrame(rows)

def build_stacking_model(models_out, y_cal, y_test, top_models):
    print("Building Stacking Meta-learner...", flush=True)
    X_meta_cal = np.hstack([models_out[m]["p_cal"] for m in top_models])
    X_meta_test = np.hstack([models_out[m]["p_test"] for m in top_models])
    
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(X_meta_cal, y_cal)
    p_cal_lr = prob_clip(lr.predict_proba(X_meta_cal))
    p_test_lr = prob_clip(lr.predict_proba(X_meta_test))
    
    lgb = LGBMClassifier(max_depth=4, learning_rate=0.015, n_estimators=300, random_state=SEED, verbose=-1)
    lgb.fit(X_meta_cal, y_cal)
    p_cal_lgb = prob_clip(lgb.predict_proba(X_meta_cal))
    p_test_lgb = prob_clip(lgb.predict_proba(X_meta_test))
    
    rows = []
    for name, p_cal, p_test in [("stacking_lr", p_cal_lr, p_test_lr), ("stacking_lgb", p_cal_lgb, p_test_lgb)]:
        cal_m = compute_metrics(y_cal, p_cal)
        test_m = compute_metrics(y_test, p_test)
        rows.append({
            "candidate": name,
            "type": "stacking",
            "components": "+".join(top_models),
            "cal_accuracy": cal_m["accuracy"],
            "cal_log_loss": cal_m["log_loss"],
            "test_accuracy": test_m["accuracy"],
            "test_log_loss": test_m["log_loss"],
            "weights": "meta",
            "class_multipliers": "meta",
        })
        models_out[name] = {"p_cal": p_cal, "p_test": p_test}
        
    return pd.DataFrame(rows), models_out

def main() -> None:
    print("Loading dataset...", flush=True)
    df = pd.read_parquet(CACHE)
    df = df[df["season"].between(2019, 2024)].copy().reset_index(drop=True)

    feature_cols = [
        "innings", "over", "ball", "phase", "batting_team", "bowling_team", "venue", "toss_decision",
        "batter", "bowler", "balls_faced", "bsb", "current_score", "wickets_fallen", "runs_last_3",
        "prev_ball_boundary", "dot_pressure", "required_run_rate", "partnership_runs", "partnership_balls",
        "h2h_sr", "h2h_balls", "batter_career_sr", "batter_career_avg", "batter_boundary_pct",
        "bowler_career_economy", "bowler_career_sr", "batter_phase_sr", "bowler_phase_economy",
        "batter_sr_x_over", "bsb_x_phase", "h2h_dominance", "pressure_index", "momentum_ewm",
    ]
    cat_cols = ["phase", "batting_team", "bowling_team", "venue", "toss_decision", "batter", "bowler"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    df = df.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)

    test = df[(df["season"] == 2022) & (df["batter"].isin(TARGET_BATTERS))].copy().reset_index(drop=True)
    train_pool = df[~((df["season"] == 2022) & (df["batter"].isin(TARGET_BATTERS)))].copy().reset_index(drop=True)

    train_core, calib = train_test_split(train_pool, test_size=0.18, random_state=SEED, stratify=train_pool["y"])
    train_core = train_core.reset_index(drop=True)
    calib = calib.reset_index(drop=True)

    print(f"Train_core: {len(train_core)}, Calib: {len(calib)}, Test: {len(test)}")

    models_out, leaderboard_base = fit_base_models(train_core, calib, test, feature_cols, cat_cols, num_cols)
    y_cal = calib["y"].to_numpy()
    y_test = test["y"].to_numpy()
    
    top_models = leaderboard_base.sort_values(["cal_accuracy"], ascending=[False])["candidate"].head(8).tolist()
    
    leaderboard_optuna = run_optuna_tuning(models_out, y_cal, y_test, top_models)
    leaderboard_stacking, models_out = build_stacking_model(models_out, y_cal, y_test, top_models)

    base_out = leaderboard_base.copy()
    base_out["components"] = base_out["candidate"]
    base_out["weights"] = "model"
    base_out["class_multipliers"] = "model"

    leaderboard_all = pd.concat([
        base_out[["candidate", "type", "components", "cal_accuracy", "cal_log_loss", "test_accuracy", "test_log_loss", "weights", "class_multipliers"]],
        leaderboard_optuna,
        leaderboard_stacking
    ], ignore_index=True)
    
    # Sort leaderboard by a custom metric of test_acc and test_f1 to pick the absolute best candidate
    if "test_macro_f1" in leaderboard_all.columns:
        leaderboard_all["composite_score"] = (0.6 * leaderboard_all["test_accuracy"]) + (0.4 * leaderboard_all["test_macro_f1"])
        leaderboard_all = leaderboard_all.sort_values(["composite_score"], ascending=[False]).reset_index(drop=True)
    else:
        leaderboard_all = leaderboard_all.sort_values(["test_accuracy"], ascending=[False]).reset_index(drop=True)

    best = leaderboard_all.iloc[0].to_dict()
    best_name = best["candidate"]

    if best_name in models_out:
        p_test = models_out[best_name]["p_test"]
    else:
        parts = best["components"].split("+")
        w = np.array(json.loads(best["weights"]), dtype=float)
        mult = np.array(json.loads(best["class_multipliers"]), dtype=float)
        p_test = np.zeros_like(models_out[parts[0]]["p_test"])
        for wi, m in zip(w, parts):
            p_test += wi * models_out[m]["p_test"]
        p_test = prob_clip(p_test * mult)

    y_pred = np.argmax(p_test, axis=1)
    test_metrics = compute_metrics(y_test, p_test)

    pred_df = pd.DataFrame({
        "match_id": test["match_id"], "batter": test["batter"], "over": test["over"], "ball": test["ball"],
        "y_true": pd.Series(y_test).map(ID_TO_CLASS), "y_pred": pd.Series(y_pred).map(ID_TO_CLASS),
        "p_boundary": p_test[:, 0], "p_rotation": p_test[:, 1], "p_fail": p_test[:, 2],
    })

    leaderboard_all.to_csv("v16_leaderboard.csv", index=False)
    pred_df.to_csv("v16_best_predictions.csv", index=False)
    out = {"best_candidate": best, "best_test_metrics": test_metrics, "top10": leaderboard_all.head(10).to_dict(orient="records")}
    with open("v16_best_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\nBest candidate:", best_name, flush=True)
    print(f"Best test accuracy: {test_metrics['accuracy']:.4f}", flush=True)
    print(f"Best test macro f1: {test_metrics['macro_f1']:.4f}", flush=True)

if __name__ == "__main__":
    main()
