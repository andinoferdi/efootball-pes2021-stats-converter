#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML-only Converter: eFootball (hub-style JSON) -> PES 2021
- Always ML (no linear / no rule-only mode exposed).
- Stronger model: Ensemble (ExtraTrees + Ridge) with feature engineering.
- Optional dataset builder: scrape PESMaster eFootball player pages, auto find matching PES 2021 page, build training pairs.
- Overall clamp: if ef_overall > 99 -> target PES overall 99, then boost key stats until computed overall reaches target (<= 99).

Deps:
  pip install flask numpy scikit-learn joblib requests beautifulsoup4 lxml

Run server:
  python app_ml_only.py

Build dataset from PESMaster eFootball player URLs:
  python app_ml_only.py --scrape --urls urls.txt --out dataset.jsonl

Train model from dataset:
  python app_ml_only.py --train --data dataset.jsonl --model model.joblib

API:
  POST /api/convert
  {
    "ef_stats": {...},
    "position": "RMF",
    "ef_overall": 100
  }
"""

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request
from joblib import dump, load
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Config paths
# -----------------------------
DEFAULT_DATASET_PATH = os.getenv("DATASET_PATH", "dataset.jsonl")
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")

HTTP_TIMEOUT = 20


# -----------------------------
# PES attributes & EF canonical keys
# -----------------------------
PES_ATTR_ORDER: List[str] = [
    "Offensive Awareness",
    "Ball Control",
    "Dribbling",
    "Tight Possession",
    "Low Pass",
    "Lofted Pass",
    "Finishing",
    "Heading",
    "Place Kicking",
    "Curl",
    "Speed",
    "Acceleration",
    "Kicking Power",
    "Jump",
    "Physical Contact",
    "Balance",
    "Stamina",
    "Defensive Awareness",
    "Ball Winning",
    "Aggression",
    "GK Awareness",
    "GK Catching",
    "GK Clearing",
    "GK Reflexes",
    "GK Reach",
    "Weak Foot Usage",
    "Weak Foot Accuracy",
    "Form",
    "Injury Resistance",
]

SMALL_RANGE_ATTRS = {
    "Weak Foot Usage": (1, 4),
    "Weak Foot Accuracy": (1, 4),
    "Form": (1, 8),
    "Injury Resistance": (1, 3),
}

CORE_ATTRS = [a for a in PES_ATTR_ORDER if a not in SMALL_RANGE_ATTRS]

# Canonical EF keys expected by your web JSON
EF_KEYS: List[str] = [
    "offensive_awareness",
    "ball_control",
    "dribbling",
    "tight_possession",
    "low_pass",
    "lofted_pass",
    "finishing",
    "heading",
    "place_kicking",
    "curl",
    "speed",
    "acceleration",
    "kicking_power",
    "jump",
    "physical_contact",
    "balance",
    "stamina",
    "defensive_awareness",
    "defensive_engagement",
    "tackling",
    "aggression",
    "goalkeeping",
    "gk_catching",
    "gk_parrying",
    "gk_reflexes",
    "gk_reach",
    "weak_foot_usage",
    "weak_foot_acc",
    "form",
    "injury_resistance",
]

# Position handling
POS_LIST = ["GK", "CB", "LB", "RB", "DMF", "CMF", "AMF", "LMF", "RMF", "LWF", "RWF", "SS", "CF"]


def pos_group(pos: str) -> str:
    p = (pos or "").upper()
    if p == "GK":
        return "GK"
    if p in {"CB", "LB", "RB"}:
        return "DEF"
    if p in {"DMF", "CMF", "AMF", "LMF", "RMF"}:
        return "MID"
    if p in {"LWF", "RWF", "SS", "CF"}:
        return "FWD"
    return "MID"


# -----------------------------
# Basic utils
# -----------------------------
def clamp_int(x: float, lo: int, hi: int) -> int:
    v = int(round(float(x)))
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def ensure_int_dict(d: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k, v in d.items():
        out[k] = clamp_int(v, -10_000, 10_000)
    return out


def normalize_ef_stats(raw: Dict[str, Any]) -> Dict[str, float]:
    """
    Normalize keys from various sources into EF_KEYS.
    Accepts some common aliases.
    """
    if not isinstance(raw, dict):
        return {k: 0.0 for k in EF_KEYS}

    alias = {
        "set_piece_taking": "place_kicking",
        "jumping": "jump",
        "gk_awareness": "goalkeeping",
        "gk_clearing": "gk_parrying",
        "weak_foot_accuracy": "weak_foot_acc",

        # sometimes camel case
        "offensiveAwareness": "offensive_awareness",
        "ballControl": "ball_control",
        "tightPossession": "tight_possession",
        "lowPass": "low_pass",
        "loftedPass": "lofted_pass",
        "kickingPower": "kicking_power",
        "physicalContact": "physical_contact",
        "defensiveAwareness": "defensive_awareness",
        "defensiveEngagement": "defensive_engagement",
        "weakFootUsage": "weak_foot_usage",
        "weakFootAcc": "weak_foot_acc",
        "injuryResistance": "injury_resistance",
    }

    norm: Dict[str, float] = {k: 0.0 for k in EF_KEYS}
    for k, v in raw.items():
        kk = alias.get(k, k)
        if kk in norm:
            norm[kk] = safe_float(v, 0.0)
    return norm


# -----------------------------
# Feature engineering (ML, not exposed as mode)
# -----------------------------
def _nonlinear_map(v: float, max_ef: float = 103.0, exp: float = 0.33) -> float:
    """
    Smooth monotonic squashing, designed for scale up to 103.
    Output is roughly 1..99.
    """
    v = max(0.0, float(v))
    if v <= 1.0:
        return 1.0
    # keep low stats less distorted, high stats saturate
    scaled = 99.0 * pow(min(v, max_ef) / max_ef, exp)
    return float(np.clip(scaled, 1.0, 99.0))


def _def_map(v: float) -> float:
    """
    Defence tends to be more "generous" on PES in many cards, so use slightly different curve.
    """
    v = max(0.0, float(v))
    x = _nonlinear_map(v, exp=0.28)
    return float(np.clip(x + 6.0, 1.0, 99.0))


def _nonlinear_map_gk_outfield(v: float, max_ef: float = 103.0, exp: float = 0.38, reduction: float = 0.88, bonus: float = 8.0) -> float:
    """
    Special formula for non-GK stats on goalkeepers.
    More conservative with reduction factor, but adds small bonus (8-10) to avoid being too low.
    """
    v = max(0.0, float(v))
    if v <= 1.0:
        return 1.0
    scaled = 99.0 * pow(min(v, max_ef) / max_ef, exp)
    result = scaled * reduction + min(bonus, 10.0)
    return float(np.clip(result, 1.0, 99.0))


def baseline_pes_guess(ef: Dict[str, float], position: str) -> Dict[str, float]:
    """
    Prior guess used as extra features. Not an alternate mode.
    """
    gk = pos_group(position) == "GK" or ef.get("goalkeeping", 0.0) >= 60.0

    if gk:
        map_func = lambda v: _nonlinear_map_gk_outfield(v, exp=0.38, reduction=0.88, bonus=8.0)
        def_map_func = lambda v: _nonlinear_map_gk_outfield(v, exp=0.36, reduction=0.85, bonus=8.0)
    else:
        map_func = _nonlinear_map
        def_map_func = _def_map

    out: Dict[str, float] = {}
    out["Offensive Awareness"] = map_func(ef["offensive_awareness"])
    out["Ball Control"] = map_func(ef["ball_control"])
    out["Dribbling"] = map_func(ef["dribbling"])
    out["Tight Possession"] = map_func(ef["tight_possession"])
    out["Low Pass"] = map_func(ef["low_pass"])
    out["Lofted Pass"] = map_func(ef["lofted_pass"])
    out["Finishing"] = map_func(ef["finishing"])
    out["Heading"] = map_func(ef["heading"])
    out["Place Kicking"] = map_func(ef["place_kicking"])
    out["Curl"] = map_func(ef["curl"])

    out["Speed"] = map_func(ef["speed"])
    out["Acceleration"] = map_func(ef["acceleration"])
    out["Kicking Power"] = map_func(ef["kicking_power"])
    out["Jump"] = map_func(ef["jump"])
    out["Physical Contact"] = map_func(ef["physical_contact"])
    out["Balance"] = map_func(ef["balance"])
    out["Stamina"] = map_func(ef["stamina"])

    out["Defensive Awareness"] = def_map_func(ef["defensive_awareness"])
    out["Ball Winning"] = def_map_func(0.5 * (ef["defensive_engagement"] + ef["tackling"]))
    out["Aggression"] = def_map_func(ef["aggression"])

    # GK
    if gk:
        out["GK Awareness"] = _nonlinear_map(ef["goalkeeping"], exp=0.30)
        out["GK Catching"] = _nonlinear_map(ef["gk_catching"], exp=0.30)
        out["GK Clearing"] = _nonlinear_map(ef["gk_parrying"], exp=0.30)
        out["GK Reflexes"] = _nonlinear_map(ef["gk_reflexes"], exp=0.30)
        out["GK Reach"] = _nonlinear_map(ef["gk_reach"], exp=0.30)
    else:
        out["GK Awareness"] = 40.0
        out["GK Catching"] = 40.0
        out["GK Clearing"] = 40.0
        out["GK Reflexes"] = 40.0
        out["GK Reach"] = 40.0

    # Small-range fields are handled separately (not predicted)
    return out


def one_hot_position(pos: str) -> List[int]:
    p = (pos or "").upper()
    return [1 if p == x else 0 for x in POS_LIST]


def build_features(ef: Dict[str, float], position: str, ef_overall: Optional[float]) -> np.ndarray:
    """
    Features:
      - raw EF stats (EF_KEYS)
      - baseline guess for CORE_ATTRS
      - position one-hot
      - group one-hot (FWD/MID/DEF/GK)
      - ef_overall (if provided else 0)
      - a few interactions
    """
    pos = (position or "CF").upper()
    ef_overall_val = safe_float(ef_overall, 0.0)

    raw_vec = np.array([ef[k] for k in EF_KEYS], dtype=np.float32)

    base = baseline_pes_guess(ef, pos)
    base_vec = np.array([base.get(a, 0.0) for a in CORE_ATTRS], dtype=np.float32)

    pos_oh = np.array(one_hot_position(pos), dtype=np.float32)

    g = pos_group(pos)
    grp_oh = np.array(
        [1.0 if g == "FWD" else 0.0, 1.0 if g == "MID" else 0.0, 1.0 if g == "DEF" else 0.0, 1.0 if g == "GK" else 0.0],
        dtype=np.float32,
    )

    # interactions
    spd = ef["speed"]
    acc = ef["acceleration"]
    phy = ef["physical_contact"]
    dri = ef["dribbling"]
    pas = 0.5 * (ef["low_pass"] + ef["lofted_pass"])
    defmix = 0.5 * (ef["defensive_engagement"] + ef["tackling"])
    interactions = np.array(
        [
            spd - acc,            # pace profile
            acc - spd,
            dri * 0.01 * spd,     # dribble-speed synergy
            pas * 0.01 * ef["curl"],
            phy * 0.01 * ef["jump"],
            defmix,
            ef["goalkeeping"],
            float(ef_overall_val),
        ],
        dtype=np.float32,
    )

    return np.concatenate([raw_vec, base_vec, pos_oh, grp_oh, interactions], axis=0)


# -----------------------------
# Model: Ensemble (ExtraTrees + Ridge), multi-output
# -----------------------------
@dataclass
class TrainedModel:
    model_trees: Any
    model_ridge: Any
    feature_dim: int
    target_attrs: List[str]


def train_ensemble(samples: List[Dict[str, Any]]) -> TrainedModel:
    """
    samples item:
      {
        "position": "RMF",
        "ef": {...},   # canonical ef dict or raw, normalize_ef_stats will handle
        "pes": {...}   # including keys in PES_ATTR_ORDER
      }
    """
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []

    for s in samples:
        pos = (s.get("position") or "CF").upper()
        ef = normalize_ef_stats(s.get("ef") or {})
        pes = s.get("pes") or {}
        if not isinstance(pes, dict):
            continue

        # ensure required targets exist
        if any(a not in pes for a in CORE_ATTRS):
            continue

        x = build_features(ef, pos, s.get("ef_overall"))
        y = np.array([safe_float(pes[a], 0.0) for a in CORE_ATTRS], dtype=np.float32)

        X_list.append(x)
        Y_list.append(y)

    if len(X_list) < 8:
        raise RuntimeError(f"Dataset too small for stable ML. Minimum 8 valid samples required. Got: {len(X_list)}")

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)

    # Tree model (strong non-linear)
    trees = ExtraTreesRegressor(
        n_estimators=800,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features="auto",
        bootstrap=False,
    )
    trees.fit(X, Y)

    # Ridge model (stabilizer)
    ridge = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=3.0, random_state=42)),
        ]
    )
    ridge.fit(X, Y)

    return TrainedModel(model_trees=trees, model_ridge=ridge, feature_dim=X.shape[1], target_attrs=CORE_ATTRS)


def predict_ensemble(tm: TrainedModel, ef: Dict[str, Any], position: str, ef_overall: Optional[float]) -> Dict[str, int]:
    ef_n = normalize_ef_stats(ef)
    pos = (position or "CF").upper()

    x = build_features(ef_n, pos, ef_overall).reshape(1, -1)
    if x.shape[1] != tm.feature_dim:
        raise RuntimeError("Feature dimension mismatch. You trained the model with a different code version.")

    pred_t = tm.model_trees.predict(x)
    pred_r = tm.model_ridge.predict(x)

    # average
    pred = 0.6 * pred_t + 0.4 * pred_r
    pred = pred.reshape(-1)

    out: Dict[str, int] = {}
    for i, a in enumerate(tm.target_attrs):
        out[a] = clamp_int(pred[i], 1, 99)

    # Post rules (bukan mode, cuma constraint PES)
    gk_mode = pos_group(pos) == "GK" or ef_n.get("goalkeeping", 0.0) >= 60.0

    # Non-GK must be 40 for GK stats (PES 2021 common behavior)
    if not gk_mode:
        out["GK Awareness"] = 40
        out["GK Catching"] = 40
        out["GK Clearing"] = 40
        out["GK Reflexes"] = 40
        out["GK Reach"] = 40
    else:
        non_gk_attrs = [
            "Offensive Awareness", "Ball Control", "Dribbling", "Tight Possession",
            "Low Pass", "Lofted Pass", "Finishing", "Heading", "Place Kicking", "Curl",
            "Speed", "Acceleration", "Kicking Power", "Jump", "Physical Contact",
            "Balance", "Stamina", "Defensive Awareness", "Ball Winning", "Aggression"
        ]
        for attr in non_gk_attrs:
            if attr in out:
                ef_key_map = {
                    "Offensive Awareness": "offensive_awareness",
                    "Ball Control": "ball_control",
                    "Dribbling": "dribbling",
                    "Tight Possession": "tight_possession",
                    "Low Pass": "low_pass",
                    "Lofted Pass": "lofted_pass",
                    "Finishing": "finishing",
                    "Heading": "heading",
                    "Place Kicking": "place_kicking",
                    "Curl": "curl",
                    "Speed": "speed",
                    "Acceleration": "acceleration",
                    "Kicking Power": "kicking_power",
                    "Jump": "jump",
                    "Physical Contact": "physical_contact",
                    "Balance": "balance",
                    "Stamina": "stamina",
                    "Defensive Awareness": "defensive_awareness",
                    "Aggression": "aggression",
                }
                if attr == "Ball Winning":
                    de = ef_n.get("defensive_engagement", 0.0)
                    tk = ef_n.get("tackling", 0.0)
                    ef_val = 0.5 * (de + tk)
                else:
                    ef_key = ef_key_map.get(attr)
                    if ef_key:
                        ef_val = ef_n.get(ef_key, 0.0)
                    else:
                        continue
                baseline_gk = _nonlinear_map_gk_outfield(ef_val)
                if out[attr] > baseline_gk:
                    out[attr] = clamp_int(baseline_gk, 1, 99)

    # Small-range fields: clamp directly from ef
    for a, (lo, hi) in SMALL_RANGE_ATTRS.items():
        key = {
            "Weak Foot Usage": "weak_foot_usage",
            "Weak Foot Accuracy": "weak_foot_acc",
            "Form": "form",
            "Injury Resistance": "injury_resistance",
        }[a]
        out[a] = clamp_int(ef_n.get(key, lo), lo, hi)

    return out


def apply_ef_max_clamp(pes_stats: Dict[str, int], ef_stats: Dict[str, float]) -> Dict[str, int]:
    """
    If eFootball stat >= 99, PES result must be 99 (maximum).
    """
    out = dict(pes_stats)
    
    mapping = {
    "Offensive Awareness": "offensive_awareness",
    "Ball Control": "ball_control",
    "Dribbling": "dribbling",
    "Tight Possession": "tight_possession",
    "Low Pass": "low_pass",
    "Lofted Pass": "lofted_pass",
    "Finishing": "finishing",
    "Heading": "heading",
    "Place Kicking": "place_kicking",
    "Curl": "curl",
    "Speed": "speed",
    "Acceleration": "acceleration",
    "Kicking Power": "kicking_power",
    "Jump": "jump",
    "Physical Contact": "physical_contact",
    "Balance": "balance",
    "Stamina": "stamina",
    "Defensive Awareness": "defensive_awareness",
    "Aggression": "aggression",
    "GK Awareness": "goalkeeping",
    "GK Catching": "gk_catching",
    "GK Clearing": "gk_parrying",
    "GK Reflexes": "gk_reflexes",
    "GK Reach": "gk_reach",
    }
    
    for pes_attr, ef_key in mapping.items():
        if pes_attr in out:
            ef_val = ef_stats.get(ef_key, 0.0)
            if ef_val >= 99.0:
                out[pes_attr] = 99
    
    if "Ball Winning" in out:
        de = ef_stats.get("defensive_engagement", 0.0)
        tk = ef_stats.get("tackling", 0.0)
        ef_bw = 0.5 * (de + tk)
        if ef_bw >= 99.0:
            out["Ball Winning"] = 99
    
    return out


# -----------------------------
# PES overall estimator + calibration (overall clamp)
# -----------------------------
def compute_pes_overall(pes_stats: Dict[str, int], position: str) -> int:
    """
    Estimator, not Konami's internal formula.
    Used for calibrating target 99 when ef_overall > 99.
    """
    pos = (position or "CF").upper()
    g = pos_group(pos)

    def wavg(items: List[Tuple[str, float]]) -> float:
        s = 0.0
        w = 0.0
        for k, ww in items:
            v = float(pes_stats.get(k, 40))
            s += v * ww
            w += ww
        return s / max(w, 1e-9)

    if g == "GK":
        base = wavg(
            [
                ("GK Awareness", 0.25),
                ("GK Catching", 0.20),
                ("GK Reflexes", 0.25),
                ("GK Reach", 0.20),
                ("GK Clearing", 0.10),
            ]
        )
        return clamp_int(base, 40, 99)

    if g == "DEF":
        base = wavg(
            [
                ("Defensive Awareness", 0.23),
                ("Ball Winning", 0.22),
                ("Aggression", 0.10),
                ("Speed", 0.10),
                ("Physical Contact", 0.12),
                ("Heading", 0.08),
                ("Jump", 0.07),
                ("Low Pass", 0.08),
            ]
        )
        return clamp_int(base, 40, 99)

    if g == "MID":
        base = wavg(
            [
                ("Ball Control", 0.12),
                ("Dribbling", 0.10),
                ("Tight Possession", 0.10),
                ("Low Pass", 0.18),
                ("Lofted Pass", 0.14),
                ("Offensive Awareness", 0.08),
                ("Speed", 0.08),
                ("Acceleration", 0.06),
                ("Stamina", 0.10),
                ("Curl", 0.04),
            ]
        )
        return clamp_int(base, 40, 99)

    # FWD
    base = wavg(
        [
            ("Offensive Awareness", 0.18),
            ("Finishing", 0.18),
            ("Ball Control", 0.10),
            ("Dribbling", 0.08),
            ("Speed", 0.10),
            ("Acceleration", 0.10),
            ("Kicking Power", 0.10),
            ("Physical Contact", 0.08),
            ("Heading", 0.04),
            ("Jump", 0.04),
        ]
    )
    return clamp_int(base, 40, 99)


def boost_to_target_overall(pes_stats: Dict[str, int], position: str, target: int = 99) -> Dict[str, int]:
    """
    Gradually increase important stats until overall estimator reaches target (max 99).
    """
    pos = (position or "CF").upper()
    g = pos_group(pos)
    target = clamp_int(target, 40, 99)

    # priority list per group
    if g == "GK":
        prio = ["GK Awareness", "GK Reflexes", "GK Reach", "GK Catching", "GK Clearing"]
    elif g == "DEF":
        prio = ["Defensive Awareness", "Ball Winning", "Physical Contact", "Speed", "Aggression", "Heading", "Jump", "Low Pass", "Balance", "Stamina"]
    elif g == "MID":
        prio = ["Low Pass", "Lofted Pass", "Ball Control", "Tight Possession", "Dribbling", "Stamina", "Speed", "Acceleration", "Offensive Awareness", "Curl", "Balance"]
    else:
        prio = ["Offensive Awareness", "Finishing", "Speed", "Acceleration", "Ball Control", "Dribbling", "Kicking Power", "Physical Contact", "Heading", "Jump", "Balance"]

    out = dict(pes_stats)
    if compute_pes_overall(out, pos) >= target:
        return out

    # try up to 2000 single-step bumps
    for _ in range(2000):
        cur = compute_pes_overall(out, pos)
        if cur >= target:
            break
        bumped = False
        for k in prio:
            if k in SMALL_RANGE_ATTRS:
                continue
            if k.startswith("GK") and g != "GK":
                continue
            if out.get(k, 40) < 99:
                out[k] = out.get(k, 40) + 1
                bumped = True
                break
        if not bumped:
            break

    return out


# -----------------------------
# Dataset IO (JSONL)
# -----------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Scraper: PESMaster (build training pairs)
# -----------------------------
PESMASTER_BASE = "https://www.pesmaster.com"

LABEL_TO_EF_KEY = {
    # attacking
    "Offensive Awareness": "offensive_awareness",
    "Ball Control": "ball_control",
    "Dribbling": "dribbling",
    "Tight Possession": "tight_possession",
    "Low Pass": "low_pass",
    "Lofted Pass": "lofted_pass",
    "Finishing": "finishing",
    "Heading": "heading",
    "Set Piece Taking": "place_kicking",
    "Place Kicking": "place_kicking",
    "Curl": "curl",
    # physical
    "Speed": "speed",
    "Acceleration": "acceleration",
    "Kicking Power": "kicking_power",
    "Jump": "jump",
    "Jumping": "jump",
    "Physical Contact": "physical_contact",
    "Balance": "balance",
    "Stamina": "stamina",
    # defence
    "Defensive Awareness": "defensive_awareness",
    "Defensive Engagement": "defensive_engagement",
    "Tackling": "tackling",
    "Aggression": "aggression",
    # gk
    "GK Awareness": "goalkeeping",
    "GK Catching": "gk_catching",
    "GK Parrying": "gk_parrying",
    "GK Reflexes": "gk_reflexes",
    "GK Reach": "gk_reach",
    # small
    "Weak Foot Usage": "weak_foot_usage",
    "Weak Foot Acc.": "weak_foot_acc",
    "Weak Foot Acc": "weak_foot_acc",
    "Form": "form",
    "Injury Resistance": "injury_resistance",
}

PES_LABELS = set(PES_ATTR_ORDER) | {"Ball Winning", "GK Parrying", "Weak Foot Acc."}


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.8,id;q=0.7",
    }
    r = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.text


def find_first_href(html: str, pattern: str) -> Optional[str]:
    m = re.search(pattern, html, re.IGNORECASE)
    if not m:
        return None
    href = m.group(1)
    if href.startswith("http"):
        return href
    if href.startswith("/"):
        return PESMASTER_BASE + href
    return PESMASTER_BASE + "/" + href


def parse_stats_by_labels(html: str, wanted: Dict[str, str]) -> Dict[str, int]:
    """
    Parse page by scanning text lines. Works across different PESMaster layouts.
    wanted: label -> output_key
    """
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text("\n")
    lines = [x.strip() for x in text.split("\n") if x.strip()]
    out: Dict[str, int] = {}

    for i, line in enumerate(lines):
        for label, key in wanted.items():
            if label.lower() in line.lower():
                parts = line.split()
                for j, part in enumerate(parts):
                    if part.isdigit() or (part.replace(".", "").isdigit() and "." in part):
                        val = int(float(part))
                        if 1 <= val <= 120:
                            out[key] = val
                            break
                break
    return out


def scrape_ef_pes_pair(ef_url: str) -> Optional[Dict[str, Any]]:
    """
    Scrape eFootball player page, find matching PES 2021 page, return pair.
    Returns None if failed.
    """
    try:
        html_ef = fetch_html(ef_url)
        soup_ef = BeautifulSoup(html_ef, "lxml")
        
        name_elem = soup_ef.find("h1") or soup_ef.find("title")
        if not name_elem:
            return None
        name = name_elem.get_text().strip()
        if not name:
            return None
        
        pos_elem = soup_ef.find(string=re.compile(r"Position|Pos", re.I))
        position = "CF"
        if pos_elem:
            parent = pos_elem.find_parent()
            if parent:
                pos_text = parent.get_text()
                for p in POS_LIST:
                    if p in pos_text.upper():
                        position = p
                        break
        
        ef_stats_raw = parse_stats_by_labels(html_ef, {k: v for k, v in LABEL_TO_EF_KEY.items() if v in EF_KEYS})
        if not ef_stats_raw:
            return None
        
        ef_stats = normalize_ef_stats(ef_stats_raw)
        
        overall_elem = soup_ef.find(string=re.compile(r"Overall|Rating|OVR", re.I))
        ef_overall = None
        if overall_elem:
            parent = overall_elem.find_parent()
            if parent:
                text = parent.get_text()
                nums = re.findall(r"\d+", text)
                if nums:
                    ef_overall = float(nums[0])
        
        pes_url = find_first_href(html_ef, r'href=["\']([^"\']*pes[^"\']*2021[^"\']*)["\']')
        if not pes_url:
            pes_url = find_first_href(html_ef, r'href=["\']([^"\']*pes[^"\']*)["\']')
        
        if not pes_url:
            return None
        
        html_pes = fetch_html(pes_url)
        pes_stats_raw = parse_stats_by_labels(html_pes, {k: k for k in PES_ATTR_ORDER})
        if not pes_stats_raw:
            return None
        
        pes_stats: Dict[str, int] = {}
        for k in PES_ATTR_ORDER:
            pes_stats[k] = clamp_int(pes_stats_raw.get(k, 40), 1, 99)
        
        return {
            "name": name,
            "position": position,
            "ef": ef_stats,
            "pes": pes_stats,
            "ef_overall": ef_overall,
        }
    except Exception as e:
        print(f"Error scraping {ef_url}: {e}", file=sys.stderr)
        return None


# -----------------------------
# CLI & Flask App
# -----------------------------
def main_cli():
    parser = argparse.ArgumentParser(description="ML-only eFootball -> PES 2021 converter")
    parser.add_argument("--scrape", action="store_true", help="Scrape PESMaster URLs")
    parser.add_argument("--urls", type=str, default="", help="File with URLs (one per line)")
    parser.add_argument("--out", type=str, default=DEFAULT_DATASET_PATH, help="Output JSONL path")
    parser.add_argument("--train", action="store_true", help="Train model from dataset")
    parser.add_argument("--data", type=str, default=DEFAULT_DATASET_PATH, help="Dataset JSONL path")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Model output path")
    args = parser.parse_args()
    
    if args.scrape:
        if not args.urls:
            print("Error: --urls required for --scrape", file=sys.stderr)
            sys.exit(1)
        urls = []
        with open(args.urls, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)
        
        pairs = []
        for i, url in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}] Scraping {url}...", file=sys.stderr)
            pair = scrape_ef_pes_pair(url)
            if pair:
                pairs.append(pair)
                print(f"  OK: {pair.get('name', 'Unknown')}", file=sys.stderr)
            else:
                print(f"  FAILED", file=sys.stderr)
        
        if pairs:
            save_jsonl(args.out, pairs)
            print(f"Saved {len(pairs)} pairs to {args.out}", file=sys.stderr)
        else:
            print("No pairs scraped", file=sys.stderr)
    
    elif args.train:
        samples = load_jsonl(args.data)
        if len(samples) < 8:
            print(f"Error: Need at least 8 samples, got {len(samples)}", file=sys.stderr)
            sys.exit(1)
        
        print(f"Training on {len(samples)} samples...", file=sys.stderr)
        model = train_ensemble(samples)
        dump(model, args.model)
        print(f"Model saved to {args.model}", file=sys.stderr)
    
    else:
        parser.print_help()


def to_markdown_table(pes_stats: Dict[str, int], ef_stats: Dict[str, float]) -> str:
    lines = []
    lines.append("| PES 2021 Attribute | Converted Value | eFootball Value |")
    lines.append("|---|---:|---:|")
    
    for attr in PES_ATTR_ORDER:
        if attr == "Ball Winning":
            de = ef_stats.get("defensive_engagement")
            tk = ef_stats.get("tackling")
            if de is not None and tk is not None:
                ef_val = int(round(0.5 * (float(de) + float(tk))))
            else:
                ef_val = ""
        else:
            key_map = {
                "Offensive Awareness": "offensive_awareness",
                "Ball Control": "ball_control",
                "Dribbling": "dribbling",
                "Tight Possession": "tight_possession",
                "Low Pass": "low_pass",
                "Lofted Pass": "lofted_pass",
                "Finishing": "finishing",
                "Heading": "heading",
                "Place Kicking": "place_kicking",
                "Curl": "curl",
                "Speed": "speed",
                "Acceleration": "acceleration",
                "Kicking Power": "kicking_power",
                "Jump": "jump",
                "Physical Contact": "physical_contact",
                "Balance": "balance",
                "Stamina": "stamina",
                "Defensive Awareness": "defensive_awareness",
                "Aggression": "aggression",
                "GK Awareness": "goalkeeping",
                "GK Catching": "gk_catching",
                "GK Clearing": "gk_parrying",
                "GK Reflexes": "gk_reflexes",
                "GK Reach": "gk_reach",
                "Weak Foot Usage": "weak_foot_usage",
                "Weak Foot Accuracy": "weak_foot_acc",
                "Form": "form",
                "Injury Resistance": "injury_resistance",
            }
            ef_key = key_map.get(attr)
            ef_val = ef_stats.get(ef_key, "") if ef_key else ""
        
        lines.append(f"| {attr} | {pes_stats.get(attr, '')} | {ef_val} |")
    
    return "\n".join(lines)


app = Flask(__name__)
trained_model: Optional[TrainedModel] = None


def load_model_if_exists() -> Optional[TrainedModel]:
    global trained_model
    if trained_model is not None:
        return trained_model
    
    if os.path.exists(DEFAULT_MODEL_PATH):
        try:
            trained_model = load(DEFAULT_MODEL_PATH)
            return trained_model
        except Exception as e:
            print(f"Warning: Failed to load model: {e}", file=sys.stderr)
    
    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/convert", methods=["POST"])
def api_convert():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400
        
        ef_stats_raw = data.get("ef_stats", {})
        position = data.get("position", "CF")
        ef_overall = data.get("ef_overall")
        
        if not isinstance(ef_stats_raw, dict):
            return jsonify({"error": "ef_stats must be a dictionary"}), 400
        
        ef_stats = normalize_ef_stats(ef_stats_raw)
        
        model = load_model_if_exists()
        
        if model:
            pes_stats = predict_ensemble(model, ef_stats, position, ef_overall)
        else:
            base = baseline_pes_guess(ef_stats, position)
            pes_stats = {k: clamp_int(v, 1, 99) for k, v in base.items()}
            
            gk_mode = pos_group(position) == "GK" or ef_stats.get("goalkeeping", 0.0) >= 60.0
            if not gk_mode:
                pes_stats["GK Awareness"] = 40
                pes_stats["GK Catching"] = 40
                pes_stats["GK Clearing"] = 40
                pes_stats["GK Reflexes"] = 40
                pes_stats["GK Reach"] = 40
            
            for a, (lo, hi) in SMALL_RANGE_ATTRS.items():
                key = {
                    "Weak Foot Usage": "weak_foot_usage",
                    "Weak Foot Accuracy": "weak_foot_acc",
                    "Form": "form",
                    "Injury Resistance": "injury_resistance",
                }[a]
                pes_stats[a] = clamp_int(ef_stats.get(key, lo), lo, hi)
        
        pes_stats = apply_ef_max_clamp(pes_stats, ef_stats)
        
        pes_overall = compute_pes_overall(pes_stats, position)
        
        markdown_table = to_markdown_table(pes_stats, ef_stats)
        
        return jsonify({
            "pes_stats": pes_stats,
            "pes_overall": pes_overall,
            "markdown_table": markdown_table
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main_cli()
    else:
        app.run(debug=True, host="0.0.0.0", port=5000)
