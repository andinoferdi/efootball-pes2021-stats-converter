#!/usr/bin/env python3
# Konversi stat eFootball 2025 -> PES 2021 + opsi ML linear regression

import json
import math
import argparse
import sys
from typing import Dict, List, Tuple, Optional
from flask import Flask, render_template, request, jsonify


# -------------------------------------------------------
# Util umum
# -------------------------------------------------------

def clamp_int(x: float, lo: int, hi: int) -> int:
    v = int(round(x))
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


# -------------------------------------------------------
# Rumus dasar konversi (rule based)
# -------------------------------------------------------

def convert_main_stat(ef_value: float,
                      max_ef: float = 103.0,
                      exp: float = 0.30,
                      low_cut: float = 60.0,
                      low_add: float = 6.0) -> int:
    """Rumus utama untuk stat non-defence dan non-GK."""
    v = float(ef_value)
    if v < low_cut:
        return clamp_int(v + low_add, 1, 99)
    scaled = 99.0 * pow(v / max_ef, exp)
    return clamp_int(scaled, 1, 99)


def convert_def_stat(ef_value: float,
                     a: float = 0.733,
                     b: float = 23.467) -> int:
    """Rumus stat bertahan: Defensive Awareness, Aggression, dll."""
    v = float(ef_value)
    y = a * v + b
    return clamp_int(y, 1, 99)


def convert_ball_winning(def_engagement: float,
                         tackling: float) -> int:
    """Hitung Ball Winning dari Defensive Engagement dan Tackling."""
    ef_bw = 0.5 * (float(def_engagement) + float(tackling))
    return convert_def_stat(ef_bw)


def convert_gk_stat(ef_value: float,
                    is_gk: bool,
                    gk_bonus: int = 3,
                    bonus_cut: float = 90.0) -> int:
    """Konversi stat GK. Untuk pemain non-kiper, paksa 40."""
    if not is_gk:
        return 40
    base = convert_main_stat(ef_value)
    if float(ef_value) >= bonus_cut:
        base += gk_bonus
    return clamp_int(base, 1, 99)


def clamp_small_range(value: float,
                      lo: int,
                      hi: int) -> int:
    return clamp_int(value, lo, hi)


# -------------------------------------------------------
# Mapping nama atribut
# -------------------------------------------------------

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

# key di input eFootball
EF_KEY_FOR_PES_ATTR: Dict[str, str] = {
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
    # Ball Winning pakai kombinasi
    "Ball Winning": "ball_winning_helper",
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


# -------------------------------------------------------
# Konversi rule based penuh
# -------------------------------------------------------

def convert_rule_based(ef: Dict[str, float],
                       position: str = "CF") -> Dict[str, int]:
    """Konversi stat eFootball -> PES 2021 pakai rumus manual."""
    goalkeeping = float(ef.get("goalkeeping", 0))
    is_gk = position.upper() == "GK" or goalkeeping >= 60.0

    out: Dict[str, int] = {}

    # Attacking dan passing
    out["Offensive Awareness"] = convert_main_stat(ef.get("offensive_awareness", 1))
    out["Ball Control"] = convert_main_stat(ef.get("ball_control", 1))
    out["Dribbling"] = convert_main_stat(ef.get("dribbling", 1))
    out["Tight Possession"] = convert_main_stat(ef.get("tight_possession", 1))
    out["Low Pass"] = convert_main_stat(ef.get("low_pass", 1))
    out["Lofted Pass"] = convert_main_stat(ef.get("lofted_pass", 1))
    out["Finishing"] = convert_main_stat(ef.get("finishing", 1))
    out["Heading"] = convert_main_stat(ef.get("heading", 1))
    out["Place Kicking"] = convert_main_stat(ef.get("place_kicking", 1))
    out["Curl"] = convert_main_stat(ef.get("curl", 1))

    # Fisik
    out["Speed"] = convert_main_stat(ef.get("speed", 1))
    out["Acceleration"] = convert_main_stat(ef.get("acceleration", 1))
    out["Kicking Power"] = convert_main_stat(ef.get("kicking_power", 1))
    out["Jump"] = convert_main_stat(ef.get("jump", 1))
    out["Physical Contact"] = convert_main_stat(ef.get("physical_contact", 1))
    out["Balance"] = convert_main_stat(ef.get("balance", 1))
    out["Stamina"] = convert_main_stat(ef.get("stamina", 1))

    # Defence
    out["Defensive Awareness"] = convert_def_stat(ef.get("defensive_awareness", 1))
    out["Aggression"] = convert_def_stat(ef.get("aggression", 1))
    out["Ball Winning"] = convert_ball_winning(
        ef.get("defensive_engagement", 1),
        ef.get("tackling", 1),
    )

    # GK
    out["GK Awareness"] = convert_gk_stat(goalkeeping, is_gk)
    out["GK Catching"] = convert_gk_stat(ef.get("gk_catching", 0), is_gk)
    out["GK Clearing"] = convert_gk_stat(ef.get("gk_parrying", 0), is_gk)
    out["GK Reflexes"] = convert_gk_stat(ef.get("gk_reflexes", 0), is_gk)
    out["GK Reach"] = convert_gk_stat(ef.get("gk_reach", 0), is_gk)

    # Weak foot, form, injury
    out["Weak Foot Usage"] = clamp_small_range(ef.get("weak_foot_usage", 1), 1, 4)
    out["Weak Foot Accuracy"] = clamp_small_range(ef.get("weak_foot_acc", 1), 1, 4)
    out["Form"] = clamp_small_range(ef.get("form", 1), 1, 8)
    out["Injury Resistance"] = clamp_small_range(ef.get("injury_resistance", 1), 1, 3)

    return out


# -------------------------------------------------------
# Machine learning sederhana: linear regression per atribut
# -------------------------------------------------------

def fit_simple_linear(xs: List[float],
                      ys: List[float]) -> Tuple[float, float]:
    """OLS 1D: y = a*x + b."""
    n = len(xs)
    if n == 0:
        return 1.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return 1.0, 0.0
    a = num / den
    b = mean_y - a * mean_x
    return a, b


def train_models(training_samples: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    training_samples: list of dict
    {
      "name": "...",
      "position": "CF" / "RMF" / "CB" / "GK" / ...
      "ef": { ... stat eF ... },
      "pes": { ... stat PES target ... }
    }
    """
    models: Dict[str, Dict[str, float]] = {}

    for attr in PES_ATTR_ORDER:
        xs: List[float] = []
        ys: List[float] = []
        ef_key = EF_KEY_FOR_PES_ATTR.get(attr)
        if ef_key is None:
            continue

        for sample in training_samples:
            ef_stats = sample["ef"]
            pes_stats = sample["pes"]
            if attr not in pes_stats:
                continue
            if attr == "Ball Winning":
                de = ef_stats.get("defensive_engagement")
                tk = ef_stats.get("tackling")
                if de is None or tk is None:
                    continue
                ef_val = 0.5 * (float(de) + float(tk))
            else:
                if ef_key not in ef_stats:
                    continue
                ef_val = float(ef_stats[ef_key])

            xs.append(ef_val)
            ys.append(float(pes_stats[attr]))

        if len(xs) >= 2:
            a, b = fit_simple_linear(xs, ys)
            models[attr] = {"a": a, "b": b}

    return models


def convert_with_models(ef: Dict[str, float],
                        position: str,
                        models: Dict[str, Dict[str, float]]) -> Dict[str, int]:
    """
    Konversi pakai model ML kalau ada.
    Kalau suatu atribut belum punya model, pakai rule based.
    """
    base_rb = convert_rule_based(ef, position)
    out: Dict[str, int] = {}

    goalkeeping = float(ef.get("goalkeeping", 0))
    is_gk = position.upper() == "GK" or goalkeeping >= 60.0

    for attr in PES_ATTR_ORDER:
        ef_key = EF_KEY_FOR_PES_ATTR.get(attr)
        model = models.get(attr)

        if attr == "Ball Winning":
            de = ef.get("defensive_engagement")
            tk = ef.get("tackling")
            if de is None or tk is None or model is None:
                out[attr] = base_rb[attr]
            else:
                ef_val = 0.5 * (float(de) + float(tk))
                a = model["a"]
                b = model["b"]
                out[attr] = clamp_int(a * ef_val + b, 1, 99)
            continue

        if ef_key is None or ef_key not in ef:
            out[attr] = base_rb[attr]
            continue

        ef_val = float(ef[ef_key])

        # GK tetap hormati aturan 40 untuk non-kiper
        if attr.startswith("GK"):
            if not is_gk:
                out[attr] = 40
                continue
            if model is None:
                out[attr] = base_rb[attr]
            else:
                a = model["a"]
                b = model["b"]
                out[attr] = clamp_int(a * ef_val + b, 1, 99)
            continue

        # Weak foot, form, injury tetap pakai clamp saja
        if attr in ["Weak Foot Usage", "Weak Foot Accuracy", "Form", "Injury Resistance"]:
            out[attr] = base_rb[attr]
            continue

        if model is None:
            out[attr] = base_rb[attr]
        else:
            a = model["a"]
            b = model["b"]
            out[attr] = clamp_int(a * ef_val + b, 1, 99)

    return out


# -------------------------------------------------------
# Contoh struktur data training (Kamu isi sendiri)
# -------------------------------------------------------

# Default kosong supaya script tetap jalan.
# Kamu bisa isi manual nanti dengan stat asli dari gambar
# misalnya Ronaldo CF, Kahn GK, Beckenbauer CB, Beckham RMF.
TRAINING_SAMPLES_EXAMPLE: List[Dict] = [
    # Contoh kerangka satu pemain:
    # {
    #   "name": "Cristiano Ronaldo",
    #   "position": "CF",
    #   "ef": {
    #       "offensive_awareness": 96,
    #       "ball_control": 84,
    #       ...
    #   },
    #   "pes": {
    #       "Offensive Awareness": 98,
    #       "Ball Control": 95,
    #       ...
    #   }
    # },
]


# -------------------------------------------------------
# Output tabel Markdown
# -------------------------------------------------------

def to_markdown_table(pes_stats: Dict[str, int],
                      ef_stats: Dict[str, float]) -> str:
    lines = []
    lines.append("| Atribut PES 2021 | Nilai konversi | Nilai eFootball |")
    lines.append("|---|---:|---:|")
    for attr in PES_ATTR_ORDER:
        ef_key = EF_KEY_FOR_PES_ATTR.get(attr)
        if attr == "Ball Winning":
            # tampilkan helper average di kolom eFootball
            de = ef_stats.get("defensive_engagement")
            tk = ef_stats.get("tackling")
            if de is not None and tk is not None:
                ef_val = round_half_up(0.5 * (float(de) + float(tk)))
            else:
                ef_val = ""
        else:
            ef_val = ef_stats.get(ef_key, "") if ef_key else ""
        lines.append(f"| {attr} | {pes_stats.get(attr, '')} | {ef_val} |")
    return "\n".join(lines)


# -------------------------------------------------------
# CLI
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Konversi stat eFootball 2025 -> PES 2021 dengan rumus + ML sederhana"
    )
    parser.add_argument("--in",
                        dest="infile",
                        default="",
                        help="Path JSON input stat eFootball. Kosong = baca dari stdin")
    parser.add_argument("--position",
                        dest="position",
                        default="CF",
                        help="Posisi pemain, contoh: CF, RMF, CB, GK")
    parser.add_argument("--use-ml",
                        action="store_true",
                        help="Aktifkan model ML linear berdasarkan TRAINING_SAMPLES_EXAMPLE")
    parser.add_argument("--show-table",
                        action="store_true",
                        help="Cetak tabel Markdown")
    args = parser.parse_args()

    # Baca input eFootball
    if args.infile:
        with open(args.infile, "r", encoding="utf-8") as f:
            ef_stats = json.load(f)
    else:
        ef_stats = json.load(sys.stdin)

    # Pilih mode
    if args.use_ml and len(TRAINING_SAMPLES_EXAMPLE) >= 2:
        models = train_models(TRAINING_SAMPLES_EXAMPLE)
        pes_stats = convert_with_models(ef_stats, args.position, models)
    else:
        pes_stats = convert_rule_based(ef_stats, args.position)

    print(json.dumps(pes_stats, ensure_ascii=False, indent=2))

    if args.show_table:
        print()
        print(to_markdown_table(pes_stats, ef_stats))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        app = Flask(__name__)
        
        @app.route("/")
        def index():
            return render_template("index.html")
        
        @app.route("/api/convert", methods=["POST"])
        def api_convert():
            try:
                data = request.get_json()
                if not data:
                    return jsonify({"error": "Invalid JSON"}), 400
                
                ef_stats = data.get("ef_stats", {})
                position = data.get("position", "CF")
                use_ml = data.get("use_ml", False)
                
                if not isinstance(ef_stats, dict):
                    return jsonify({"error": "ef_stats must be a dictionary"}), 400
                
                if use_ml and len(TRAINING_SAMPLES_EXAMPLE) >= 2:
                    models = train_models(TRAINING_SAMPLES_EXAMPLE)
                    pes_stats = convert_with_models(ef_stats, position, models)
                else:
                    pes_stats = convert_rule_based(ef_stats, position)
                
                markdown_table = to_markdown_table(pes_stats, ef_stats)
                
                return jsonify({
                    "pes_stats": pes_stats,
                    "markdown_table": markdown_table
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        app.run(debug=True, host="0.0.0.0", port=5000)
