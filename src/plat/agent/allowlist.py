"""超参白名单（06 号票）：Agent 可修改的参数集合与数值边界，越界 clamp 并记录。"""

from __future__ import annotations

# (min, max) 边界；缺省值取 ultralytics 默认超参
ALLOWLIST: dict[str, dict] = {
    "lr0":            {"min": 0.00001, "max": 0.1,   "default": 0.01},
    "lrf":            {"min": 0.001,  "max": 0.2,   "default": 0.01},
    "momentum":       {"min": 0.6,    "max": 0.99,  "default": 0.937},
    "weight_decay":   {"min": 0.0,    "max": 0.001, "default": 0.0005},
    "warmup_epochs":  {"min": 0.0,    "max": 10.0,  "default": 3.0},
    "warmup_momentum": {"min": 0.0,   "max": 1.0,   "default": 0.8},
    "box":            {"min": 1.0,    "max": 20.0,  "default": 7.5},
    "cls":            {"min": 0.2,    "max": 4.0,   "default": 0.5},
    "dfl":            {"min": 0.5,    "max": 4.0,   "default": 1.5},
    "hsv_h":          {"min": 0.0,    "max": 0.1,   "default": 0.015},
    "hsv_s":          {"min": 0.0,    "max": 0.95,  "default": 0.7},
    "hsv_v":          {"min": 0.0,    "max": 0.95,  "default": 0.4},
    "degrees":        {"min": 0.0,    "max": 30.0,  "default": 0.0},
    "translate":      {"min": 0.0,    "max": 0.5,   "default": 0.1},
    "scale":          {"min": 0.0,    "max": 1.0,   "default": 0.5},
    "fliplr":         {"min": 0.0,    "max": 1.0,   "default": 0.5},
    "mosaic":         {"min": 0.0,    "max": 1.0,   "default": 1.0},
    "mixup":          {"min": 0.0,    "max": 1.0,   "default": 0.0},
    "close_mosaic":   {"min": 0,      "max": 50,    "default": 10},
    "patience":       {"min": 5,      "max": 500,   "default": 100},
}


def clamp_changes(changes: dict) -> tuple[dict, list[dict]]:
    """应用白名单边界，返回 (合法变更, clamp 记录)。非白名单键直接拒绝记录。"""
    applied: dict = {}
    clamps: list[dict] = []
    for key, value in (changes or {}).items():
        if key not in ALLOWLIST:
            clamps.append({"key": key, "requested": value, "reason": "not_in_allowlist"})
            continue
        spec = ALLOWLIST[key]
        try:
            v = float(value)
        except (TypeError, ValueError):
            clamps.append({"key": key, "requested": value, "reason": "not_numeric"})
            continue
        if v < spec["min"]:
            clamps.append({"key": key, "requested": v, "clamped_to": spec["min"]})
            v = spec["min"]
        elif v > spec["max"]:
            clamps.append({"key": key, "requested": v, "clamped_to": spec["max"]})
            v = spec["max"]
        applied[key] = v
    return applied, clamps
