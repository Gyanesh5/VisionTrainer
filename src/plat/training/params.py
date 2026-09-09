"""ultralytics 训练参数快照与校验——用户可设的全量官方 train 参数（对齐 ultralytics 8.4.100）。

来源：训练 venv 的 ultralytics.cfg.DEFAULT_CFG_DICT（111 键）。分类：
- FIRST_CLASS        —— 平台 API 一等字段（epochs/imgsz/...），顶层表单处理，禁止出现在 extra_args；
- PLATFORM_MANAGED   —— 平台契约参数（model/data/project/name/resume...），不开放；
- NON_TRAIN_ARGS     —— predict/val/export/track 模式专用，train 不消费，不开放；
- GROUPS             —— 其余全部 train 参数，按 UI 分组暴露，可经 extra_args 设置。

升级 ultralytics 后需重新核对（tests/test_params.py 有覆盖一致性断言）。
"""

from __future__ import annotations

ULTRALYTICS_VERSION = "8.4.100"

# API 一等字段（create_task 显形参数）
FIRST_CLASS = frozenset({"epochs", "imgsz", "batch", "device", "workers", "seed", "patience"})

# 平台契约参数——trainer_entry/平台自身管理
PLATFORM_MANAGED = frozenset({
    "task", "mode", "model", "data", "project", "name", "exist_ok", "cfg",
    "resume",   # 断点续训走平台 /runs/{uid}/resume，语义与 ultralytics resume 不同
    "workspace",
})

# train 模式不消费的参数（predict/val/export/track 专用）
NON_TRAIN_ARGS = frozenset({
    "source", "vid_stride", "stream_buffer", "visualize", "augment", "agnostic_nms",
    "classes", "retina_masks", "embed", "show", "save_frames", "save_txt", "save_conf",
    "save_crop", "show_labels", "show_conf", "show_boxes", "line_width", "format",
    "keras", "optimize", "dynamic", "simplify", "opset", "nms", "dnn", "quantize",
    "end2end", "tracker", "save_json",
})

# 分组定义：type ∈ {int, float, bool, str, int|list, bool|str}；
# 数值键可带 min/max guard（防呆，不是官方约束）；enum 键带 options。
# default 为 ultralytics 8.4.100 官方默认值（None 表示默认不设）。
GROUPS: dict[str, list[dict]] = {
    "训练基础": [
        {"key": "time", "type": "float", "default": None, "label": "time（时长上限·小时，与 epochs 二选一）", "min": 0.0},
        {"key": "save", "type": "bool", "default": True, "label": "save（保存 ckpt/结果）"},
        {"key": "save_period", "type": "int", "default": -1, "label": "save_period（checkpoint 间隔 epoch，-1 关）", "min": -1},
        {"key": "cache", "type": "bool|str", "default": False, "label": "cache（数据缓存）", "options": ["ram", "disk"]},
        {"key": "pretrained", "type": "bool", "default": True, "label": "pretrained（预训练权重）"},
        {"key": "cls_remap", "type": "bool", "default": True, "label": "cls_remap（类别重映射）"},
        {"key": "verbose", "type": "bool", "default": True, "label": "verbose（详细日志）"},
        {"key": "deterministic", "type": "bool", "default": True, "label": "deterministic（确定性训练）"},
        {"key": "single_cls", "type": "bool", "default": False, "label": "single_cls（按单类训练）"},
        {"key": "rect", "type": "bool", "default": False, "label": "rect（矩形训练）"},
        {"key": "amp", "type": "bool", "default": True, "label": "amp（混合精度）"},
        {"key": "fraction", "type": "float", "default": 1.0, "label": "fraction（数据集采样比例）", "min": 0.0, "max": 1.0},
        {"key": "profile", "type": "bool", "default": False, "label": "profile（耗时分析）"},
        {"key": "compile", "type": "bool", "default": False, "label": "compile（torch.compile）"},
        {"key": "multi_scale", "type": "float", "default": 0.0, "label": "multi_scale（多尺度变化幅度，0 关）", "min": 0.0, "max": 0.5},
        {"key": "freeze", "type": "int|list", "default": None, "label": "freeze（冻结层数，如 10 或 0,1,2）"},
        {"key": "overlap_mask", "type": "bool", "default": True, "label": "overlap_mask（seg mask 重叠）"},
        {"key": "mask_ratio", "type": "int", "default": 4, "label": "mask_ratio（seg mask 下采样）", "min": 1},
        {"key": "dropout", "type": "float", "default": 0.0, "label": "dropout", "min": 0.0, "max": 1.0},
        {"key": "val", "type": "bool", "default": True, "label": "val（训练中验证）"},
        {"key": "plots", "type": "bool", "default": True, "label": "plots（训练图表）"},
    ],
    "优化器与调度": [
        {"key": "optimizer", "type": "str", "default": "auto", "label": "optimizer",
         "options": ["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"]},
        {"key": "lr0", "type": "float", "default": 0.01, "label": "lr0（初始学习率）", "min": 0.0, "max": 0.1},
        {"key": "lrf", "type": "float", "default": 0.01, "label": "lrf（最终 lr 系数）", "min": 0.0, "max": 1.0},
        {"key": "momentum", "type": "float", "default": 0.937, "label": "momentum", "min": 0.6, "max": 0.99},
        {"key": "weight_decay", "type": "float", "default": 0.0005, "label": "weight_decay", "min": 0.0, "max": 0.01},
        {"key": "warmup_epochs", "type": "float", "default": 3.0, "label": "warmup_epochs", "min": 0.0, "max": 10.0},
        {"key": "warmup_momentum", "type": "float", "default": 0.8, "label": "warmup_momentum", "min": 0.0, "max": 1.0},
        {"key": "warmup_bias_lr", "type": "float", "default": 0.1, "label": "warmup_bias_lr", "min": 0.0, "max": 0.1},
        {"key": "cos_lr", "type": "bool", "default": False, "label": "cos_lr（余弦退火）"},
        {"key": "nbs", "type": "int", "default": 64, "label": "nbs（名义 batch）", "min": 1},
        {"key": "distill_model", "type": "str", "default": None, "label": "distill_model（蒸馏教师）"},
        {"key": "dis", "type": "float", "default": 6.0, "label": "dis（蒸馏增益）", "min": 0.0},
    ],
    "损失权重": [
        {"key": "box", "type": "float", "default": 7.5, "label": "box", "min": 0.0, "max": 20.0},
        {"key": "cls", "type": "float", "default": 0.5, "label": "cls", "min": 0.0, "max": 4.0},
        {"key": "cls_pw", "type": "float", "default": 0.0, "label": "cls_pw", "min": 0.0, "max": 4.0},
        {"key": "dfl", "type": "float", "default": 1.5, "label": "dfl", "min": 0.0, "max": 4.0},
        {"key": "pose", "type": "float", "default": 12.0, "label": "pose（姿态）", "min": 0.0, "max": 20.0},
        {"key": "kobj", "type": "float", "default": 1.0, "label": "kobj（关键点可见性）", "min": 0.0, "max": 10.0},
        {"key": "rle", "type": "float", "default": 1.0, "label": "rle（旋转回归）", "min": 0.0, "max": 10.0},
        {"key": "angle", "type": "float", "default": 1.0, "label": "angle（OBB 角度）", "min": 0.0, "max": 10.0},
    ],
    "数据增强": [
        {"key": "hsv_h", "type": "float", "default": 0.015, "label": "hsv_h（色调）", "min": 0.0, "max": 0.1},
        {"key": "hsv_s", "type": "float", "default": 0.7, "label": "hsv_s（饱和度）", "min": 0.0, "max": 0.95},
        {"key": "hsv_v", "type": "float", "default": 0.4, "label": "hsv_v（明度）", "min": 0.0, "max": 0.95},
        {"key": "degrees", "type": "float", "default": 0.0, "label": "degrees（旋转°）", "min": 0.0, "max": 180.0},
        {"key": "translate", "type": "float", "default": 0.1, "label": "translate（平移）", "min": 0.0, "max": 0.9},
        {"key": "scale", "type": "float", "default": 0.5, "label": "scale（缩放）", "min": 0.0, "max": 1.0},
        {"key": "shear", "type": "float", "default": 0.0, "label": "shear（剪切°）", "min": 0.0, "max": 90.0},
        {"key": "perspective", "type": "float", "default": 0.0, "label": "perspective（透视）", "min": 0.0, "max": 0.01},
        {"key": "flipud", "type": "float", "default": 0.0, "label": "flipud（上下翻转）", "min": 0.0, "max": 1.0},
        {"key": "fliplr", "type": "float", "default": 0.5, "label": "fliplr（左右翻转）", "min": 0.0, "max": 1.0},
        {"key": "bgr", "type": "float", "default": 0.0, "label": "bgr（通道翻转）", "min": 0.0, "max": 1.0},
        {"key": "mosaic", "type": "float", "default": 1.0, "label": "mosaic（马赛克拼接）", "min": 0.0, "max": 1.0},
        {"key": "mixup", "type": "float", "default": 0.0, "label": "mixup", "min": 0.0, "max": 1.0},
        {"key": "cutmix", "type": "float", "default": 0.0, "label": "cutmix", "min": 0.0, "max": 1.0},
        {"key": "copy_paste", "type": "float", "default": 0.0, "label": "copy_paste（实例复制粘贴）", "min": 0.0, "max": 1.0},
        {"key": "copy_paste_mode", "type": "str", "default": "flip", "label": "copy_paste_mode", "options": ["flip", "mixup"]},
        {"key": "auto_augment", "type": "str", "default": "randaugment", "label": "auto_augment（分类）",
         "options": ["randaugment", "augmix", "autoaugment"]},
        {"key": "erasing", "type": "float", "default": 0.4, "label": "erasing（分类随机擦除）", "min": 0.0, "max": 1.0},
        {"key": "close_mosaic", "type": "int", "default": 10, "label": "close_mosaic（末尾 N 关 mosaic）", "min": 0, "max": 100},
    ],
    "验证与阈值": [
        {"key": "split", "type": "str", "default": "val", "label": "split（验证划分名）"},
        {"key": "conf", "type": "float", "default": None, "label": "conf（验证置信度阈值）", "min": 0.0, "max": 1.0},
        {"key": "iou", "type": "float", "default": 0.7, "label": "iou（验证 NMS IoU）", "min": 0.0, "max": 1.0},
        {"key": "max_det", "type": "int", "default": 300, "label": "max_det（最大检出数）", "min": 1},
    ],
}

PARAMS: dict[str, dict] = {e["key"]: e | {"group": g} for g, entries in GROUPS.items() for e in entries}


def _coerce(entry: dict, value) -> tuple[object, str | None]:
    """按声明类型转换值；返回 (转换值, 错误)。None 直接放行（=不设置→移除）。"""
    if value is None:
        return None, None
    t, k = entry["type"], entry["key"]
    try:
        if t == "int":
            if isinstance(value, bool):
                return None, f"{k}: 需要整数"
            f = float(value)
            if f != int(f):
                return None, f"{k}: 需要整数"
            v = int(f)
        elif t == "float":
            if isinstance(value, bool):
                return None, f"{k}: 需要数值"
            v = float(value)
        elif t == "bool":
            if isinstance(value, bool):
                v = value
            elif isinstance(value, (int, float)) and value in (0, 1):
                v = bool(value)
            elif isinstance(value, str) and value.lower() in ("true", "false"):
                v = value.lower() == "true"
            else:
                return None, f"{k}: 需要布尔（true/false）"
        elif t == "str":
            v = str(value)
            if "options" in entry and v not in entry["options"]:
                return None, f"{k}: 只允许 {entry['options']}"
        elif t == "int|list":  # freeze：10 或 "0,1,2"
            if isinstance(value, str):
                parts = [p.strip() for p in value.split(",") if p.strip()]
                if len(parts) > 1:
                    return [int(p) for p in parts], None
                value = parts[0]
            if isinstance(value, bool):
                return None, f"{k}: 需要整数或逗号分隔整数列表"
            return int(float(value)), None
        elif t == "bool|str":  # cache：False/True/'ram'/'disk'
            if isinstance(value, bool):
                v = value
            elif isinstance(value, str) and value.lower() in entry["options"]:
                v = value.lower()
            elif isinstance(value, str) and value.lower() in ("true", "false"):
                v = value.lower() == "true"
            else:
                return None, f"{k}: 只允许 true/false/{entry['options']}"
        else:  # pragma: no cover - schema 自身错误
            return None, f"{k}: 未知类型 {t}"
    except (TypeError, ValueError):
        return None, f"{k}: 值 {value!r} 无法转为 {t}"

    if isinstance(v, (int, float)) and not isinstance(v, bool):
        lo, hi = entry.get("min"), entry.get("max")
        if lo is not None and v < lo:
            return None, f"{k}: {v} 低于下限 {lo}"
        if hi is not None and v > hi:
            return None, f"{k}: {v} 高于上限 {hi}"
    return v, None


def validate_extra_args(extra: dict | None) -> tuple[dict, list[str]]:
    """校验用户训练参数：键合法、类型可转、数值在 guard 内。

    返回 (净化后的参数, 错误列表)；错误非空时调用方应整体拒绝。
    一等字段（epochs 等）由 API 显形参数承担，出现在 extra 里视为错误。
    """
    applied: dict = {}
    errors: list[str] = []
    for key, value in (extra or {}).items():
        if key in FIRST_CLASS:
            errors.append(f"{key}: 是顶层字段，请勿放在训练参数里")
            continue
        entry = PARAMS.get(key)
        if entry is None:
            errors.append(f"{key}: 不是可设置的 ultralytics 训练参数")
            continue
        v, err = _coerce(entry, value)
        if err:
            errors.append(err)
        elif v is not None:
            applied[key] = v
    return applied, errors
