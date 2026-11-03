"""测试夹具与会话守卫：每测独立临时工作区 + 生产账本零写入（M11 01 号票）。

三层保障：

1. **换址**：`client` / `ui_client` 按测（或会话一次）构建应用，库、数据集根、产物根、
   模型库根全部落进该测的 tmp_path，并就地补丁 `config.load_config`——线程内 live 读配置
   （`get_datasets_root` / 模型库 / 自检）同源换址；LLM 端点与桌面通知一并静音。
2. **惰性装配**：`visiontrainer.main` 的模块级 app 改惰性构建，收集阶段 import 不再建引擎、
   不再建目录。
3. **对账**：会话首尾只读比对生产 SQLite 库（逐表行数 + 文件指纹）与生产数据集/产物根
   （仅 stat 清单，不读内容），任一变动即把本次 pytest 会话判失败。

坐标基线直读仓库根 `config.toml`（补丁前的 load_config 真身），守卫看到的永远是生产真值。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from visiontrainer import config as config_module
from visiontrainer.datasets import service as svc
from visiontrainer.datasets.split import build_manifest, save_manifest
from visiontrainer.datasets.stats import build_stats, save_stats
from visiontrainer.main import create_app

REPO_ROOT = Path(__file__).resolve().parents[1]
_real_load_config = config_module.load_config  # 补丁前真身：换址基线与生产坐标都从这里取

PNG_MAGIC = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPG_MAGIC = b"\xff\xd8\xff\xe0" + b"\x00" * 32


# ---------------------------------------------------------------- 生产账本对账

def prod_coords() -> tuple[Path, Path, Path]:
    """(生产库, 生产数据集根, 生产产物根)——config 缺省键按 config.py 的同名缺省补齐。"""
    training = _real_load_config(REPO_ROOT / "config.toml").get("training", {})
    db = Path(training.get("db_url", "sqlite:///data/visiontrainer.db").removeprefix("sqlite:///"))
    return (db if db.is_absolute() else REPO_ROOT / db,
            Path(training.get("datasets_dir", "~/visiontrainer/datasets")).expanduser(),
            Path(training.get("runs_dir", "~/visiontrainer/runs")).expanduser())


def _table_counts(db: Path) -> dict[str, int | str]:
    """只读打开生产库取逐表行数；不可读时记错误串，比对阶段自然暴露为差异。"""
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return {"_open_error": str(exc)}
    try:
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return {n: con.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0] for n in names}
    except sqlite3.Error as exc:
        return {"_query_error": str(exc)}
    finally:
        con.close()


def _tree_fingerprint(root: Path) -> dict:
    """目录清单指纹：相对路径 + 字节数 + mtime_ns 排序后哈希（只 stat，不读文件内容）。"""
    if not root.exists():
        return {"missing": True, "n_files": 0, "digest": ""}
    entries: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            try:
                st = os.stat(Path(dirpath, name))
            except OSError:
                continue
            entries.append(f"{Path(dirpath, name).relative_to(root)}|{st.st_size}|{st.st_mtime_ns}")
    entries.sort()
    return {"n_files": len(entries),
            "digest": hashlib.sha256("\n".join(entries).encode()).hexdigest()}


def snapshot(db: Path, datasets: Path, runs: Path) -> dict:
    db_exists = db.exists()
    return {
        "db": {"path": str(db), "exists": db_exists,
               "size": db.stat().st_size if db_exists else 0,
               "sha256": hashlib.sha256(db.read_bytes()).hexdigest() if db_exists else "",
               "counts": _table_counts(db) if db_exists else {}},
        "datasets": {"path": str(datasets), **_tree_fingerprint(datasets)},
        "runs": {"path": str(runs), **_tree_fingerprint(runs)},
    }


def diff(before: dict, after: dict) -> list[str]:
    """差异清单（中文，逐条定位到表/根）；空列表即生产账本零变动。"""
    out: list[str] = []
    b, a = before["db"], after["db"]
    if b["exists"] != a["exists"]:
        out.append(f"生产库存在性变化：{b['path']} {b['exists']} → {a['exists']}")
    for key in ("size", "sha256"):
        if b[key] != a[key]:
            out.append(f"生产库{key}变化：{b[key]} → {a[key]}（{b['path']}）")
    for table in sorted(set(b["counts"]) | set(a["counts"])):
        if b["counts"].get(table) != a["counts"].get(table):
            out.append(f"生产库表 {table} 行数变化：{b['counts'].get(table)} → {a['counts'].get(table)}")
    for area, label in (("datasets", "数据集"), ("runs", "产物")):
        b_t, a_t = before[area], after[area]
        if (b_t.get("digest"), b_t.get("n_files")) != (a_t.get("digest"), a_t.get("n_files")):
            out.append(f"生产{label}根 {b_t['path']} 变动："
                       f"文件数 {b_t.get('n_files')} → {a_t.get('n_files')}，"
                       f"指纹 {str(b_t.get('digest'))[:12]} → {str(a_t.get('digest'))[:12]}")
    return out


# ---------------------------------------------------------------- 换址

def _overrides(roots: Path) -> dict[str, dict]:
    """落到 roots 的换址覆盖 + 外呼静音（discard 端口让泄漏的 LLM 调用直接连接拒绝，不烧真实配额）。"""
    return {
        "training": {
            "db_url": f"sqlite:///{roots / 'visiontrainer-test.db'}",
            "datasets_dir": str(roots / "datasets"),
            "runs_dir": str(roots / "runs"),
            "model_lib_dir": str(roots / "models"),
            "check_occupancy": False,
        },
        "notifications": {"desktop": False},
        "llm": {"base_url": "http://127.0.0.1:9/v1", "api_key": "test-key"},
    }


def _apply(base: dict, overrides: dict[str, dict]) -> dict:
    merged = dict(base)
    for section, kv in overrides.items():
        merged[section] = {**base.get(section, {}), **kv}
    return merged


def isolated_cfg(roots: Path) -> dict:
    """生产配置真值 + 换址覆盖（深拷贝，夹具改它不伤盘上配置）。"""
    return _apply(_real_load_config(REPO_ROOT / "config.toml"), _overrides(roots))


def _prepare_roots(cfg: dict) -> None:
    for section in ("datasets_dir", "runs_dir", "model_lib_dir"):
        Path(cfg["training"][section]).mkdir(parents=True, exist_ok=True)


def _static_config_users() -> list:
    """顶部静态 `from ..config import load_config` 的模块：改绑前取到的是生产配置。"""
    from visiontrainer import config
    from visiontrainer.agent import decider, llm, tuner
    from visiontrainer.ui import pages

    return [config, decider, llm, tuner, pages]


_STATIC_CONFIG_USERS = _static_config_users()


# ---------------------------------------------------------------- 夹具

@pytest.fixture()
def datasets_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "datasets"
    root.mkdir(exist_ok=True)
    monkeypatch.setattr(svc, "get_datasets_root", lambda: root)
    return root


def make_image(dir_: Path, name: str, magic: bytes = PNG_MAGIC) -> Path:
    p = dir_ / name
    p.write_bytes(magic)
    return p


def make_annotation(dir_: Path, stem: str, shapes: list[dict], width=640, height=480, flags=None) -> Path:
    p = dir_ / f"{stem}.json"
    p.write_text(
        json.dumps({"imageWidth": width, "imageHeight": height, "shapes": shapes, "flags": flags or {}}, ensure_ascii=False),
        encoding="utf-8",
    )
    return p


def rect(label: str, x0=10, y0=10, x1=110, y1=90) -> dict:
    return {"label": label, "shape_type": "rectangle", "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


def build_dataset(name: str, task_type: str, source_dir: Path | str, ratio: float = 0.8,
                  seed: int = 42, stratify: bool = False,
                  classes: list[str] | None = None, root: Path | None = None) -> dict:
    """票 03 测试侧转换助手（替代已退役的 `svc.import_dataset`）：托管出生 + 首次切分。

    票 03 唯一出生入口 `create_dataset` 只做校验 + 画像、**不切分**；训练/审计/审查侧
    既有用例仍需真实切分清单，本助手 = labeled 态 `create_dataset` + `build_manifest` +
    `_derive` + `save_manifest` + 画像重算，保留旧导入「进来即切分」语义（切分实体归票 05
    后再由此处统一换轨）。classes 缺省时按 `create_dataset` 的阻断口径探测类别表
    （全部 shape label + flags 真键），保证转换后的用例不会意外触发外类别 409。
    """
    src = Path(source_dir)
    if classes is None:
        classes = sorted(svc._probe_source_labels(src))
    created = svc.create_dataset(name, task_type, classes, "labeled", str(src), root=root)
    base = root or svc.get_datasets_root()
    ds_dir = base / name
    source = ds_dir / svc.SOURCE_SUBDIR
    final_classes = created["classes"]
    scan = svc.scan_entries(source, task_type, final_classes)
    entries = [{"image": e["image"], "label": e["label"], "primary_label": e.get("primary_label")}
               for e in scan["entries"]]
    manifest = build_manifest(entries, ratio, seed, stratify)
    save_manifest(ds_dir, manifest)
    svc._derive(ds_dir, manifest, task_type, final_classes)
    raw = svc._read_meta(ds_dir / "dataset.json")
    raw["manifests"] = [svc._manifest_brief(manifest)]
    raw["active_manifest"] = manifest["uid"]
    svc._write_meta(ds_dir, raw)
    save_stats(name, build_stats(scan["entries"], manifest, task_type, final_classes, name))
    return svc.get_dataset(name, root)


@pytest.fixture()
def detect_source(tmp_path: Path) -> Path:
    """8 张 detect 标注（car/person 两类）+ 1 张负样本。坏 JSON 场景各测试自建
    （开训新鲜度门对清单内 bad_json 拒训，训练链路夹具须保持清单干净）。"""
    src = tmp_path / "annot"
    src.mkdir()
    for i in range(4):
        make_image(src, f"car{i}.png")
        make_annotation(src, f"car{i}", [rect("car")])
    for i in range(4):
        make_image(src, f"person{i}.png")
        make_annotation(src, f"person{i}", [rect("person", 20, 30, 200, 220)])
    make_image(src, "negative.png")
    make_annotation(src, "negative", [])
    return src


@pytest.fixture()
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """本测专属工作区配置，并把 config.load_config 换成读它的版本。

    函数内 live import 的模块（`get_datasets_root` / 模型库 / 自检 / 通知）随模块补丁生效；
    顶部 `from ..config import load_config` 静态绑定的模块（agent.llm / agent.tuner /
    agent.decider / ui.pages）逐个改绑——否则测试期仍会拿生产 config.toml 的 LLM 密钥外呼。
    """
    cfg = isolated_cfg(tmp_path)
    _prepare_roots(cfg)
    overrides = _overrides(tmp_path)
    real = _real_load_config

    def load_config(path: Path | None = None) -> dict:
        return _apply(real(path), overrides)

    monkeypatch.setattr(config_module, "load_config", load_config)
    for mod in _STATIC_CONFIG_USERS:
        if hasattr(mod, "load_config"):
            monkeypatch.setattr(mod, "load_config", load_config)
    return cfg


@pytest.fixture()
def client(isolated_config: dict) -> TestClient:
    """按测构建的 REST 客户端：独立临时库 + 独立数据集根 + 独立产物根，不跑 lifespan。"""
    return TestClient(create_app(cfg=isolated_config, with_ui=False))


@pytest.fixture(scope="session")
def ui_client():
    """会话一次构建的带 UI 客户端（NiceGUI 全局态不可重复装配）：同样落进会话临时工作区。"""
    root = Path(tempfile.mkdtemp(prefix="visiontrainer-tests-ui-"))
    cfg = isolated_cfg(root)
    _prepare_roots(cfg)
    try:
        with TestClient(create_app(cfg=cfg, with_ui=True)) as c:
            yield c
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- 会话守卫

_SNAPSHOT: dict = {}


def pytest_configure(config: pytest.Config) -> None:
    _SNAPSHOT.clear()
    _SNAPSHOT.update(snapshot(*prod_coords()))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not _SNAPSHOT:
        return
    problems = diff(_SNAPSHOT, snapshot(*prod_coords()))
    rep = session.config.pluginmanager.get_plugin("terminalreporter")
    if problems:
        if rep:
            rep.write_sep("=", "生产账本对账失败：测试会话改动了生产数据", red=True)
            for p in problems:
                rep.write_line(f"  - {p}")
        else:
            print("[生产账本对账失败] " + "；".join(problems), file=sys.stderr)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    elif rep:
        rep.write_line("生产账本对账：生产库与生产数据集/产物根零变动 ✔")


def _thread_excepthook(args):
    print(f"THREAD CRASH in {args.thread.name}: {args.exc_type.__name__}: {args.exc_value}", file=sys.stderr)


threading.excepthook = _thread_excepthook
