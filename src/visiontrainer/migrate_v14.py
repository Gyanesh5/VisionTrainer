"""migrate_v14：v1.3 → v1.4 一次性迁移（票 10；spec §3.5；ADR 0009–0012）。

CLI：`python -m visiontrainer.migrate_v14 --dry-run|--apply`（平台 venv 手动执行，
不随启动自动跑；--apply 前自动备份 SQLite 库）。硬切换：无兼容读层、无映射表、
无过渡窗——机制面全废，结果面保留（旧 Run 可续训/错误分析，靠迁移时就地改写快照）。

步骤：
  真源入位   `mv <旧 source>/ → <root>/<名>/source/`（同分区 rename）；`_delete_/` 并入
             `<ds>/_delete_/`；classes.txt 逐字节不变（迁移前断言）；dataset.json 重写为
             瘦身形态；excluded 搬 `<ds>/excluded.json`（同名取最早 at）。
  孤儿集     circle-mix-800 直接删除（唯一豁免回收区语义）。
  切分转制   旧 derived/<uid>/split_manifest.json → splits/<new_uid>/split.json（内容寻址），
             同划分去重；memo 记旧 uid 溯源；active 版本成为「最近切分」；旧树删除。
  tile2x2    dataset.json.tiling.groups 由 `_r{r}c{c}` 反推回填，任一瓦片无法映射即中止。
  库改写     schema 5→6；tasks 三列回填（dataset/split_uid/format_id）+ 删旧列；
             两本账表重建弃档（init_db 既有逻辑）。
  快照改写   每个 run_manifest.json 就地补 split/effective/format 三段（uid 换新、
             实效集按当时名单现算、class_map 按 classes 行序）。

幂等：每集 `.migrated-v14` 标记，中断重跑跳过已完成集；库与快照改写幂等
（schema 版本 / 列存在性 / 快照新键判断）。回退：恢复 DB 备份 + 反向 rename（手工，
步骤由 --dry-run 打印）。杂物不动（孤儿 task 目录、0 字节 .db）。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

SNAPSHOT_NAME = "run_manifest.json"
ORPHAN_SETS = {"circle-mix-800"}        # 用户点名直接删除（唯一豁免回收区）
TILE_RE = "_r"                          # 瓦片名反推源图的分隔（<源stem>_r{r}c{c}）
KEEP_SIBLINGS = ("labels", "backup", "duplicates", "test", "scanning", "plate_images",
                 "__pycache__")         # 原地不入平台（.backup-* 前缀另判）


def classes_sha(classes: list[str]) -> str:
    import hashlib

    return hashlib.sha256("\n".join(classes).encode("utf-8")).hexdigest()


def split_uid_of(train: list[str], val: list[str]) -> str:
    import hashlib

    payload = json.dumps(sorted(set(train)), ensure_ascii=False) + \
        json.dumps(sorted(set(val)), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(p: Path, data: dict) -> None:
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- 发现

def _managed_sets(root: Path) -> list[Path]:
    """平台根下的旧托管集目录：有 dataset.json 且非外部注册。"""
    out = []
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        if d.is_dir() and (d / "dataset.json").is_file() and d.name != ".external":
            out.append(d)
    return out


def _old_source_dir(meta: dict, ds_dir: Path) -> Path | None:
    """v1.3 真源：优先 dataset.json.source_dir；缺省视作根内 images/。"""
    src = meta.get("source_dir")
    if src and Path(src).is_dir():
        return Path(src)
    if (ds_dir / "images").is_dir():
        return ds_dir / "images"
    if (ds_dir / "source").is_dir():
        return ds_dir / "source"
    return None


def _manifests_of(ds_dir: Path) -> list[tuple[str, dict]]:
    """旧派生树里的全部清单 [(uid, manifest)]。"""
    out = []
    derived = ds_dir / "derived"
    if not derived.is_dir():
        return out
    for d in sorted(derived.iterdir()):
        m = d / "split_manifest.json"
        if d.is_dir() and m.is_file():
            try:
                out.append((d.name, _read_json(m)))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    return out


# ---------------------------------------------------------------- 单集迁移

def migrate_dataset(ds_dir: Path, actions: list[str], apply: bool) -> dict:
    """迁移一个托管集；返回 {name, old_uid→new_uid 映射} 供库改写。"""
    meta = _read_json(ds_dir / "dataset.json")
    name = meta.get("name") or ds_dir.name
    marker = ds_dir / ".migrated-v14"
    if marker.exists():
        return {"name": name, "uid_map": {}, "skipped": True}
    actions.append(f"[{name}] 开始迁移（origin={meta.get('origin') or '旧账'}）")

    # ① 真源入位
    src = _old_source_dir(meta, ds_dir)
    if src is None:
        raise SystemExit(f"[{name}] 找不到真源目录（source_dir 缺失且无 images/）——中止")
    source = ds_dir / "source"
    if src.resolve() != source.resolve():
        if apply:
            source.parent.mkdir(parents=True, exist_ok=True)
            if source.exists():
                raise SystemExit(f"[{name}] {source} 已存在——入位前须清场，中止")
            shutil.move(str(src), str(source))
        actions.append(f"[{name}] mv {src} → {source}")
    # _delete_/ 并入 <ds>/_delete_/
    old_del = src.parent / "_delete_"
    if old_del.is_dir() and any(old_del.iterdir()):
        dst_del = ds_dir / "_delete_"
        if apply:
            dst_del.mkdir(parents=True, exist_ok=True)
            for p in old_del.iterdir():
                shutil.move(str(p), str(dst_del / p.name))
        actions.append(f"[{name}] 合并 XAL 回收 {old_del} → {dst_del}")

    # ② classes.txt 断言（逐字节不变的前提是它已存在）
    classes_txt = ds_dir / "classes.txt"
    if not classes_txt.is_file():
        classes = meta.get("classes") or []
        if not classes:
            raise SystemExit(f"[{name}] classes.txt 缺失且旧账无类别——中止")
        if apply:
            classes_txt.write_text("".join(f"{c}\n" for c in classes), encoding="utf-8")
        actions.append(f"[{name}] 从旧账写 classes.txt（{len(classes)} 类）")
    classes = [ln.strip() for ln in classes_txt.read_text(encoding="utf-8").splitlines()
               if ln.strip()]

    # ③ excluded → excluded.json（同名取最早 at）
    excluded_file = ds_dir / "excluded.json"
    entries: dict[str, dict] = {}
    if excluded_file.is_file():
        try:
            old_ex = _read_json(excluded_file)
            entries = {k: dict(v) for k, v in (old_ex.get("files") or {}).items()}
        except (json.JSONDecodeError, UnicodeDecodeError):
            entries = {}
    for item in meta.get("excluded") or []:
        if isinstance(item, dict) and item.get("file"):
            rec = {"at": item.get("at") or "", "reason": item.get("reason") or ""}
            cur = entries.get(item["file"])
            if cur is None or (rec["at"] and cur.get("at") and rec["at"] < cur["at"]):
                entries[item["file"]] = rec
    if apply and entries:
        _write_json(excluded_file, {"version": 1, "files": entries})
    if entries:
        actions.append(f"[{name}] excluded.json {len(entries)} 条")

    # ⑤ tile2x2 groups 回填（_r{r}c{c} 反推；任一失败即中止）
    tiling = meta.get("tiling")
    if tiling is not None and not tiling.get("groups") and source.is_dir():
        groups: dict[str, list[str]] = {}
        miss: list[str] = []
        for p in sorted(source.iterdir()):
            if not p.is_file() or p.suffix.lower() == ".json":
                continue
            if TILE_RE not in p.stem:
                miss.append(p.name)
                continue
            src_stem = p.stem.rsplit(TILE_RE, 1)[0]
            if not src_stem:
                miss.append(p.name)
                continue
            groups.setdefault(f"{src_stem}.png", []).append(p.name)
        if miss:
            raise SystemExit(f"[{name}] tiling.groups 回填失败：{len(miss)} 个瓦片无法映射"
                             f"（如 {miss[0]}）——中止")
        meta["tiling"] = {**tiling, "groups": groups}
        actions.append(f"[{name}] tiling.groups 回填 {sum(len(v) for v in groups.values())} 瓦片"
                       f" / {len(groups)} 源图")

    # ④ 切分转制（两遍：先算 new_uid 映射，再写实体；lineage 由事件推断）
    manifests = _manifests_of(ds_dir)
    uid_map: dict[str, str] = {}
    for old_uid, m in manifests:
        train = [Path(p).name for p in m.get("train") or []]
        val = [Path(p).name for p in m.get("val") or []]
        uid_map[old_uid] = split_uid_of(train, val)
    active = str(meta.get("active_manifest") or "")
    now = datetime.now().isoformat(timespec="seconds")
    seen_new: dict[str, str] = {}
    for old_uid, m in manifests:
        train = [Path(p).name for p in m.get("train") or []]
        val = [Path(p).name for p in m.get("val") or []]
        new_uid = uid_map[old_uid]
        created = m.get("created_at") or ""
        if old_uid == active and created != max(
                (mm.get("created_at") or "") for _, mm in manifests):
            created = now                     # active 版本成为「最近切分」
        events = bool(m.get("excluded") or m.get("absorbed") or m.get("restored"))
        parent_old = m.get("prev_uid")
        entity = {
            "uid": new_uid,
            "created_at": created or now,
            "train": sorted(set(train)),
            "val": sorted(set(val)),
            "params": {"seed": m.get("seed", 42), "ratio": m.get("ratio", 0.8),
                       "stratify": bool(m.get("stratify")),
                       "approved_only": bool(m.get("approved_only"))},
            "lineage": {"mode": "inherit" if (events or parent_old) else "fresh",
                        "parent_uid": uid_map.get(parent_old)},
            "grouped": "source_image" if (meta.get("tiling") or {}).get("groups") else None,
            "counts": {"train": len(set(train)), "val": len(set(val))},
            "memo": f"迁移自 {old_uid}（{m.get('created_at') or '?'}）",
        }
        if new_uid in seen_new:
            continue                          # 同划分去重（首份留痕）
        seen_new[new_uid] = old_uid
        if apply:
            path = ds_dir / "splits" / new_uid / "split.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.is_file():
                _write_json(path, entity)
    if manifests:
        actions.append(f"[{name}] 切分转制 {len(manifests)} 份 → {len(seen_new)} 实体"
                       f"（active {active} → {uid_map.get(active)}）")

    # ⑥ dataset.json 重写为瘦身形态（classes.txt 唯一真源；excluded 进 excluded.json）
    lv = meta.get("last_validation") or {}
    old_drops = meta.get("drops") or {}
    new_meta = {
        "name": name,
        "task_type": meta.get("task_type", "detect"),
        "form": "managed",
        "created_at": meta.get("created_at") or now,
        "origin": {"mode": "annotated_dir", "path": str(src),
                   "at": meta.get("created_at") or now, "transfer": "move"},
        "validation_summary": {
            "ok": bool(lv.get("ok", meta.get("validation_ok", True))),
            "total_images": int(lv.get("total_images") or meta.get("total_images") or 0),
            "negatives": int(lv.get("background") or meta.get("negatives") or 0),
            "checked_at": lv.get("checked_at"),
            "source_fingerprint": lv.get("source_fingerprint") or "",
            "drops": {"count": int(old_drops.get("count") or (lv.get("drops") or 0)),
                      "signature": old_drops.get("signature") or "",
                      "acked": old_drops.get("acked") or ""},
        },
    }
    if (meta.get("tiling") or {}).get("groups"):
        new_meta["tiling"] = meta["tiling"]
    if meta.get("flip_idx"):
        new_meta["flip_idx"] = meta["flip_idx"]
    if apply:
        _write_json(ds_dir / "dataset.json", new_meta)
        _write_json(marker, {"migrated_at": now, "old_uids": sorted(uid_map),
                             "new_uids": sorted(seen_new)})
        # ⑦ 旧树删除（池按需重建；以 format_id 名与 8 位 uid 名区分新旧）
        derived = ds_dir / "derived"
        if derived.is_dir():
            shutil.rmtree(derived)
    if manifests:
        actions.append(f"[{name}] dataset.json 瘦身 + 旧 derived/ 树删除 + .migrated-v14 标记")
    return {"name": name, "uid_map": uid_map, "skipped": False,
            "classes": classes, "source": source}


# ---------------------------------------------------------------- 库与快照

def _migrate_db(db: Path, root: Path, uid_maps: dict[str, dict[str, str]],
                actions: list[str], apply: bool) -> None:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    ver = con.execute("SELECT version FROM schema_version WHERE id=1").fetchone()[0]
    cols = [r[1] for r in con.execute("PRAGMA table_info(tasks)")]
    if ver >= 6 and "dataset_manifest_path" not in cols:
        con.close()
        return                            # 已迁移（幂等）
    # 两本账重建弃档（票 06 一行制新形状；判据 = 旧 manifest_uid 列还在）
    for table, ddl in (
        ("dataset_stats",
         "CREATE TABLE IF NOT EXISTS dataset_stats (id INTEGER PRIMARY KEY, "
         "dataset VARCHAR(255) NOT NULL, source_fingerprint VARCHAR(64) NOT NULL, "
         "task_type VARCHAR(16) NOT NULL, computed_at DATETIME, payload JSON)"),
        ("dataset_audits",
         "CREATE TABLE IF NOT EXISTS dataset_audits (id INTEGER PRIMARY KEY, "
         "dataset VARCHAR(255) NOT NULL, source_fingerprint VARCHAR(64) NOT NULL, "
         "created_at DATETIME, payload JSON)"),
    ):
        tcols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
        if tcols and "manifest_uid" in tcols:
            if apply:
                con.execute(f"DROP TABLE {table}")
                con.execute(ddl)
                con.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS ux_{table}_dataset "
                            f"ON {table} (dataset)")
            actions.append(f"[db] {table} 重建为一行制（历史行清零）")
    # tasks：三列回填 + 删旧列
    if "dataset_manifest_path" in cols:
        rows = con.execute("SELECT id, dataset_manifest_path FROM tasks "
                           "WHERE dataset_manifest_path IS NOT NULL").fetchall()
        if apply:
            con.execute("ALTER TABLE tasks ADD COLUMN dataset VARCHAR(255)")
            con.execute("ALTER TABLE tasks ADD COLUMN split_uid VARCHAR(32)")
            con.execute("ALTER TABLE tasks ADD COLUMN format_id VARCHAR(32) NOT NULL DEFAULT 'yolo'")
        for r in rows:
            path = Path(r["dataset_manifest_path"])
            # <root>/<名>/derived/<uid>/data.yaml
            ds_name = path.parent.parent.parent.name
            old_uid = path.parent.name
            new_uid = uid_maps.get(ds_name, {}).get(old_uid)
            if apply:
                if new_uid is None:
                    con.close()
                    raise SystemExit(f"[db] task#{r['id']} 的旧清单 uid {old_uid} 无转制映射——中止")
                con.execute("UPDATE tasks SET dataset=?, split_uid=?, format_id='yolo' WHERE id=?",
                            (ds_name, new_uid, r["id"]))
            actions.append(f"[db] task#{r['id']} → ({ds_name}, {new_uid}, yolo)")
        if apply:
            con.execute("ALTER TABLE tasks DROP COLUMN dataset_manifest_path")
        actions.append("[db] tasks 三列落位 + dataset_manifest_path 删除")
    if apply:
        con.execute("UPDATE schema_version SET version=6 WHERE id=1")
        con.commit()
    else:
        actions.append("[db] schema_version 5 → 6")
    con.close()


def _migrate_snapshots(runs_root: Path, uid_maps: dict[str, dict[str, str]],
                       actions: list[str], apply: bool) -> None:
    """旧 run_manifest.json 就地补写 split/effective/format 三段（幂等：新键已齐即跳过）。"""
    if not runs_root.is_dir():
        return
    for task_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            snap_path = run_dir / SNAPSHOT_NAME
            if not snap_path.is_file():
                continue
            snap = _read_json(snap_path)
            if "format" in snap and "split_manifest" not in snap:
                continue                    # 已是新格式（幂等）
            ds_dir = Path(snap.get("dataset_dir") or "")
            ds_name = ds_dir.name
            sm = snap.get("split_manifest") or {}
            old_uid = sm.get("uid")
            new_uid = uid_maps.get(ds_name, {}).get(old_uid)
            if new_uid is None:
                actions.append(f"[snap {run_dir.name}] 数据集 {ds_name} 无映射（外部集/孤儿）——跳过")
                continue
            split_path = ds_dir / "splits" / new_uid / "split.json"
            if not split_path.is_file():
                actions.append(f"[snap {run_dir.name}] 切分实体缺失（{ds_name}/{new_uid}）——跳过")
                continue
            split = _read_json(split_path)
            members = set(split["train"]) | set(split["val"])
            excluded = set()
            ex_file = ds_dir / "excluded.json"
            if ex_file.is_file():
                try:
                    excluded = set(_read_json(ex_file).get("files") or {})
                except (json.JSONDecodeError, UnicodeDecodeError):
                    excluded = set()
            source = ds_dir / "source"
            dropped = {"missing": [], "excluded": []}
            eff = {"train": [], "val": []}
            for side in ("train", "val"):
                for n in split[side]:
                    if not (source / n).is_file():
                        dropped["missing"].append(n)
                    elif n in excluded:
                        dropped["excluded"].append(n)
                    else:
                        eff[side].append(n)
            classes = [ln.strip() for ln in (ds_dir / "classes.txt").read_text(
                encoding="utf-8").splitlines() if ln.strip()]
            new_snap = dict(snap)
            new_snap.pop("split_manifest", None)
            new_snap["split"] = {"uid": new_uid,
                                 "params": split.get("params") or {},
                                 "memo": split.get("memo") or "",
                                 "grouped": split.get("grouped")}
            new_snap["effective"] = {"train": sorted(eff["train"]), "val": sorted(eff["val"]),
                                     "dropped": {k: sorted(v) for k, v in dropped.items()}}
            new_snap["format"] = {"format_id": "yolo", "converter_version": "yolo-1",
                                  "projection_version": "v14-1",
                                  "class_map": {c: i for i, c in enumerate(classes)},
                                  "entry_path": str(run_dir / "data")}
            if apply:
                _write_json(snap_path, new_snap)
            actions.append(f"[snap {run_dir.name}] split/effective/format 三段就地补写"
                           f"（{old_uid} → {new_uid}，实效 t/v "
                           f"{len(eff['train'])}/{len(eff['val'])}）")


# ---------------------------------------------------------------- 编排

def _rollback_steps(db: Path, root: Path) -> list[str]:
    return [
        "手工回退步骤（无自动 --rollback）：",
        f"  1. 停 8600；恢复库备份：cp {db}.bak-<ts> {db}",
        f"  2. 反向 rename：对每集 mv <root>/<名>/source → 原真源目录"
        f"（原路径见各集 dataset.json origin.path 的迁移前备份）",
        "  3. 删除各集 splits/ excluded.json .migrated-v14；dataset.json 无法自动还原——",
        "     依赖第 1 步的库备份与 exports 之外的旧账副本（迁移前建议整集 tar 备份）",
    ]


def main(argv: list[str] | None = None) -> int:
    from visiontrainer.config import load_config

    ap = argparse.ArgumentParser(prog="migrate_v14",
                                 description="v1.3 → v1.4 一次性迁移（票 10，spec §3.5）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="只打印将执行的动作与回退步骤")
    g.add_argument("--apply", action="store_true", help="执行迁移（先自动备份 SQLite 库）")
    ap.add_argument("--datasets-root", default=None, help="覆盖 config 的 datasets_dir（测试用）")
    ap.add_argument("--runs-root", default=None, help="覆盖 config 的 runs_dir（测试用）")
    ap.add_argument("--db", default=None, help="覆盖 config 的 db_url 指向的库文件（测试用）")
    args = ap.parse_args(argv)

    cfg = load_config().get("training", {})
    root = Path(args.datasets_root or cfg.get("datasets_dir", "~/visiontrainer/datasets")).expanduser()
    runs_root = Path(args.runs_root or cfg.get("runs_dir", "~/visiontrainer/runs")).expanduser()
    db_url = cfg.get("db_url", "sqlite:///data/visiontrainer.db")
    db = Path(args.db or db_url.removeprefix("sqlite:///"))
    if not db.is_absolute():
        db = Path.cwd() / db

    actions: list[str] = []
    apply = bool(args.apply)
    print(f"migrate_v14 {'--apply' if apply else '--dry-run'} · root={root} · db={db}")

    if apply:
        if db.is_file():
            bak = db.with_name(f"{db.name}.bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            shutil.copy2(db, bak)
            print(f"[db] 备份 {db} → {bak}")
        else:
            print(f"[db] 库不存在（{db}）——只做文件面迁移")

    uid_maps: dict[str, dict[str, str]] = {}
    classes_map: dict[str, list[str]] = {}
    # ① 孤儿集直接删除（唯一豁免回收区）
    for name in ORPHAN_SETS:
        d = root / name
        if d.is_dir():
            actions.append(f"[{name}] 孤儿集整目录删除（真源已消失，豁免回收区）")
            if apply:
                shutil.rmtree(d)
    # ② 逐集迁移
    for ds_dir in _managed_sets(root):
        if not (ds_dir / ".migrated-v14").exists() and \
                (ds_dir / "derived").is_dir() or (ds_dir / "images").is_dir() or \
                (ds_dir / "splits").is_dir():
            info = migrate_dataset(ds_dir, actions, apply)
            uid_maps[info["name"]] = info.get("uid_map") or {}
            if info.get("classes"):
                classes_map[info["name"]] = info["classes"]
        else:
            actions.append(f"[{ds_dir.name}] 已迁移或无旧账（跳过）")
            uid_maps[ds_dir.name] = {}

    # ③ 库与快照
    if db.is_file():
        _migrate_db(db, root, uid_maps, actions, apply)
    _migrate_snapshots(runs_root, uid_maps, actions, apply)

    print("\n".join(actions) if actions else "（无事可做）")
    print("\n".join(_rollback_steps(db, root)))
    if not apply:
        print("\ndry-run 结束：以上为将执行的动作清单，未改动任何文件。")
    else:
        print("\napply 完成：重启平台（schema 6 代码）后按 spec §3.5 验收八条逐项过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
