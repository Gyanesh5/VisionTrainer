"""票 05 切片：内容寻址的切分实体——`<数据集>/splits/<uid>/split.json`。

契约来源：docs/spec.md §2.2（切分实体与选择器 / 可比性不变式 / 实效集 / 新建两模式）与
§3.2（split.json 字段契约）、ADR 0010、docs/api.md 的 `/splits` 系列端点——本切片只做
端点消费的**纯文件层函数**（REST/service/UI/训练三列归集成波）。

关键口径：uid = sha256(json(train)+json(val))[:8]，参数、分组、memo 一律不进哈希；
成员/参数/血缘不可变，只有 memo 可改；无 active 指针，「最近」＝ created_at 最新。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import make_annotation, make_image, rect

from visiontrainer.datasets import review as rv
from visiontrainer.datasets import split as sp


CLASSES = ["car", "person"]


def _ds(tmp_path: Path, name: str = "demo") -> Path:
    """最小托管数据集目录（纯文件层，不经 service 建行）：source/ + classes.txt。"""
    ds = tmp_path / name
    (ds / "source").mkdir(parents=True)
    (ds / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    return ds


def _add(ds: Path, stem: str, label: str | None = None, *, approved: bool = True) -> str:
    """入图：label=None 即负样本（无同名 JSON）；approved 走平台审查窄写（ADR 0003）。"""
    source = ds / "source"
    make_image(source, f"{stem}.png")
    if label is None:
        return f"{stem}.png"
    json_path = make_annotation(source, stem, [rect(label)])
    if approved:
        rv.write_review(json_path, "approved")
    return f"{stem}.png"


def _write_excluded(ds: Path, names: list[str]) -> None:
    payload = {"version": 1, "files": {n: {"at": "2026-11-03T10:00:00", "reason": "脏标"} for n in names}}
    (ds / "excluded.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _split_dirs(ds: Path) -> list[str]:
    root = ds / "splits"
    return sorted(p.name for p in root.iterdir()) if root.is_dir() else []


def _plan(ds: Path, **kw) -> dict:
    params = {"mode": "fresh", "seed": 42, "ratio": 0.8, "stratify": False, "approved_only": False}
    params.update(kw)
    return sp.plan_split(ds, **params)


def _entity(train: list[str], val: list[str], created_at: str, memo: str = "") -> dict:
    """手造实体字典（历史列表用例需要可控的 created_at）。"""
    return {
        "uid": sp.split_uid(train, val),
        "created_at": created_at,
        "train": list(train),
        "val": list(val),
        "params": {"seed": 42, "ratio": 0.8, "stratify": False, "approved_only": False},
        "lineage": {"mode": "fresh", "parent_uid": None},
        "grouped": None,
        "memo": memo,
    }


# ---------------------------------------------------------------- uid 与落盘（验收 1）

def test_同参数同成员重复落盘_命中同一uid不产生第二份文件(tmp_path):
    ds = _ds(tmp_path)
    for i in range(6):
        _add(ds, f"car{i}", "car")
    for i in range(2):
        _add(ds, f"neg{i}")

    first = _plan(ds)
    second = _plan(ds)
    assert first["uid"] == second["uid"]  # 同划分 → 同 uid（可复现）

    path = sp.save_split(ds, first)
    again = sp.save_split(ds, second)
    assert path == again
    assert path.name == "split.json"
    assert path.parent.parent.name == "splits"
    assert _split_dirs(ds) == [first["uid"]]                     # 不产生第二份
    assert list(path.parent.iterdir()) == [path]
    assert sp.load_split(ds, first["uid"])["train"] == second["train"]


def test_改一个成员即换uid(tmp_path):
    ds = _ds(tmp_path)
    for i in range(5):
        _add(ds, f"car{i}", "car")
    _add(ds, "neg0")

    before = _plan(ds)
    sp.save_split(ds, before)
    _add(ds, "car9", "car")
    after = _plan(ds)

    assert after["uid"] != before["uid"]
    sp.save_split(ds, after)
    assert _split_dirs(ds) == sorted([before["uid"], after["uid"]])
    assert set(after["train"]) | set(after["val"]) == set(sp.eligible_names(ds, approved_only=False))


def test_参数不进哈希_划分内容不变则uid不变(tmp_path):
    """与旧清单口径的关键差异：uid 不再吃 seed/ratio，参数只是溯源记账。"""
    ds = _ds(tmp_path)
    for i in range(4):
        _add(ds, f"neg{i}")  # 全负样本 → 恒进 train，划分与 seed/ratio 无关

    a = _plan(ds, seed=42, ratio=0.8)
    b = _plan(ds, seed=7, ratio=0.5, stratify=True)
    assert a["train"] == b["train"] and a["val"] == b["val"]
    assert a["uid"] == b["uid"]
    sp.save_split(ds, a)
    sp.save_split(ds, b)
    assert _split_dirs(ds) == [a["uid"]]                          # 命中既有实体，复用

    edited = dict(b)
    edited["params"] = {"seed": 1, "ratio": 0.1, "stratify": False, "approved_only": True}
    assert sp.save_split(ds, edited) == ds / "splits" / a["uid"] / "split.json"
    assert sp.load_split(ds, a["uid"])["params"] == a["params"]   # 复用不覆盖既有记账


def test_split_uid_只由名单决定(tmp_path):
    assert sp.split_uid(["b.png", "a.png"], ["c.png"]) == sp.split_uid(["a.png", "b.png"], ["c.png"])
    assert sp.split_uid(["a.png"], ["b.png"]) != sp.split_uid(["a.png", "b.png"], [])


# ---------------------------------------------------------------- 合格成员（验收 4 前置）

def test_eligible_names_在盘且未排除_恢复即回效(tmp_path):
    ds = _ds(tmp_path)
    keep = _add(ds, "car0", "car")
    out = _add(ds, "car1", "car")
    ghost = _add(ds, "car2", "car")
    (ds / "source" / "car2.png").unlink()

    assert sp.eligible_names(ds, approved_only=False) == [keep, out]

    _write_excluded(ds, [out])
    assert sp.eligible_names(ds, approved_only=False) == [keep]
    _write_excluded(ds, [])
    assert sp.eligible_names(ds, approved_only=False) == [keep, out]


def test_approved_only_只烘焙审查通过_负样本恒合格(tmp_path):
    ds = _ds(tmp_path)
    approved = _add(ds, "car0", "car", approved=True)
    _add(ds, "person0", "person", approved=False)   # 有 JSON 但未过审
    _add(ds, "person1", "person", approved=False)
    negative = _add(ds, "neg0")                     # 无同名 JSON → 恒合格

    assert sp.eligible_names(ds, approved_only=True) == [approved, negative]
    assert set(sp.eligible_names(ds, approved_only=False)) == {
        approved, "person0.png", "person1.png", negative}

    plan = _plan(ds, approved_only=True)
    assert negative in plan["train"]                # 负样本只进 train
    assert negative not in plan["val"]
    assert plan["params"]["approved_only"] is True


def test_负样本恒只进train_标注图按ratio分布(tmp_path):
    ds = _ds(tmp_path)
    for i in range(10):
        _add(ds, f"car{i}", "car")
    negatives = [_add(ds, f"neg{i}") for i in range(3)]

    plan = _plan(ds, ratio=0.8)
    assert set(negatives) <= set(plan["train"])
    assert set(negatives).isdisjoint(plan["val"])
    assert plan["counts"] == {"train": len(plan["train"]), "val": len(plan["val"])}
    assert len(plan["val"]) == 2                    # int(10*0.8) → 标注图各归其位


# ---------------------------------------------------------------- 新建两模式（验收 4）

def test_inherit_锁定参照参数_既有归属一字不动只放新文件(tmp_path):
    ds = _ds(tmp_path)
    for i in range(10):
        _add(ds, f"car{i}", "car")
    base = _plan(ds, ratio=0.8)
    sp.save_split(ds, base)

    for i in range(2):
        _add(ds, f"new{i}", "car")
    inher = _plan(ds, mode="inherit", parent_uid=base["uid"], seed=999, ratio=0.2, stratify=True)

    assert inher["params"]["seed"] == base["params"]["seed"]          # 以参照为准
    assert inher["params"]["ratio"] == base["params"]["ratio"]
    assert inher["params"]["stratify"] is base["params"]["stratify"]
    assert inher["lineage"] == {"mode": "inherit", "parent_uid": base["uid"]}
    for name in base["train"]:                                        # 既有归属不动
        assert name in inher["train"]
    for name in base["val"]:
        assert name in inher["val"]
    added = (set(inher["train"]) | set(inher["val"])) - (set(base["train"]) | set(base["val"]))
    assert added == {"new0.png", "new1.png"}
    assert inher["uid"] != base["uid"]


def test_inherit_approved_only与memo可改_既有成员不被追溯剔除(tmp_path):
    ds = _ds(tmp_path)
    car = _add(ds, "car0", "car", approved=True)
    loose = _add(ds, "person0", "person", approved=False)
    base = _plan(ds, approved_only=False)
    assert set(base["train"]) | set(base["val"]) == {car, loose}
    sp.save_split(ds, base)

    inher = _plan(ds, mode="inherit", parent_uid=base["uid"], approved_only=True, memo="只训已审")
    assert inher["params"]["approved_only"] is True                   # 可改
    assert inher["memo"] == "只训已审"
    assert {car, loose} <= set(inher["train"]) | set(inher["val"])    # 追溯不剔除（实效集才管）


def test_fresh_无参照与非法入参(tmp_path):
    ds = _ds(tmp_path)
    _add(ds, "car0", "car")
    with pytest.raises(ValueError):
        _plan(ds, mode="inherit")            # 继承必须有参照
    with pytest.raises(ValueError):
        _plan(ds, mode="shuffle")            # 模式白名单
    assert _plan(ds)["lineage"]["parent_uid"] is None    # fresh 无 parent 即首轮


def test_分层_按主类别分桶且组内按seed洗牌可复现(tmp_path):
    ds = _ds(tmp_path)
    for i in range(20):
        _add(ds, f"car{i}", "car")
    for i in range(4):
        _add(ds, f"person{i}", "person")

    a = _plan(ds, ratio=0.8, stratify=True)
    b = _plan(ds, ratio=0.8, stratify=True)
    assert a["uid"] == b["uid"] and a["val"] == b["val"]     # 同 seed 可复现

    val_names = set(a["val"])
    assert any(n.startswith("person") for n in val_names)    # 稀有类在 val 有代表
    assert a["params"]["stratify"] is True

    c = _plan(ds, ratio=0.8, stratify=True, seed=2026)
    assert c["uid"] != a["uid"]                              # 洗牌真的由 seed 驱动
    assert sorted(c["val"]) != sorted(a["val"])


# ---------------------------------------------------------------- 分组（验收 5）

def test_分组随机单位是源图_同源瓦片恒同侧_无组键各自成组(tmp_path):
    ds = _ds(tmp_path)
    tiles = []
    for src in range(4):
        for col in range(2):
            tiles.append(_add(ds, f"s{src}r{src}c{col}", "car"))
    lone = _add(ds, "loose0", "car")
    groups = {f"src{src}": [f"s{src}r{src}c0.png", f"s{src}r{src}c1.png"] for src in range(4)}

    for seed in (1, 7, 42, 2026):
        plan = _plan(ds, ratio=0.5, seed=seed, groups=groups)
        assert plan["grouped"] == "source_image"
        train, val = set(plan["train"]), set(plan["val"])
        for members in groups.values():
            assert not (set(members) & train and set(members) & val)   # 同源不跨侧
        assert lone in train | val                                     # 无组键自成一单位

    plan = _plan(ds, ratio=0.5, groups=groups)
    assert sp._leak_groups(groups, plan["train"], plan["val"]) == 0
    assert sp.preview_split(ds, mode="fresh", seed=42, ratio=0.5, stratify=False,
                            approved_only=False, groups=groups)["leak_groups"] == 0


def test_inherit_新瓦片跟随同组既有归属(tmp_path):
    ds = _ds(tmp_path)
    _add(ds, "s0c0", "car")
    _add(ds, "s0c1", "car")
    _add(ds, "s1c0", "car")
    _add(ds, "s1c1", "car")
    groups = {"src0": ["s0c0.png", "s0c1.png"], "src1": ["s1c0.png", "s1c1.png"]}
    base = _plan(ds, ratio=0.5, seed=42, groups=groups)
    sp.save_split(ds, base)

    src0_side = "train" if "s0c0.png" in base["train"] else "val"
    _add(ds, "s0c2", "car")                       # 已有源图的新瓦片
    groups["src0"].append("s0c2.png")
    inher = _plan(ds, mode="inherit", parent_uid=base["uid"], seed=42, ratio=0.5,
                  stratify=False, approved_only=False, groups=groups)

    assert ("s0c2.png" in inher["train"]) == (src0_side == "train")
    assert set(inher[src0_side]) >= {"s0c0.png", "s0c1.png", "s0c2.png"}   # 同源恒同侧
    for name in base["train"]:
        assert name in inher["train"]
    for name in base["val"]:
        assert name in inher["val"]
    assert sp._leak_groups(groups, inher["train"], inher["val"]) == 0


# ---------------------------------------------------------------- 实效集（验收 4/5）

def test_effective_set_只剔不加_缺图与排除各计一次(tmp_path):
    ds = _ds(tmp_path)
    for i in range(8):
        _add(ds, f"car{i}", "car")
    _add(ds, "neg0")
    plan = _plan(ds, ratio=0.75)
    sp.save_split(ds, plan)
    stored = sp.load_split(ds, plan["uid"])

    (ds / "source" / "car0.png").unlink()         # 盘上缺失
    _write_excluded(ds, ["car1.png"])             # 排除在册
    _add(ds, "car9", "car")                       # 新文件：只剔不加，不得混进实效集
    (ds / "source" / "car2.json").write_text("{ 坏文件", encoding="utf-8")

    eff = sp.effective_set(ds, stored)
    assert eff["dropped"]["missing"] == ["car0.png"]
    assert eff["dropped"]["excluded"] == ["car1.png"]
    assert "car9.png" not in eff["train"] and "car9.png" not in eff["val"]
    assert sorted(eff["train"] + eff["val"]) == sorted(set(stored["train"]) | set(stored["val"])
                                                        - {"car0.png", "car1.png"})
    assert eff["counts"] == {"train": len(eff["train"]), "val": len(eff["val"])}


def test_effective_set_缺图优先于排除_一名只计一次(tmp_path):
    ds = _ds(tmp_path)
    for i in range(4):
        _add(ds, f"car{i}", "car")
    plan = _plan(ds, ratio=0.5)
    (ds / "source" / "car0.png").unlink()
    _write_excluded(ds, ["car0.png"])

    eff = sp.effective_set(ds, plan)
    assert eff["dropped"]["missing"] == ["car0.png"]
    assert eff["dropped"]["excluded"] == []


def test_effective_set_空实效只报不判(tmp_path):
    ds = _ds(tmp_path)
    _add(ds, "car0", "car")
    _add(ds, "car1", "car")
    plan = _plan(ds, ratio=0.5)
    for name in plan["val"]:
        (ds / "source" / name).unlink()
    eff = sp.effective_set(ds, plan)
    assert eff["val"] == [] and eff["dropped"]["missing"]           # 拒训是调用方的裁决
    assert eff["counts"]["val"] == 0


# ---------------------------------------------------------------- 历史与可变字段（验收 1/3）

def test_list_splits_按created_at倒序_无active指针(tmp_path):
    ds = _ds(tmp_path)
    for i in range(6):
        _add(ds, f"car{i}", "car")
        _add(ds, f"person{i}", "person")

    # 落盘顺序刻意与时间顺序相反：证明「最近」读的是 created_at，而不是写入先后或某个指针
    newest = _entity([f"car{i}.png" for i in range(6)], [f"person{i}.png" for i in range(6)],
                     "2026-11-03T10:00:00", "最新")
    mid = _entity([f"car{i}.png" for i in range(4)], ["car4.png", "car5.png"],
                  "2026-11-02T10:00:00")
    oldest = _entity(["car0.png", "car1.png"], ["car2.png"], "2026-11-01T10:00:00")
    for item in (newest, mid, oldest):
        sp.save_split(ds, item)

    (ds / "splits" / "ffffffff").mkdir()          # 空目录（半成品）不入历史
    listed = sp.list_splits(ds)
    assert [s["uid"] for s in listed] == [newest["uid"], mid["uid"], oldest["uid"]]
    assert listed[0]["memo"] == "最新"
    assert sp.latest_split(ds)["uid"] == newest["uid"]
    assert not any("active" in k for s in listed for k in s)   # 无 active 指针字段
    assert sp.latest_split(tmp_path / "空集") is None
    assert sp.list_splits(tmp_path / "空集") == []


def test_set_memo_是唯一的可变字段(tmp_path):
    ds = _ds(tmp_path)
    for i in range(5):
        _add(ds, f"car{i}", "car")
    plan = _plan(ds, memo="初稿")
    uid = plan["uid"]
    sp.save_split(ds, plan)

    updated = sp.set_memo(ds, uid, "稀有类补标后重切")
    assert updated["memo"] == "稀有类补标后重切"
    reloaded = sp.load_split(ds, uid)
    assert reloaded["memo"] == "稀有类补标后重切"
    assert reloaded["uid"] == uid and _split_dirs(ds) == [uid]      # memo 不进哈希
    assert reloaded["train"] == plan["train"] and reloaded["params"] == plan["params"]

    with pytest.raises(ValueError):
        sp.set_memo(ds, uid, 42)                                    # 非字符串不走这条路
    with pytest.raises(ValueError):
        sp.set_memo(ds, uid, {"train": []})
    tampered = dict(reloaded)
    tampered["train"] = list(reloaded["train"]) + ["car9.png"]
    tampered_path = sp.save_split(ds, tampered)
    assert tampered_path != ds / "splits" / uid / "split.json"      # 改名单即换实体
    assert sp.load_split(ds, uid)["train"] == plan["train"]         # 原实体不受影响
    with pytest.raises(FileNotFoundError):
        sp.set_memo(ds, "ffffffff", "不存在")


def test_split_json_字段契约与落盘形状(tmp_path):
    ds = _ds(tmp_path)
    for i in range(5):
        _add(ds, f"car{i}", "car")
    _add(ds, "neg0")
    plan = _plan(ds, seed=7, ratio=0.8, stratify=True, approved_only=True, memo="首轮")
    path = sp.save_split(ds, plan)

    assert set(plan) == {"uid", "created_at", "train", "val", "params",
                         "lineage", "grouped", "counts", "memo"}
    assert set(plan["params"]) == {"seed", "ratio", "stratify", "approved_only"}
    assert set(plan["lineage"]) == {"mode", "parent_uid"}
    assert set(plan["counts"]) == {"train", "val"}
    assert plan["grouped"] is None                                  # 非切片集
    assert plan["uid"] == sp.split_uid(plan["train"], plan["val"])
    assert plan["train"] == sorted(plan["train"])                   # 相对 source/ 的排序文件名
    assert all("/" not in n for n in plan["train"] + plan["val"])
    assert json.loads(path.read_text(encoding="utf-8"))["uid"] == plan["uid"]


# ---------------------------------------------------------------- 预览与错误面（验收 2/3）

def test_preview_与落盘同口径_重合与可比性(tmp_path):
    ds = _ds(tmp_path)
    for i in range(8):
        _add(ds, f"car{i}", "car")
    first = sp.preview_split(ds, mode="fresh", seed=42, ratio=0.75, stratify=False,
                             approved_only=False)
    assert first["comparable"] is True and first["overlap"] == {"val_shared": 0, "val_total": 0}
    assert first["leak_groups"] is None                             # 无分组信息算不出
    sp.save_split(ds, first)

    _add(ds, "car8", "car")
    again = sp.preview_split(ds, mode="inherit", parent_uid=first["uid"], seed=1, ratio=0.1,
                             stratify=True, approved_only=False, memo="补一张")
    assert again["comparable"] is False                             # 换划分 → 与既有 Run 不可比
    assert again["overlap"] == {"val_shared": len(first["val"]), "val_total": len(first["val"])}
    assert again["memo"] == "补一张"

    saved_uid = sp.save_split(ds, again).parent.name
    stored = sp.load_split(ds, saved_uid)
    assert saved_uid == again["uid"]
    assert stored["train"] == again["train"] and stored["val"] == again["val"]
    assert stored["counts"] == again["counts"] == {"train": len(again["train"]),
                                                   "val": len(again["val"])}
    assert stored["params"]["seed"] == first["params"]["seed"]      # 参照锁参数（预览同口径）

    # 无新合格成员时继承 → 同一划分 → 同 uid → 可比
    steady = sp.preview_split(ds, mode="inherit", parent_uid=again["uid"], seed=99, ratio=0.5,
                              stratify=True, approved_only=False)
    assert steady["comparable"] is True and steady["uid"] == again["uid"]
    assert steady["overlap"] == {"val_shared": len(again["val"]), "val_total": len(again["val"])}


def test_load_split_缺失与坏文件(tmp_path):
    ds = _ds(tmp_path)
    for i in range(4):
        _add(ds, f"car{i}", "car")
    plan = _plan(ds)
    path = sp.save_split(ds, plan)

    with pytest.raises(FileNotFoundError):
        sp.load_split(ds, "deadbeef")
    with pytest.raises(FileNotFoundError):
        sp.plan_split(ds, mode="inherit", parent_uid="deadbeef", seed=42, ratio=0.8,
                      stratify=False, approved_only=False)

    path.write_text("{ 不是 JSON", encoding="utf-8")
    with pytest.raises(ValueError):
        sp.load_split(ds, plan["uid"])

    path.write_text(json.dumps({**plan, "train": ["car0.png"]}, ensure_ascii=False),
                    encoding="utf-8")
    with pytest.raises(ValueError):            # uid 与划分内容不符（被手改过）
        sp.load_split(ds, plan["uid"])

    with pytest.raises(ValueError):            # params / lineage 缺失拒绝落盘
        sp.save_split(ds, {**plan, "params": {}})
    with pytest.raises(ValueError):
        sp.save_split(ds, {**plan, "train": plan["train"] + [n for n in plan["val"]][:1]})


def test_save_split_原子落盘不留残骸(tmp_path, monkeypatch):
    ds = _ds(tmp_path)
    for i in range(4):
        _add(ds, f"car{i}", "car")
    plan = _plan(ds)
    sp.save_split(ds, plan)
    before = sp.load_split(ds, plan["uid"])

    _add(ds, "car4", "car")                    # 新成员 → 必然是新实体
    other = _plan(ds, seed=9, stratify=True)
    assert other["uid"] != plan["uid"]

    def 爆炸(*_a, **_kw):
        raise OSError("磁盘写不下")

    monkeypatch.setattr(sp.os, "replace", 爆炸)
    with pytest.raises(OSError):
        sp.save_split(ds, other)

    tmp_dir = ds / "splits" / other["uid"]
    assert not tmp_dir.exists()                                   # 空壳目录一并收回
    assert list((ds / "splits" / plan["uid"]).iterdir()) == [ds / "splits" / plan["uid"] / "split.json"]
    assert sp.load_split(ds, plan["uid"]) == before                 # 既有实体不受影响
    assert _split_dirs(ds) == [plan["uid"]]                         # 失败的实体不留半个账


def test_旧清单层已硬删():
    """票 05 硬切：活跃清单机制（build_manifest / save_manifest / excluded_manifest 等）无残留。"""
    for legacy in ("build_manifest", "build_grouped_manifest", "excluded_manifest",
                   "restored_manifest", "absorbed_manifest", "save_manifest",
                   "_manifest_uid", "_file_label"):
        assert not hasattr(sp, legacy), legacy
