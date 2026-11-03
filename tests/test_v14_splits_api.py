"""票 05 REST：/datasets/{id}/splits 五端点（docs/api.md v1.4）。

- GET  /splits                切分历史（created_at 倒序，完整实体）
- POST /splits/preview        新建预览（零写入，数字与落盘同口径）
- POST /splits                独立确认落盘（同划分命中既有 uid 复用，不随任务取消回滚）
- PATCH /splits/{uid}         仅编辑 memo
- GET  /splits/{uid}/effective 实效集预览（与出队同口径）
"""

from __future__ import annotations

from pathlib import Path

from conftest import make_annotation, make_image, rect

from visiontrainer.datasets import service as svc


def _born(client, datasets_root, detect_source) -> str:
    """出生一个托管集（不切分），返回名字。"""
    r = client.post("/api/v1/datasets", json={
        "name": "spx", "task_type": "detect", "classes": ["car", "person"],
        "source_mode": "labeled", "source_path": str(detect_source)})
    assert r.status_code == 200, r.text
    return "spx"


def test_splits_crud_roundtrip(client, datasets_root, detect_source):
    name = _born(client, datasets_root, detect_source)

    # 出生不切分：历史为空
    r = client.get(f"/api/v1/datasets/{name}/splits")
    assert r.status_code == 200 and r.json() == []

    # 预览：零写入；数字与落盘同口径；命中提示与实效计数齐备
    body = {"mode": "fresh", "seed": 42, "ratio": 0.75, "stratify": False,
            "approved_only": False, "memo": "首轮"}
    pv = client.post(f"/api/v1/datasets/{name}/splits/preview", json=body)
    assert pv.status_code == 200, pv.text
    p = pv.json()
    assert p["counts"] == {"train": p["counts"]["train"], "val": p["counts"]["val"]}
    assert p["comparable"] is True and p["overlap"] == {"val_shared": 0, "val_total": 0}
    assert p["reuses_existing"] is False
    eff0 = p["effective"]
    assert eff0["counts"] == {"train": len(eff0["train"]), "val": len(eff0["val"])}
    assert client.get(f"/api/v1/datasets/{name}/splits").json() == []  # 预览零写入

    # 落盘：同划分命中既有 uid 直接复用（第二次 create 不产生第二份、不改 memo）
    cr = client.post(f"/api/v1/datasets/{name}/splits", json=body)
    assert cr.status_code == 200, cr.text
    s = cr.json()
    assert s["uid"] == p["uid"] and s["memo"] == "首轮"
    again = client.post(f"/api/v1/datasets/{name}/splits",
                        json={**body, "memo": "改不掉"})
    assert again.json()["uid"] == s["uid"]
    hist = client.get(f"/api/v1/datasets/{name}/splits").json()
    assert [h["uid"] for h in hist] == [s["uid"]]
    assert hist[0]["memo"] == "首轮"

    # 票 07：切分落盘零派生——格式池与切分无关，出队时懒建
    assert not (datasets_root / name / "derived").exists()

    # PATCH 仅 memo：uid 不变；GET 单条不存在 404
    pa = client.patch(f"/api/v1/datasets/{name}/splits/{s['uid']}",
                      json={"memo": "补标后重切"})
    assert pa.status_code == 200 and pa.json()["memo"] == "补标后重切"
    assert client.get(f"/api/v1/datasets/{name}/splits").json()[0]["memo"] == "补标后重切"
    assert client.patch(f"/api/v1/datasets/{name}/splits/deadbeef",
                        json={"memo": "x"}).status_code == 404

    # effective：与出队同口径；缺图与排除只剔不加
    (datasets_root / name / "source" / "car0.png").unlink()
    svc.exclude_files(name, ["car1.png"], "测试")
    ef = client.get(f"/api/v1/datasets/{name}/splits/{s['uid']}/effective").json()
    assert ef["dropped"]["missing"] == ["car0.png"]
    assert ef["dropped"]["excluded"] == ["car1.png"]
    assert "car0.png" not in ef["train"] + ef["val"]
    assert client.get(f"/api/v1/datasets/{name}/splits/deadbeef/effective").status_code == 404


def test_splits_history_order_and_inherit_preview(client, datasets_root, detect_source):
    name = _born(client, datasets_root, detect_source)
    first = client.post(f"/api/v1/datasets/{name}/splits",
                        json={"mode": "fresh", "seed": 42, "ratio": 0.75}).json()

    # 加一张新图 → 继承归属预览：既有归属不动、只放新图、参照锁参、不可比警示
    src = datasets_root / name / "source"
    make_image(src, "zz_new.png")
    make_annotation(src, "zz_new", [rect("car")])
    pv = client.post(f"/api/v1/datasets/{name}/splits/preview",
                     json={"mode": "inherit", "parent_uid": first["uid"],
                           "seed": 999, "ratio": 0.1, "stratify": True}).json()
    assert pv["params"]["seed"] == 42 and pv["params"]["ratio"] == 0.75  # 以参照为准
    assert set(first["train"]) <= set(pv["train"])
    assert "zz_new.png" in set(pv["train"]) | set(pv["val"])
    assert pv["uid"] != first["uid"]
    assert pv["comparable"] is False and "不可直接比较" in pv["incomparable_warning"]
    ov = pv["overlap"]
    assert ov["val_shared"] == len(first["val"]) and ov["val_total"] == len(first["val"])

    second = client.post(f"/api/v1/datasets/{name}/splits",
                         json={"mode": "inherit", "parent_uid": first["uid"],
                               "memo": "放新图"}).json()
    hist = client.get(f"/api/v1/datasets/{name}/splits").json()
    assert [h["uid"] for h in hist][:2] == [second["uid"], first["uid"]] or \
        [h["created_at"] for h in hist] == sorted((h["created_at"] for h in hist), reverse=True)

    # 继承必须有参照：400
    bad = client.post(f"/api/v1/datasets/{name}/splits/preview",
                      json={"mode": "inherit"})
    assert bad.status_code == 400


def test_splits_endpoints_reject_external(client, datasets_root, tmp_path):
    ext = tmp_path / "extds"
    ext.mkdir()
    (ext / "data.yaml").write_text("path: .\ntrain: t\nval: v\nnames: {0: car}\n",
                                   encoding="utf-8")
    svc.register_external("exts", "detect", str(ext), root=datasets_root)
    for method, path in (("get", "/api/v1/datasets/exts/splits"),
                         ("post", "/api/v1/datasets/exts/splits"),
                         ("post", "/api/v1/datasets/exts/splits/preview")):
        if method == "get":
            r = client.get(path)
        else:
            r = client.post(path, json={"mode": "fresh"})
        assert r.status_code == 404, (method, path, r.text)  # api.md v1.4：外部数据集无切分 → 404
