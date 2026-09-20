"""切分清单测试：可复现性（M1 验收核心）、比例、分层、负样本去向。"""

from visiontrainer.datasets.split import build_manifest


def _entries(n: int, label_prefix: str = "car") -> list[dict]:
    return [
        {"image": f"/data/{label_prefix}{i}.png", "label": f"/data/{label_prefix}{i}.json", "primary_label": label_prefix}
        for i in range(n)
    ]


def test_same_seed_reproducible():
    entries = _entries(50)
    m1 = build_manifest(entries, 0.8, seed=42, stratify=False)
    m2 = build_manifest(entries, 0.8, seed=42, stratify=False)
    assert m1["train"] == m2["train"] and m1["val"] == m2["val"]
    assert m1["uid"] == m2["uid"]


def test_different_seed_different_split():
    entries = _entries(50)
    m1 = build_manifest(entries, 0.8, seed=42, stratify=False)
    m2 = build_manifest(entries, 0.8, seed=7, stratify=False)
    assert m1["train"] != m2["train"]


def test_ratio_approx():
    entries = _entries(100)
    m = build_manifest(entries, 0.8, seed=1, stratify=False)
    assert len(m["train"]) == 80 and len(m["val"]) == 20


def test_stratify_keeps_rare_class_in_val():
    entries = _entries(30, "car") + _entries(3, "rare_person")
    m = build_manifest(entries, 0.8, seed=1, stratify=True)
    val_names = "\n".join(m["val"])
    assert "rare_person" in val_names


def test_background_only_in_train():
    entries = _entries(10) + [
        {"image": f"/data/neg{i}.png", "label": f"/data/neg{i}.json", "primary_label": None} for i in range(5)
    ]
    m = build_manifest(entries, 0.8, seed=3, stratify=False)
    assert all("neg" not in v for v in m["val"])
    assert sum(1 for t in m["train"] if "neg" in t) == 5


def test_manifest_records_snapshot():
    entries = _entries(10)
    m = build_manifest(entries, 0.8, seed=1, stratify=False)
    assert {f["image"] for f in m["files"]} == {e["image"] for e in entries}
