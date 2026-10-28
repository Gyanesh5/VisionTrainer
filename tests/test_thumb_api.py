"""缩略图 HTTP 端点（架构路线一）：浏览器原生缓存 + 签名路径安全。"""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from conftest import make_image

from visiontrainer import thumbs


def _thumb_path(p: str, box: int) -> str:
    """从 thumb_url 拆出路由路径 + 查询串（TestClient 用）。"""
    u = urlparse(thumbs.thumb_url(p, box))
    return u.path + "?" + u.query


def test_thumb_endpoint_200_and_304(tmp_path, client):
    from PIL import Image

    p = tmp_path / "ok.jpg"
    Image.new("RGB", (400, 300), (9, 99, 199)).save(p, format="JPEG")

    r = client.get(_thumb_path(str(p), 260))
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/jpeg")
    assert r.content[:2] == b"\xff\xd8"
    assert "etag" in r.headers and "max-age" in r.headers["cache-control"]

    # 带上一次的 ETag 再验证 → 304 空体（重进页面不重传字节）
    r2 = client.get(_thumb_path(str(p), 260), headers={"if-none-match": r.headers["etag"]})
    assert r2.status_code == 304
    assert r2.content == b""


def test_thumb_endpoint_original_box0(tmp_path, client):
    """box=0 原图直出（大图查看器）：字节即原文件、mime 按后缀、304 再验证。"""
    from PIL import Image

    p = tmp_path / "orig.png"
    Image.new("RGB", (300, 200), (7, 77, 177)).save(p)
    r = client.get(_thumb_path(str(p), 0))
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content == p.read_bytes()  # 原字节不缩放
    r2 = client.get(_thumb_path(str(p), 0), headers={"if-none-match": r.headers["etag"]})
    assert r2.status_code == 304


def test_thumb_endpoint_security(tmp_path, client):
    from PIL import Image

    p = tmp_path / "ok.png"
    Image.new("RGB", (64, 64)).save(p)

    # 无签名 / 坏签名 / 篡改路径：403（任意文件读取必须不可行）
    assert client.get(f"/api/v1/thumb?p={tmp_path}/ok.png&box=260").status_code == 403
    assert client.get(f"/api/v1/thumb?p={tmp_path}/ok.png&box=260&t=badbadbad").status_code == 403
    tok = parse_qs(urlparse(thumbs.thumb_url(str(p), 260)).query)["t"][0]
    from urllib.parse import quote
    evil = f"/api/v1/thumb?p={quote('/etc/passwd', safe='')}&box=260&t={tok}"  # 换路径留旧签名
    assert client.get(evil).status_code == 403

    # 尺寸白名单外：400
    assert client.get(f"/api/v1/thumb?p=/x&box=123&t={'0'*11+'x'*20}").status_code == 400

    # 签名有效但文件不存在：404
    missing = _thumb_path(str(tmp_path / "gone.png"), 72)
    assert client.get(missing).status_code == 404

    # 签名有效但文件不可解码：404（magic 字节非图）
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\nbroken")
    assert client.get(_thumb_path(str(bad), 72)).status_code == 404


def test_thumb_endpoint_cache_hit_same_bytes(tmp_path, client):
    """两次请求同一张图：第二次命中服务端 LRU（字节一致），ETag 稳定。"""
    from PIL import Image

    p = tmp_path / "c.png"
    Image.new("RGB", (200, 200), (1, 2, 3)).save(p)
    r1 = client.get(_thumb_path(str(p), 96))
    r2 = client.get(_thumb_path(str(p), 96))
    assert r1.status_code == r2.status_code == 200
    assert r1.content == r2.content
    assert r1.headers["etag"] == r2.headers["etag"]
