import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { runsApi, ApiRequestError } from "./client";

/** request() 错误信封解析：直挂 body.error 与 FastAPI HTTPException 的 body.detail.error 两种落点 */
function stubFetchOnce(payload: unknown, status = 400) {
  const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  }));
  vi.stubGlobal("fetch", fetchMock);
}

beforeEach(() => {});
afterEach(() => {
  vi.unstubAllGlobals();
});

describe("client request 错误信封", () => {
  it("parses_envelope_nested_under_httpexception_detail", async () => {
    stubFetchOnce({ detail: { error: { code: "invalid_args", message: "图片不存在：/tmp/x.png", detail: {} } } });
    const err = await runsApi.testInfer("u1", { image: "/tmp/x.png" }).catch((e) => e);
    expect(err).toBeInstanceOf(ApiRequestError);
    expect((err as ApiRequestError).message).toBe("图片不存在：/tmp/x.png");
    expect((err as ApiRequestError).code).toBe("invalid_args");
    expect((err as ApiRequestError).status).toBe(400);
  });

  it("parses_envelope_direct_on_body_and_keeps_detail_payload", async () => {
    const conflicts = [{ file: "a.jpg", reason: "exists_in_source", sources: [] }];
    stubFetchOnce({ error: { code: "blocked", message: "同名冲突", detail: { conflicts } } }, 409);
    const err = await runsApi.artifacts("u1").catch((e) => e);
    expect((err as ApiRequestError).message).toBe("同名冲突");
    expect((err as ApiRequestError).detail).toEqual({ conflicts });
  });

  it("falls_back_to_http_line_on_non_json_error_body", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("nginx 502", { status: 502 })));
    const err = await runsApi.get("u1").catch((e) => e);
    expect((err as ApiRequestError).message).toBe("请求失败（HTTP 502）");
  });
});
