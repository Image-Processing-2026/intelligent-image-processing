import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, NETWORK_MESSAGE, apiFetch, errorMessage, setUnauthorizedHandler } from "./client";

function mockFetch(response: Response | Error) {
  const fn = vi.fn(async () => {
    if (response instanceof Error) throw response;
    return response;
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

afterEach(() => {
  vi.unstubAllGlobals();
  setUnauthorizedHandler(null);
});

describe("apiFetch", () => {
  it("sends JSON with same-origin credentials under /api/v1", async () => {
    const fetchMock = mockFetch(new Response(JSON.stringify({ ok: true }), { status: 200 }));
    await expect(apiFetch("/auth/login", { method: "POST", json: { a: 1 } })).resolves.toEqual({
      ok: true,
    });
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/v1/auth/login");
    expect(init.credentials).toBe("same-origin");
    expect(init.body).toBe('{"a":1}');
    expect(init.signal).toBeUndefined();
  });

  it("maps the error envelope to ApiError", async () => {
    mockFetch(
      new Response(
        JSON.stringify({ error: { code: "username_taken", message: "Tên đăng nhập đã được dùng." } }),
        { status: 409 },
      ),
    );
    const error = await apiFetch("/auth/register").catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe("username_taken");
    expect(errorMessage(error)).toBe("Tên đăng nhập đã được dùng.");
  });

  it("never surfaces raw bodies without an envelope", async () => {
    mockFetch(new Response("Traceback (most recent call last): secret", { status: 500 }));
    const error = (await apiFetch("/x").catch((e: unknown) => e)) as ApiError;
    expect(error.code).toBe("internal");
    expect(error.message).not.toContain("Traceback");
  });

  it("reports network failures in Vietnamese", async () => {
    mockFetch(new TypeError("Failed to fetch"));
    const error = (await apiFetch("/x").catch((e: unknown) => e)) as ApiError;
    expect(error.code).toBe("network");
    expect(error.message).toBe(NETWORK_MESSAGE);
  });

  it("calls the 401 handler unless skipped", async () => {
    const handler = vi.fn();
    setUnauthorizedHandler(handler);
    const unauthorized = () =>
      new Response(JSON.stringify({ error: { code: "unauthenticated", message: "m" } }), {
        status: 401,
      });
    mockFetch(unauthorized());
    await apiFetch("/workspace/sessions").catch(() => undefined);
    expect(handler).toHaveBeenCalledTimes(1);
    mockFetch(unauthorized());
    await apiFetch("/auth/me", { skipAuthRedirect: true }).catch(() => undefined);
    expect(handler).toHaveBeenCalledTimes(1);
  });

  it("returns undefined for 204", async () => {
    mockFetch(new Response(null, { status: 204 }));
    await expect(apiFetch("/auth/logout", { method: "POST" })).resolves.toBeUndefined();
  });
});
