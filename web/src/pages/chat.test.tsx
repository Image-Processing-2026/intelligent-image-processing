import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { SessionDetail } from "@/api/types";
import { DETAIL, SESSION_ID, SUMMARY, WITH_EDIT } from "@/test/fixtures";
import { ALICE, mockApi, renderApp } from "@/test/utils";

const BASE = [
  { path: "/auth/me", body: ALICE },
  { path: "/health", body: { status: "ok" } },
  { path: "/workspace/sessions", body: [SUMMARY] },
];
const SESSION_PATH = `/workspace/sessions/${SESSION_ID}`;

const WITH_PROMPT: SessionDetail = {
  ...DETAIL,
  messages: [
    {
      id: 0,
      role: "user",
      kind: "user_prompt",
      created_at: "2026-10-22T10:00:00+00:00",
      content: { text: "Mặt trong ảnh bị tối", original_artifact_id: DETAIL.original!.id },
    },
    ...DETAIL.messages,
  ],
};

describe("chat thread", () => {
  it("opens with the original image and the initial prompt", async () => {
    mockApi([...BASE, { path: SESSION_PATH, body: WITH_PROMPT }]);
    renderApp(`/s/${SESSION_ID}`);
    expect(await screen.findByText("Mặt trong ảnh bị tối")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Ảnh gốc đính kèm" })).toBeInTheDocument();
  });

  it("still shows the original image for sessions without an initial prompt", async () => {
    mockApi([...BASE, { path: SESSION_PATH, body: DETAIL }]);
    renderApp(`/s/${SESSION_ID}`);
    expect(await screen.findByRole("img", { name: "Ảnh gốc đính kèm" })).toBeInTheDocument();
    expect(screen.getByText("Ảnh trong nhà thiếu sáng")).toBeInTheDocument();
  });

  it("does not crash on an unknown message kind", async () => {
    const unknown = {
      ...DETAIL,
      messages: [
        ...DETAIL.messages,
        {
          id: 50,
          role: "assistant" as const,
          kind: "future_kind" as never,
          created_at: "2026-10-22T10:00:09+00:00",
          content: { text: "Tin nhắn loại mới" },
        },
      ],
    };
    mockApi([...BASE, { path: SESSION_PATH, body: unknown }]);
    renderApp(`/s/${SESSION_ID}`);
    expect(await screen.findByText("Tin nhắn loại mới")).toBeInTheDocument();
  });

  it("sends a quick suggestion as an analysis note", async () => {
    const fetchMock = mockApi([
      ...BASE,
      { path: SESSION_PATH, body: DETAIL },
      { method: "POST", path: `${SESSION_PATH}/chat`, body: WITH_EDIT },
    ]);
    const user = userEvent.setup();
    renderApp(`/s/${SESSION_ID}`);
    await user.click(await screen.findByRole("button", { name: "Làm sáng khuôn mặt" }));
    await waitFor(() => {
      const post = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/chat"));
      expect(JSON.parse(String(post?.[1]?.body))).toEqual({ revision: 1, message: "Làm sáng khuôn mặt" });
    });
  });
});

describe("new chat page", () => {
  it("sends the prompt with the image and opens the created session", async () => {
    const fetchMock = mockApi([
      ...BASE.slice(0, 2),
      { path: "/workspace/samples", body: [] },
      { path: "/workspace/sessions", body: [] },
      { method: "POST", path: "/workspace/sessions", status: 201, body: WITH_PROMPT },
      { path: SESSION_PATH, body: WITH_PROMPT },
    ]);
    const user = userEvent.setup();
    renderApp("/");
    await user.upload(
      await screen.findByLabelText("Tải ảnh lên"),
      new File(["png"], "coffee.png", { type: "image/png" }),
    );
    await user.type(screen.getByPlaceholderText(/Mô tả yêu cầu chỉnh sửa ảnh/), "Mặt trong ảnh bị tối");
    await user.click(screen.getByRole("button", { name: "Bắt đầu chẩn đoán" }));

    expect(await screen.findByRole("heading", { name: "Chẩn đoán ảnh" })).toBeInTheDocument();
    const post = fetchMock.mock.calls.find(([, init]) => init?.method === "POST");
    expect((post?.[1]?.body as FormData).get("user_prompt")).toBe("Mặt trong ảnh bị tối");
  });
});

const APPROACH = {
  id: "approach",
  text: "AI đề xuất xử lý như sau. Sửa: tối. Bạn muốn xử lý theo đề xuất này hay theo cách khác?",
  choices: [
    { value: "recommend", label: "Theo đề xuất của AI" },
    { value: "custom", label: "Cách khác (mô tả yêu cầu của bạn ở khung chat bên dưới)" },
  ],
  default: "recommend",
};
const ASKING_APPROACH: SessionDetail = {
  ...DETAIL,
  messages: DETAIL.messages.map((m) =>
    m.kind === "questions" ? { ...m, content: { questions: [APPROACH] } } : m,
  ),
};

describe("conversation replaces approach selection", () => {
  it("offers execution without recommend/custom radios", async () => {
    mockApi([...BASE, {path: SESSION_PATH, body: ASKING_APPROACH}]);
    renderApp(`/s/${SESSION_ID}`);
    expect(await screen.findByRole("button", {name: "Xử lý ảnh"})).toBeEnabled();
    expect(screen.queryByRole("radio", {name: /Theo đề xuất/})).toBeNull();
  });
});

describe("running sessions", () => {
  it("keeps polling a running session and shows the result when it finishes", async () => {
    const running: SessionDetail = { ...DETAIL, status: "running" };
    const finished: SessionDetail = { ...DETAIL, status: "done" };
    let reads = 0;
    mockApi([
      ...BASE,
      { path: SESSION_PATH, get body() { reads += 1; return reads <= 1 ? running : finished; } },
    ]);
    renderApp(`/s/${SESSION_ID}`);
    expect(await screen.findByText("Đang xử lý, thời gian phụ thuộc ảnh và số vòng")).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByText(/Đang xử lý, thời gian/)).toBeNull(), {
      timeout: 6000,
    });
    expect(reads).toBeGreaterThan(1);
  }, 10000);
});
