import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { QueryClientProvider } from "@tanstack/react-query";

import { createQueryClient } from "@/App";
import { ChatInputBar } from "./ChatInputBar";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "@/lib/theme";
import { ALICE, mockApi } from "@/test/utils";
import { DETAIL, SESSION_ID } from "@/test/fixtures";
import { ApiError } from "@/api/client";

const SAMPLES = [
  {
    id: "backlit-portrait",
    label: "Chân dung ngược sáng",
    thumbnail_url: "/samples/backlit.jpg",
  },
  {
    id: "night-scene",
    label: "Cảnh đêm thiếu sáng",
    thumbnail_url: "/samples/night.jpg",
  },
];

const BASE_ROUTES = [
  { path: "/auth/me", body: ALICE },
  { path: "/workspace/samples", body: SAMPLES },
];

function renderInputBar(props: React.ComponentProps<typeof ChatInputBar> = {}) {
  const queryClient = createQueryClient();
  return render(
    <ThemeProvider>
      <QueryClientProvider client={queryClient}>
        <TooltipProvider>
          <MemoryRouter>
            <ChatInputBar {...props} />
          </MemoryRouter>
        </TooltipProvider>
      </QueryClientProvider>
    </ThemeProvider>,
  );
}

describe("ChatInputBar", () => {
  it("renders in new session mode with disabled submit button when empty", async () => {
    mockApi(BASE_ROUTES);
    renderInputBar();

    const textarea = screen.getByPlaceholderText(
      /Mô tả yêu cầu chỉnh sửa ảnh \(tùy chọn\)\.\.\./,
    );
    expect(textarea).toBeInTheDocument();

    const submitBtn = screen.getByRole("button", { name: "Bắt đầu chẩn đoán" });
    expect(submitBtn).toBeDisabled();

    // Verify upload button & advanced options button exist
    expect(screen.getByRole("button", { name: "Đính kèm ảnh" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Tùy chọn nâng cao" })).toBeInTheDocument();
  });

  it("attaches an image via file input and enables submit button", async () => {
    mockApi(BASE_ROUTES);
    const user = userEvent.setup();
    renderInputBar();

    const fileInput = screen.getByLabelText("Tải ảnh lên") as HTMLInputElement;
    const testFile = new File(["test-image-content"], "portrait.png", {
      type: "image/png",
    });

    await user.upload(fileInput, testFile);

    // Thumbnail chip appears
    expect(screen.getByText("portrait.png")).toBeInTheDocument();
    const submitBtn = screen.getByRole("button", { name: "Bắt đầu chẩn đoán" });
    expect(submitBtn).toBeEnabled();

    // Clicking remove chip removes the image and disables submit
    const removeBtn = screen.getByRole("button", { name: "Gỡ ảnh portrait.png" });
    await user.click(removeBtn);

    expect(screen.queryByText("portrait.png")).toBeNull();
    expect(submitBtn).toBeDisabled();
  });

  it("handles image paste from clipboard (Ctrl+V)", async () => {
    mockApi(BASE_ROUTES);
    renderInputBar();

    const testFile = new File(["clipboard-image"], "clipboard.jpg", {
      type: "image/jpeg",
    });

    const textarea = screen.getByPlaceholderText(
      /Mô tả yêu cầu chỉnh sửa ảnh \(tùy chọn\)\.\.\./,
    );

    // Simulate paste with clipboardData.files
    fireEvent.paste(textarea, {
      clipboardData: {
        files: [testFile],
        items: [
          {
            type: "image/jpeg",
            getAsFile: () => testFile,
          },
        ],
      },
    });

    await waitFor(() => {
      expect(screen.getByText("clipboard.jpg")).toBeInTheDocument();
    });

    expect(screen.getByRole("button", { name: "Bắt đầu chẩn đoán" })).toBeEnabled();
  });

  it("rejects unsupported MIME type on paste", async () => {
    mockApi(BASE_ROUTES);
    renderInputBar();

    const gifFile = new File(["gif-data"], "animated.gif", { type: "image/gif" });
    const textarea = screen.getByPlaceholderText(
      /Mô tả yêu cầu chỉnh sửa ảnh \(tùy chọn\)\.\.\./,
    );

    fireEvent.paste(textarea, {
      clipboardData: {
        files: [gifFile],
        items: [{ type: "image/gif", getAsFile: () => gifFile }],
      },
    });

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(
        "Chỉ hỗ trợ ảnh JPEG, PNG hoặc WEBP.",
      );
    });
    expect(screen.getByRole("button", { name: "Bắt đầu chẩn đoán" })).toBeDisabled();
  });

  it("rejects file exceeding size limit (20MB)", async () => {
    mockApi(BASE_ROUTES);
    renderInputBar();

    const bigFile = new File(["x"], "huge.png", { type: "image/png" });
    Object.defineProperty(bigFile, "size", { value: 25 * 1024 * 1024 });

    const textarea = screen.getByPlaceholderText(
      /Mô tả yêu cầu chỉnh sửa ảnh \(tùy chọn\)\.\.\./,
    );

    fireEvent.paste(textarea, {
      clipboardData: {
        files: [bigFile],
        items: [{ type: "image/png", getAsFile: () => bigFile }],
      },
    });

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent("Tệp vượt quá 20 MB.");
    });
  });

  it("handles drag-and-drop of image files", async () => {
    mockApi(BASE_ROUTES);
    renderInputBar();

    const testFile = new File(["dropped-content"], "dropped.webp", {
      type: "image/webp",
    });

    const textarea = screen.getByPlaceholderText(
      /Mô tả yêu cầu chỉnh sửa ảnh \(tùy chọn\)\.\.\./,
    );

    // Drag over shows overlay
    fireEvent.dragOver(textarea, {
      dataTransfer: { files: [testFile] },
    });
    expect(screen.getByText("Thả ảnh vào đây để tải lên")).toBeInTheDocument();

    // Drop attaches file
    fireEvent.drop(textarea, {
      dataTransfer: { files: [testFile] },
    });

    await waitFor(() => {
      expect(screen.getByText("dropped.webp")).toBeInTheDocument();
    });
    expect(screen.queryByText("Thả ảnh vào đây để tải lên")).toBeNull();
    expect(screen.getByRole("button", { name: "Bắt đầu chẩn đoán" })).toBeEnabled();
  });

  it("allows selecting sample, ground truth, and max_iterations via Advanced Options dialog", async () => {
    mockApi(BASE_ROUTES);
    const user = userEvent.setup();
    renderInputBar();

    // Click options button
    const optionsBtn = screen.getByRole("button", { name: "Tùy chọn nâng cao" });
    await user.click(optionsBtn);

    // Dialog opens
    expect(await screen.findByRole("heading", { name: "Tùy chọn nâng cao" })).toBeInTheDocument();
    expect(screen.getByText("Chân dung ngược sáng")).toBeInTheDocument();

    // Select sample
    await user.click(screen.getByText("Chân dung ngược sáng"));
    expect(screen.getByText(/Đã chọn ảnh mẫu: Chân dung ngược sáng/)).toBeInTheDocument();

    // Select max_iterations 5
    const btn5 = screen.getByRole("radio", { name: "5" });
    await user.click(btn5);

    // Attach ground truth file
    const gtInput = screen.getByLabelText("Chọn tệp ảnh chuẩn") as HTMLInputElement;
    const gtFile = new File(["gt-data"], "truth.png", { type: "image/png" });
    await user.upload(gtInput, gtFile);
    expect(screen.getAllByText("truth.png").length).toBeGreaterThan(0);

    // Close dialog
    await user.click(screen.getByRole("button", { name: "Xong" }));

    // Chips are displayed in the input bar
    expect(screen.getByText("Chân dung ngược sáng")).toBeInTheDocument();
    expect(screen.getByText("truth.png")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Bắt đầu chẩn đoán" })).toBeEnabled();
  });

  it("submits a new session with image and prompt on Enter key", async () => {
    mockApi([
      ...BASE_ROUTES,
      { path: "/workspace/sessions", body: [] },
      { method: "POST", path: "/workspace/sessions", status: 201, body: DETAIL },
    ]);
    const user = userEvent.setup();
    const onNewSession = vi.fn().mockResolvedValue(DETAIL);

    renderInputBar({ onNewSession });

    const fileInput = screen.getByLabelText("Tải ảnh lên") as HTMLInputElement;
    const testFile = new File(["png"], "camera.png", { type: "image/png" });
    await user.upload(fileInput, testFile);

    const textarea = screen.getByPlaceholderText(
      /Mô tả yêu cầu chỉnh sửa ảnh \(tùy chọn\)\.\.\./,
    );
    await user.type(textarea, "Mặt hơi tối, hãy làm sáng mặt nhưng giữ nguyên bầu trời");

    // Press Enter to submit
    await user.keyboard("{Enter}");

    await waitFor(() => {
      expect(onNewSession).toHaveBeenCalledTimes(1);
    });

    expect(onNewSession).toHaveBeenCalledWith(
      expect.objectContaining({
        file: testFile,
        userPrompt: "Mặt hơi tối, hãy làm sáng mặt nhưng giữ nguyên bầu trời",
      }),
    );
  });

  it("does not submit on Shift+Enter (allows multiline input)", async () => {
    mockApi(BASE_ROUTES);
    const user = userEvent.setup();
    const onNewSession = vi.fn();
    renderInputBar({ onNewSession });

    const fileInput = screen.getByLabelText("Tải ảnh lên") as HTMLInputElement;
    await user.upload(fileInput, new File(["png"], "camera.png", { type: "image/png" }));

    const textarea = screen.getByPlaceholderText(
      /Mô tả yêu cầu chỉnh sửa ảnh \(tùy chọn\)\.\.\./,
    );
    await user.type(textarea, "Dòng 1{Shift>}{Enter}{/Shift}Dòng 2");

    expect(onNewSession).not.toHaveBeenCalled();
    expect(textarea).toHaveValue("Dòng 1\nDòng 2");
  });

  it("handles analysis_note mode when sessionStatus is awaiting_answers", async () => {
    mockApi(BASE_ROUTES);
    const user = userEvent.setup();
    const onSendNote = vi.fn().mockResolvedValue({});

    renderInputBar({
      sessionId: SESSION_ID,
      sessionStatus: "awaiting_answers",
      onSendNote,
    });

    // In analysis_note mode, upload button and options button are hidden
    expect(screen.queryByRole("button", { name: "Đính kèm ảnh" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Tùy chọn nâng cao" })).toBeNull();

    const textarea = screen.getByPlaceholderText(
      /Nhập phản hồi hoặc yêu cầu chỉnh sửa chẩn đoán\.\.\./,
    );
    const submitBtn = screen.getByRole("button", { name: "Gửi bổ sung phân tích" });
    expect(submitBtn).toBeDisabled();

    // Type note
    await user.type(textarea, "Chỉ làm sáng mặt thôi nhé");
    expect(submitBtn).toBeEnabled();

    // Submit
    await user.click(submitBtn);

    await waitFor(() => {
      expect(onSendNote).toHaveBeenCalledWith("Chỉ làm sáng mặt thôi nhé");
    });
    // Clears textarea after send
    expect(textarea).toHaveValue("");
  });

  it("handles refine mode when sessionStatus is done", async () => {
    mockApi(BASE_ROUTES);
    const user = userEvent.setup();
    const onRefine = vi.fn().mockResolvedValue({});

    renderInputBar({
      sessionId: SESSION_ID,
      sessionStatus: "done",
      onRefine,
    });

    const textarea = screen.getByPlaceholderText(/Nhập góp ý tinh chỉnh cho kết quả\.\.\./);
    const submitBtn = screen.getByRole("button", { name: "Gửi góp ý tinh chỉnh" });

    await user.type(textarea, "Tông da hơi bị ám vàng");
    await user.click(submitBtn);

    await waitFor(() => {
      expect(onRefine).toHaveBeenCalledWith("Tông da hơi bị ám vàng");
    });
    expect(textarea).toHaveValue("");
  });

  it("disables input when sessionStatus is running", () => {
    mockApi(BASE_ROUTES);
    renderInputBar({
      sessionId: SESSION_ID,
      sessionStatus: "running",
    });

    const textarea = screen.getByPlaceholderText("Hệ thống đang xử lý, vui lòng chờ...");
    expect(textarea).toBeDisabled();
    expect(screen.getByRole("button", { name: "Đang bận" })).toBeDisabled();
  });

  it("preserves typed text when sending fails", async () => {
    mockApi(BASE_ROUTES);
    const user = userEvent.setup();
    const onSendNote = vi.fn().mockRejectedValue(new ApiError(400, "invalid", "Lỗi kết nối máy chủ"));

    renderInputBar({
      sessionId: SESSION_ID,
      sessionStatus: "awaiting_answers",
      onSendNote,
    });

    const textarea = screen.getByPlaceholderText(
      /Nhập phản hồi hoặc yêu cầu chỉnh sửa chẩn đoán\.\.\./,
    );
    await user.type(textarea, "Ghi chú phân tích quan trọng");
    await user.click(screen.getByRole("button", { name: "Gửi bổ sung phân tích" }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent("Lỗi kết nối máy chủ");
    });
    // Text remains in input for user to retry
    expect(textarea).toHaveValue("Ghi chú phân tích quan trọng");
  });
});
