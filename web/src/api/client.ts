// Client HTTP cho API /api/v1 (cùng origin, cookie httpOnly). Không đặt timeout: pipeline chạy
// đồng bộ và có thể mất vài phút (D3).

export const API_PREFIX = "/api/v1";

export const NETWORK_MESSAGE =
  "Không kết nối được máy chủ. Kiểm tra backend đang chạy rồi thử lại.";
const GENERIC_MESSAGE = "Có lỗi phía máy chủ. Vui lòng thử lại.";

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly fields?: Record<string, string>;

  constructor(status: number, code: string, message: string, fields?: Record<string, string>) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.fields = fields;
  }
}

// Trạng thái kết nối tới backend, suy từ chính các request (không poll, D3)
type ConnectivityListener = (online: boolean) => void;
const connectivityListeners = new Set<ConnectivityListener>();

/** Đăng ký nghe trạng thái kết nối; trả hàm hủy đăng ký. */
export function onConnectivity(listener: ConnectivityListener): () => void {
  connectivityListeners.add(listener);
  return () => connectivityListeners.delete(listener);
}

function reportConnectivity(online: boolean) {
  connectivityListeners.forEach((listener) => listener(online));
}

type UnauthorizedHandler = (error: ApiError) => void;
let onUnauthorized: UnauthorizedHandler | null = null;

/** Đăng ký hàm xử lý 401 toàn cục (xóa cache, chuyển về trang đăng nhập). */
export function setUnauthorizedHandler(handler: UnauthorizedHandler | null) {
  onUnauthorized = handler;
}

export interface ApiRequestInit extends Omit<RequestInit, "body"> {
  json?: unknown;
  body?: BodyInit | null;
  /** Không gọi handler 401 toàn cục (dùng cho /auth/me, /auth/login). */
  skipAuthRedirect?: boolean;
}

interface ErrorEnvelope {
  error?: { code?: unknown; message?: unknown; fields?: unknown };
}

const STATUS_FALLBACK: Record<number, string> = {
  401: "unauthenticated",
  404: "not_found",
  409: "invalid_state",
  413: "file_too_large",
  415: "unsupported_format",
  422: "validation",
};

async function toApiError(response: Response): Promise<ApiError> {
  let envelope: ErrorEnvelope | null = null;
  try {
    envelope = (await response.json()) as ErrorEnvelope;
  } catch {
    envelope = null;
  }
  const error = envelope?.error;
  if (error && typeof error.code === "string" && typeof error.message === "string") {
    const fields =
      error.fields && typeof error.fields === "object"
        ? (error.fields as Record<string, string>)
        : undefined;
    return new ApiError(response.status, error.code, error.message, fields);
  }
  // Không có envelope (proxy lỗi, route cũ): không hiện nội dung thô lên UI
  if (response.status === 502 || response.status === 503 || response.status === 504) {
    return new ApiError(response.status, "network", NETWORK_MESSAGE);
  }
  return new ApiError(
    response.status,
    STATUS_FALLBACK[response.status] ?? "internal",
    GENERIC_MESSAGE,
  );
}

export async function apiFetch<T>(path: string, init: ApiRequestInit = {}): Promise<T> {
  const { json, skipAuthRedirect, headers, ...rest } = init;
  const finalHeaders = new Headers(headers);
  let body = init.body;
  if (json !== undefined) {
    finalHeaders.set("Content-Type", "application/json");
    body = JSON.stringify(json);
  }
  finalHeaders.set("Accept", "application/json");

  let response: Response;
  try {
    response = await fetch(`${API_PREFIX}${path}`, {
      ...rest,
      body,
      headers: finalHeaders,
      credentials: "same-origin",
    });
  } catch {
    reportConnectivity(false);
    throw new ApiError(0, "network", NETWORK_MESSAGE);
  }
  // Proxy của Vite trả 502/503/504 khi backend không chạy
  reportConnectivity(![502, 503, 504].includes(response.status));

  if (!response.ok) {
    const error = await toApiError(response);
    if (error.status === 401 && !skipAuthRedirect) onUnauthorized?.(error);
    throw error;
  }
  if (response.status === 204) return undefined as T;
  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError(response.status, "internal", GENERIC_MESSAGE);
  }
}

/** Thông điệp tiếng Việt an toàn để hiện lên UI cho mọi loại lỗi. */
export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  return GENERIC_MESSAGE;
}
