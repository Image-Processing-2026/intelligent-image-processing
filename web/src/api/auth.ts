import { useQuery } from "@tanstack/react-query";

import { ApiError, apiFetch } from "./client";
import type { User, UserSettings } from "./types";

export const ME_QUERY_KEY = ["auth", "me"] as const;

export interface Credentials {
  username: string;
  password: string;
}

export function login(credentials: Credentials): Promise<User> {
  return apiFetch<User>("/auth/login", {
    method: "POST",
    json: credentials,
    skipAuthRedirect: true,
  });
}

export function register(credentials: Credentials): Promise<User> {
  return apiFetch<User>("/auth/register", {
    method: "POST",
    json: credentials,
    skipAuthRedirect: true,
  });
}

export function logout(): Promise<void> {
  return apiFetch<void>("/auth/logout", { method: "POST", skipAuthRedirect: true });
}

export function updateSettings(settings: Partial<UserSettings>): Promise<User> {
  return apiFetch<User>("/auth/me/settings", { method: "PATCH", json: settings });
}

/** Người dùng hiện tại; null khi chưa đăng nhập (401), lỗi khác (mạng) giữ nguyên là lỗi. */
export async function fetchMe(): Promise<User | null> {
  try {
    return await apiFetch<User>("/auth/me", { skipAuthRedirect: true });
  } catch (error) {
    if (error instanceof ApiError && error.status === 401) return null;
    throw error;
  }
}

export function useMe() {
  return useQuery({ queryKey: ME_QUERY_KEY, queryFn: fetchMe, staleTime: 5 * 60 * 1000 });
}

/** Chỉ cho phép quay về đường dẫn nội bộ (chống open redirect qua ?next=). */
export function safeNext(next: string | null | undefined): string {
  if (!next || !next.startsWith("/") || next.startsWith("//") || next.startsWith("/\\")) {
    return "/";
  }
  return next;
}

// Quy tắc giống backend (src/api/auth.py) để báo lỗi sớm; backend vẫn là nơi kiểm tra cuối.
export const USERNAME_RULE = "Tên đăng nhập gồm 3–32 ký tự: chữ cái không dấu, số, dấu _ . -";
export const PASSWORD_RULE = "Mật khẩu cần 8–128 ký tự.";

export function validateUsername(username: string): string | null {
  return /^[a-z0-9_.-]{3,32}$/.test(username.trim().toLowerCase()) ? null : USERNAME_RULE;
}

export function validatePassword(password: string): string | null {
  return password.length >= 8 && password.length <= 128 ? null : PASSWORD_RULE;
}
