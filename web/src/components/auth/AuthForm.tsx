import { useMutation, useQueryClient } from "@tanstack/react-query";
import { AlertCircle, Loader2 } from "lucide-react";
import { useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import {
  ME_QUERY_KEY,
  login,
  register,
  validatePassword,
  validateUsername,
} from "@/api/auth";
import { ApiError, errorMessage } from "@/api/client";
import type { User } from "@/api/types";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

type Mode = "login" | "register";
type FieldName = "username" | "password" | "confirm";
type FieldErrors = Partial<Record<FieldName, string>>;

const COPY: Record<Mode, { title: string; description: string; submit: string }> = {
  login: {
    title: "Đăng nhập",
    description: "Đăng nhập để xem và tiếp tục các phiên xử lý của bạn.",
    submit: "Đăng nhập",
  },
  register: {
    title: "Tạo tài khoản",
    description: "Mỗi tài khoản có phiên xử lý và bộ nhớ ca bệnh riêng.",
    submit: "Tạo tài khoản",
  },
};

function clientErrors(mode: Mode, username: string, password: string, confirm: string): FieldErrors {
  if (mode === "login") {
    const errors: FieldErrors = {};
    if (!username.trim()) errors.username = "Nhập tên đăng nhập.";
    if (!password) errors.password = "Nhập mật khẩu.";
    return errors;
  }
  const errors: FieldErrors = {};
  const usernameError = validateUsername(username);
  const passwordError = validatePassword(password);
  if (usernameError) errors.username = usernameError;
  if (passwordError) errors.password = passwordError;
  if (!passwordError && confirm !== password) errors.confirm = "Mật khẩu xác nhận không khớp.";
  return errors;
}

export function AuthForm({ mode }: { mode: Mode }) {
  const copy = COPY[mode];
  const [params] = useSearchParams();
  const queryClient = useQueryClient();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({});
  const refs = {
    username: useRef<HTMLInputElement>(null),
    password: useRef<HTMLInputElement>(null),
    confirm: useRef<HTMLInputElement>(null),
  };

  const focusFirst = (errors: FieldErrors) => {
    const first = (["username", "password", "confirm"] as FieldName[]).find((name) => errors[name]);
    if (first) refs[first].current?.focus();
  };

  const mutation = useMutation({
    mutationFn: () => (mode === "login" ? login : register)({ username, password }),
    // RedirectIfAuthed chuyển trang khi đã có người dùng (tới ?next= nếu hợp lệ)
    onSuccess: (user: User) => queryClient.setQueryData(ME_QUERY_KEY, user),
    onError: (error) => {
      if (error instanceof ApiError && error.fields) {
        const errors: FieldErrors = {};
        if (error.fields.username) errors.username = error.fields.username;
        if (error.fields.password) errors.password = error.fields.password;
        setFieldErrors(errors);
        focusFirst(errors);
      }
    },
  });

  const onSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    if (mutation.isPending) return;
    const errors = clientErrors(mode, username, password, confirm);
    setFieldErrors(errors);
    if (Object.keys(errors).length > 0) {
      focusFirst(errors);
      return;
    }
    mutation.mutate();
  };

  // Lỗi theo trường đã hiện cạnh ô nhập; Alert chỉ dành cho lỗi chung
  const serverError =
    mutation.error && !(mutation.error instanceof ApiError && mutation.error.fields)
      ? errorMessage(mutation.error)
      : null;
  const busy = mutation.isPending;

  const field = (
    name: FieldName,
    label: string,
    value: string,
    setValue: (value: string) => void,
    props: React.ComponentProps<"input">,
  ) => {
    const error = fieldErrors[name];
    const errorId = `${name}-error`;
    return (
      <div className="grid gap-2">
        <Label htmlFor={name}>{label}</Label>
        <Input
          id={name}
          name={name}
          ref={refs[name]}
          value={value}
          onChange={(event) => setValue(event.target.value)}
          aria-invalid={error ? true : undefined}
          aria-describedby={error ? errorId : undefined}
          disabled={busy}
          {...props}
        />
        {error ? (
          <p id={errorId} className="text-sm text-destructive">
            {error}
          </p>
        ) : null}
      </div>
    );
  };

  return (
    <div className="flex min-h-svh items-center justify-center bg-background p-4">
      <Card className="w-full max-w-sm">
        <CardHeader>
          <p className="text-sm font-semibold text-primary">AI Image Doctor</p>
          <CardTitle>
            <h1 className="text-xl">{copy.title}</h1>
          </CardTitle>
          <CardDescription>{copy.description}</CardDescription>
        </CardHeader>
        <CardContent>
          <form className="grid gap-4" onSubmit={onSubmit} noValidate>
            {serverError ? (
              <Alert variant="destructive">
                <AlertCircle aria-hidden />
                <AlertDescription>{serverError}</AlertDescription>
              </Alert>
            ) : null}
            {field("username", "Tên đăng nhập", username, setUsername, {
              autoComplete: "username",
              autoCapitalize: "none",
              spellCheck: false,
              autoFocus: true,
            })}
            {field("password", "Mật khẩu", password, setPassword, {
              type: "password",
              autoComplete: mode === "login" ? "current-password" : "new-password",
            })}
            {mode === "register"
              ? field("confirm", "Xác nhận mật khẩu", confirm, setConfirm, {
                  type: "password",
                  autoComplete: "new-password",
                })
              : null}
            <Button type="submit" disabled={busy} className="w-full">
              {busy ? <Loader2 className="animate-spin" aria-hidden /> : null}
              {copy.submit}
            </Button>
          </form>
          <p className="mt-4 text-center text-sm text-muted-foreground">
            {mode === "login" ? (
              <>
                Chưa có tài khoản?{" "}
                <Link
                  className="font-medium text-primary underline-offset-4 hover:underline"
                  to={{ pathname: "/register", search: params.toString() ? `?${params}` : "" }}
                >
                  Tạo tài khoản
                </Link>
              </>
            ) : (
              <>
                Đã có tài khoản?{" "}
                <Link
                  className="font-medium text-primary underline-offset-4 hover:underline"
                  to={{ pathname: "/login", search: params.toString() ? `?${params}` : "" }}
                >
                  Đăng nhập
                </Link>
              </>
            )}
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
