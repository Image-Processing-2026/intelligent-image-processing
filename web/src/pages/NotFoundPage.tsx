import { Link } from "react-router-dom";

import { Button } from "@/components/ui/button";

export default function NotFoundPage() {
  return (
    <div className="mx-auto grid max-w-xl justify-items-center gap-3 p-6 text-center">
      <h1 className="text-lg font-semibold">Không tìm thấy trang</h1>
      <p className="text-sm text-muted-foreground">Đường dẫn không tồn tại.</p>
      <Button asChild>
        <Link to="/">Về Phiên mới</Link>
      </Button>
    </div>
  );
}
