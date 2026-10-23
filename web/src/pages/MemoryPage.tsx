import { errorMessage } from "@/api/client";
import { useMemoryStats } from "@/api/workspace";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { ADJUSTMENT_LABELS } from "@/lib/actions";
import { label } from "@/lib/labels";

const STYLE_LABELS: Record<string, string> = {
  natural: "Tự nhiên",
  balanced: "Cân bằng",
  vivid: "Đậm nét",
};

function CountList({
  counts,
  labels,
  empty,
}: {
  counts: Record<string, number>;
  labels: Record<string, string>;
  empty: string;
}) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) return <p className="text-sm text-muted-foreground">{empty}</p>;
  return (
    <ul className="grid gap-1 text-sm">
      {entries.map(([key, count]) => (
        <li key={key} className="flex justify-between gap-4 border-b py-1 last:border-b-0">
          <span>{label(labels, key)}</span>
          <span className="tabular-nums text-muted-foreground">{count} lần</span>
        </li>
      ))}
    </ul>
  );
}

export default function MemoryPage() {
  const memory = useMemoryStats();
  return (
    <div className="mx-auto grid max-w-3xl gap-4 p-4 sm:p-6">
      <div>
        <h1 className="text-xl font-semibold">Bộ nhớ của tôi</h1>
        <p className="text-sm text-muted-foreground">
          Hệ thống ghi lại số liệu ảnh, phác đồ, phiên bản bạn chọn và góp ý của bạn (không lưu
          ảnh) để gợi ý phong cách và học từ góp ý ở các lần sau. Chỉ bạn thấy bộ nhớ của mình.
        </p>
      </div>
      {memory.isPending ? (
        <Skeleton className="h-48 w-full" />
      ) : memory.isError ? (
        <div className="grid justify-items-start gap-2">
          <p role="alert" className="text-sm">
            {errorMessage(memory.error)}
          </p>
          <Button variant="outline" size="sm" onClick={() => memory.refetch()}>
            Thử lại
          </Button>
        </div>
      ) : !memory.data.enabled ? (
        <p className="rounded-md border p-4 text-sm">Bộ nhớ ca bệnh đang tắt trên máy chủ.</p>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2">
          <Card className="sm:col-span-2">
            <CardHeader>
              <CardTitle>
                <h2 className="text-base">Tổng quan</h2>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-sm">
                <dt className="text-muted-foreground">Số ca đã lưu</dt>
                <dd className="tabular-nums">{memory.data.cases}</dd>
                <dt className="text-muted-foreground">Phong cách ưa thích</dt>
                <dd>
                  {memory.data.preferred_style
                    ? label(STYLE_LABELS, memory.data.preferred_style)
                    : "Chưa đủ dữ liệu: cần ít nhất 3 lần chọn, một phong cách chiếm ≥ 60%"}
                </dd>
              </dl>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>
                <h2 className="text-base">Phiên bản đã chọn</h2>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <CountList
                counts={memory.data.choices}
                labels={STYLE_LABELS}
                empty="Chưa chọn phiên bản nào."
              />
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>
                <h2 className="text-base">Góp ý hay gặp</h2>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <CountList
                counts={memory.data.feedback}
                labels={ADJUSTMENT_LABELS}
                empty="Chưa có góp ý nào được hiểu thành điều chỉnh."
              />
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
