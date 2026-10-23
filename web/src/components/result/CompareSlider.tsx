import { ZoomIn } from "lucide-react";
import { useId, useState } from "react";

import type { Artifact } from "@/api/types";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cn } from "@/lib/utils";

type ZoomLevel = "fit" | "100" | "200";
type Side = "before" | "after";

/** Xem ảnh phóng to: vừa khung / 100% / 200%, cuộn để di chuyển, chuyển trước ↔ sau. */
function ZoomDialog({
  before,
  after,
  afterLabel,
}: {
  before: Artifact;
  after: Artifact;
  afterLabel: string;
}) {
  const [zoom, setZoom] = useState<ZoomLevel>("fit");
  const [side, setSide] = useState<Side>("after");
  const image = side === "before" ? before : after;
  const scale = zoom === "200" ? 2 : 1;
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm">
          <ZoomIn aria-hidden />
          Phóng to
        </Button>
      </DialogTrigger>
      <DialogContent className="flex max-h-[92svh] w-[calc(100vw-2rem)] max-w-5xl flex-col sm:max-w-5xl">
        <DialogHeader>
          <DialogTitle>Xem chi tiết</DialogTitle>
          <DialogDescription>Cuộn để di chuyển khi phóng to.</DialogDescription>
        </DialogHeader>
        <div className="flex flex-wrap items-center gap-2">
          <ToggleGroup
            type="single"
            variant="outline"
            size="sm"
            value={side}
            onValueChange={(value) => value && setSide(value as Side)}
            aria-label="Ảnh hiển thị"
          >
            <ToggleGroupItem value="before">Ảnh gốc</ToggleGroupItem>
            <ToggleGroupItem value="after">{afterLabel}</ToggleGroupItem>
          </ToggleGroup>
          <ToggleGroup
            type="single"
            variant="outline"
            size="sm"
            value={zoom}
            onValueChange={(value) => value && setZoom(value as ZoomLevel)}
            aria-label="Mức phóng"
          >
            <ToggleGroupItem value="fit">Vừa khung</ToggleGroupItem>
            <ToggleGroupItem value="100">100%</ToggleGroupItem>
            <ToggleGroupItem value="200">200%</ToggleGroupItem>
          </ToggleGroup>
        </div>
        <div
          className="min-h-0 flex-1 overflow-auto rounded-md bg-muted"
          tabIndex={0}
          aria-label="Vùng xem ảnh, dùng phím mũi tên để cuộn"
        >
          {zoom === "fit" ? (
            <img
              src={image.url}
              alt={side === "before" ? "Ảnh gốc" : afterLabel}
              className="mx-auto max-h-[70svh] w-full object-contain"
            />
          ) : (
            <img
              src={image.url}
              alt={side === "before" ? "Ảnh gốc" : afterLabel}
              width={image.width * scale}
              height={image.height * scale}
              style={{ width: image.width * scale, height: image.height * scale, maxWidth: "none" }}
              className="block [image-rendering:pixelated]"
            />
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

/**
 * So sánh trước/sau: ảnh sau phủ lên ảnh gốc, cắt theo vị trí thanh trượt. Thanh trượt là
 * input range gốc nên dùng được bằng bàn phím (←/→, Home/End).
 */
export function CompareSlider({
  before,
  after,
  afterLabel,
}: {
  before: Artifact;
  after: Artifact;
  afterLabel: string;
}) {
  const [position, setPosition] = useState(50);
  const inputId = useId();
  const sameSize = before.width === after.width && before.height === after.height;
  return (
    <div className="grid gap-2">
      <div
        className="relative mx-auto w-full overflow-hidden rounded-md bg-muted"
        style={{ aspectRatio: `${before.width} / ${before.height}`, maxHeight: 480 }}
      >
        <img
          src={before.url}
          alt="Ảnh gốc"
          className="absolute inset-0 h-full w-full object-contain"
        />
        <img
          src={after.url}
          alt={afterLabel}
          className="absolute inset-0 h-full w-full object-contain"
          style={{ clipPath: `inset(0 0 0 ${position}%)` }}
        />
        <div
          aria-hidden
          className="pointer-events-none absolute inset-y-0 w-0.5 bg-primary shadow"
          style={{ left: `${position}%` }}
        />
        <span className="pointer-events-none absolute top-2 left-2 rounded bg-background/85 px-1.5 py-0.5 text-xs">
          Gốc
        </span>
        <span className="pointer-events-none absolute top-2 right-2 rounded bg-background/85 px-1.5 py-0.5 text-xs">
          {afterLabel}
        </span>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <label htmlFor={inputId} className="text-xs text-muted-foreground">
          Vị trí so sánh
        </label>
        <input
          id={inputId}
          type="range"
          min={0}
          max={100}
          step={1}
          value={position}
          onChange={(event) => setPosition(Number(event.target.value))}
          aria-valuetext={`${position}% bên trái là ảnh gốc`}
          className={cn("h-2 min-w-32 flex-1 cursor-pointer accent-[var(--primary)]")}
        />
        <ZoomDialog before={before} after={after} afterLabel={afterLabel} />
      </div>
      {!sameSize ? (
        <p className="text-xs text-muted-foreground">
          Hai ảnh khác kích thước ({before.width}×{before.height} và {after.width}×{after.height});
          ảnh được co theo khung để so sánh.
        </p>
      ) : null}
    </div>
  );
}
