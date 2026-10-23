import { ChevronDown, Copy } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

/** Khu "Nâng cao": JSON kỹ thuật thu gọn mặc định, cuộn trong khung, có nút sao chép. */
export function TechnicalJson({ data }: { data: unknown }) {
  const [open, setOpen] = useState(false);
  const text = open ? JSON.stringify(data, null, 2) : "";
  return (
    <Collapsible open={open} onOpenChange={setOpen} className="rounded-lg border">
      <CollapsibleTrigger asChild>
        <button
          type="button"
          className="flex w-full items-center justify-between px-3 py-2 text-left text-sm font-medium"
        >
          Nâng cao: dữ liệu kỹ thuật (JSON)
          <ChevronDown
            aria-hidden
            className={`size-4 transition-transform ${open ? "rotate-180" : ""}`}
          />
        </button>
      </CollapsibleTrigger>
      <CollapsibleContent className="grid gap-2 border-t p-3">
        <div>
          <Button
            size="sm"
            variant="outline"
            onClick={() => {
              navigator.clipboard
                ?.writeText(JSON.stringify(data, null, 2))
                .then(() => toast.success("Đã sao chép JSON."))
                .catch(() => toast.error("Không sao chép được. Hãy chọn và sao chép thủ công."));
            }}
          >
            <Copy aria-hidden />
            Sao chép
          </Button>
        </div>
        <pre className="max-h-96 overflow-auto rounded bg-muted p-3 text-xs leading-relaxed">
          {text}
        </pre>
      </CollapsibleContent>
    </Collapsible>
  );
}
