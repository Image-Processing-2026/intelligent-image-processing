import { Menu } from "lucide-react";
import { useState } from "react";
import { Link, Outlet } from "react-router-dom";

import { SessionSidebar } from "@/components/session/SessionSidebar";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";

import { BackendStatus } from "./BackendStatus";
import { ThemeToggle } from "./ThemeToggle";
import { UserMenu } from "./UserMenu";

export function AppShell() {
  const [menuOpen, setMenuOpen] = useState(false);
  return (
    <div className="flex h-svh flex-col overflow-hidden">
      <header className="z-20 flex h-14 shrink-0 items-center gap-2 border-b bg-background px-3 sm:px-4">
        <Button
          variant="ghost"
          size="icon"
          className="md:hidden"
          aria-label="Mở danh sách phiên"
          onClick={() => setMenuOpen(true)}
        >
          <Menu className="size-5" aria-hidden />
        </Button>
        <Link to="/" className="min-w-0 truncate rounded-sm font-semibold tracking-tight">
          AI Image Doctor
        </Link>
        <div className="ml-auto flex shrink-0 items-center gap-1 sm:gap-2">
          <BackendStatus />
          <ThemeToggle />
          <UserMenu />
        </div>
      </header>
      <div className="flex min-h-0 flex-1">
        <aside className="hidden w-56 shrink-0 border-r bg-sidebar text-sidebar-foreground md:block">
          <SessionSidebar />
        </aside>
        <main className="relative min-w-0 flex-1 overflow-y-auto">
          <Outlet />
        </main>
      </div>
      <Sheet open={menuOpen} onOpenChange={setMenuOpen}>
        <SheetContent side="left" className="w-80 max-w-[85vw] gap-0 p-0">
          <SheetHeader className="border-b">
            <SheetTitle>Phiên xử lý</SheetTitle>
          </SheetHeader>
          <SessionSidebar onNavigate={() => setMenuOpen(false)} />
        </SheetContent>
      </Sheet>
    </div>
  );
}
