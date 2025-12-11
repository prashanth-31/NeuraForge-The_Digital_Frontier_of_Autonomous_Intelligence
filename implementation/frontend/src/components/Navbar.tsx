import { useMemo } from "react";
import { Search, User, Sparkles } from "lucide-react";

import { useTaskContext } from "@/contexts/TaskContext";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Input } from "./ui/input";

/**
 * Navbar — Premium Teal Design System
 * -------------------------------------------------
 * Fixed glass header with teal accent logo, search bar, session badge.
 */
const Navbar = () => {
  const { resetSession, isStreaming, currentTaskId } = useTaskContext();

  const sessionLabel = useMemo(() => {
    if (isStreaming) return "Streaming";
    if (currentTaskId) return `Task ${currentTaskId.slice(0, 6)}…`;
    return "Idle";
  }, [isStreaming, currentTaskId]);

  return (
    <nav className="flex-shrink-0 h-14 z-50 bg-white border-b border-slate-200">
      <div className="h-full max-w-7xl mx-auto px-6 flex items-center justify-between gap-6">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center shadow-sm">
            <Sparkles className="h-4 w-4 text-white" />
          </div>
          <h1 className="text-lg font-bold tracking-tight text-foreground">
            NeuraForge
          </h1>
        </div>

        {/* Search */}
        <div className="flex-1 max-w-xl hidden md:block">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              type="text"
              placeholder="Search interactions, documents, agents…"
              className="pl-10 h-9 bg-slate-50 border-slate-200 focus:bg-white text-sm"
            />
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-3">
          <Badge
            variant={isStreaming ? "default" : "secondary"}
            className="hidden sm:inline-flex text-xs"
          >
            {sessionLabel}
          </Badge>
          <Button
            variant="outline"
            size="sm"
            onClick={resetSession}
            disabled={isStreaming}
            className="h-8 text-sm"
          >
            New Session
          </Button>
          <Button variant="ghost" size="icon" className="rounded-full h-8 w-8">
            <User className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
