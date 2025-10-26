import { useMemo } from "react";
import { Search, User } from "lucide-react";

import { useTaskContext } from "@/contexts/TaskContext";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Input } from "./ui/input";

const Navbar = () => {
  const { resetSession, isStreaming, currentTaskId } = useTaskContext();

  const sessionLabel = useMemo(() => {
    if (isStreaming) {
      return "Streaming";
    }
    if (currentTaskId) {
      return `Task ${currentTaskId.slice(0, 6)}…`;
    }
    return "Idle";
  }, [isStreaming, currentTaskId]);

  return (
    <nav className="fixed top-0 left-0 right-0 h-16 bg-card border-b border-border z-50 shadow-soft">
      <div className="h-full px-6 flex items-center justify-between">
        <div className="flex items-center gap-8">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-primary to-secondary rounded-lg" />
            <h1 className="text-xl font-semibold text-foreground">NeuraForge</h1>
          </div>
        </div>
        
        <div className="flex-1 max-w-xl mx-8">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              type="text"
              placeholder="Search past interactions or documents..."
              className="pl-10 bg-muted/50 border-border/50 focus:bg-background transition-smooth"
            />
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          <Badge variant={isStreaming ? "default" : "secondary"} className="hidden md:inline-flex">
            {sessionLabel}
          </Badge>
          <Button
            variant="outline"
            size="sm"
            className="transition-smooth"
            onClick={resetSession}
            disabled={isStreaming}
          >
            New Session
          </Button>
          <Button variant="ghost" size="icon" className="rounded-full">
            <User className="h-5 w-5" />
          </Button>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
