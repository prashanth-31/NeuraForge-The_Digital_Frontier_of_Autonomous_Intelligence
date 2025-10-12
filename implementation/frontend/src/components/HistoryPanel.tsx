import { Clock, History, Loader } from "lucide-react";

import { useTaskContext } from "@/contexts/TaskContext";
import { cn } from "@/lib/utils";
import { ScrollArea } from "./ui/scroll-area";

interface HistoryPanelProps {
  className?: string;
}

const HistoryPanel = ({ className }: HistoryPanelProps) => {
  const { history, currentTaskId, isStreaming } = useTaskContext();
  const hasHistory = history.length > 0;

  return (
    <aside
      className={cn(
        "hidden lg:flex w-80 xl:w-96 flex-col border-l border-border bg-card shadow-soft sticky top-16 h-[calc(100vh-4rem)]",
        className,
      )}
    >
      <div className="p-4 border-b border-border">
        <h2 className="font-semibold text-sm text-foreground">History</h2>
        <p className="text-xs text-muted-foreground mt-1">
          {currentTaskId ? `Task ${currentTaskId.slice(0, 8)}…` : "Past conversations"}
        </p>
      </div>

      <ScrollArea className="flex-1 p-3">
        {hasHistory ? (
          <div className="space-y-3 pr-2">
            {history.map((entry, index) => (
              <div key={`${entry.agent}-${index}`} className="p-3 rounded-lg border border-border/40 bg-muted/30">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-foreground">{entry.agent}</span>
                  {typeof entry.confidence === "number" && (
                    <span className="text-[11px] font-semibold text-primary">
                      {(entry.confidence * 100).toFixed(0)}% confidence
                    </span>
                  )}
                </div>
                <p className="text-xs text-muted-foreground leading-relaxed whitespace-pre-wrap">
                  {entry.content}
                </p>
                <div className="flex items-center gap-1 text-[11px] text-muted-foreground mt-2">
                  <Clock className="h-3 w-3" />
                  <span>{entry.timestamp}</span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-center text-xs text-muted-foreground gap-3 px-4">
            <div className="w-10 h-10 rounded-full bg-muted flex items-center justify-center">
              {isStreaming ? <Loader className="h-5 w-5 animate-spin" /> : <History className="h-5 w-5" />}
            </div>
            {isStreaming && currentTaskId ? (
              <p>Agents are working. Results will appear here shortly.</p>
            ) : (
              <p>Submit a task to build a timeline of agent insights.</p>
            )}
          </div>
        )}
      </ScrollArea>
    </aside>
  );
};

export default HistoryPanel;
