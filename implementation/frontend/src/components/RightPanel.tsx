import { useState } from "react";
import { Activity, History, ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { useTaskContext } from "@/contexts/TaskContext";
import { AgentActivityPanel } from "./AgentActivityPanel";
import HistoryPanel from "./HistoryPanel";
import { Badge } from "./ui/badge";

type ActiveTab = "activity" | "history";

const RightPanel = () => {
  const [activeTab, setActiveTab] = useState<ActiveTab>("activity");
  const [collapsed, setCollapsed] = useState(false);
  const { thinkingEvents, isStreaming } = useTaskContext();

  const hasActivity = thinkingEvents.length > 0;

  if (collapsed) {
    return (
      <div className="w-12 border-l border-slate-200/60 bg-white/80 flex flex-col items-center py-4 gap-2">
        <button
          onClick={() => setCollapsed(false)}
          className="w-8 h-8 rounded-lg bg-slate-100 hover:bg-slate-200 flex items-center justify-center transition-colors"
          title="Expand panel"
        >
          <ChevronLeft className="h-4 w-4 text-slate-600" />
        </button>
        <div className="flex flex-col gap-2 mt-2">
          <button
            onClick={() => {
              setActiveTab("activity");
              setCollapsed(false);
            }}
            className={cn(
              "w-8 h-8 rounded-lg flex items-center justify-center transition-colors relative",
              activeTab === "activity" ? "bg-primary-100 text-primary-600" : "bg-slate-100 text-slate-600 hover:bg-slate-200"
            )}
            title="Agent Activity"
          >
            <Activity className="h-4 w-4" />
            {hasActivity && isStreaming && (
              <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-primary-500 animate-pulse" />
            )}
          </button>
          <button
            onClick={() => {
              setActiveTab("history");
              setCollapsed(false);
            }}
            className={cn(
              "w-8 h-8 rounded-lg flex items-center justify-center transition-colors",
              activeTab === "history" ? "bg-primary-100 text-primary-600" : "bg-slate-100 text-slate-600 hover:bg-slate-200"
            )}
            title="Session History"
          >
            <History className="h-4 w-4" />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="w-80 border-l border-slate-200/60 bg-white/80 backdrop-blur-sm flex flex-col">
      {/* Tab Headers */}
      <div className="flex items-center border-b border-slate-200/60 px-2 py-2 gap-1">
        <button
          onClick={() => setActiveTab("activity")}
          className={cn(
            "flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
            activeTab === "activity"
              ? "bg-primary-50 text-primary-700"
              : "text-muted-foreground hover:bg-slate-100"
          )}
        >
          <Activity className="h-4 w-4" />
          <span>Activity</span>
          {hasActivity && (
            <Badge 
              variant={isStreaming ? "default" : "secondary"} 
              className={cn(
                "text-[10px] px-1.5 py-0 min-w-[20px]",
                isStreaming && "animate-pulse"
              )}
            >
              {thinkingEvents.length}
            </Badge>
          )}
        </button>
        <button
          onClick={() => setActiveTab("history")}
          className={cn(
            "flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
            activeTab === "history"
              ? "bg-primary-50 text-primary-700"
              : "text-muted-foreground hover:bg-slate-100"
          )}
        >
          <History className="h-4 w-4" />
          <span>History</span>
        </button>
        <button
          onClick={() => setCollapsed(true)}
          className="w-8 h-8 rounded-lg bg-slate-100 hover:bg-slate-200 flex items-center justify-center transition-colors ml-1"
          title="Collapse panel"
        >
          <ChevronRight className="h-4 w-4 text-slate-600" />
        </button>
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === "activity" ? (
          <div className="h-full overflow-y-auto p-4">
            <AgentActivityPanel
              thinkingEvents={thinkingEvents}
              isStreaming={isStreaming}
            />
          </div>
        ) : (
          <HistoryPanel embedded />
        )}
      </div>
    </div>
  );
};

export default RightPanel;
