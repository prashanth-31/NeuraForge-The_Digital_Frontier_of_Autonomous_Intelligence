import { Clock, Star } from "lucide-react";
import { Button } from "./ui/button";
import { ScrollArea } from "./ui/scroll-area";

const historyItems = [
  {
    id: 1,
    title: "Multi-agent reasoning analysis",
    summary: "Explored collaborative decision-making patterns",
    timestamp: "2 hours ago",
    pinned: true,
  },
  {
    id: 2,
    title: "Document retrieval optimization",
    summary: "Improved RAG source accuracy",
    timestamp: "5 hours ago",
    pinned: false,
  },
  {
    id: 3,
    title: "System architecture review",
    summary: "Analyzed orchestration patterns",
    timestamp: "Yesterday",
    pinned: false,
  },
];

const HistoryPanel = () => {
  return (
    <aside className="fixed right-0 top-16 bottom-0 w-72 bg-card border-l border-border shadow-soft">
      <div className="h-full flex flex-col">
        <div className="p-4 border-b border-border">
          <h2 className="font-semibold text-sm text-foreground">History</h2>
          <p className="text-xs text-muted-foreground mt-1">Past conversations</p>
        </div>
        
        <ScrollArea className="flex-1 p-3">
          <div className="space-y-2">
            {historyItems.map((item) => (
              <button
                key={item.id}
                className="w-full text-left p-3 rounded-lg bg-muted/30 hover:bg-muted/50 transition-smooth border border-transparent hover:border-border/50 group"
              >
                <div className="flex items-start justify-between gap-2 mb-2">
                  <h3 className="text-sm font-medium text-foreground line-clamp-1 group-hover:text-primary transition-smooth">
                    {item.title}
                  </h3>
                  {item.pinned && (
                    <Star className="h-3.5 w-3.5 text-accent fill-accent flex-shrink-0" />
                  )}
                </div>
                <p className="text-xs text-muted-foreground line-clamp-2 mb-2">
                  {item.summary}
                </p>
                <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Clock className="h-3 w-3" />
                  <span>{item.timestamp}</span>
                </div>
              </button>
            ))}
          </div>
        </ScrollArea>
      </div>
    </aside>
  );
};

export default HistoryPanel;
