import { LayoutDashboard, MessageSquare, BookOpen, Settings, ChevronLeft } from "lucide-react";
import { Button } from "./ui/button";
import { useState } from "react";
import { cn } from "@/lib/utils";

const menuItems = [
  { icon: LayoutDashboard, label: "Dashboard", active: false },
  { icon: MessageSquare, label: "Conversations", active: true },
  { icon: BookOpen, label: "Knowledge Base", active: false },
  { icon: Settings, label: "Settings", active: false },
];

const Sidebar = () => {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside 
      className={cn(
        "fixed left-0 top-16 bottom-0 bg-sidebar-background border-r border-sidebar-border transition-all duration-300 shadow-soft",
        collapsed ? "w-16" : "w-56"
      )}
    >
      <div className="h-full flex flex-col p-3">
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setCollapsed(!collapsed)}
          className="ml-auto mb-4 transition-smooth"
        >
          <ChevronLeft className={cn("h-4 w-4 transition-transform", collapsed && "rotate-180")} />
        </Button>
        
        <nav className="flex-1 space-y-1">
          {menuItems.map((item) => (
            <Button
              key={item.label}
              variant={item.active ? "secondary" : "ghost"}
              className={cn(
                "w-full justify-start gap-3 transition-smooth",
                collapsed && "justify-center px-2"
              )}
            >
              <item.icon className="h-5 w-5 flex-shrink-0" />
              {!collapsed && <span className="text-sm font-medium">{item.label}</span>}
            </Button>
          ))}
        </nav>
      </div>
    </aside>
  );
};

export default Sidebar;
