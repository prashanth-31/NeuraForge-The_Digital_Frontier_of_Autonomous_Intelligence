import { LayoutDashboard, MessageSquare, BookOpen, Settings, ChevronLeft, ClipboardCheck } from "lucide-react";
import { NavLink } from "react-router-dom";

import { cn } from "@/lib/utils";
import { Button } from "./ui/button";

const menuItems = [
  { icon: LayoutDashboard, label: "Dashboard", to: "/dashboard" },
  { icon: MessageSquare, label: "Workspace", to: "/" },
  { icon: BookOpen, label: "Knowledge Base", to: "/knowledge" },
  { icon: ClipboardCheck, label: "Reviews", to: "/reviews" },
  { icon: Settings, label: "Settings", to: "/settings" },
];

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

const Sidebar = ({ collapsed, onToggle }: SidebarProps) => {
  return (
    <aside
      className={cn(
        "bg-sidebar-background border-r border-sidebar-border transition-all duration-300 shadow-soft hidden md:flex sticky top-16 h-[calc(100vh-4rem)]",
        collapsed ? "w-16" : "w-60"
      )}
    >
      <div className="h-full flex flex-col p-3 w-full">
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggle}
          className="ml-auto mb-4 transition-smooth"
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          <ChevronLeft className={cn("h-4 w-4 transition-transform", collapsed && "rotate-180")} />
        </Button>

        <nav className="flex-1 space-y-1">
          {menuItems.map((item) => (
            <NavLink key={item.label} to={item.to} className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-smooth hover:bg-muted/60",
                collapsed && "justify-center px-2",
                isActive ? "bg-secondary text-secondary-foreground" : "text-muted-foreground"
              )
            }>
              <item.icon className="h-5 w-5 flex-shrink-0" />
              {!collapsed && <span>{item.label}</span>}
            </NavLink>
          ))}
        </nav>
      </div>
    </aside>
  );
};

export default Sidebar;
