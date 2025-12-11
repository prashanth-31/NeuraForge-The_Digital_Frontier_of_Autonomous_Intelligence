import {
  LayoutDashboard,
  MessageSquare,
  BookOpen,
  Settings,
  ChevronLeft,
  ClipboardCheck,
} from "lucide-react";
import { NavLink } from "react-router-dom";

import { cn } from "@/lib/utils";
import { Button } from "./ui/button";

/**
 * Sidebar — Premium Teal Design System
 * -------------------------------------------------
 * Light glass surface, teal-highlighted active state, collapsible.
 */
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
        "hidden md:flex flex-col sticky top-16 h-[calc(100vh-4rem)]",
        "bg-sidebar border-r border-sidebar-border shadow-soft transition-all duration-300",
        collapsed ? "w-[72px]" : "w-60"
      )}
    >
      {/* Toggle button */}
      <div className="p-3 flex justify-end">
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggle}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          className="rounded-lg"
        >
          <ChevronLeft
            className={cn(
              "h-4 w-4 transition-transform",
              collapsed && "rotate-180"
            )}
          />
        </Button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 space-y-1">
        {menuItems.map((item) => (
          <NavLink
            key={item.label}
            to={item.to}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium",
                "transition-smooth",
                collapsed && "justify-center px-2",
                isActive
                  ? "bg-sidebar-accent text-sidebar-accent-foreground shadow-soft"
                  : "text-sidebar-foreground/70 hover:bg-sidebar-accent/60 hover:text-sidebar-foreground"
              )
            }
          >
            <item.icon className="h-5 w-5 flex-shrink-0" />
            {!collapsed && <span>{item.label}</span>}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
};

export default Sidebar;
