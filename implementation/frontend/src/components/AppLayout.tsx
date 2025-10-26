import { ReactNode, useState } from "react";

import Navbar from "@/components/Navbar";
import Sidebar from "@/components/Sidebar";
import HistoryPanel from "@/components/HistoryPanel";
import { cn } from "@/lib/utils";

interface AppLayoutProps {
  children: ReactNode;
  rightPanel?: ReactNode;
  showHistory?: boolean;
  mainClassName?: string;
}

const AppLayout = ({ children, rightPanel, showHistory = false, mainClassName }: AppLayoutProps) => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Navbar />
      <div className="flex flex-1 pt-16 overflow-hidden">
        <Sidebar collapsed={sidebarCollapsed} onToggle={() => setSidebarCollapsed((prev) => !prev)} />
        <div className="flex flex-1 overflow-hidden">
          <main className={cn("flex-1 flex flex-col overflow-hidden", mainClassName)}>
            {children}
          </main>
          {showHistory ? rightPanel ?? <HistoryPanel /> : rightPanel}
        </div>
      </div>
    </div>
  );
};

export default AppLayout;
