import { useCallback, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Activity, Brain, Zap, CheckCircle, Loader2 } from "lucide-react";

import AppLayout from "@/components/AppLayout";
import { Card } from "@/components/ui/card";
import { useTaskContext } from "@/contexts/TaskContext";
import { API_BASE_URL } from "@/lib/api";

interface RecentActivityItem {
  task_id: string;
  summary: string;
  agent?: string | null;
  status?: string | null;
  confidence?: number | null;
  timestamp?: string | null;
}

interface HealthStatus {
  status: string;
}

const RECENT_ACTIVITY_LIMIT = 5;

const fetchRecentActivity = async (limit: number): Promise<RecentActivityItem[]> => {
  const response = await fetch(`${API_BASE_URL}/api/v1/history/recent?limit=${limit}`);
  if (!response.ok) {
    throw new Error(`Failed to load recent activity (${response.status})`);
  }
  return response.json() as Promise<RecentActivityItem[]>;
};

const fetchHealthStatus = async (): Promise<HealthStatus> => {
  const response = await fetch(`${API_BASE_URL}/api/v1/health`);
  if (!response.ok) {
    throw new Error("Health check failed");
  }
  return response.json();
};

const formatRelativeTime = (iso?: string | null) => {
  if (!iso) {
    return "Unknown";
  }
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return "Unknown";
  }
  const diffMs = Date.now() - date.getTime();
  const diffMinutes = Math.max(0, Math.floor(diffMs / 60000));
  if (diffMinutes < 1) return "Just now";
  if (diffMinutes < 60) return `${diffMinutes}m ago`;
  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays}d ago`;
};

const getStatusAccent = (status?: string | null) => {
  if (!status) return "bg-slate-300";
  const normalized = status.toLowerCase();
  if (normalized.includes("fail")) return "bg-rose-500";
  if (normalized.includes("progress") || normalized.includes("running")) return "bg-amber-500";
  if (normalized.includes("complete")) return "bg-emerald-500";
  return "bg-primary-500";
};

const AGENT_COUNT = 5; // finance, research, enterprise, creative, general

const Dashboard = () => {
  const navigate = useNavigate();
  const { loadTaskById } = useTaskContext();

  const { data: recentActivity = [], isLoading, isError } = useQuery({
    queryKey: ["recent-history", RECENT_ACTIVITY_LIMIT],
    queryFn: () => fetchRecentActivity(RECENT_ACTIVITY_LIMIT),
    staleTime: 30_000,
  });

  const { data: healthData } = useQuery({
    queryKey: ["health-status"],
    queryFn: fetchHealthStatus,
    staleTime: 60_000,
    retry: 1,
  });

  // Compute stats from real activity data
  const stats = useMemo(() => {
    const completedTasks = recentActivity.filter(
      (a) => a.status?.toLowerCase().includes("complete")
    ).length;
    const inProgressTasks = recentActivity.filter(
      (a) => a.status?.toLowerCase().includes("progress") || a.status?.toLowerCase().includes("running")
    ).length;
    const uniqueAgents = new Set(recentActivity.map((a) => a.agent).filter(Boolean)).size;

    return [
      { label: "Tasks in Progress", value: inProgressTasks.toString(), icon: Activity, color: "text-primary-600", bg: "bg-primary-50" },
      { label: "Agents Available", value: AGENT_COUNT.toString(), icon: Brain, color: "text-violet-600", bg: "bg-violet-50" },
      { label: "Recent Completed", value: completedTasks.toString(), icon: Zap, color: "text-amber-600", bg: "bg-amber-50" },
      { label: "Agents Active", value: uniqueAgents.toString(), icon: CheckCircle, color: "text-emerald-600", bg: "bg-emerald-50" },
    ];
  }, [recentActivity]);

  const isHealthy = healthData?.status === "ok";

  const handleActivitySelect = useCallback(async (taskId: string) => {
    await loadTaskById(taskId);
    navigate("/");
  }, [loadTaskById, navigate]);

  return (
    <AppLayout>
      <div className="flex-1 overflow-auto bg-gradient-to-b from-slate-50/50 to-white">
        <div className="max-w-7xl mx-auto space-y-8 p-8">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-foreground mb-1 tracking-tight">Dashboard</h1>
              <p className="text-muted-foreground">Monitor your AI collaboration metrics</p>
            </div>
            <div className="hidden sm:flex items-center gap-2 text-xs text-slate-400 bg-white border border-slate-200/60 px-3 py-1.5 rounded-full shadow-xs">
              <div className={`w-2 h-2 rounded-full ${isHealthy ? "bg-emerald-400 animate-pulse" : "bg-rose-400"}`} />
              <span>{isHealthy ? "All systems operational" : "System degraded"}</span>
            </div>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
            {stats.map((stat) => (
              <Card key={stat.label} className="p-6 hover:shadow-elevated transition-all duration-300 group cursor-pointer">
                <div className="flex items-center justify-between mb-4">
                  <div className={`w-12 h-12 rounded-xl ${stat.bg} flex items-center justify-center transition-transform duration-300 group-hover:scale-110`}>
                    <stat.icon className={`h-6 w-6 ${stat.color}`} />
                  </div>
                </div>
                <div className="text-3xl font-bold text-foreground mb-0.5 tracking-tight">{stat.value}</div>
                <div className="text-sm text-muted-foreground">{stat.label}</div>
              </Card>
            ))}
          </div>

          {/* Activity and Status Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 pb-16">
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-5 text-foreground">Recent Activity</h3>
              {isLoading && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground py-8 justify-center">
                  <Loader2 className="h-4 w-4 animate-spin text-primary-500" />
                  Loading activity…
                </div>
              )}
              {isError && (
                <p className="text-sm text-rose-600 bg-rose-50 border border-rose-200/60 rounded-lg px-4 py-3">Unable to load recent activity. Try again soon.</p>
              )}
              {!isLoading && !isError && recentActivity.length === 0 && (
                <p className="text-sm text-muted-foreground text-center py-8">No recent tasks have been recorded yet.</p>
              )}
              {!isLoading && !isError && recentActivity.length > 0 && (
                <div className="space-y-1 max-h-80 overflow-y-auto pr-2">
                  {recentActivity.map((activity) => (
                    <button
                      key={activity.task_id}
                      type="button"
                      onClick={() => handleActivitySelect(activity.task_id)}
                      className="w-full text-left hover:bg-slate-50 rounded-xl px-3 py-3 -mx-3 transition-colors duration-200"
                    >
                      <div className="flex items-start gap-3">
                        <div className={`w-2.5 h-2.5 rounded-full ${getStatusAccent(activity.status)} mt-1.5 flex-shrink-0`} />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-start justify-between gap-3">
                            <p className="text-sm font-medium text-foreground line-clamp-2">{activity.summary}</p>
                            <span className="text-[11px] text-slate-400 whitespace-nowrap bg-slate-100 px-1.5 py-0.5 rounded">
                              {formatRelativeTime(activity.timestamp)}
                            </span>
                          </div>
                          <p className="text-xs text-muted-foreground mt-1">
                            {activity.agent ?? "Unknown agent"}
                            {activity.status ? ` • ${activity.status.replace(/_/g, " ")}` : ""}
                          </p>
                          {typeof activity.confidence === "number" && (
                            <span className="inline-block text-[10px] font-semibold text-primary-600 bg-primary-50 px-1.5 py-0.5 rounded mt-1.5">
                              {(activity.confidence * 100).toFixed(0)}% confidence
                            </span>
                          )}
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </Card>

            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-5 text-foreground">System Status</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 rounded-xl bg-slate-50/80">
                  <span className="text-sm text-muted-foreground">AI Processing</span>
                  <span className="text-sm font-semibold text-emerald-600">Optimal</span>
                </div>
                <div className="flex items-center justify-between p-3 rounded-xl bg-slate-50/80">
                  <span className="text-sm text-muted-foreground">Memory Usage</span>
                  <div className="flex items-center gap-2">
                    <div className="w-24 h-2 bg-slate-200 rounded-full overflow-hidden">
                      <div className="w-[68%] h-full bg-gradient-to-r from-primary-400 to-primary-500 rounded-full" />
                    </div>
                    <span className="text-sm font-semibold text-foreground">68%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between p-3 rounded-xl bg-slate-50/80">
                  <span className="text-sm text-muted-foreground">Response Time</span>
                  <span className="text-sm font-semibold text-primary-600">125ms</span>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </AppLayout>
  );
};

export default Dashboard;
