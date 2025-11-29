import { useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Activity, Brain, Zap, Users, Loader2 } from "lucide-react";

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

const RECENT_ACTIVITY_LIMIT = 5;

const fetchRecentActivity = async (limit: number): Promise<RecentActivityItem[]> => {
  const response = await fetch(`${API_BASE_URL}/api/v1/history/recent?limit=${limit}`);
  if (!response.ok) {
    throw new Error(`Failed to load recent activity (${response.status})`);
  }
  return response.json() as Promise<RecentActivityItem[]>;
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
  if (!status) return "bg-muted";
  const normalized = status.toLowerCase();
  if (normalized.includes("fail")) return "bg-destructive";
  if (normalized.includes("progress") || normalized.includes("running")) return "bg-amber-500";
  if (normalized.includes("complete")) return "bg-emerald-500";
  return "bg-primary";
};

const stats = [
  { label: "Active Sessions", value: "12", icon: Activity, color: "text-primary" },
  { label: "Agents Deployed", value: "4", icon: Brain, color: "text-secondary" },
  { label: "Tasks Completed", value: "47", icon: Zap, color: "text-accent" },
  { label: "Collaborators", value: "3", icon: Users, color: "text-primary" },
];

const Dashboard = () => {
  const navigate = useNavigate();
  const { loadTaskById } = useTaskContext();

  const { data: recentActivity = [], isLoading, isError } = useQuery({
    queryKey: ["recent-history", RECENT_ACTIVITY_LIMIT],
    queryFn: () => fetchRecentActivity(RECENT_ACTIVITY_LIMIT),
    staleTime: 30_000,
  });

  const handleActivitySelect = useCallback(async (taskId: string) => {
    await loadTaskById(taskId);
    navigate("/");
  }, [loadTaskById, navigate]);

  return (
    <AppLayout>
      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-8">
          <div>
            <h1 className="text-3xl font-bold text-foreground mb-2">Dashboard</h1>
            <p className="text-muted-foreground">Monitor your AI collaboration metrics</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {stats.map((stat) => (
              <Card key={stat.label} className="p-6 hover-lift transition-smooth">
                <div className="flex items-center justify-between mb-4">
                  <stat.icon className={`h-8 w-8 ${stat.color}`} />
                </div>
                <div className="text-3xl font-bold text-foreground mb-1">{stat.value}</div>
                <div className="text-sm text-muted-foreground">{stat.label}</div>
              </Card>
            ))}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 pb-16">
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4 text-foreground">Recent Activity</h3>
              {isLoading && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Loading activity…
                </div>
              )}
              {isError && (
                <p className="text-sm text-destructive">Unable to load recent activity. Try again soon.</p>
              )}
              {!isLoading && !isError && recentActivity.length === 0 && (
                <p className="text-sm text-muted-foreground">No recent tasks have been recorded yet.</p>
              )}
              {!isLoading && !isError && recentActivity.length > 0 && (
                <div className="space-y-3 max-h-72 overflow-y-auto pr-2">
                  {recentActivity.map((activity) => (
                    <button
                      key={activity.task_id}
                      type="button"
                      onClick={() => handleActivitySelect(activity.task_id)}
                      className="w-full text-left"
                    >
                      <div className="flex items-start gap-3 pb-4 border-b border-border last:border-0 last:pb-0">
                        <div className={`w-2 h-2 rounded-full ${getStatusAccent(activity.status)} mt-2`} />
                        <div className="flex-1">
                          <div className="flex items-center justify-between gap-3">
                            <p className="text-sm font-medium text-foreground line-clamp-2">{activity.summary}</p>
                            <span className="text-xs text-muted-foreground whitespace-nowrap">
                              {formatRelativeTime(activity.timestamp)}
                            </span>
                          </div>
                          <p className="text-xs text-muted-foreground mt-1">
                            {activity.agent ?? "Unknown agent"}
                            {activity.status ? ` • ${activity.status.replace(/_/g, " ")}` : ""}
                          </p>
                          {typeof activity.confidence === "number" && (
                            <p className="text-[11px] font-semibold text-primary mt-1">
                              Confidence {(activity.confidence * 100).toFixed(0)}%
                            </p>
                          )}
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </Card>

            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4 text-foreground">System Status</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">AI Processing</span>
                  <span className="text-sm font-medium text-accent">Optimal</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Memory Usage</span>
                  <span className="text-sm font-medium text-foreground">68%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Response Time</span>
                  <span className="text-sm font-medium text-accent">125ms</span>
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
