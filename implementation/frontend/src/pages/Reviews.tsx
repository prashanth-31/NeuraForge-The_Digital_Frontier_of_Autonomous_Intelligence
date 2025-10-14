import { useEffect, useMemo, useRef } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { formatDistanceToNow } from "date-fns";
import {
  ClipboardList,
  MessageSquareText,
  CheckCircle2,
  UserPlus,
  RotateCcw,
  Loader2,
  AlertTriangle,
} from "lucide-react";

import AppLayout from "@/components/AppLayout";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { useToast } from "@/components/ui/use-toast";
import { useAuth } from "@/contexts/AuthContext";
import { API_BASE_URL, REVIEW_API_TOKEN, buildReviewHeaders } from "@/lib/api";
import { cn } from "@/lib/utils";

type ReviewStatus = "open" | "in_review" | "resolved" | "dismissed";

type ReviewNote = {
  note_id: string;
  author: string;
  content: string;
  created_at: string;
};

type ReviewTicket = {
  ticket_id: string;
  task_id: string;
  status: ReviewStatus;
  summary: string | null;
  created_at: string;
  updated_at: string;
  assigned_to: string | null;
  sources: string[];
  escalation_payload?: {
    prompt?: string;
    meta?: Record<string, unknown>;
    negotiation?: Record<string, unknown>;
  };
  notes: ReviewNote[];
};

type EnrichedTicket = ReviewTicket & {
  latest_note: ReviewNote | null;
};

type ActionParams = {
  ticketId: string;
  body?: Record<string, unknown>;
};

const fetchTickets = async (token: string | undefined): Promise<ReviewTicket[]> => {
  if (!token) {
    throw new Error("Reviewer token not configured");
  }
  const response = await fetch(`${API_BASE_URL}/api/v1/reviews`, {
    headers: buildReviewHeaders(token),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "Failed to load review tickets");
  }

  return response.json();
};

const Reviews = () => {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  const { toast } = useToast();
  const previousTicketsRef = useRef<Map<string, EnrichedTicket> | null>(null);

  const reviewerToken = useMemo(() => {
    const metadataToken = typeof user?.user_metadata?.review_api_token === "string"
      ? user.user_metadata.review_api_token.trim()
      : undefined;
    return metadataToken || REVIEW_API_TOKEN || undefined;
  }, [user]);

  const reviewerLabel = user?.email ?? user?.id ?? "unknown";

  const {
    data: tickets = [],
    isLoading,
    error,
  } = useQuery({
    queryKey: ["review-tickets", reviewerToken],
    queryFn: () => fetchTickets(reviewerToken),
    enabled: Boolean(reviewerToken),
    refetchInterval: reviewerToken ? 15000 : false,
  });

  const enrichedTickets = useMemo<EnrichedTicket[]>(
    () =>
      tickets.map((ticket) => ({
        ...ticket,
        latest_note: ticket.notes.length ? ticket.notes[ticket.notes.length - 1] : null,
      })),
    [tickets],
  );

  const groupedTickets = useMemo(() => {
    return {
      open: enrichedTickets.filter((ticket) => ticket.status === "open"),
      in_review: enrichedTickets.filter((ticket) => ticket.status === "in_review"),
      closed: enrichedTickets.filter((ticket) => ticket.status === "resolved" || ticket.status === "dismissed"),
    };
  }, [enrichedTickets]);

  const invalidateTickets = () =>
    queryClient.invalidateQueries({ queryKey: ["review-tickets", reviewerToken] });

  const performAction = async (endpoint: string, { ticketId, body }: ActionParams, method: "POST" | "PATCH" = "POST") => {
    if (!reviewerToken) {
      throw new Error("Reviewer token not configured");
    }
    const response = await fetch(`${API_BASE_URL}/api/v1/reviews/${ticketId}/${endpoint}`, {
      method,
      headers: buildReviewHeaders(reviewerToken),
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || "Action failed");
    }
    return response.json();
  };

  const assignMutation = useMutation({
    mutationFn: (params: ActionParams) =>
      performAction("assign", {
        ...params,
        body: { reviewer_id: reviewerLabel, ...(params.body ?? {}) },
      }),
    onSuccess: invalidateTickets,
    onError: (err: Error) => toast({ title: "Assignment failed", description: err.message, variant: "destructive" }),
  });

  const unassignMutation = useMutation({
    mutationFn: (params: ActionParams) => performAction("unassign", params, "PATCH"),
    onSuccess: invalidateTickets,
    onError: (err: Error) => toast({ title: "Unassign failed", description: err.message, variant: "destructive" }),
  });

  const noteMutation = useMutation({
    mutationFn: (params: ActionParams) =>
      performAction("notes", {
        ...params,
        body: { author: reviewerLabel, ...(params.body ?? {}) },
      }),
    onSuccess: invalidateTickets,
    onError: (err: Error) => toast({ title: "Note failed", description: err.message, variant: "destructive" }),
  });

  const resolveMutation = useMutation({
    mutationFn: (params: ActionParams & { status: Exclude<ReviewStatus, "open" | "in_review"> }) =>
      performAction(
        "resolve",
        {
          ticketId: params.ticketId,
          body: {
            status: params.status,
            reviewer_id: reviewerLabel,
            ...(params.body ?? {}),
          },
        },
      ),
    onSuccess: invalidateTickets,
    onError: (err: Error) => toast({ title: "Resolution failed", description: err.message, variant: "destructive" }),
  });

  useEffect(() => {
    if (!enrichedTickets.length) {
      previousTicketsRef.current = new Map();
      return;
    }

    const previousMap = previousTicketsRef.current;
    const currentMap = new Map(enrichedTickets.map((ticket) => [ticket.ticket_id, ticket]));

    if (!previousMap) {
      previousTicketsRef.current = currentMap;
      return;
    }

    enrichedTickets.forEach((ticket) => {
      const previous = previousMap.get(ticket.ticket_id);
      if (!previous) {
        toast({
          title: "New review ticket",
          description: buildToastDescription(ticket, "created"),
        });
        return;
      }

      if (previous.status !== ticket.status) {
        toast({
          title: ticket.status === "resolved" ? "Ticket resolved" : ticket.status === "dismissed" ? "Ticket dismissed" : "Status updated",
          description: buildToastDescription(ticket, ticket.status),
        });
        return;
      }

      if (previous.assigned_to !== ticket.assigned_to) {
        toast({
          title: ticket.assigned_to ? "Ticket assigned" : "Ticket unassigned",
          description: buildToastDescription(ticket, ticket.assigned_to ? "assigned" : "unassigned"),
        });
        return;
      }

      if (previous.notes.length < ticket.notes.length && ticket.latest_note) {
        toast({
          title: "New reviewer note",
          description: `${ticket.latest_note.author}: ${ticket.latest_note.content}`,
        });
      }
    });

    previousTicketsRef.current = currentMap;
  }, [enrichedTickets, toast]);

  const ensureNote = (action: string) => {
    const content = window.prompt(`Add a note for ${action}`);
    if (!content) {
      return undefined;
    }
    const trimmed = content.trim();
    return trimmed.length ? trimmed : undefined;
  };

  const renderTicketHeader = (ticket: EnrichedTicket, showSummary = true) => {
    const title = ticket.summary?.trim() || `Ticket ${ticket.ticket_id.slice(0, 8)}`;
    const prompt = typeof ticket.escalation_payload?.prompt === "string" ? ticket.escalation_payload.prompt : "Prompt unavailable";
    return (
      <div className="space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="text-lg font-semibold text-foreground">{title}</h3>
          <Badge variant="outline" className="capitalize">
            {ticket.status.replace("_", " ")}
          </Badge>
          {ticket.assigned_to && <Badge variant="secondary">Assigned to {ticket.assigned_to}</Badge>}
        </div>
        {showSummary && (
          <p className="text-sm text-muted-foreground whitespace-pre-wrap">{prompt}</p>
        )}
      </div>
    );
  };

  const openTicketsContent = groupedTickets.open.length ? (
    <div className="space-y-4">
      {groupedTickets.open.map((ticket) => (
        <Card key={ticket.ticket_id} className="p-4 transition-smooth hover-lift">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div className="space-y-4">
              {renderTicketHeader(ticket)}
              <div className="space-y-2 text-xs text-muted-foreground">
                <div className="flex items-center gap-2">
                  <ClipboardList className="h-4 w-4" />
                  Created {formatDistanceToNow(new Date(ticket.created_at), { addSuffix: true })}
                </div>
                <div className="flex items-center gap-2">
                  <MessageSquareText className="h-4 w-4" />
                  {ticket.notes.length} note{ticket.notes.length === 1 ? "" : "s"}
                </div>
              </div>
            </div>
            <div className="flex flex-col gap-2 min-w-[200px]">
              <Button
                variant="default"
                size="sm"
                onClick={() => assignMutation.mutate({ ticketId: ticket.ticket_id })}
                disabled={assignMutation.isPending}
              >
                <UserPlus className="mr-2 h-4 w-4" /> Assign to me
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => {
                  const content = ensureNote("context update");
                  if (!content) return;
                  noteMutation.mutate({ ticketId: ticket.ticket_id, body: { content } });
                }}
                disabled={noteMutation.isPending}
              >
                <MessageSquareText className="mr-2 h-4 w-4" /> Add note
              </Button>
              <Button
                variant="destructive"
                size="sm"
                onClick={() => resolveMutation.mutate({ ticketId: ticket.ticket_id, status: "dismissed" })}
                disabled={resolveMutation.isPending}
              >
                <RotateCcw className="mr-2 h-4 w-4" /> Dismiss
              </Button>
            </div>
          </div>
        </Card>
      ))}
    </div>
  ) : (
    <Card className="p-8 text-center text-muted-foreground">
      <p>No open review tickets 🎉</p>
    </Card>
  );

  const activeTicketsContent = groupedTickets.in_review.length ? (
    <div className="space-y-4">
      {groupedTickets.in_review.map((ticket) => (
        <Card key={ticket.ticket_id} className="p-4 transition-smooth hover-lift">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div className="space-y-3">
              {renderTicketHeader(ticket)}
              <div className="space-y-2 text-xs text-muted-foreground">
                <div className="flex items-center gap-2">
                  <ClipboardList className="h-4 w-4" />
                  Updated {formatDistanceToNow(new Date(ticket.updated_at), { addSuffix: true })}
                </div>
                {ticket.latest_note && (
                  <div className="flex items-center gap-2">
                    <MessageSquareText className="h-4 w-4" />
                    Latest note by {ticket.latest_note.author}: {ticket.latest_note.content}
                  </div>
                )}
              </div>
            </div>
            <div className="flex flex-col gap-2 min-w-[200px]">
              <Button
                variant="default"
                size="sm"
                onClick={() => resolveMutation.mutate({ ticketId: ticket.ticket_id, status: "resolved" })}
                disabled={resolveMutation.isPending}
              >
                <CheckCircle2 className="mr-2 h-4 w-4" /> Resolve
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => unassignMutation.mutate({ ticketId: ticket.ticket_id })}
                disabled={unassignMutation.isPending}
              >
                Unassign
              </Button>
            </div>
          </div>
        </Card>
      ))}
    </div>
  ) : (
    <Card className="p-8 text-center text-muted-foreground">
      <p>No tickets actively under review.</p>
    </Card>
  );

  const closedTicketsContent = groupedTickets.closed.length ? (
    <div className="grid gap-4 md:grid-cols-2">
      {groupedTickets.closed.map((ticket) => (
        <Card key={ticket.ticket_id} className="p-4">
          <div className="flex items-center gap-2 mb-2">
            <Badge
              variant={ticket.status === "resolved" ? "default" : "outline"}
              className={cn("capitalize")}
            >
              {ticket.status.replace("_", " ")}
            </Badge>
            {ticket.assigned_to && <span className="text-xs text-muted-foreground">by {ticket.assigned_to}</span>}
          </div>
          <h4 className="text-sm font-semibold text-foreground">
            {ticket.summary?.trim() || `Ticket ${ticket.ticket_id.slice(0, 8)}`}
          </h4>
          <p className="text-xs text-muted-foreground mt-1 line-clamp-3">
            {typeof ticket.escalation_payload?.prompt === "string"
              ? ticket.escalation_payload.prompt
              : "Prompt unavailable"}
          </p>
        </Card>
      ))}
    </div>
  ) : (
    <Card className="p-6 text-center text-muted-foreground">
      <p>No closed tickets yet.</p>
    </Card>
  );

  const loadingState = isLoading ? (
    <div className="flex items-center justify-center py-12 text-muted-foreground">
      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
      Loading tickets...
    </div>
  ) : error ? (
    <div className="text-sm text-destructive">{(error as Error).message}</div>
  ) : null;

  return (
    <AppLayout>
      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-6xl mx-auto space-y-8">
          <div>
            <h1 className="text-3xl font-bold text-foreground">Reviewer Console</h1>
            <p className="text-muted-foreground mt-2">
              Track escalations, collaborate with agents, and keep service levels on track.
            </p>
          </div>

          {!reviewerToken && (
            <Card className="border-dashed border-2 border-destructive/40 bg-destructive/10 p-6">
              <div className="flex items-start gap-3">
                <AlertTriangle className="h-5 w-5 text-destructive mt-0.5" />
                <div className="space-y-2 text-sm text-destructive">
                  <p className="font-semibold">Reviewer API token missing.</p>
                  <p>
                    Provide a token via <code>VITE_REVIEW_API_TOKEN</code> or add <code>review_api_token</code> to your
                    Supabase user metadata, then refresh the page.
                  </p>
                </div>
              </div>
            </Card>
          )}

          {loadingState}

          {reviewerToken && !isLoading && !error && (
            <div className="space-y-10 pb-16">
              <section>
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-semibold text-foreground">New tickets</h2>
                  <Badge variant="secondary">{groupedTickets.open.length} waiting</Badge>
                </div>
                {openTicketsContent}
              </section>

              <Separator />

              <section>
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-semibold text-foreground">Active reviews</h2>
                  <Badge variant="secondary">{groupedTickets.in_review.length} in progress</Badge>
                </div>
                {activeTicketsContent}
              </section>

              <Separator />

              <section>
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-semibold text-foreground">Recently closed</h2>
                  <Badge variant="outline">{groupedTickets.closed.length}</Badge>
                </div>
                {closedTicketsContent}
              </section>
            </div>
          )}
        </div>
      </div>
    </AppLayout>
  );
};

function buildToastDescription(ticket: EnrichedTicket, reason: string): string {
  const title = ticket.summary?.trim() || ticket.task_id;
  switch (reason) {
    case "created":
      return `${title} opened${ticket.assigned_to ? ` (default ${ticket.assigned_to})` : ""}`;
    case "assigned":
      return `${title} assigned to ${ticket.assigned_to ?? "unknown"}`;
    case "unassigned":
      return `${title} returned to queue`;
    case "resolved":
      return `${title} verified and closed`;
    case "dismissed":
      return `${title} dismissed after review`;
    default:
      return title;
  }
}

export default Reviews;
