import { useMemo, useState } from "react";
import { API_BASE_URL } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";
import { Clipboard, ClipboardCheck, FileText, Loader2, Upload, AlertTriangle } from "lucide-react";

interface DocumentMetadata {
  filename: string;
  content_type: string | null;
  extension: string | null;
  line_count: number;
  character_count: number;
  filesize_bytes: number;
}

interface DocumentAnalysisResponse {
  output: string;
  document: DocumentMetadata;
  truncated: boolean;
  persisted: boolean;
  memory_task_id: string | null;
  preview: string | null;
}

const formatBytes = (value: number) => {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
};

const DocumentUploader = () => {
  const { toast } = useToast();
  const [file, setFile] = useState<File | null>(null);
  const [persist, setPersist] = useState(false);
  const [response, setResponse] = useState<DocumentAnalysisResponse | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [copyState, setCopyState] = useState<"idle" | "copied">("idle");

  const analysisPreview = useMemo(() => {
    if (!response?.preview) return null;
    return response.preview.length > 0 ? response.preview : null;
  }, [response]);

  const handleCopyTaskId = async () => {
    if (!response?.memory_task_id) return;
    try {
      await navigator.clipboard.writeText(response.memory_task_id);
      setCopyState("copied");
      setTimeout(() => setCopyState("idle"), 1500);
      toast({
        title: "Task ID copied",
        description: "You can reference this analysis in future workflows.",
      });
    } catch (error) {
      toast({
        title: "Copy failed",
        description: "Unable to copy task identifier to the clipboard.",
      });
    }
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!file) {
      toast({
        title: "Choose a document",
        description: "Select a PDF, DOCX, TXT, CSV, JSON, or Markdown file to analyze.",
      });
      return;
    }

    const formData = new FormData();
    formData.append("document", file);

    const url = new URL(`${API_BASE_URL}/api/v1/upload_document`);
    if (persist) {
      url.searchParams.set("persist", "true");
    }

    setIsUploading(true);
    setCopyState("idle");
    try {
      const response = await fetch(url.toString(), {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let detail = "Document upload failed.";
        try {
          const payload = await response.json();
          detail = payload?.detail ?? detail;
        } catch (error) {
          // ignore json parse issues and fall back to default detail
        }
        toast({
          title: "Upload failed",
          description: detail,
        });
        return;
      }

      const payload: DocumentAnalysisResponse = await response.json();
      setResponse(payload);
      toast({
        title: "Analysis complete",
        description: payload.persisted
          ? "Document summarized and stored in memory."
          : "Document summarized without persistence.",
      });
    } catch (error) {
      toast({
        title: "Upload failed",
        description: error instanceof Error ? error.message : "Unexpected error analyzing document.",
      });
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <Card className="p-6 space-y-6">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div className="space-y-1">
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <FileText className="h-5 w-5 text-primary" />
            Upload knowledge asset
          </h2>
          <p className="text-sm text-muted-foreground">
            Parse a document, generate an LLM summary, and optionally persist the snapshot for future tasks.
          </p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-5">
        <div className="grid gap-4 sm:grid-cols-[minmax(0,_2fr)_minmax(0,_1fr)]">
          <div className="space-y-3">
            <div className="flex flex-col gap-2">
              <Label htmlFor="document">Document</Label>
              <Input
                id="document"
                type="file"
                accept=".pdf,.docx,.txt,.csv,.json,.md,.markdown"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                disabled={isUploading}
              />
              <p className="text-xs text-muted-foreground">
                Supported: PDF, DOCX, TXT, CSV, JSON, Markdown • max 8 MB
              </p>
            </div>

            <div className="flex items-center justify-between rounded-lg border p-4">
              <div>
                <Label htmlFor="persist-toggle" className="flex items-center gap-2">
                  Persist analysis
                </Label>
                <p className="text-xs text-muted-foreground">
                  Store the LLM summary and raw text in the memory service for downstream agents.
                </p>
              </div>
              <Switch
                id="persist-toggle"
                checked={persist}
                onCheckedChange={(checked) => setPersist(checked)}
                disabled={isUploading}
              />
            </div>
          </div>

          <div className="rounded-lg border bg-muted/30 p-4 text-sm text-muted-foreground space-y-2">
            <p>
              <span className="font-semibold text-foreground">Tip:</span> Persisting creates a task snapshot so agents can
              recall this document by Task ID.
            </p>
            <p>
              Documents stay in-memory during the session. For long-term storage, schedule ingestion via the Knowledge
              API.
            </p>
          </div>
        </div>

        <Button type="submit" disabled={isUploading} className="gap-2">
          {isUploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
          {isUploading ? "Analyzing..." : "Analyze document"}
        </Button>
      </form>

      {response && (
        <div className="space-y-5">
          <div className="flex items-center gap-2 text-sm">
            <Badge variant="outline" className={cn("gap-1", response.persisted && "bg-emerald-50 text-emerald-600")}> 
              {response.persisted ? "Persisted" : "Ephemeral"}
            </Badge>
            {response.truncated && (
              <span className="flex items-center gap-1 text-amber-600">
                <AlertTriangle className="h-3.5 w-3.5" />
                Document truncated before analysis
              </span>
            )}
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <Card className="p-4 space-y-2">
              <h3 className="text-sm font-semibold text-foreground">Document details</h3>
              <dl className="grid grid-cols-2 gap-x-3 gap-y-2 text-sm">
                <dt className="text-muted-foreground">Filename</dt>
                <dd className="text-foreground break-all">{response.document.filename}</dd>
                <dt className="text-muted-foreground">Type</dt>
                <dd className="text-foreground">{response.document.content_type ?? "unknown"}</dd>
                <dt className="text-muted-foreground">Lines</dt>
                <dd className="text-foreground">{response.document.line_count.toLocaleString()}</dd>
                <dt className="text-muted-foreground">Characters</dt>
                <dd className="text-foreground">{response.document.character_count.toLocaleString()}</dd>
                <dt className="text-muted-foreground">Size</dt>
                <dd className="text-foreground">{formatBytes(response.document.filesize_bytes)}</dd>
              </dl>

              {analysisPreview && (
                <div className="mt-3 space-y-1">
                  <h4 className="text-xs font-semibold uppercase tracking-wide">Preview</h4>
                  <p className="rounded-md border bg-muted/50 p-3 text-xs leading-relaxed text-muted-foreground">
                    {analysisPreview}
                  </p>
                </div>
              )}
            </Card>

            <Card className="p-4 space-y-3">
              <div className="flex items-center justify-between gap-2">
                <div>
                  <h3 className="text-sm font-semibold text-foreground">Memory reference</h3>
                  <p className="text-xs text-muted-foreground">
                    Share this ID with agents to thread the document into future tasks.
                  </p>
                </div>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={handleCopyTaskId}
                  disabled={!response.memory_task_id}
                  className="gap-2"
                >
                  {copyState === "copied" ? <ClipboardCheck className="h-4 w-4" /> : <Clipboard className="h-4 w-4" />}
                  {copyState === "copied" ? "Copied" : "Copy ID"}
                </Button>
              </div>

              <p className="text-sm font-mono break-all rounded-md border bg-muted/50 p-3">
                {response.memory_task_id ?? "Persistence disabled for this analysis."}
              </p>
            </Card>
          </div>

          <div className="space-y-2">
            <Label htmlFor="analysis-output">LLM summary</Label>
            <Textarea
              id="analysis-output"
              value={response.output}
              readOnly
              className="min-h-[200px] font-mono text-sm"
            />
          </div>
        </div>
      )}
    </Card>
  );
};

export default DocumentUploader;
