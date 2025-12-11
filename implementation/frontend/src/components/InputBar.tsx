import { Send, Paperclip, Settings2, Loader2, X } from "lucide-react";
import { useRef, useState } from "react";

import { useTaskContext } from "@/contexts/TaskContext";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";
import { API_BASE_URL } from "@/lib/api";
import { DocumentAnalysisResponse } from "@/types/documents";
import { useToast } from "@/hooks/use-toast";

type AttachmentStatus = "uploading" | "ready" | "error";

interface AttachmentRecord {
  id: string;
  fileName: string;
  status: AttachmentStatus;
  memoryTaskId?: string | null;
  documentId?: string | null;
  persisted?: boolean;
  document?: DocumentAnalysisResponse["document"];
  preview?: string | null;
  truncated?: boolean;
  ingestion?: DocumentAnalysisResponse["ingestion"];
  error?: string;
}

const randomId = () => (typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2));

const InputBar = () => {
  const [message, setMessage] = useState("");
  const [attachments, setAttachments] = useState<AttachmentRecord[]>([]);
  const { submitTask, isStreaming } = useTaskContext();
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const isUploadingAttachment = attachments.some((attachment) => attachment.status === "uploading");
  const readyAttachments = attachments.filter((attachment) => attachment.status === "ready");

  const removeAttachment = (id: string) => {
    setAttachments((previous) => previous.filter((attachment) => attachment.id !== id));
  };

  const handleAttachmentUpload = async (file: File) => {
    const attachmentId = randomId();
    setAttachments((previous) => [
      ...previous,
      {
        id: attachmentId,
        fileName: file.name,
        status: "uploading",
      },
    ]);

    const formData = new FormData();
    formData.append("document", file);

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/upload_document?persist=true`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let detail = "Upload failed.";
        try {
          const payload = await response.json();
          detail = payload?.detail ?? detail;
        } catch (error) {
          // ignore parse errors
        }
        throw new Error(detail);
      }

      const payload: DocumentAnalysisResponse = await response.json();
      setAttachments((previous) =>
        previous.map((attachment) =>
          attachment.id === attachmentId
            ? {
                ...attachment,
                status: "ready",
                persisted: payload.persisted,
                memoryTaskId: payload.memory_task_id,
                documentId: payload.ingestion?.document_id ?? null,
                document: payload.document,
                preview: payload.preview,
                truncated: payload.truncated,
                ingestion: payload.ingestion,
              }
            : attachment,
        ),
      );
      toast({
        title: "Document attached",
        description: payload.memory_task_id
          ? `${file.name} analyzed and stored as ${payload.memory_task_id}.`
          : `${file.name} analyzed without persistence.`,
      });
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Unexpected error while uploading.";
      setAttachments((previous) =>
        previous.map((attachment) =>
          attachment.id === attachmentId
            ? {
                ...attachment,
                status: "error",
                error: detail,
              }
            : attachment,
        ),
      );
      toast({
        title: "Failed to attach file",
        description: detail,
        variant: "destructive",
      });
    }
  };

  const handleFileSelection = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files ? Array.from(event.target.files) : [];
    event.target.value = "";
    for (const file of files) {
      void handleAttachmentUpload(file);
    }
  };

  const handleSubmit = async () => {
    if (!message.trim()) {
      return;
    }
    if (isUploadingAttachment) {
      toast({
        title: "Documents still processing",
        description: "Wait for uploads to finish before sending your prompt.",
      });
      return;
    }

    const metadata: Record<string, unknown> = {};
    if (readyAttachments.length > 0) {
      const attachmentPayload = readyAttachments.map((attachment) => ({
        id: attachment.memoryTaskId ?? attachment.id,
        filename: attachment.fileName,
        memory_task_id: attachment.memoryTaskId,
        document_id: attachment.documentId,
        persisted: attachment.persisted ?? false,
        truncated: attachment.truncated ?? false,
        preview: attachment.preview,
        document: attachment.document,
        ingestion: attachment.ingestion,
      }));
      metadata.attachments = attachmentPayload;
      const documentIds = attachmentPayload
        .map((item) => item.document_id ?? item.memory_task_id)
        .filter((value): value is string => typeof value === "string" && value.length > 0);
      if (documentIds.length > 0) {
        metadata.documents = documentIds;
      }
    }

    await submitTask(message, Object.keys(metadata).length > 0 ? metadata : undefined);
    setMessage("");
    setAttachments([]);
  };

  return (
    <div className="border-t border-slate-200/60 bg-white/95 backdrop-blur-sm shadow-elevated sticky bottom-0">
      <div className="p-5">
        <div className="max-w-4xl mx-auto">
          <div className="relative flex items-end gap-3">
            <div className="flex-1 relative">
              <Textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Ask NeuraForge anything..."
                className="min-h-[56px] max-h-36 resize-none pr-24 rounded-2xl border-slate-200 bg-slate-50/50 focus:bg-white focus:border-primary-400 focus:shadow-glow transition-all duration-200"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void handleSubmit();
                  }
                }}
                disabled={isStreaming}
              />
              <div className="absolute right-3 bottom-3 flex items-center gap-1.5">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,.docx,.txt,.csv,.json,.md,.markdown"
                  multiple
                  className="hidden"
                  onChange={handleFileSelection}
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 rounded-lg text-slate-400 hover:text-primary-600 hover:bg-primary-50"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isStreaming}
                >
                  {isUploadingAttachment ? <Loader2 className="h-4 w-4 animate-spin text-primary-500" /> : <Paperclip className="h-4 w-4" />}
                </Button>
                <Button 
                  type="button" 
                  variant="ghost" 
                  size="icon" 
                  className="h-8 w-8 rounded-lg text-slate-400 hover:text-primary-600 hover:bg-primary-50" 
                  disabled
                >
                  <Settings2 className="h-4 w-4" />
                </Button>
              </div>
            </div>
            <Button
              size="icon"
              className="h-14 w-14 rounded-2xl bg-gradient-to-br from-primary-500 to-primary-600 hover:from-primary-600 hover:to-primary-700 shadow-lg shadow-primary-500/25 transition-all duration-200 hover:scale-105 hover:shadow-xl hover:shadow-primary-500/30"
              onClick={() => void handleSubmit()}
              disabled={isStreaming || isUploadingAttachment}
            >
              <Send className="h-5 w-5" />
            </Button>
          </div>
          {attachments.length > 0 && (
            <div className="mt-4 flex flex-wrap gap-2">
              {attachments.map((attachment) => (
                <div
                  key={attachment.id}
                  className="flex items-center gap-2.5 rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-xs text-foreground shadow-sm transition-all duration-200 hover:shadow-md"
                >
                  <span className="font-medium truncate max-w-[180px]" title={attachment.fileName}>
                    {attachment.fileName}
                  </span>
                  {attachment.status === "uploading" && <Loader2 className="h-3.5 w-3.5 animate-spin text-primary-500" />}
                  {attachment.status === "ready" && (
                    <span className="text-emerald-600 font-semibold text-[10px] uppercase tracking-wide bg-emerald-50 px-1.5 py-0.5 rounded">Ready</span>
                  )}
                  {attachment.status === "error" && (
                    <span className="text-rose-600 font-semibold text-[10px] uppercase tracking-wide bg-rose-50 px-1.5 py-0.5 rounded" title={attachment.error}>
                      Failed
                    </span>
                  )}
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="h-5 w-5 rounded-md text-slate-400 hover:text-slate-600 hover:bg-slate-100"
                    onClick={() => removeAttachment(attachment.id)}
                    disabled={attachment.status === "uploading"}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </div>
              ))}
            </div>
          )}
          <div className="mt-3 flex items-center justify-between text-xs text-slate-400">
            <span className="flex items-center gap-1.5">
              <kbd className="px-1.5 py-0.5 bg-slate-100 rounded text-[10px] font-mono">Shift</kbd>
              <span>+</span>
              <kbd className="px-1.5 py-0.5 bg-slate-100 rounded text-[10px] font-mono">↵</kbd>
              <span className="ml-1">new line</span>
              <span className="mx-2">•</span>
              <kbd className="px-1.5 py-0.5 bg-slate-100 rounded text-[10px] font-mono">↵</kbd>
              <span className="ml-1">send</span>
            </span>
            <span className={isStreaming ? "text-primary-500 font-medium" : ""}>
              {isStreaming ? "⚡ Agents responding..." : "Collaborative Mode"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default InputBar;
