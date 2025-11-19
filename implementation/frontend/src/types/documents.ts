export interface DocumentMetadata {
  filename: string;
  content_type: string | null;
  extension: string | null;
  line_count: number;
  character_count: number;
  filesize_bytes: number;
}

export interface DocumentAnalysisResponse {
  output: string;
  document: DocumentMetadata;
  truncated: boolean;
  persisted: boolean;
  memory_task_id: string | null;
  preview: string | null;
}
