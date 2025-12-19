import { useState } from "react";
import AppLayout from "@/components/AppLayout";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Search, FileText, Brain, BookOpen, Settings2 } from "lucide-react";
import DocumentUploader from "@/components/knowledge/DocumentUploader";

const quickGuides = [
  {
    title: "Document Upload",
    description: "Upload PDFs, Word docs, or text files to be analyzed and summarized by AI agents. Persisted documents can be referenced in future conversations.",
    icon: FileText,
    color: "text-primary-600",
    bg: "bg-primary-50",
  },
  {
    title: "Agent Memory",
    description: "When you persist a document, agents can recall its contents using the Task ID. Reference documents in your prompts for contextual conversations.",
    icon: Brain,
    color: "text-violet-600",
    bg: "bg-violet-50",
  },
  {
    title: "Supported Formats",
    description: "PDF, DOCX, TXT, CSV, JSON, and Markdown files up to 8MB. Large documents are automatically chunked for efficient retrieval.",
    icon: BookOpen,
    color: "text-amber-600",
    bg: "bg-amber-50",
  },
  {
    title: "Best Practices",
    description: "Enable 'Persist analysis' for documents you'll reference often. Copy the Task ID or Document ID to thread context into agent workflows.",
    icon: Settings2,
    color: "text-emerald-600",
    bg: "bg-emerald-50",
  },
];

const Knowledge = () => {
  const [searchQuery, setSearchQuery] = useState("");

  const filteredGuides = quickGuides.filter(
    (guide) =>
      searchQuery === "" ||
      guide.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      guide.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <AppLayout>
      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-5xl mx-auto space-y-8">
          <div className="space-y-2">
            <h1 className="text-3xl font-bold text-foreground">Knowledge Base</h1>
            <p className="text-muted-foreground">
              Upload documents to analyze with AI and persist for agent memory. Reference uploaded content in your conversations.
            </p>
          </div>

          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search guides and documentation..."
              className="pl-10"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          <DocumentUploader />

          <div className="space-y-4">
            <h2 className="text-lg font-semibold text-foreground">Quick Guide</h2>
            <div className="grid gap-4 md:grid-cols-2">
              {filteredGuides.map((guide) => (
                <Card key={guide.title} className="p-5 hover:shadow-md transition-shadow duration-200">
                  <div className="flex items-start gap-4">
                    <div className={`w-10 h-10 rounded-lg ${guide.bg} flex items-center justify-center flex-shrink-0`}>
                      <guide.icon className={`h-5 w-5 ${guide.color}`} />
                    </div>
                    <div className="space-y-1">
                      <h3 className="text-base font-semibold text-foreground">{guide.title}</h3>
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        {guide.description}
                      </p>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          </div>
        </div>
      </div>
    </AppLayout>
  );
};

export default Knowledge;
