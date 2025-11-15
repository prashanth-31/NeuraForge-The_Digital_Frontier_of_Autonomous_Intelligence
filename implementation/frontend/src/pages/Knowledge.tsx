import AppLayout from "@/components/AppLayout";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Search } from "lucide-react";
import DocumentUploader from "@/components/knowledge/DocumentUploader";

const Knowledge = () => {
  return (
    <AppLayout>
      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-5xl mx-auto space-y-8">
          <div className="space-y-2">
            <h1 className="text-3xl font-bold text-foreground">Knowledge Base</h1>
            <p className="text-muted-foreground">
              Centralize research, playbooks, and reference materials. Upload a document to prime the agent workspace.
            </p>
          </div>

          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input placeholder="Search documents, tasks, or agents" className="pl-10" />
          </div>

          <DocumentUploader />

          <div className="grid gap-4 md:grid-cols-2">
            {["Agent Playbooks", "Recent Discoveries", "Operational Runbooks", "Design Principles"].map((title) => (
              <Card key={title} className="p-5 hover-lift transition-smooth">
                <h3 className="text-lg font-semibold text-foreground mb-2">{title}</h3>
                <p className="text-sm text-muted-foreground">
                  Coming soon: richer document browsing and semantic retrieval tailored to your workspace.
                </p>
              </Card>
            ))}
          </div>
        </div>
      </div>
    </AppLayout>
  );
};

export default Knowledge;
