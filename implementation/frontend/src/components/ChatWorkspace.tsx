import { ScrollArea } from "./ui/scroll-area";
import MessageCard from "./MessageCard";

const sampleMessages = [
  {
    id: 1,
    role: "user" as const,
    content: "How does multi-agent collaboration improve reasoning accuracy?",
    timestamp: "10:32 AM",
  },
  {
    id: 2,
    role: "assistant" as const,
    content: "Multi-agent collaboration enhances reasoning accuracy through several key mechanisms:\n\n1. **Diverse Perspectives**: Different agents bring unique reasoning approaches, reducing bias and blind spots.\n\n2. **Verification & Validation**: Agents cross-check each other's outputs, identifying inconsistencies early.\n\n3. **Specialized Expertise**: Each agent focuses on specific domains, allowing deeper analysis in their area of strength.\n\n4. **Iterative Refinement**: Collaborative rounds enable progressive improvement of hypotheses and conclusions.\n\nThis approach is particularly effective when dealing with complex, multi-faceted problems requiring both breadth and depth of analysis.",
    timestamp: "10:32 AM",
  },
  {
    id: 3,
    role: "user" as const,
    content: "Can you provide a concrete example?",
    timestamp: "10:34 AM",
  },
  {
    id: 4,
    role: "assistant" as const,
    content: "Consider a medical diagnosis scenario:\n\n**Agent A (Symptom Analyzer)** identifies patterns in patient symptoms.\n**Agent B (Literature Retrieval)** searches medical databases for similar cases.\n**Agent C (Reasoning Engine)** synthesizes findings and proposes differential diagnoses.\n**Agent D (Validator)** checks for logical consistency and evidence gaps.\n\nEach agent contributes specialized knowledge, while the orchestrator ensures coherent collaboration. The result is a more comprehensive, reliable diagnosis than any single agent could provide.\n\nThis demonstrates how task decomposition and collaborative synthesis create emergent intelligence.",
    timestamp: "10:34 AM",
  },
];

const ChatWorkspace = () => {
  return (
    <div className="flex-1 flex flex-col h-full">
      <div className="border-b border-border p-4 bg-card shadow-soft">
        <h2 className="text-sm font-semibold text-foreground">Current Session</h2>
        <p className="text-xs text-muted-foreground mt-1">Collaborative Intelligence Mode</p>
      </div>
      
      <ScrollArea className="flex-1 p-6">
        <div className="max-w-4xl mx-auto space-y-4">
          {sampleMessages.map((message) => (
            <MessageCard
              key={message.id}
              role={message.role}
              content={message.content}
              timestamp={message.timestamp}
            />
          ))}
        </div>
      </ScrollArea>
    </div>
  );
};

export default ChatWorkspace;
