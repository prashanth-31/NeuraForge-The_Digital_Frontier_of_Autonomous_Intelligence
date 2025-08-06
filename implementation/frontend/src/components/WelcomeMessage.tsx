import { motion } from "framer-motion";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Brain, Sparkles, Code, FileText, Database } from "lucide-react";

interface WelcomeMessageProps {
  onExampleClick: (example: string) => void;
}

export const WelcomeMessage = ({ onExampleClick }: WelcomeMessageProps) => {
  const examples = [
    "Analyze the current market trends for renewable energy investments",
    "Help me understand the main concepts of machine learning",
    "Create a marketing strategy for a new sustainable product",
    "Explain the benefits of microservices architecture",
    "Summarize the key features of NeuraForge's architecture"
  ];

  const capabilities = [
    {
      icon: <Brain className="w-5 h-5 text-neura-blue" />,
      title: "Multi-Agent System",
      description: "Specialized agents collaborate to solve complex problems"
    },
    {
      icon: <Database className="w-5 h-5 text-green-500" />,
      title: "Advanced Memory",
      description: "Remembers context across conversations"
    },
    {
      icon: <FileText className="w-5 h-5 text-purple-500" />,
      title: "Knowledge Base",
      description: "Retrieval augmented generation for accurate information"
    },
    {
      icon: <Code className="w-5 h-5 text-orange-500" />,
      title: "Code Understanding",
      description: "Analyze and explain code across various languages"
    }
  ];

  return (
    <div className="flex flex-col items-center justify-center h-full p-6 max-w-4xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center mb-8"
      >
        <div className="flex justify-center mb-4">
          <div className="h-16 w-16 rounded-2xl bg-gradient-to-br from-neura-blue to-neura-glow flex items-center justify-center">
            <Sparkles className="h-8 w-8 text-white" />
          </div>
        </div>
        <h1 className="text-2xl font-bold mb-2 text-foreground">Welcome to NeuraForge</h1>
        <p className="text-muted-foreground max-w-lg mx-auto">
          The intelligent multi-agent system designed to handle complex tasks through specialized agent collaboration.
        </p>
      </motion.div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full mb-8">
        {capabilities.map((capability, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 + index * 0.1 }}
          >
            <Card className="p-4 h-full border-border/50 bg-background/50 backdrop-blur-sm hover:shadow-md transition-shadow">
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 bg-accent/50">
                  {capability.icon}
                </div>
                <div>
                  <h3 className="font-medium">{capability.title}</h3>
                  <p className="text-sm text-muted-foreground">{capability.description}</p>
                </div>
              </div>
            </Card>
          </motion.div>
        ))}
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.5 }}
        className="w-full"
      >
        <h2 className="text-lg font-medium mb-4 text-center">Try an example</h2>
        <div className="flex flex-wrap gap-2 justify-center">
          {examples.map((example, index) => (
            <Button
              key={index}
              variant="outline"
              size="sm"
              onClick={() => onExampleClick(example)}
              className="text-xs h-8 transition-all hover:bg-accent/80 hover:text-accent-foreground hover:border-neura-blue/40"
            >
              {example}
            </Button>
          ))}
        </div>
      </motion.div>
    </div>
  );
};
