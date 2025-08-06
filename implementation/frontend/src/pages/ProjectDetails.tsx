import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { NeuraForgeHeader } from "@/components/NeuraForgeHeader";
import { Search, Palette, TrendingUp, Building, Code, Users, Database, Scale, Zap } from "lucide-react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

export default function ProjectDetails() {
  const [activeTab, setActiveTab] = useState("agents");
  
  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Grid pattern background */}
      <div 
        className="fixed inset-0 opacity-20"
        style={{
          backgroundImage: 'radial-gradient(circle at 1px 1px, hsl(var(--neura-blue)) 1px, transparent 0)',
          backgroundSize: '20px 20px'
        }}
      />
      
      <div className="relative z-10 flex flex-col h-screen">
        <NeuraForgeHeader />
        
        <div className="flex-1 container mx-auto py-6 overflow-hidden">
          <div className="mb-6">
            <h1 className="text-2xl font-bold mb-2">Project Overview</h1>
            <p className="text-muted-foreground">
              Detailed information about your NeuraForge multi-agent system
            </p>
          </div>
          
          <Tabs value={activeTab} onValueChange={setActiveTab} className="h-[calc(100vh-12rem)]">
            <TabsList className="grid w-full grid-cols-2 mb-6">
              <TabsTrigger value="agents">Agent System</TabsTrigger>
              <TabsTrigger value="developers">Developers</TabsTrigger>
            </TabsList>
            
            <TabsContent value="agents" className="h-full overflow-auto p-1">
              <div className="grid md:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <div className="flex items-center gap-3 mb-2">
                      <div className="w-8 h-8 rounded-lg bg-agent-research/10 border border-agent-research/20 flex items-center justify-center">
                        <Search className="w-4 h-4 text-agent-research" />
                      </div>
                      <CardTitle>Research Agent</CardTitle>
                    </div>
                    <CardDescription>
                      Knowledge retrieval and information analysis
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Accuracy Rating</span>
                          <span className="font-medium">94%</span>
                        </div>
                        <Progress value={94} className="h-2" />
                      </div>
                      
                      <div className="space-y-1">
                        <span className="text-sm font-medium">Capabilities:</span>
                        <div className="flex flex-wrap gap-2">
                          <Badge variant="outline" className="text-xs">Web Retrieval</Badge>
                          <Badge variant="outline" className="text-xs">Academic Search</Badge>
                          <Badge variant="outline" className="text-xs">Data Validation</Badge>
                          <Badge variant="outline" className="text-xs">Fact Checking</Badge>
                          <Badge variant="outline" className="text-xs">Citation Generation</Badge>
                        </div>
                      </div>
                      
                      <div className="space-y-1">
                        <span className="text-sm font-medium">Technology:</span>
                        <div className="flex flex-wrap gap-2">
                          <Badge className="bg-agent-research/10 text-agent-research border-agent-research/20 text-xs">
                            LLaMA 3.1 70B
                          </Badge>
                          <Badge className="bg-agent-research/10 text-agent-research border-agent-research/20 text-xs">
                            Qdrant Vector DB
                          </Badge>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <div className="flex items-center gap-3 mb-2">
                      <div className="w-8 h-8 rounded-lg bg-agent-creative/10 border border-agent-creative/20 flex items-center justify-center">
                        <Palette className="w-4 h-4 text-agent-creative" />
                      </div>
                      <CardTitle>Creative Agent</CardTitle>
                    </div>
                    <CardDescription>
                      Content generation and creative problem solving
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Creativity Rating</span>
                          <span className="font-medium">91%</span>
                        </div>
                        <Progress value={91} className="h-2" />
                      </div>
                      
                      <div className="space-y-1">
                        <span className="text-sm font-medium">Capabilities:</span>
                        <div className="flex flex-wrap gap-2">
                          <Badge variant="outline" className="text-xs">Content Creation</Badge>
                          <Badge variant="outline" className="text-xs">Ideation</Badge>
                          <Badge variant="outline" className="text-xs">Storytelling</Badge>
                          <Badge variant="outline" className="text-xs">Marketing Copy</Badge>
                          <Badge variant="outline" className="text-xs">Design Concepts</Badge>
                        </div>
                      </div>
                      
                      <div className="space-y-1">
                        <span className="text-sm font-medium">Technology:</span>
                        <div className="flex flex-wrap gap-2">
                          <Badge className="bg-agent-creative/10 text-agent-creative border-agent-creative/20 text-xs">
                            LLaMA 3.1 70B
                          </Badge>
                          <Badge className="bg-agent-creative/10 text-agent-creative border-agent-creative/20 text-xs">
                            Creative Database
                          </Badge>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <div className="flex items-center gap-3 mb-2">
                      <div className="w-8 h-8 rounded-lg bg-agent-finance/10 border border-agent-finance/20 flex items-center justify-center">
                        <TrendingUp className="w-4 h-4 text-agent-finance" />
                      </div>
                      <CardTitle>Finance Agent</CardTitle>
                    </div>
                    <CardDescription>
                      Financial analysis and market insights
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Accuracy Rating</span>
                          <span className="font-medium">89%</span>
                        </div>
                        <Progress value={89} className="h-2" />
                      </div>
                      
                      <div className="space-y-1">
                        <span className="text-sm font-medium">Capabilities:</span>
                        <div className="flex flex-wrap gap-2">
                          <Badge variant="outline" className="text-xs">Market Analysis</Badge>
                          <Badge variant="outline" className="text-xs">Financial Modeling</Badge>
                          <Badge variant="outline" className="text-xs">Risk Assessment</Badge>
                          <Badge variant="outline" className="text-xs">Investment Strategy</Badge>
                          <Badge variant="outline" className="text-xs">Trend Forecasting</Badge>
                        </div>
                      </div>
                      
                      <div className="space-y-1">
                        <span className="text-sm font-medium">Technology:</span>
                        <div className="flex flex-wrap gap-2">
                          <Badge className="bg-agent-finance/10 text-agent-finance border-agent-finance/20 text-xs">
                            LLaMA 3.1 70B
                          </Badge>
                          <Badge className="bg-agent-finance/10 text-agent-finance border-agent-finance/20 text-xs">
                            Financial Database
                          </Badge>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <div className="flex items-center gap-3 mb-2">
                      <div className="w-8 h-8 rounded-lg bg-agent-enterprise/10 border border-agent-enterprise/20 flex items-center justify-center">
                        <Building className="w-4 h-4 text-agent-enterprise" />
                      </div>
                      <CardTitle>Enterprise Agent</CardTitle>
                    </div>
                    <CardDescription>
                      Business strategy and organizational insights
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Strategic Rating</span>
                          <span className="font-medium">87%</span>
                        </div>
                        <Progress value={87} className="h-2" />
                      </div>
                      
                      <div className="space-y-1">
                        <span className="text-sm font-medium">Capabilities:</span>
                        <div className="flex flex-wrap gap-2">
                          <Badge variant="outline" className="text-xs">Strategic Planning</Badge>
                          <Badge variant="outline" className="text-xs">Process Optimization</Badge>
                          <Badge variant="outline" className="text-xs">Organizational Design</Badge>
                          <Badge variant="outline" className="text-xs">Business Intelligence</Badge>
                          <Badge variant="outline" className="text-xs">Competitive Analysis</Badge>
                        </div>
                      </div>
                      
                      <div className="space-y-1">
                        <span className="text-sm font-medium">Technology:</span>
                        <div className="flex flex-wrap gap-2">
                          <Badge className="bg-agent-enterprise/10 text-agent-enterprise border-agent-enterprise/20 text-xs">
                            LLaMA 3.1 70B
                          </Badge>
                          <Badge className="bg-agent-enterprise/10 text-agent-enterprise border-agent-enterprise/20 text-xs">
                            Enterprise Database
                          </Badge>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
              
              <Card className="mt-6">
                <CardHeader>
                  <CardTitle>System Architecture</CardTitle>
                  <CardDescription>
                    How NeuraForge agents work together
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="grid grid-cols-3 gap-4">
                      <div className="space-y-2 border border-border/50 rounded-lg p-3">
                        <div className="flex items-center gap-2">
                          <Database className="w-4 h-4 text-neura-blue" />
                          <span className="font-medium text-sm">Memory System</span>
                        </div>
                        <p className="text-xs text-muted-foreground">
                          Long-term and short-term memory for context preservation across agent interactions
                        </p>
                      </div>
                      
                      <div className="space-y-2 border border-border/50 rounded-lg p-3">
                        <div className="flex items-center gap-2">
                          <Code className="w-4 h-4 text-neura-blue" />
                          <span className="font-medium text-sm">Reasoning Engine</span>
                        </div>
                        <p className="text-xs text-muted-foreground">
                          Chain-of-thought processing for complex problem decomposition and solution synthesis
                        </p>
                      </div>
                      
                      <div className="space-y-2 border border-border/50 rounded-lg p-3">
                        <div className="flex items-center gap-2">
                          <Scale className="w-4 h-4 text-neura-blue" />
                          <span className="font-medium text-sm">Negotiation Protocol</span>
                        </div>
                        <p className="text-xs text-muted-foreground">
                          Allows agents to debate, refine, and reach consensus on complex topics
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex justify-center mt-2">
                      <Badge className="bg-neura-blue/10 text-neura-blue border-neura-blue/20">
                        Powered by LLaMA 3.1 Technology
                      </Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="developers" className="h-full overflow-auto p-1">
              <div className="grid md:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Development Team</CardTitle>
                    <CardDescription>
                      The people behind NeuraForge
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-6">
                      <div className="flex items-center gap-4">
                        <Avatar className="h-12 w-12 border border-border bg-gradient-to-br from-neura-blue/20 to-neura-glow/20">
                          <AvatarImage src="/placeholder.svg" />
                          <AvatarFallback>PS</AvatarFallback>
                        </Avatar>
                        <div>
                          <p className="font-medium">Prashanth S</p>
                          <p className="text-sm text-muted-foreground">Lead AI Architect</p>
                          <div className="flex gap-1 mt-1">
                            <Badge variant="outline" className="text-xs">LLM Systems</Badge>
                            <Badge variant="outline" className="text-xs">Architecture</Badge>
                            <Badge variant="outline" className="text-xs">Team Lead</Badge>
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-4">
                        <Avatar className="h-12 w-12 border border-border bg-gradient-to-br from-neura-blue/20 to-neura-glow/20">
                          <AvatarImage src="/placeholder.svg" />
                          <AvatarFallback>SS</AvatarFallback>
                        </Avatar>
                        <div>
                          <p className="font-medium">Suhaas S</p>
                          <p className="text-sm text-muted-foreground">AI Engineer</p>
                          <div className="flex gap-1 mt-1">
                            <Badge variant="outline" className="text-xs">ML Models</Badge>
                            <Badge variant="outline" className="text-xs">Backend</Badge>
                            <Badge variant="outline" className="text-xs">Vector DBs</Badge>
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-4">
                        <Avatar className="h-12 w-12 border border-border bg-gradient-to-br from-neura-blue/20 to-neura-glow/20">
                          <AvatarImage src="/placeholder.svg" />
                          <AvatarFallback>YN</AvatarFallback>
                        </Avatar>
                        <div>
                          <p className="font-medium">Yashas Nandan S</p>
                          <p className="text-sm text-muted-foreground">AI Engineer</p>
                          <div className="flex gap-1 mt-1">
                            <Badge variant="outline" className="text-xs">LLM Integration</Badge>
                            <Badge variant="outline" className="text-xs">Agent Systems</Badge>
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-4">
                        <Avatar className="h-12 w-12 border border-border bg-gradient-to-br from-neura-blue/20 to-neura-glow/20">
                          <AvatarImage src="/placeholder.svg" />
                          <AvatarFallback>LD</AvatarFallback>
                        </Avatar>
                        <div>
                          <p className="font-medium">Lavanya DL</p>
                          <p className="text-sm text-muted-foreground">AI Engineer</p>
                          <div className="flex gap-1 mt-1">
                            <Badge variant="outline" className="text-xs">Deep Learning</Badge>
                            <Badge variant="outline" className="text-xs">Model Optimization</Badge>
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <CardTitle>Technology Stack</CardTitle>
                    <CardDescription>
                      Tools and frameworks used
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-6">
                      <div>
                        <h3 className="text-sm font-medium mb-2">Frontend</h3>
                        <div className="flex flex-wrap gap-2">
                          <Badge variant="outline">React</Badge>
                          <Badge variant="outline">TypeScript</Badge>
                          <Badge variant="outline">Tailwind CSS</Badge>
                          <Badge variant="outline">Shadcn UI</Badge>
                          <Badge variant="outline">Framer Motion</Badge>
                        </div>
                      </div>
                      
                      <div>
                        <h3 className="text-sm font-medium mb-2">Backend</h3>
                        <div className="flex flex-wrap gap-2">
                          <Badge variant="outline">FastAPI</Badge>
                          <Badge variant="outline">Python</Badge>
                          <Badge variant="outline">Ollama</Badge>
                          <Badge variant="outline">Redis</Badge>
                          <Badge variant="outline">PostgreSQL</Badge>
                        </div>
                      </div>
                      
                      <div>
                        <h3 className="text-sm font-medium mb-2">AI & ML</h3>
                        <div className="flex flex-wrap gap-2">
                          <Badge className="bg-neura-blue/10 text-neura-blue border-neura-blue/20">
                            LLaMA 3.1 Models
                          </Badge>
                          <Badge variant="outline">Qdrant Vector DB</Badge>
                          <Badge variant="outline">LangChain</Badge>
                          <Badge variant="outline">PyTorch</Badge>
                        </div>
                      </div>
                      
                      <div>
                        <h3 className="text-sm font-medium mb-2">DevOps</h3>
                        <div className="flex flex-wrap gap-2">
                          <Badge variant="outline">Docker</Badge>
                          <Badge variant="outline">GitHub Actions</Badge>
                          <Badge variant="outline">Kubernetes</Badge>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                
                <Card className="md:col-span-2">
                  <CardHeader>
                    <CardTitle>Development Roadmap</CardTitle>
                    <CardDescription>
                      Upcoming features and improvements
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex items-center gap-3">
                        <div className={cn(
                          "w-10 h-10 rounded-full flex items-center justify-center border",
                          "bg-green-500/10 border-green-500/20"
                        )}>
                          <Zap className="w-5 h-5 text-green-500" />
                        </div>
                        <div>
                          <p className="font-medium">August 2025 (Current)</p>
                          <p className="text-sm text-muted-foreground">
                            LLaMA 3.1 integration, UI redesign, enhanced memory system
                          </p>
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-3">
                        <div className={cn(
                          "w-10 h-10 rounded-full flex items-center justify-center border",
                          "bg-amber-500/10 border-amber-500/20"
                        )}>
                          <Zap className="w-5 h-5 text-amber-500" />
                        </div>
                        <div>
                          <p className="font-medium">September - October 2025</p>
                          <p className="text-sm text-muted-foreground">
                            Multi-modal agent capabilities, custom agent workflows, enterprise API integrations
                          </p>
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-3">
                        <div className={cn(
                          "w-10 h-10 rounded-full flex items-center justify-center border",
                          "bg-blue-500/10 border-blue-500/20"
                        )}>
                          <Zap className="w-5 h-5 text-blue-500" />
                        </div>
                        <div>
                          <p className="font-medium">November - December 2025</p>
                          <p className="text-sm text-muted-foreground">
                            Final release with advanced security, compliance features, and optimized performance for production
                          </p>
                        </div>
                      </div>
                      
                      <div className="mt-4 p-3 rounded-md bg-neura-blue/5 border border-neura-blue/20">
                        <h4 className="text-sm font-medium text-neura-blue mb-1">Project Completion: December 2025</h4>
                        <p className="text-xs text-muted-foreground">
                          The NeuraForge platform will be fully production-ready with all planned features, optimizations, 
                          and enterprise capabilities by the end of December 2025.
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}
