import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Brain } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import "@/components/animations.css";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isVisible, setIsVisible] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    // Add a slight delay for a smoother entrance animation
    const timer = setTimeout(() => {
      setIsVisible(true);
    }, 100);
    return () => clearTimeout(timer);
  }, []);

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    
    // Simulate authentication
    setTimeout(() => {
      setIsLoading(false);
      // Redirect to main dashboard after successful login
      navigate("/dashboard");
    }, 1500);
  };

  // Container animation
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { 
      opacity: 1,
      transition: { 
        duration: 0.5,
        when: "beforeChildren",
        staggerChildren: 0.1
      }
    }
  };

  // Child animation
  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.5 }
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      {/* Grid pattern background with animation */}
      <div 
        className="fixed inset-0 animate-fade-in"
        style={{
          backgroundImage: 'radial-gradient(circle at 1px 1px, hsl(var(--neura-blue)) 1px, transparent 0)',
          backgroundSize: '20px 20px',
        }}
      />
      
      <motion.div 
        className="relative z-10 w-full max-w-md px-4"
        variants={containerVariants}
        initial="hidden"
        animate={isVisible ? "visible" : "hidden"}
      >
        <motion.div className="text-center mb-8" variants={itemVariants}>
          <motion.div 
            className="inline-flex items-center justify-center gap-3 mb-4"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <motion.div 
              className="w-12 h-12 rounded-xl bg-gradient-to-br from-neura-blue to-neura-glow flex items-center justify-center"
              animate={{ 
                boxShadow: ["0 0 0px rgba(var(--neura-blue-rgb), 0)", "0 0 20px rgba(var(--neura-blue-rgb), 0.3)", "0 0 0px rgba(var(--neura-blue-rgb), 0)"]
              }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <Brain className="w-7 h-7 text-white dark:text-neura-dark" />
            </motion.div>
            <h1 className="text-3xl font-bold animated-gradient-text">
              NeuraForge
            </h1>
          </motion.div>
          <motion.p 
            className="text-muted-foreground"
            variants={itemVariants}
          >
            Advanced Multi-Agent Interface
          </motion.p>
        </motion.div>
        
        <motion.div variants={itemVariants}>
          <Card className="border-border/50 bg-card/80 backdrop-blur-sm card-hover">
            <CardHeader>
              <CardTitle>Sign In</CardTitle>
              <CardDescription>
                Enter your credentials to access your agents
              </CardDescription>
            </CardHeader>
            
            <form onSubmit={handleLogin}>
              <CardContent className="space-y-4">
                <motion.div 
                  className="space-y-2"
                  variants={itemVariants}
                >
                  <Label htmlFor="email">Email</Label>
                  <Input 
                    id="email" 
                    type="email" 
                    placeholder="name@example.com" 
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    className="transition-all duration-200 focus:ring-2 focus:ring-neura-blue/20"
                  />
                </motion.div>
                
                <motion.div 
                  className="space-y-2"
                  variants={itemVariants}
                >
                  <div className="flex items-center justify-between">
                    <Label htmlFor="password">Password</Label>
                    <Button variant="link" size="sm" className="p-0 h-auto text-xs">
                      Forgot password?
                    </Button>
                  </div>
                  <Input 
                    id="password" 
                    type="password" 
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    className="transition-all duration-200 focus:ring-2 focus:ring-neura-blue/20"
                  />
                </motion.div>
              </CardContent>
              
              <CardFooter>
                <motion.div 
                  className="w-full"
                  variants={itemVariants}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <Button 
                    type="submit" 
                    className="w-full bg-neura-blue hover:bg-neura-blue/90"
                    disabled={isLoading}
                  >
                    {isLoading ? "Signing in..." : "Sign In"}
                  </Button>
                </motion.div>
              </CardFooter>
            </form>
          </Card>
        </motion.div>
        
        <motion.div 
          className="mt-4 text-center text-sm text-muted-foreground"
          variants={itemVariants}
        >
          Don't have an account?{" "}
          <Button variant="link" className="p-0 h-auto" onClick={() => navigate("/dashboard")}>
            Sign up
          </Button>
        </motion.div>
      </motion.div>
    </div>
  );
}
