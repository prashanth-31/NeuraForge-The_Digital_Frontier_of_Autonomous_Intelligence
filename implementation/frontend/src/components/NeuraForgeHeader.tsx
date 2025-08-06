import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Upload, Plus, Brain, Map, Settings, Search, Info, LogOut } from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";
import { Input } from "./ui/input";
import { useNavigate, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import "@/components/animations.css";
import { ReactNode } from "react";

interface NeuraForgeHeaderProps {
  children?: ReactNode;
}

export const NeuraForgeHeader = ({ children }: NeuraForgeHeaderProps) => {
  const navigate = useNavigate();
  const location = useLocation();
  const isProjectDetails = location.pathname === "/project-details";
  
  return (
    <header className="h-16 flex items-center justify-between px-6 border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
      <motion.div 
        className="flex items-center gap-6"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: "easeOut" }}
      >
        <div className="flex items-center gap-3">
          <motion.div 
            className="w-8 h-8 rounded-lg bg-gradient-to-br from-neura-blue to-neura-glow flex items-center justify-center"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Brain className="w-5 h-5 text-white dark:text-neura-dark" />
          </motion.div>
          <h1 className="text-xl font-bold animated-gradient-text">
            NeuraForge
          </h1>
        </div>
        
        <div className="flex items-center gap-3 md:flex hidden">
          {!isProjectDetails ? (
            <>
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Button variant="outline" size="sm" className="gap-2 card-hover">
                  <Plus className="w-4 h-4" />
                  New Task
                </Button>
              </motion.div>
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Button variant="outline" size="sm" className="gap-2 card-hover">
                  <Upload className="w-4 h-4" />
                  Upload File
                </Button>
              </motion.div>
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Button variant="outline" size="sm" className="gap-2 card-hover">
                  <Brain className="w-4 h-4" />
                  Memory View
                </Button>
              </motion.div>
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="gap-2 card-hover"
                  onClick={() => navigate("/project-details")}
                >
                  <Info className="w-4 h-4" />
                  Project Details
                </Button>
              </motion.div>
            </>
          ) : (
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button 
                variant="outline" 
                size="sm" 
                className="gap-2 card-hover"
                onClick={() => navigate("/dashboard")}
              >
                <Map className="w-4 h-4" />
                Dashboard
              </Button>
            </motion.div>
          )}
        </div>
      </motion.div>
      
      <motion.div 
        className="flex items-center gap-4"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.1, ease: "easeOut" }}
      >
        {!isProjectDetails && (
          <div className="relative max-w-sm hidden md:block">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              type="search"
              placeholder="Search conversations..."
              className="pl-8 w-[250px] bg-background/50"
            />
          </div>
        )}
        
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3, delay: 0.2 }}
        >
          <Badge variant="secondary" className="gap-2 bg-neura-blue/10 text-neura-blue border-neura-blue/20 hidden md:flex">
            <div className="w-2 h-2 rounded-full bg-neura-blue animate-pulse" />
            Connected to LLaMA 3.1 (via Ollama)
          </Badge>
        </motion.div>
        
        {children}
        
        <ThemeToggle />
        
        <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}>
          <Button 
            variant="ghost" 
            size="icon" 
            className="rounded-full"
            onClick={() => navigate("/login")}
          >
            <LogOut className="h-[1.2rem] w-[1.2rem]" />
            <span className="sr-only">Log Out</span>
          </Button>
        </motion.div>
      </motion.div>
    </header>
  );
};