'use client';

import { useEffect, useState } from 'react';
import ChatInterface from '../components/ChatInterface';

export default function Home() {
  const [mounted, setMounted] = useState(false);

  // Set mounted to true on client-side
  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <main className="flex flex-col h-screen">
      <header className="bg-gradient-to-r from-neural-dark via-neural-main to-forge-main text-white py-4 px-6 shadow-md relative overflow-hidden">
        {/* Background pattern overlay */}
        <div className="absolute inset-0 opacity-10 bg-neural-pattern"></div>
        
        {/* Animated glow effects */}
        <div className="absolute -top-20 -left-20 w-60 h-60 bg-neural-accent rounded-full filter blur-3xl opacity-20 pulse-animation"></div>
        <div className="absolute -bottom-20 -right-20 w-60 h-60 bg-forge-accent rounded-full filter blur-3xl opacity-20 pulse-animation"></div>
        
        <div className="max-w-6xl mx-auto flex justify-between items-center relative z-10">
          <div className="flex items-center space-x-4">
            <div className="relative">
              {/* Neural network logo */}
              <div className="relative h-12 w-12 rounded-xl bg-white/20 flex items-center justify-center backdrop-blur-sm p-2 shadow-glow">
                <svg 
                  className="h-10 w-10 text-white" 
                  viewBox="0 0 24 24" 
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path 
                    d="M12 3C7.03 3 3 7.03 3 12C3 16.97 7.03 21 12 21C16.97 21 21 16.97 21 12C21 7.03 16.97 3 12 3ZM12 19C8.13 19 5 15.87 5 12C5 8.13 8.13 5 12 5C15.87 5 19 8.13 19 12C19 15.87 15.87 19 12 19Z" 
                    fill="currentColor"
                  />
                  <path 
                    d="M12 8C9.79 8 8 9.79 8 12C8 14.21 9.79 16 12 16C14.21 16 16 14.21 16 12C16 9.79 14.21 8 12 8ZM18 12C18 15.31 15.31 18 12 18C8.69 18 6 15.31 6 12C6 8.69 8.69 6 12 6C15.31 6 18 8.69 18 12Z" 
                    fill="currentColor" 
                    className="opacity-70"
                  />
                  <path 
                    d="M12 10C10.9 10 10 10.9 10 12C10 13.1 10.9 14 12 14C13.1 14 14 13.1 14 12C14 10.9 13.1 10 12 10Z" 
                    fill="currentColor" 
                  />
                </svg>
                <div className="absolute -top-1 -right-1 w-4 h-4 bg-forge-accent rounded-full shadow-lg shadow-forge-accent/30 animate-pulse"></div>
              </div>
            </div>
            
            <div>
              <h1 className="text-3xl font-display font-bold tracking-tight">
                <span className="gradient-text">NeuraForge</span>
                <span className="absolute -mt-1 ml-1 text-xs bg-neural-dark px-1.5 py-0.5 rounded font-mono">BETA</span>
              </h1>
              <div className="text-sm font-medium text-white/80">Advanced AI-Powered Intelligence Platform</div>
            </div>
          </div>
          
          <div className="hidden md:flex items-center space-x-4">
            <div className="glassmorphism px-3 py-1.5 rounded-lg text-sm flex items-center">
              <span className="inline-block w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse"></span>
              <span className="font-medium">Ollama Connected</span>
              <span className="ml-2 text-xs bg-white/20 px-1.5 py-0.5 rounded">LLaMA 3.1:8b</span>
            </div>
            
            <div className="flex space-x-2">
              <button className="bg-white/10 hover:bg-white/20 text-white px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200 flex items-center">
                <svg className="w-4 h-4 mr-1.5" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                  <path fillRule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd"></path>
                </svg>
                Settings
              </button>
              
              <button className="bg-white/10 hover:bg-white/20 text-white px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200">
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd"></path>
                </svg>
              </button>
            </div>
          </div>
        </div>
      </header>
      
      {/* Animated nodes and connections - purely decorative */}
      <div className="absolute left-0 top-0 w-full h-full pointer-events-none overflow-hidden opacity-10 z-0">
        <div className="absolute left-[10%] top-[20%] w-2 h-2 bg-neural-main rounded-full"></div>
        <div className="absolute left-[30%] top-[10%] w-2 h-2 bg-forge-main rounded-full"></div>
        <div className="absolute left-[60%] top-[15%] w-2 h-2 bg-neural-accent rounded-full"></div>
        <div className="absolute left-[80%] top-[25%] w-2 h-2 bg-forge-accent rounded-full"></div>
        <div className="absolute left-[20%] top-[50%] w-2 h-2 bg-neural-main rounded-full"></div>
        <div className="absolute left-[40%] top-[60%] w-2 h-2 bg-forge-main rounded-full"></div>
        <div className="absolute left-[70%] top-[50%] w-2 h-2 bg-neural-accent rounded-full"></div>
        <div className="absolute left-[90%] top-[70%] w-2 h-2 bg-forge-accent rounded-full"></div>
        <div className="absolute left-[15%] top-[80%] w-2 h-2 bg-neural-main rounded-full"></div>
        <div className="absolute left-[50%] top-[90%] w-2 h-2 bg-forge-main rounded-full"></div>
      </div>
      
      <div className="flex-1 overflow-hidden bg-gradient-to-b from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 relative">
        {/* Subtle background patterns */}
        <div className="absolute inset-0 neural-pattern opacity-[0.03] pointer-events-none"></div>
        <div className="absolute inset-0 neural-bg pointer-events-none"></div>
        
        <ChatInterface />
      </div>
      
      <footer className="py-4 px-6 text-center text-sm text-gray-600 dark:text-gray-400 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-800 backdrop-blur-sm bg-opacity-80 dark:bg-opacity-80 relative z-10">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-center gap-3">
          <div className="flex items-center space-x-3">
            <div className="flex items-center glassmorphism p-1 px-2 rounded-md shadow-sm">
              <svg 
                className="h-4 w-4 text-neural-main mr-1" 
                viewBox="0 0 24 24" 
                fill="currentColor"
              >
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 16h-2v-6h2v6zm4 0h-2v-6h2v6z" />
              </svg>
              <span className="text-neural-dark dark:text-neural-light font-medium">NeuraForge v0.1.0</span>
            </div>
            
            {mounted && (
              <div className="text-xs bg-gray-100 dark:bg-gray-800 px-2 py-0.5 rounded font-mono">
                {new Date().toLocaleDateString()}
              </div>
            )}
          </div>
          
          <div className="flex flex-wrap justify-center gap-4 text-xs">
            <div className="flex items-center space-x-1">
              <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                <path fillRule="evenodd" d="M12.316 3.051a1 1 0 01.633 1.265l-4 12a1 1 0 11-1.898-.632l4-12a1 1 0 011.265-.633zM5.707 6.293a1 1 0 010 1.414L3.414 10l2.293 2.293a1 1 0 11-1.414 1.414l-3-3a1 1 0 010-1.414l3-3a1 1 0 011.414 0zm8.586 0a1 1 0 011.414 0l3 3a1 1 0 010 1.414l-3 3a1 1 0 11-1.414-1.414L16.586 10l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd"></path>
              </svg>
              <span>Python + FastAPI</span>
            </div>
            <div className="flex items-center space-x-1">
              <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                <path fillRule="evenodd" d="M12.316 3.051a1 1 0 01.633 1.265l-4 12a1 1 0 11-1.898-.632l4-12a1 1 0 011.265-.633zM5.707 6.293a1 1 0 010 1.414L3.414 10l2.293 2.293a1 1 0 11-1.414 1.414l-3-3a1 1 0 010-1.414l3-3a1 1 0 011.414 0zm8.586 0a1 1 0 011.414 0l3 3a1 1 0 010 1.414l-3 3a1 1 0 11-1.414-1.414L16.586 10l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd"></path>
              </svg>
              <span>Next.js + React</span>
            </div>
            <div className="flex items-center space-x-1">
              <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                <path d="M9 6a3 3 0 11-6 0 3 3 0 016 0zM17 6a3 3 0 11-6 0 3 3 0 016 0zM12.93 17c.046-.327.07-.66.07-1a6.97 6.97 0 00-1.5-4.33A5 5 0 0119 16v1h-6.07zM6 11a5 5 0 015 5v1H1v-1a5 5 0 015-5z"></path>
              </svg>
              <span>LLaMA 3.1:8b</span>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <a href="#" className="text-neural-main hover:text-neural-dark transition-colors font-medium">Docs</a>
            <a href="#" className="text-neural-main hover:text-neural-dark transition-colors font-medium">GitHub</a>
            <span>© {mounted ? new Date().getFullYear() : 2023} NeuraForge</span>
          </div>
        </div>
      </footer>
    </main>
  );
}
