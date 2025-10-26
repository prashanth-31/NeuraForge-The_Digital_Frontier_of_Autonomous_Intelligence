import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";

import ProtectedRoute from "@/components/ProtectedRoute";
import { AuthProvider } from "@/contexts/AuthContext";
import { TaskProvider } from "@/contexts/TaskContext";
import { ThemeProvider } from "@/contexts/ThemeContext";
import Auth from "./pages/Auth";
import Dashboard from "./pages/Dashboard";
import Index from "./pages/Index";
import Knowledge from "./pages/Knowledge";
import NotFound from "./pages/NotFound";
import Settings from "./pages/Settings";
import Reviews from "./pages/Reviews";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <AuthProvider>
          <ThemeProvider>
            <TaskProvider>
              <Routes>
                <Route path="/auth" element={<Auth />} />
                <Route
                  path="/"
                  element={(
                    <ProtectedRoute>
                      <Index />
                    </ProtectedRoute>
                  )}
                />
                <Route
                  path="/dashboard"
                  element={(
                    <ProtectedRoute>
                      <Dashboard />
                    </ProtectedRoute>
                  )}
                />
                <Route
                  path="/knowledge"
                  element={(
                    <ProtectedRoute>
                      <Knowledge />
                    </ProtectedRoute>
                  )}
                />
                <Route
                  path="/reviews"
                  element={(
                    <ProtectedRoute>
                      <Reviews />
                    </ProtectedRoute>
                  )}
                />
                <Route
                  path="/settings"
                  element={(
                    <ProtectedRoute>
                      <Settings />
                    </ProtectedRoute>
                  )}
                />
                {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
                <Route path="*" element={<NotFound />} />
              </Routes>
            </TaskProvider>
          </ThemeProvider>
        </AuthProvider>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
