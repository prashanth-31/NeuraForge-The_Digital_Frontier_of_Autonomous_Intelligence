import Navbar from "@/components/Navbar";
import Sidebar from "@/components/Sidebar";
import ChatWorkspace from "@/components/ChatWorkspace";
import HistoryPanel from "@/components/HistoryPanel";
import InputBar from "@/components/InputBar";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <Sidebar />
      
      <main className="fixed top-16 left-56 right-72 bottom-0 overflow-hidden">
        <ChatWorkspace />
      </main>
      
      <HistoryPanel />
      <InputBar />
    </div>
  );
};

export default Index;
