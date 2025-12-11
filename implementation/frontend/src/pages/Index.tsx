import AppLayout from "@/components/AppLayout";
import ChatWorkspace from "@/components/ChatWorkspace";
import InputBar from "@/components/InputBar";
import RightPanel from "@/components/RightPanel";

const Index = () => {
  return (
    <AppLayout rightPanel={<RightPanel />}>
      <ChatWorkspace />
      <InputBar />
    </AppLayout>
  );
};

export default Index;
