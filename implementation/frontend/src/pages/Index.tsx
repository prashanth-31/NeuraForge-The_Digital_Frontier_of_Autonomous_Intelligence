import AppLayout from "@/components/AppLayout";
import ChatWorkspace from "@/components/ChatWorkspace";
import InputBar from "@/components/InputBar";

const Index = () => {
  return (
    <AppLayout showHistory>
      <ChatWorkspace />
      <InputBar />
    </AppLayout>
  );
};

export default Index;
