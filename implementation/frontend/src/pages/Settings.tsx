import AppLayout from "@/components/AppLayout";
import { Card } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";

const settings = [
  {
    title: "Notifications",
    options: [
      { id: "alerts", label: "Alert me when tasks finish", enabled: true },
      { id: "failures", label: "Notify on agent failures", enabled: true },
      { id: "weekly", label: "Send weekly insight summary", enabled: false },
    ],
  },
  {
    title: "Interface",
    options: [
      { id: "animations", label: "Enable workspace animations", enabled: true },
      { id: "compact", label: "Compact message layout", enabled: false },
    ],
  },
];

const Settings = () => {
  return (
    <AppLayout>
      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-4xl mx-auto space-y-8">
          <div>
            <h1 className="text-3xl font-bold text-foreground mb-2">Settings</h1>
            <p className="text-muted-foreground">Configure how NeuraForge collaborates with your team.</p>
          </div>

          {settings.map((section) => (
            <Card key={section.title} className="p-6 space-y-4">
              <div>
                <h2 className="text-xl font-semibold text-foreground">{section.title}</h2>
                <p className="text-sm text-muted-foreground">Fine-tune preferences for your workspace.</p>
              </div>
              <Separator className="bg-border/60" />
              <div className="space-y-4">
                {section.options.map((option) => (
                  <div key={option.id} className="flex items-center justify-between">
                    <div>
                      <Label htmlFor={option.id} className="text-sm font-medium">{option.label}</Label>
                    </div>
                    <Switch id={option.id} checked={option.enabled} aria-readonly className="pointer-events-none opacity-60" />
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </div>
      </div>
    </AppLayout>
  );
};

export default Settings;
