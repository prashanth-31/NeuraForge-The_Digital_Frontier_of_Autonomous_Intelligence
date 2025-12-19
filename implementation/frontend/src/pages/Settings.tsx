import { useCallback, useEffect, useState } from "react";
import AppLayout from "@/components/AppLayout";
import { Card } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { useToast } from "@/hooks/use-toast";

const STORAGE_KEY = "neuraforge-settings";

type SettingsState = {
  alerts: boolean;
  failures: boolean;
  weekly: boolean;
  animations: boolean;
  compact: boolean;
};

const defaultSettings: SettingsState = {
  alerts: true,
  failures: true,
  weekly: false,
  animations: true,
  compact: false,
};

const loadSettings = (): SettingsState => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      return { ...defaultSettings, ...JSON.parse(stored) };
    }
  } catch {
    // Ignore parse errors
  }
  return defaultSettings;
};

const saveSettings = (settings: SettingsState) => {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // Ignore storage errors
  }
};

const settingsConfig = [
  {
    title: "Notifications",
    description: "Control how NeuraForge alerts you about task progress.",
    options: [
      { id: "alerts" as const, label: "Alert me when tasks finish", description: "Receive notifications when agent tasks complete" },
      { id: "failures" as const, label: "Notify on agent failures", description: "Get alerted when an agent encounters an error" },
      { id: "weekly" as const, label: "Send weekly insight summary", description: "Receive a weekly digest of AI collaboration activity" },
    ],
  },
  {
    title: "Interface",
    description: "Customize the look and feel of your workspace.",
    options: [
      { id: "animations" as const, label: "Enable workspace animations", description: "Show smooth transitions and effects" },
      { id: "compact" as const, label: "Compact message layout", description: "Display messages in a denser format" },
    ],
  },
];

const Settings = () => {
  const { toast } = useToast();
  const [settings, setSettings] = useState<SettingsState>(defaultSettings);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    setSettings(loadSettings());
    setLoaded(true);
  }, []);

  const handleToggle = useCallback((key: keyof SettingsState) => {
    setSettings((prev) => {
      const updated = { ...prev, [key]: !prev[key] };
      saveSettings(updated);
      return updated;
    });
    toast({
      title: "Setting updated",
      description: "Your preference has been saved.",
    });
  }, [toast]);

  return (
    <AppLayout>
      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-4xl mx-auto space-y-8">
          <div>
            <h1 className="text-3xl font-bold text-foreground mb-2">Settings</h1>
            <p className="text-muted-foreground">Configure how NeuraForge collaborates with your team.</p>
          </div>

          {settingsConfig.map((section) => (
            <Card key={section.title} className="p-6 space-y-4">
              <div>
                <h2 className="text-xl font-semibold text-foreground">{section.title}</h2>
                <p className="text-sm text-muted-foreground">{section.description}</p>
              </div>
              <Separator className="bg-border/60" />
              <div className="space-y-5">
                {section.options.map((option) => (
                  <div key={option.id} className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor={option.id} className="text-sm font-medium cursor-pointer">{option.label}</Label>
                      <p className="text-xs text-muted-foreground">{option.description}</p>
                    </div>
                    <Switch
                      id={option.id}
                      checked={loaded ? settings[option.id] : false}
                      onCheckedChange={() => handleToggle(option.id)}
                      disabled={!loaded}
                    />
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
