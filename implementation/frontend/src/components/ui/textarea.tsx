import * as React from "react";

import { cn } from "@/lib/utils";

export type TextareaProps = React.TextareaHTMLAttributes<HTMLTextAreaElement>;

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(({ className, ...props }, ref) => {
  return (
    <textarea
      className={cn(
        "flex min-h-[80px] w-full rounded-xl border border-slate-200 bg-white/80 px-4 py-3 text-sm",
        "placeholder:text-slate-400",
        "transition-all duration-200",
        "focus:outline-none focus:border-primary-400 focus:bg-white focus:shadow-glow",
        "hover:border-slate-300 hover:bg-white",
        "disabled:cursor-not-allowed disabled:opacity-50 disabled:bg-slate-50",
        className,
      )}
      ref={ref}
      {...props}
    />
  );
});
Textarea.displayName = "Textarea";

export { Textarea };
