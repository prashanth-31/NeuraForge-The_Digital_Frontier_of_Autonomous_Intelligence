import * as React from "react";

import { cn } from "@/lib/utils";

/**
 * Input — Premium Teal Design System
 * -------------------------------------------------
 * Subtle background tint, soft border, teal glow on focus.
 */
const Input = React.forwardRef<HTMLInputElement, React.ComponentProps<"input">>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          "flex h-10 w-full rounded-lg border border-border bg-input px-4 py-2",
          "text-sm text-foreground placeholder:text-muted-foreground",
          "transition-smooth",
          "focus:outline-none focus:border-primary focus:shadow-glow",
          "disabled:cursor-not-allowed disabled:opacity-50",
          "file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-foreground",
          className
        )}
        ref={ref}
        {...props}
      />
    );
  }
);
Input.displayName = "Input";

export { Input };
