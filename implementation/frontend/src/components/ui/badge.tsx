import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

/**
 * Badge — Premium Teal Design System
 * -------------------------------------------------
 * Pill-shaped with semantic colors, subtle opacity variants.
 */
const badgeVariants = cva(
  [
    "inline-flex items-center justify-center rounded-full border px-3 py-0.5",
    "text-xs font-semibold tracking-wide transition-smooth",
    "focus:outline-none focus:shadow-glow",
  ].join(" "),
  {
    variants: {
      variant: {
        default: [
          "border-transparent bg-primary text-primary-foreground",
          "hover:bg-primary-600",
        ].join(" "),
        secondary: [
          "border-transparent bg-muted text-muted-foreground",
          "hover:bg-primary-100 hover:text-primary-700",
        ].join(" "),
        outline: [
          "border-border bg-transparent text-foreground",
          "hover:border-primary hover:text-primary",
        ].join(" "),
        success: [
          "border-transparent bg-success text-success-foreground",
          "hover:bg-success/90",
        ].join(" "),
        warning: [
          "border-transparent bg-warning text-warning-foreground",
          "hover:bg-warning/90",
        ].join(" "),
        destructive: [
          "border-transparent bg-destructive text-destructive-foreground",
          "hover:bg-destructive/90",
        ].join(" "),
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className, variant, ...props }, ref) => {
    return (
      <span
        ref={ref}
        className={cn(badgeVariants({ variant }), className)}
        {...props}
      />
    );
  }
);

Badge.displayName = "Badge";

export { Badge, badgeVariants };
