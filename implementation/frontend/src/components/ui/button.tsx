import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

/**
 * Button Variants — Premium Teal Design System
 * -------------------------------------------------
 * Uses softer radii (rounded-lg), refined shadows on hover,
 * and consistent focus-visible ring using teal glow.
 */
const buttonVariants = cva(
  [
    "inline-flex items-center justify-center gap-2 whitespace-nowrap",
    "rounded-lg text-sm font-semibold",
    "ring-offset-background transition-smooth",
    "focus-visible:outline-none focus-visible:shadow-glow",
    "disabled:pointer-events-none disabled:opacity-50",
    "[&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  ].join(" "),
  {
    variants: {
      variant: {
        default: [
          "bg-primary text-primary-foreground",
          "hover:bg-primary-600 hover:shadow-elevated",
          "active:bg-primary-700",
        ].join(" "),
        secondary: [
          "bg-muted text-foreground",
          "hover:bg-primary-100 hover:text-primary-700",
          "active:bg-primary-200",
        ].join(" "),
        outline: [
          "border border-border bg-transparent text-foreground",
          "hover:border-primary hover:text-primary hover:bg-primary-50",
          "active:bg-primary-100",
        ].join(" "),
        ghost: [
          "bg-transparent text-foreground",
          "hover:bg-primary-50 hover:text-primary-700",
          "active:bg-primary-100",
        ].join(" "),
        destructive: [
          "bg-destructive text-destructive-foreground",
          "hover:bg-destructive/90 hover:shadow-elevated",
          "active:bg-destructive/80",
        ].join(" "),
        link: [
          "text-primary underline-offset-4",
          "hover:underline",
        ].join(" "),
        success: [
          "bg-success text-success-foreground",
          "hover:bg-success/90 hover:shadow-elevated",
          "active:bg-success/80",
        ].join(" "),
      },
      size: {
        default: "h-10 px-5 py-2",
        sm: "h-9 px-4 text-xs",
        lg: "h-12 px-8 text-base",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };
