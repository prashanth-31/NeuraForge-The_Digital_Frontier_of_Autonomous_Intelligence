import { motion } from "framer-motion";

// Fade in animation for content 
export const FadeIn = ({ children, delay = 0, duration = 0.5, className = "" }) => (
  <motion.div
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration, delay, ease: "easeOut" }}
    className={className}
  >
    {children}
  </motion.div>
);

// Scale animation for cards and interactive elements
export const ScaleIn = ({ children, delay = 0, duration = 0.4, className = "" }) => (
  <motion.div
    initial={{ opacity: 0, scale: 0.95 }}
    animate={{ opacity: 1, scale: 1 }}
    transition={{ duration, delay, ease: "easeOut" }}
    className={className}
  >
    {children}
  </motion.div>
);

// Slide in animation for sidebars
export const SlideIn = ({ 
  children, 
  direction = "left", 
  delay = 0, 
  duration = 0.5,
  className = "" 
}) => {
  const xValue = direction === "left" ? -20 : direction === "right" ? 20 : 0;
  const yValue = direction === "top" ? -20 : direction === "bottom" ? 20 : 0;
  
  return (
    <motion.div
      initial={{ opacity: 0, x: xValue, y: yValue }}
      animate={{ opacity: 1, x: 0, y: 0 }}
      transition={{ duration, delay, ease: "easeOut" }}
      className={className}
    >
      {children}
    </motion.div>
  );
};

// Staggered list animation for items in a list
export const StaggeredList = ({ children, delay = 0, staggerDelay = 0.1 }) => (
  <motion.div initial="hidden" animate="visible" variants={{
    visible: {
      transition: {
        staggerChildren: staggerDelay,
        delayChildren: delay,
      }
    }
  }}>
    {children}
  </motion.div>
);

// Item animation for use with StaggeredList
export const StaggerItem = ({ children, className = "" }) => (
  <motion.div
    variants={{
      hidden: { opacity: 0, y: 10 },
      visible: { opacity: 1, y: 0 }
    }}
    transition={{ duration: 0.4 }}
    className={className}
  >
    {children}
  </motion.div>
);

// Pulse animation for accent elements
export const PulseEffect = ({ children, className = "" }) => (
  <motion.div
    animate={{ 
      boxShadow: [
        "0 0 0 0 rgba(var(--neura-blue-rgb), 0)",
        "0 0 0 8px rgba(var(--neura-blue-rgb), 0.1)",
        "0 0 0 0 rgba(var(--neura-blue-rgb), 0)"
      ] 
    }}
    transition={{ 
      duration: 2,
      repeat: Infinity,
      repeatType: "loop"
    }}
    className={className}
  >
    {children}
  </motion.div>
);

// Loading spinner with animation
export const LoadingSpinner = ({ size = "md", className = "" }) => {
  const sizeClass = 
    size === "sm" ? "w-4 h-4" : 
    size === "lg" ? "w-8 h-8" : 
    "w-6 h-6";
  
  return (
    <motion.div 
      className={`${sizeClass} rounded-full border-2 border-neura-blue/30 border-t-neura-blue ${className}`}
      animate={{ rotate: 360 }}
      transition={{ 
        duration: 1.2, 
        repeat: Infinity, 
        ease: "linear" 
      }}
    />
  );
};
