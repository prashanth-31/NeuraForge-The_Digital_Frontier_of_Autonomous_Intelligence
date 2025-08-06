/**
 * Environment variables for the application
 */

export const ENV = {
  // API URL with fallback to localhost
  API_URL: (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000',
  
  // Development mode
  IS_DEV: (import.meta as any).env?.MODE === 'development' || true,
  
  // Version
  VERSION: '0.1.0',
};

export default ENV;
