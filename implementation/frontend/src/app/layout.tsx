import './globals.css';
import { Inter, Lexend, Fira_Code } from 'next/font/google';
import type { Metadata, Viewport } from 'next';

const inter = Inter({ 
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

const lexend = Lexend({
  subsets: ['latin'],
  variable: '--font-lexend',
  display: 'swap',
});

const firaCode = Fira_Code({
  subsets: ['latin'],
  variable: '--font-fira-code',
  display: 'swap',
});

// Define viewport export separately
export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#7c3aed' },
    { media: '(prefers-color-scheme: dark)', color: '#4338ca' },
  ],
};

export const metadata: Metadata = {
  title: 'NeuraForge - Advanced AI Assistant Platform',
  description: 'Next-generation AI-powered multi-agent system with LLaMA 3.1 integration',
  keywords: 'AI, artificial intelligence, LLM, LLaMA, multi-agent, NeuraForge, assistant, neural networks',
  authors: [{ name: 'NeuraForge Team' }],
  metadataBase: new URL('http://localhost:3000'),
  openGraph: {
    type: 'website',
    title: 'NeuraForge - Advanced AI Assistant Platform',
    description: 'Next-generation AI-powered multi-agent system with LLaMA 3.1 integration',
    siteName: 'NeuraForge',
    images: [{ url: '/og-image.png', width: 1200, height: 630 }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'NeuraForge - Advanced AI Assistant Platform',
    description: 'Next-generation AI-powered multi-agent system with LLaMA 3.1 integration',
    images: ['/twitter-image.png'],
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} ${lexend.variable} ${firaCode.variable} scroll-smooth`}>
      <head>
        <link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png" />
        <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png" />
        <link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png" />
        <link rel="mask-icon" href="/safari-pinned-tab.svg" color="#7c3aed" />
        <meta name="msapplication-TileColor" content="#7c3aed" />
        <meta name="theme-color" content="#ffffff" />
      </head>
      <body className="font-sans antialiased">
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                // Theme handling
                function getThemePreference() {
                  if (typeof window !== 'undefined' && window.localStorage) {
                    const storedPrefs = window.localStorage.getItem('theme-preference');
                    if (typeof storedPrefs === 'string') return storedPrefs;
                    
                    const userMedia = window.matchMedia('(prefers-color-scheme: dark)');
                    if (userMedia.matches) return 'dark';
                  }
                  
                  return 'light'; // Default theme
                }
                
                const theme = getThemePreference();
                document.documentElement.classList.toggle('dark', theme === 'dark');
              })();
            `,
          }}
        />
        {children}
      </body>
    </html>
  );
}
