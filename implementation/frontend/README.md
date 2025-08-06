# NeuraForge Frontend

<p align="center">
  <img src="public/favicon.svg" alt="NeuraForge Logo" width="120" height="120" />
</p>

<p align="center">
  <b>Advanced AI-powered chat interface with LLaMA 3.1 integration</b><br/>
  Built with Next.js 14, React and TailwindCSS
</p>

<p align="center">
  <img src="https://img.shields.io/badge/next.js-14.0.0-black?style=flat-square" alt="Next.js" />
  <img src="https://img.shields.io/badge/react-18.0.0-blue?style=flat-square" alt="React" />
  <img src="https://img.shields.io/badge/tailwindcss-3.3.0-38bdf8?style=flat-square" alt="TailwindCSS" />
  <img src="https://img.shields.io/badge/typescript-5.0.0-3178c6?style=flat-square" alt="TypeScript" />
</p>

---

## 🚀 Features

- ✨ Modern, responsive UI with glassmorphism design
- 💬 Real-time chat interface with NeuraForge AI
- 🖥️ Code block formatting with syntax highlighting
- 🌙 Automatic dark mode support
- 🔄 Connection status indicators
- 📊 AI confidence scoring visualization
- ⚡ Performance optimized animations
- 📱 Mobile-friendly responsive layout

![Screenshot of NeuraForge UI](https://via.placeholder.com/800x450?text=NeuraForge+UI+Screenshot)

## 🛠️ Getting Started

### Prerequisites

- **Node.js 18+** (LTS recommended)
- **npm** or **yarn**
- **NeuraForge backend server** running (default: http://localhost:8000)
- **Ollama** with LLaMA 3.1:8b model loaded

### Installation

1. Clone the repository (if not done already):

```bash
git clone https://github.com/yourusername/neuraforge.git
cd neuraforge/implementation/frontend
```

2. Install dependencies:

```bash
npm install
# or
yarn install
```

3. Run the development server:

```bash
npm run dev
# or
yarn dev
```

4. Open [http://localhost:3000](http://localhost:3000) with your browser to see the app.

## ⚙️ Environment Variables

Configure the API URL by creating a `.env.local` file in the root directory:

```env
# API URL (default: http://localhost:8000)
NEXT_PUBLIC_API_URL=http://localhost:8000

# Enable/disable additional debugging (optional)
NEXT_PUBLIC_DEBUG_MODE=false
```

## 🏗️ Project Structure

```
src/
├── app/                # Next.js App Router
│   ├── globals.css     # Global styles
│   ├── layout.tsx      # Root layout
│   └── page.tsx        # Home page
├── components/         # React components
│   └── ChatInterface.tsx  # Main chat interface
└── ...
```

## 📦 Building for Production

```bash
npm run build
# or
yarn build
```

Then start the production server:

```bash
npm run start
# or
yarn start
```

## 🧩 Tech Stack

- **Next.js 14** - React framework with server-side rendering
- **React 18** - UI library
- **TypeScript** - Type safety
- **Tailwind CSS** - Utility-first styling
- **Socket.io** - For future WebSocket support

## 🔮 Upcoming Features

- [ ] Streaming responses with WebSockets
- [ ] Multi-modal capabilities (image generation, understanding)
- [ ] Conversation history and export
- [ ] Custom themes and UI personalization
- [ ] Offline mode with local storage

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

<p align="center">
  Made with ❤️ by the NeuraForge Team
</p>
