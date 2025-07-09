# Dynamic Agent System Frontend

A React frontend with neomorphic design for the Dynamic Agent System.

## Features

- **Neomorphic Design**: Modern dark theme with soft shadows
- **Three-Panel Layout**: 
  - Left: KB Sources and Persona Management
  - Center: Chat Interface with Query Processing
  - Right: Metadata and Source Preview
- **Live Query Processing**: Real-time query processing with visual flow
- **File Upload**: Support for PDFs, CSVs, and database connections
- **Multi-Persona Support**: Switch between different AI personas
- **Visual Flow Debugging**: See query processing steps in real-time

## Tech Stack

- React 18
- Tailwind CSS
- Ant Design Icons
- Axios for API calls

## Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                        Dynamic Agent System                      │
├─────────────┬─────────────────────────────┬─────────────────────┤
│             │                             │                     │
│  Left Panel │      Center Panel           │    Right Panel      │
│             │                             │                     │
│ • Personas  │ • Chat Interface            │ • Metadata          │
│ • Sources   │ • Query Input               │ • Processing Flow   │
│ • LLM Config│ • Suggested Queries         │ • Source Preview    │
│ • File Upload│ • Response Display         │ • Performance Stats │
│             │                             │                     │
└─────────────┴─────────────────────────────┴─────────────────────┘
```

## Components

### LeftPanel
- **Persona Management**: Select and configure AI personas
- **File Upload**: Upload PDFs, CSVs, Excel files
- **Database Connections**: Connect to SQL/NoSQL databases
- **LLM Configuration**: Manage API keys and providers

### CenterPanel
- **Chat Interface**: Real-time chat with the AI system
- **Query Processing**: Submit queries and see responses
- **Suggested Queries**: Pre-built example queries
- **Response Display**: Formatted responses with suggestions

### RightPanel
- **Metadata**: Query processing information
- **Processing Flow**: Visual representation of query pipeline
- **Source Files**: Preview and manage uploaded files
- **Performance Stats**: Response times and system metrics

## Setup Instructions

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure Backend

Make sure your backend is running on `http://localhost:8000`

### 3. Start Development Server

```bash
npm start
```

The app will open at `http://localhost:3000`

## Available Scripts

- `npm start` - Start development server
- `npm build` - Build for production
- `npm test` - Run tests

## API Integration

The frontend connects to the backend at `http://localhost:8000` with the following endpoints:

- `POST /query` - Submit queries
- `POST /upload` - Upload files
- `GET /health` - Check backend status

## Customization

### Colors
Edit `tailwind.config.js` to modify the dark theme colors:

```javascript
colors: {
  dark: {
    primary: '#1a1a1a',    // Background
    secondary: '#2a2a2a',   // Panels
    tertiary: '#3a3a3a',    // Cards
    // ... more colors
  }
}
```

### Neomorphic Effects
Modify shadow styles in `src/index.css`:

```css
.neo-panel {
  @apply bg-dark-secondary shadow-neo-outset rounded-2xl;
}
```

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+ 