/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          primary: '#1a1a1a',
          secondary: '#2a2a2a',
          tertiary: '#3a3a3a',
          accent: '#4a4a4a',
          text: '#e0e0e0',
          textSecondary: '#b0b0b0',
          border: '#404040',
          success: '#10b981',
          warning: '#f59e0b',
          error: '#ef4444',
          info: '#3b82f6'
        }
      },
      boxShadow: {
        'neo-inset': 'inset 8px 8px 16px #151515, inset -8px -8px 16px #1f1f1f',
        'neo-outset': '8px 8px 16px #151515, -8px -8px 16px #1f1f1f',
        'neo-pressed': 'inset 4px 4px 8px #151515, inset -4px -4px 8px #1f1f1f',
        'neo-hover': '6px 6px 12px #151515, -6px -6px 12px #1f1f1f',
        'neo-small': '4px 4px 8px #151515, -4px -4px 8px #1f1f1f'
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'bounce-slow': 'bounce 2s infinite',
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'slide-down': 'slideDown 0.3s ease-out'
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' }
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' }
        },
        slideDown: {
          '0%': { transform: 'translateY(-10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' }
        }
      },
      backdropBlur: {
        xs: '2px',
      }
    },
  },
  plugins: [

    require('@tailwindcss/typography'),

  ],
} 