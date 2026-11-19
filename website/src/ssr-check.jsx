// Renders the app in Node without a browser; effects (data loading, audio) do not run under SSR.
import { renderToString } from 'react-dom/server'
import App from './App.jsx'

const html = renderToString(<App />)
if (!html.includes('Conditional Similarity Probe')) { console.error('App FAILED'); process.exit(1) }
console.log('App  OK ', html.length, 'chars')
