// Builds dist/index.html: one self-contained page (CSS and JS inlined) for drag-and-drop deploys.
// Run: npm install && npm run build
import { build } from 'esbuild';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';

const js = await build({
  entryPoints: ['src/app.js'], bundle: true, format: 'iife', charset: 'utf8',
  minifyWhitespace: true, minifySyntax: true, write: false,
});
const css = await build({ entryPoints: ['styles.css'], minify: true, write: false });
const jsText = js.outputFiles[0].text, cssText = css.outputFiles[0].text;
if (jsText.includes('</script')) throw new Error('bundle contains </script');

let html = readFileSync('index.html', 'utf8');
html = html.replace('<link rel="stylesheet" href="styles.css">', () => `<style>${cssText}</style>`)
  .replace('<script type="module" src="src/app.js"></script>', () => `<script>${jsText}</script>`);
mkdirSync('dist', { recursive: true });
writeFileSync('dist/index.html', html);
console.log(`dist/index.html written (${(html.length / 1024).toFixed(1)} KB)`);
