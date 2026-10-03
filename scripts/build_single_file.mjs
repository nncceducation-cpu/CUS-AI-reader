#!/usr/bin/env node
/* Build the single-file edition from docs/index.html and docs/assets/*.

   Each module is wrapped in its own IIFE with its import statements stripped
   and its exports destructured into the enclosing scope, so module-private
   names cannot collide the way they would under naive concatenation.

   The inline script and style hashes are then computed and written into the
   Content-Security-Policy. This is the step that cannot be done by hand: the
   0.7.0 single-file edition pins a script hash, and any edit to the script
   invalidates it, which blocks the entire page with no visible failure beyond
   a console message.

   Importable as buildSingleFile(root) so tests can check that the committed
   bundle is in sync without spawning a subprocess.

   Usage: node scripts/build_single_file.mjs [repoRoot] [outputPath]
*/
import {readFileSync, writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';

/* Dependency order. app.js must come last: it is the only module with side
   effects at evaluation time. */
const ORDER = ['core.js', 'media.js', 'store.js', 'learning.js', 'checks.js', 'app.js'];
const NAMESPACE_IMPORT = {'store.js': 'db'};   // app.js does `import * as db`

const sha = text => createHash('sha256').update(text, 'utf8').digest('base64');

function stripImports(source) {
  const lines = source.split('\n'), kept = [];
  for (let i = 0; i < lines.length; i++) {
    if (/^\s*import\b/.test(lines[i])) {
      while (i < lines.length && !/;\s*$/.test(lines[i])) i++;   // multi-line import
      continue;
    }
    kept.push(lines[i]);
  }
  return kept.join('\n');
}

/* Collect exported binding names. Handles `export function`, `export async
   function`, and `export const/let/var` including several declarators in one
   statement. Other export forms are rejected rather than mishandled. */
function exportNames(source, file) {
  if (/^export\s+default/m.test(source) || /^export\s*\{/m.test(source) || /^export\s*\*/m.test(source))
    throw Error(`${file}: export form not supported by this builder`);
  const names = new Set();
  for (const m of source.matchAll(/^export\s+(?:async\s+)?function\s+([A-Za-z_$][\w$]*)/gm))
    names.add(m[1]);
  for (const m of source.matchAll(/^export\s+(?:const|let|var)\s+/gm)) {
    let i = m.index + m[0].length, depth = 0, body = '';
    while (i < source.length) {
      const c = source[i];
      if ('([{'.includes(c)) depth++;
      else if (')]}'.includes(c)) depth--;
      else if (c === ';' && depth === 0) break;
      body += c; i++;
    }
    let d = 0, part = '';
    const parts = [];
    for (const c of body) {
      if ('([{'.includes(c)) d++;
      else if (')]}'.includes(c)) d--;
      if (c === ',' && d === 0) { parts.push(part); part = ''; continue; }
      part += c;
    }
    parts.push(part);
    for (const p of parts) {
      const name = p.split('=')[0].trim();
      if (/^[A-Za-z_$][\w$]*$/.test(name)) names.add(name);
    }
  }
  const statements = (source.match(/^export\s/gm) || []).length;
  if (names.size < statements)
    throw Error(`${file}: parsed ${names.size} exported names for ${statements} export statements`);
  return [...names];
}

export function buildSingleFile(root = '.') {
  const asset = name => join(root, 'docs', 'assets', name);
  const chunks = [];
  const claimed = new Map();

  for (const file of ORDER) {
    const raw = readFileSync(asset(file), 'utf8');
    const names = exportNames(raw, file);
    for (const name of names) {
      if (claimed.has(name))
        throw Error(`export name collision: ${name} exported by both ${claimed.get(name)} and ${file}`);
      claimed.set(name, file);
    }
    const body = stripImports(raw).replace(/^export\s+/gm, '');
    const id = '__' + file.replace(/\W/g, '_');
    if (!names.length) {
      chunks.push(`/* ${file} */\n(() => {\n${body}\n})();`);
      continue;
    }
    chunks.push(`/* ${file} */\nconst ${id} = (() => {\n${body}\nreturn {${names.join(',')}};\n})();`);
    chunks.push(NAMESPACE_IMPORT[file]
      ? `const ${NAMESPACE_IMPORT[file]} = ${id};`
      : `const {${names.join(',')}} = ${id};`);
  }

  /* The hashed text must be byte-identical to what lands between the tags, so
     both are built once here and inserted verbatim below. */
  const style = readFileSync(asset('reader.css'), 'utf8').trim() + '\n';
  const script = `document.addEventListener('DOMContentLoaded', () => {\n'use strict';\n${chunks.join('\n')}\n});`;
  const scriptHash = `'sha256-${sha(script)}'`;
  const styleHash = `'sha256-${sha(style)}'`;

  const csp = [
    "default-src 'self'",
    `script-src 'self' 'unsafe-eval' ${scriptHash} https://cdn.jsdelivr.net`,
    `style-src 'self' ${styleHash}`,
    "img-src 'self' data: blob:",
    "media-src blob:",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'none'"
  ].join('; ');

  let html = readFileSync(join(root, 'docs', 'index.html'), 'utf8');
  const before = html;
  html = html.replace(/<link rel="stylesheet" href="assets\/reader\.css">/, `<style>${style}</style>`);
  html = html.replace(/<script type="module" src="assets\/app\.js"><\/script>/, '');
  html = html.replace(/<meta http-equiv="Content-Security-Policy" content="[^"]*">/,
    `<meta http-equiv="Content-Security-Policy" content="${csp}">`);
  html = html.replace('</body>', `<script>${script}</script></body>`);
  if (html === before) throw Error('single-file assembly replaced nothing; check index.html markers');
  for (const marker of ['assets/reader.css', 'assets/app.js'])
    if (html.includes(marker)) throw Error(`single-file build still references ${marker}`);

  const banner = `<!-- CUS Reader single-file edition, generated by scripts/build_single_file.mjs.
Paste this entire document into a custom HTML/JavaScript page. Host over HTTPS.
Requires the two integrity-pinned CDN libraries and browser IndexedDB.
Do not hand-edit: the Content-Security-Policy pins the inline script and style
hashes, so any edit invalidates them and blocks the page. Rebuild instead.
Cases/models are private browser data and are not included in this code. -->`;
  html = html.replace('<!doctype html>', `<!doctype html>\n${banner}`);

  /* Re-extract from the assembled document and re-hash, so the pinned values
     are checked against the real inline content rather than the source strings. */
  const inlineScript = html.match(/<script>([\s\S]*)<\/script>/)[1];
  const inlineStyle = html.match(/<style>([\s\S]*?)<\/style>/)[1];
  if (`'sha256-${sha(inlineScript)}'` !== scriptHash || `'sha256-${sha(inlineStyle)}'` !== styleHash)
    throw Error('inline hash verification failed during assembly');

  return {html, scriptHash, styleHash, modules: ORDER.length, exportedNames: claimed.size};
}

/* CLI */
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const root = process.argv[2] || '.';
  const out = process.argv[3] || join(root, 'docs', 'CUS-reader-single-file.html');
  const built = buildSingleFile(root);
  writeFileSync(out, built.html, 'utf8');
  console.log(`wrote ${out}`);
  console.log(`  ${(built.html.length / 1024).toFixed(1)} kB, ${built.modules} modules, ${built.exportedNames} exported names`);
  console.log(`  script-src ${built.scriptHash}`);
  console.log(`  style-src  ${built.styleHash}`);
  console.log('  inline hashes verified against the assembled document');
}
