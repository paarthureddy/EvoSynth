// Build one HTML page from handbook/*.md: node build.mjs <handbook_dir> <out.html> <commit>
import fs from "fs";
import path from "path";
import MarkdownIt from "markdown-it";
import katex from "katex";

const [, , DIR, OUT, COMMIT] = process.argv;

const FILES = [
  ["README.md", "overview", "Overview & glossary", "handbook"],
  ["00a-prerequisites-ml.md", "pre-a", "Prerequisites 1: ML basics", "prereq"],
  ["00b-prerequisites-automl.md", "pre-b", "Prerequisites 2: AutoML & search", "prereq"],
  ["00c-prerequisites-evolutionary.md", "pre-c", "Prerequisites 3: EAs & bandits", "prereq"],
  ["01-overview-and-motivation.md", "ch-01", null, "handbook"],
  ["02-architecture.md", "ch-02", null, "handbook"],
  ["03-engine-internals.md", "ch-03", null, "handbook"],
  ["04-optimizers.md", "ch-04", null, "handbook"],
  ["05-search-space-and-evaluators.md", "ch-05", null, "handbook"],
  ["06-api-and-dashboard.md", "ch-06", null, "handbook"],
  ["07-experiments.md", "ch-07", null, "handbook"],
  ["08-results.md", "ch-08", null, "handbook"],
  ["09-comparative-analysis.md", "ch-09", null, "handbook"],
  ["10-paper.md", "ch-10", null, "handbook"],
  ["11-papers.md", "ch-11", null, "handbook"],
  ["12-setup-and-usage.md", "ch-12", null, "handbook"],
  ["commands.md", "commands", "Commands", "handbook"],
  ["13-panel-defense-qa.md", "ch-13", null, "defence"],
];
const ID_OF = Object.fromEntries(FILES.map(([f, id]) => [f, id]));

const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const slug = (s) => s.toLowerCase().replace(/<[^>]+>/g, "").replace(/[`*_~$\\]/g, "")
  .replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 60) || "s";

// ---- math: pull $$..$$ / $..$ out of non-code text, render as MathML afterwards
function extractMath(src, store) {
  const parts = src.split(/(```[\s\S]*?```)/g);
  return parts.map((part, i) => {
    if (i % 2 === 1) return part;                                // fenced code
    const codes = [];
    let t = part.replace(/`[^`\n]*`/g, (m) => { codes.push(m); return `CODEPH${codes.length - 1}X`; });
    t = t.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => { store.push([tex, true]); return `MATHPH${store.length - 1}X`; });
    t = t.replace(/\$(?!\s)([^$\n]+?)(?<!\s)\$/g, (_, tex) => { store.push([tex, false]); return `MATHPH${store.length - 1}X`; });
    return t.replace(/CODEPH(\d+)X/g, (_, n) => codes[+n]);
  }).join("");
}
const renderMath = (html, store) => html.replace(/MATHPH(\d+)X/g, (_, n) => {
  const [tex, display] = store[+n];
  return katex.renderToString(tex, { output: "mathml", displayMode: display, throwOnError: false });
});

function makeMd(chapId) {
  const md = new MarkdownIt({ html: true, linkify: false, typographer: false });
  const used = new Set();
  md.renderer.rules.heading_open = (tokens, idx, opts, env, self) => {
    const t = tokens[idx];
    const text = tokens[idx + 1].content;
    let id = t.tag === "h1" ? chapId : `${chapId}--${slug(text)}`;
    let k = 2;
    while (used.has(id)) id = `${chapId}--${slug(text)}-${k++}`;
    used.add(id);
    t.attrSet("id", id);
    return self.renderToken(tokens, idx, opts);
  };
  md.renderer.rules.link_open = (tokens, idx, opts, env, self) => {
    const t = tokens[idx];
    const href = t.attrGet("href") || "";
    const m = href.match(/^(?:\.\/)?([\w.-]+\.md)(#.*)?$/);
    if (m && ID_OF[m[1]]) t.attrSet("href", "#" + ID_OF[m[1]]);
    else if (/^https?:/.test(href)) { t.attrSet("target", "_blank"); t.attrSet("rel", "noopener"); }
    return self.renderToken(tokens, idx, opts);
  };
  md.renderer.rules.fence = (tokens, idx) => {
    const t = tokens[idx];
    const lang = (t.info || "").trim().split(/\s+/)[0];
    if (lang === "mermaid") return `<div class="diagram"><pre class="mermaid">${esc(t.content)}</pre></div>\n`;
    return `<div class="code"><pre><code>${esc(t.content)}</code></pre></div>\n`;
  };
  md.renderer.rules.image = (tokens, idx) => {
    const t = tokens[idx];
    const src = t.attrGet("src") || "";
    const alt = esc(t.content || "");
    const file = path.join(DIR, src);
    if (!/^https?:/.test(src) && fs.existsSync(file)) {
      const ext = path.extname(file).slice(1).toLowerCase();
      const mime = ext === "svg" ? "image/svg+xml" : ext === "jpg" ? "image/jpeg" : "image/" + ext;
      return `<figure class="fig"><img src="data:${mime};base64,${fs.readFileSync(file).toString("base64")}" alt="${alt}" loading="lazy"></figure>`;
    }
    return `<img src="${esc(src)}" alt="${alt}">`;
  };
  md.renderer.rules.table_open = () => `<div class="table-wrap"><table>\n`;
  md.renderer.rules.table_close = () => `</table></div>\n`;
  return md;
}

const sections = [];
const nav = { handbook: [], defence: [], prereq: [] };
for (const [file, id, label, group] of FILES) {
  const src = fs.readFileSync(path.join(DIR, file), "utf8");
  const h1 = (src.match(/^#\s+(.+)$/m) || [, file])[1].trim();
  const num = (h1.match(/^(\d\d[a-z]?)\s+—\s+/) || [])[1] || "";
  const title = label || h1.replace(/^\d\d[a-z]?\s+—\s+/, "");
  const store = [];
  const html = renderMath(makeMd(id).render(extractMath(src, store)), store);
  sections.push(`<section class="chapter" id="sec-${id}" data-num="${num}">\n${html}</section>`);
  nav[group].push(`<li><a href="#${id}" data-target="${id}"><span class="n">${num || "·"}</span><span>${esc(title)}</span></a></li>`);
}

const template = fs.readFileSync(new URL("./template.html", import.meta.url), "utf8");
const out = template
  .replace("{{NAV_HANDBOOK}}", nav.handbook.join("\n"))
  .replace("{{NAV_DEFENCE}}", nav.defence.join("\n"))
  .replace("{{NAV_PREREQ}}", nav.prereq.join("\n"))
  .replace("{{SECTIONS}}", sections.join("\n"))
  .replace(/{{COMMIT}}/g, COMMIT);
fs.writeFileSync(OUT, out);
console.log("wrote", OUT, (out.length / 1024).toFixed(0) + " KB");
