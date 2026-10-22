/**
 * sync-statslibx.mjs
 *
 * Bundles the real statslibx source code (plus small bundled datasets)
 * into web/public/pyodide/manifest.json so the browser playground can
 * load the actual library into the Pyodide virtual filesystem.
 *
 * Usage: node scripts/sync-statslibx.mjs
 */

import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const WEB_ROOT = process.cwd();

/** Locate the statslibx package directory (works whether the web app
 *  is the repo root, a subdirectory, or the Vercel project root). */
function findLibRoot() {
  const candidates = [
    path.join(WEB_ROOT, "statslibx"),
    path.join(WEB_ROOT, "..", "statslibx"),
    path.join(WEB_ROOT, "Libreria_Estadistica", "statslibx"),
  ];
  for (const c of candidates) {
    if (fs.existsSync(path.join(c, "__init__.py"))) return path.resolve(c);
  }
  throw new Error(
    "Could not locate the statslibx package. Run this script from the web/ directory of the repository."
  );
}

const LIB_ROOT = findLibRoot();

/** Datasets bundled into the browser build (keep the browser payload small). */
const BUNDLED_DATASETS = new Set(["iris.csv", "penguins.csv", "titanic.csv"]);

/** Directories to skip entirely. */
const SKIP_DIRS = new Set(["__pycache__", ".pytest_cache", ".git"]);

function walk(dir, base = "") {
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      if (SKIP_DIRS.has(entry.name)) continue;
      out.push(...walk(path.join(dir, entry.name), `${base}${entry.name}/`));
    } else if (entry.isFile()) {
      const rel = `${base}${entry.name}`;
      if (rel.endsWith(".py") || rel.endsWith("py.typed")) {
        out.push({ abs: path.join(dir, entry.name), rel });
      }
    }
  }
  return out;
}

function readVersion() {
  const init = fs.readFileSync(path.join(LIB_ROOT, "__init__.py"), "utf-8");
  const m = init.match(/__version__\s*=\s*["']([^"']+)["']/);
  return m ? m[1] : "unknown";
}

function main() {
  const version = readVersion();
  const files = [];

  // Python source + py.typed (package-relative paths like statslibx/__init__.py)
  for (const f of walk(LIB_ROOT)) {
    files.push({
      path: `statslibx/${f.rel}`,
      content: fs.readFileSync(f.abs, "utf-8"),
    });
  }

  // Small bundled datasets so load_iris()/load_penguins()/load_dataset() work
  const datasetsDir = path.join(LIB_ROOT, "datasets");
  for (const name of fs.readdirSync(datasetsDir)) {
    if (BUNDLED_DATASETS.has(name)) {
      files.push({
        path: `statslibx/datasets/${name}`,
        content: fs.readFileSync(path.join(datasetsDir, name), "utf-8"),
      });
    }
  }

  const outDir = path.join(WEB_ROOT, "public", "pyodide");
  fs.mkdirSync(outDir, { recursive: true });

  const manifest = {
    version,
    generatedAt: new Date().toISOString(),
    fileCount: files.length,
    files,
  };

  const outPath = path.join(outDir, "manifest.json");
  fs.writeFileSync(outPath, JSON.stringify(manifest), "utf-8");

  const kb = (fs.statSync(outPath).size / 1024).toFixed(1);
  console.log(`[sync-statslibx] statslibx v${version}: ${files.length} files -> ${path.relative(WEB_ROOT, outPath)} (${kb} KB)`);
}

main();
