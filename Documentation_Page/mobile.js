/**
 * StatsLibX — mobile.js
 * ─────────────────────────────────────────────────────────────
 * Pega esto justo antes de </body> en cada HTML:
 *
 *   <script src="https://cdn.jsdelivr.net/gh/GhostAnalyst30/StatsLibX@latest/Documentation_Page/mobile.js"></script>
 *
 * Se encarga de:
 *  1. Inyectar CSS responsive completo
 *  2. Corregir desbordamiento de código en móvil (fix Prism.js)
 *  3. Crear botón hamburger + menú móvil fullscreen
 *  4. Toggle / cierre del menú
 * ─────────────────────────────────────────────────────────────
 */

(function () {
  "use strict";

  /* ═══════════════════════════════════════════════════════════
     1. CSS
  ═══════════════════════════════════════════════════════════ */
  const CSS = `
/* ── StatsLibX mobile.js ── */

#slx-hamburger {
  display: none;
  flex-direction: column;
  gap: 5px;
  cursor: pointer;
  padding: 6px;
  border: none;
  background: transparent;
  z-index: 200;
  margin-left: auto;
  flex-shrink: 0;
}
#slx-hamburger span {
  display: block;
  width: 22px;
  height: 2px;
  background: #fff;
  border-radius: 2px;
  transition: all 0.3s ease;
}
#slx-hamburger.open span:nth-child(1) { transform: translateY(7px) rotate(45deg); }
#slx-hamburger.open span:nth-child(2) { opacity: 0; transform: scaleX(0); }
#slx-hamburger.open span:nth-child(3) { transform: translateY(-7px) rotate(-45deg); }

#slx-mob-nav {
  display: none;
  position: fixed;
  top: 60px; left: 0; right: 0; bottom: 0;
  background: rgba(9, 9, 15, 0.98);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  z-index: 99;
  padding: 1.75rem 1.5rem 2rem;
  flex-direction: column;
  gap: 0.35rem;
  border-top: 1px solid rgba(255,255,255,0.07);
  overflow-y: auto;
  animation: slxSlideDown 0.22s ease;
}
#slx-mob-nav.open { display: flex; }

@keyframes slxSlideDown {
  from { opacity: 0; transform: translateY(-8px); }
  to   { opacity: 1; transform: translateY(0); }
}

#slx-mob-nav a {
  font-family: 'DM Sans', sans-serif;
  font-size: 1rem;
  font-weight: 500;
  color: #cbd5e1;
  text-decoration: none;
  padding: 0.75rem 1rem;
  border-radius: 8px;
  transition: background 0.15s, color 0.15s, border-color 0.15s;
  border: 1px solid transparent;
}
#slx-mob-nav a:hover,
#slx-mob-nav a.active {
  color: #fff;
  background: rgba(255,255,255,0.06);
  border-color: rgba(255,255,255,0.1);
}
.slx-mob-divider {
  height: 1px;
  background: rgba(255,255,255,0.07);
  margin: 0.5rem 0;
}
#slx-mob-pip {
  font-family: 'DM Mono', monospace;
  font-size: 0.88rem;
  background: rgba(124,106,247,0.12);
  border: 1px solid rgba(124,106,247,0.3);
  color: #7c6af7;
  padding: 0.75rem 1rem;
  border-radius: 8px;
  cursor: pointer;
  text-align: center;
  margin-top: 0.25rem;
  transition: background 0.2s;
}
#slx-mob-pip:hover { background: rgba(124,106,247,0.22); }

#slx-copy-toast {
  position: fixed;
  bottom: 1.5rem;
  left: 50%;
  transform: translateX(-50%) translateY(12px);
  background: #7c6af7;
  color: #fff;
  padding: 0.55rem 1.25rem;
  border-radius: 8px;
  font-size: 0.85rem;
  font-weight: 600;
  opacity: 0;
  transition: opacity 0.25s, transform 0.25s;
  pointer-events: none;
  z-index: 9999;
  white-space: nowrap;
}
#slx-copy-toast.show {
  opacity: 1;
  transform: translateX(-50%) translateY(0);
}

/* ════════════════════════════════════════════════
   FIX BASE (todos los tamaños) — código nunca
   desborda su contenedor, siempre hace scroll
   ════════════════════════════════════════════════ */

/* method-card no debe clipear el scroll del código */
.method-card {
  overflow: visible !important;
}
/* pero el code-block dentro SÍ contiene el scroll */
.method-card .code-block {
  overflow: hidden !important;
  border-radius: 10px !important;
}

/* pre siempre scrolleable */
.code-block pre,
.code-section pre,
pre[class*="language-"] {
  overflow-x: auto  !important;
  overflow-y: visible !important;
  white-space: pre  !important;
  word-wrap: normal !important;
  max-width: 100%   !important;
  box-sizing: border-box !important;
  -webkit-overflow-scrolling: touch !important;
}


/* ════════════════════════════════════════════════
   RESPONSIVE — 768px
   ════════════════════════════════════════════════ */
@media (max-width: 768px) {

  /* Nav */
  .nav-pip   { display: none !important; }
  .nav-links { display: none !important; }
  #slx-hamburger { display: flex !important; }
  .nav-inner { padding: 0 0.5rem !important; }

  /* Eliminar padding del wrapper externo que acumula con .content */
  div[style*="max-width:1240px"] {
    padding-left:  0 !important;
    padding-right: 0 !important;
  }

  /* Layout: una columna, sin padding lateral extra */
  .layout {
    display:               block  !important;
    grid-template-columns: 1fr    !important;
    padding:               0      !important;
    width:                 100%   !important;
    overflow-x:            hidden !important;
  }

  /* Sidebar oculto */
  .sidebar { display: none !important; }

  /* Content: padding lateral controlado y sin desborde */
  .content {
    padding:    1.5rem 1rem 3rem 1rem !important;
    width:      100%  !important;
    max-width:  100%  !important;
    box-sizing: border-box !important;
    overflow-x: hidden !important;
  }

  /* Method card: visible para no clipear código */
  .method-card {
    overflow:   visible    !important;
    margin:     0 0 1rem 0 !important;
    max-width:  100%       !important;
    box-sizing: border-box !important;
  }
  .method-body {
    padding:    0 1rem 1.25rem !important;
    overflow-x: hidden         !important;
  }
  .method-header {
    padding:        0.9rem 1rem  !important;
    flex-direction: column       !important;
    gap:            0.5rem       !important;
    align-items:    flex-start   !important;
  }
  .method-returns   { align-self: flex-start !important; }
  .method-name      { font-size: 0.88rem !important; }
  .method-signature {
    font-size:   0.72rem     !important;
    word-break:  break-word  !important;
    white-space: normal      !important;
    line-height: 1.5         !important;
  }

  /* ── Bloques de código ──
     Sangría negativa para llegar al borde de la pantalla
     y dar máximo espacio horizontal al código             */
  .code-block {
    margin:        0.75rem -1rem 0 -1rem   !important;
    border-radius: 0                       !important;
    border-left:   none                    !important;
    border-right:  none                    !important;
    overflow:      hidden                  !important;
    width:         calc(100% + 2rem)       !important;
    box-sizing:    border-box              !important;
  }

  .code-block pre,
  .code-section pre,
  pre[class*="language-"] {
    font-size:   0.72rem !important;
    padding:     0.9rem 1rem !important;
    overflow-x:  auto    !important;
    white-space: pre     !important;
    -webkit-overflow-scrolling: touch !important;
    /* scrollbar visible en iOS */
    scrollbar-width: thin !important;
  }

  /* Scrollbar styling en webkit */
  .code-block pre::-webkit-scrollbar {
    height: 4px !important;
  }
  .code-block pre::-webkit-scrollbar-track {
    background: rgba(255,255,255,0.03) !important;
  }
  .code-block pre::-webkit-scrollbar-thumb {
    background: rgba(124,106,247,0.4) !important;
    border-radius: 2px !important;
  }

  /* Tabla de parámetros: scroll horizontal */
  .params-table {
    display:    block    !important;
    width:      100%     !important;
    overflow-x: auto     !important;
    -webkit-overflow-scrolling: touch !important;
    font-size:  0.78rem  !important;
  }
  .params-table th,
  .params-table td {
    padding:     0.4rem 0.5rem !important;
    white-space: nowrap        !important;
  }
  .param-desc {
    white-space: normal  !important;
    min-width:   100px   !important;
  }

  /* Page header */
  .page-header  { padding: 2.5rem 1.25rem 2rem !important; }
  .page-title   { font-size: 1.9rem  !important; }
  .page-desc    { font-size: 0.9rem  !important; }
  .init-block {
    font-size:   0.73rem     !important;
    padding:     0.5rem 0.8rem !important;
    flex-wrap:   wrap          !important;
    max-width:   100%          !important;
    overflow-x:  auto          !important;
    white-space: pre           !important;
    display:     block         !important;
  }

  /* Hero */
  .hero { min-height: auto !important; padding: 2.5rem 1.25rem 3rem !important; }
  .hero h1 { font-size: clamp(2.6rem, 14vw, 4rem) !important; line-height: 1 !important; }
  .hero-sub   { font-size: 0.92rem !important; }
  .pip-block {
    font-size:  0.8rem  !important;
    padding:    0.55rem 0.9rem !important;
    max-width:  100%    !important;
    flex-wrap:  wrap    !important;
    justify-content: center !important;
  }
  .pip-copy  { display: none !important; }
  .hero-btns { flex-direction: column !important; align-items: center !important; width: 100% !important; gap: 0.75rem !important; }
  .btn-primary, .btn-ghost { width: 100% !important; max-width: 280px !important; justify-content: center !important; }
  .hero-glow { width: 260px !important; height: 260px !important; }

  /* Stats bar */
  .stats-inner { gap: 1.25rem !important; }
  .stat-num    { font-size: 1.35rem !important; }

  /* Sections */
  section        { padding: 2.5rem 1.25rem !important; }
  .section-title { font-size: 1.55rem !important; }

  /* Grids */
  .modules-grid  { grid-template-columns: 1fr !important; gap: 0 !important; }
  .module-card   { padding: 1.25rem !important; }
  .datasets-grid { grid-template-columns: 1fr !important; }
  .attr-grid, .result-methods, .dist-grid { grid-template-columns: 1fr 1fr !important; }
  .guide-grid, .backends-grid { grid-template-columns: 1fr !important; }

  /* Footer */
  .footer-links { flex-wrap: wrap !important; gap: 1rem !important; justify-content: center !important; }

  /* Misc */
  .note-box { font-size: 0.82rem !important; }
  .result-badge { font-size: 0.7rem !important; }
}


/* ════════════════════════════════════════════════
   RESPONSIVE — 420px (móviles pequeños)
   ════════════════════════════════════════════════ */
@media (max-width: 420px) {

  .attr-grid, .result-methods, .dist-grid { grid-template-columns: 1fr !important; }
  .hero h1    { font-size: 2.5rem !important; }
  .page-title { font-size: 1.65rem !important; }

  .content {
    padding: 1.25rem 0.75rem 3rem 0.75rem !important;
  }

  /* Más ancho aún para el código en pantallas muy pequeñas */
  .code-block {
    margin: 0.75rem -0.75rem 0 -0.75rem !important;
    width:  calc(100% + 1.5rem) !important;
  }
  .code-block pre,
  pre[class*="language-"] {
    font-size: 0.67rem !important;
    padding:   0.75rem !important;
  }
}
`;

  /* ═══════════════════════════════════════════════════════════
     2. Inyectar CSS
  ═══════════════════════════════════════════════════════════ */
  function injectCSS() {
    if (document.getElementById("slx-mobile-css")) return;
    const style = document.createElement("style");
    style.id = "slx-mobile-css";
    style.textContent = CSS;
    document.head.appendChild(style);
  }

  /* ═══════════════════════════════════════════════════════════
     3. Fix overflow en bloques de código
        Se llama dos veces: inmediato + diferido (post-Prism)
  ═══════════════════════════════════════════════════════════ */
  function fixCodeBlocks() {
    // method-card: overflow visible para no clipear scroll
    document.querySelectorAll(".method-card").forEach(function (card) {
      card.style.overflow = "visible";
    });

    // code-block dentro de card: contiene el scroll
    document.querySelectorAll(".method-card .code-block").forEach(function (block) {
      block.style.overflow     = "hidden";
      block.style.borderRadius = "10px";
    });

    // Todos los <pre>: siempre scrolleables
    document.querySelectorAll(
      ".code-block pre, .code-section pre, pre[class*='language-']"
    ).forEach(function (pre) {
      pre.style.overflowX  = "auto";
      pre.style.overflowY  = "visible";
      pre.style.whiteSpace = "pre";
      pre.style.wordWrap   = "normal";
      pre.style.maxWidth   = "100%";
      pre.style.boxSizing  = "border-box";
    });
  }

  /* ═══════════════════════════════════════════════════════════
     4. Hamburger button
  ═══════════════════════════════════════════════════════════ */
  function buildHamburger() {
    if (document.getElementById("slx-hamburger")) return;
    const btn = document.createElement("button");
    btn.id = "slx-hamburger";
    btn.setAttribute("aria-label", "Menú");
    btn.setAttribute("aria-expanded", "false");
    btn.innerHTML = "<span></span><span></span><span></span>";
    btn.addEventListener("click", toggleMenu);
    const navInner = document.querySelector(".nav-inner");
    if (navInner) navInner.appendChild(btn);
  }

  /* ═══════════════════════════════════════════════════════════
     5. Menú móvil
  ═══════════════════════════════════════════════════════════ */
  function buildMobileMenu() {
    if (document.getElementById("slx-mob-nav")) return;

    const pages = [
      { href: "index.html",         label: "Inicio" },
      { href: "descriptive.html",   label: "Descriptive" },
      { href: "inferential.html",   label: "Inferential" },
      { href: "computational.html", label: "Computational" },
      { href: "utils.html",         label: "Utils" },
      { href: "preprocessing.html", label: "Preprocessing" },
      { href: "datasets.html",      label: "Datasets" },
    ];

    const current = window.location.pathname.split("/").pop() || "index.html";

    const menu = document.createElement("div");
    menu.id = "slx-mob-nav";
    menu.setAttribute("role", "navigation");

    pages.forEach(function (page) {
      const a = document.createElement("a");
      a.href = page.href;
      a.textContent = page.label;
      if (page.href === current) a.classList.add("active");
      a.addEventListener("click", closeMenu);
      menu.appendChild(a);
    });

    const divider = document.createElement("div");
    divider.className = "slx-mob-divider";
    menu.appendChild(divider);

    const pip = document.createElement("div");
    pip.id = "slx-mob-pip";
    pip.textContent = "$ pip install statslibx  ⎘";
    pip.addEventListener("click", function () {
      navigator.clipboard.writeText("pip install statslibx").then(showToast);
    });
    menu.appendChild(pip);

    const nav = document.querySelector("nav");
    if (nav) nav.insertAdjacentElement("afterend", menu);
  }

  /* ═══════════════════════════════════════════════════════════
     6. Toast
  ═══════════════════════════════════════════════════════════ */
  function buildToast() {
    if (document.getElementById("slx-copy-toast")) return;
    const toast = document.createElement("div");
    toast.id = "slx-copy-toast";
    toast.textContent = "¡Copiado al portapapeles!";
    document.body.appendChild(toast);
  }

  function showToast() {
    const t = document.getElementById("slx-copy-toast");
    if (!t) return;
    t.classList.add("show");
    setTimeout(function () { t.classList.remove("show"); }, 2200);
  }

  /* ═══════════════════════════════════════════════════════════
     7. Toggle / cerrar menú
  ═══════════════════════════════════════════════════════════ */
  function toggleMenu() {
    const btn  = document.getElementById("slx-hamburger");
    const menu = document.getElementById("slx-mob-nav");
    if (!btn || !menu) return;
    const isOpen = menu.classList.toggle("open");
    btn.classList.toggle("open", isOpen);
    btn.setAttribute("aria-expanded", String(isOpen));
    document.body.style.overflow = isOpen ? "hidden" : "";
  }

  function closeMenu() {
    const btn  = document.getElementById("slx-hamburger");
    const menu = document.getElementById("slx-mob-nav");
    if (!btn || !menu) return;
    menu.classList.remove("open");
    btn.classList.remove("open");
    btn.setAttribute("aria-expanded", "false");
    document.body.style.overflow = "";
  }

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") closeMenu();
  });

  document.addEventListener("click", function (e) {
    const menu = document.getElementById("slx-mob-nav");
    const btn  = document.getElementById("slx-hamburger");
    if (!menu || !menu.classList.contains("open")) return;
    if (!menu.contains(e.target) && btn && !btn.contains(e.target)) closeMenu();
  });

  /* ═══════════════════════════════════════════════════════════
     8. Init
  ═══════════════════════════════════════════════════════════ */
  function init() {
    injectCSS();
    buildHamburger();
    buildMobileMenu();
    buildToast();

    // Fix inmediato
    fixCodeBlocks();

    // Fix diferido: Prism.js puede sobreescribir estilos al colorear
    setTimeout(fixCodeBlocks, 200);
    setTimeout(fixCodeBlocks, 800);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

})();