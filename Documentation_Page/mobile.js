/**
 * StatsLibX — mobile.js
 * ─────────────────────────────────────────────
 * Pega este script en cualquier página HTML de
 * StatsLibX justo antes de </body>:
 *
 *   <script src="mobile.js"></script>
 *
 * Se encarga de:
 *  1. Inyectar el CSS responsive completo
 *  2. Crear el botón hamburger en el nav
 *  3. Crear el menú móvil fullscreen
 *  4. Gestionar el toggle / cierre del menú
 * ─────────────────────────────────────────────
 */

(function () {
  "use strict";

  /* ─── 1. CSS ─────────────────────────────────────────────────── */
  const CSS = `
/* ── StatsLibX mobile.js styles ── */
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
  top: 60px;
  left: 0;
  right: 0;
  bottom: 0;
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
#slx-mob-nav .mob-divider {
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

@media (max-width: 768px) {
  /* Nav */
  .nav-pip   { display: none !important; }
  .nav-links { display: none !important; }
  #slx-hamburger { display: flex !important; }

  /* Hero */
  .hero { min-height: auto !important; padding: 3rem 1.25rem 3.5rem !important; }
  .hero h1 { font-size: clamp(2.8rem, 15vw, 4.5rem) !important; line-height: 1 !important; }
  .hero-sub { font-size: 0.95rem !important; }
  .pip-block {
    font-size: 0.82rem !important;
    padding: 0.6rem 1rem !important;
    max-width: 100% !important;
    flex-wrap: wrap !important;
    justify-content: center !important;
  }
  .pip-copy { display: none !important; }
  .hero-btns { flex-direction: column !important; align-items: center !important; width: 100% !important; }
  .btn-primary, .btn-ghost {
    width: 100% !important;
    max-width: 280px !important;
    justify-content: center !important;
  }
  .hero-glow { width: 280px !important; height: 280px !important; }

  /* Stats bar */
  .stats-inner { gap: 1.5rem !important; }
  .stat-num    { font-size: 1.4rem !important; }

  /* Sections */
  section { padding: 3rem 1.25rem !important; }
  .section-title { font-size: 1.6rem !important; }

  /* Module / dataset grids */
  .modules-grid  { grid-template-columns: 1fr !important; gap: 0 !important; }
  .module-card   { padding: 1.5rem !important; }
  .datasets-grid { grid-template-columns: 1fr !important; }

  /* Code blocks */
  .code-section pre, .code-block pre {
    font-size: 0.75rem !important;
    padding: 1rem !important;
    overflow-x: auto !important;
  }

  /* Page header */
  .page-header { padding: 2.5rem 1.25rem 2rem !important; }
  .page-title  { font-size: 2rem !important; }
  .page-desc   { font-size: 0.9rem !important; }
  .init-block  {
    font-size: 0.75rem !important;
    padding: 0.55rem 0.9rem !important;
    flex-wrap: wrap !important;
    max-width: 100% !important;
    overflow: hidden !important;
  }

  /* Sidebar layout */
  .layout  { grid-template-columns: 1fr !important; padding: 0 1.25rem !important; }
  .sidebar { display: none !important; }
  .content { padding: 2rem 0 !important; }

  /* Method cards */
  .method-header    { flex-direction: column !important; gap: 0.5rem !important; }
  .method-returns   { align-self: flex-start !important; }
  .method-name      { font-size: 0.88rem !important; }
  .method-signature { font-size: 0.72rem !important; word-break: break-word !important; }

  /* Params table */
  .params-table {
    display: block !important;
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch !important;
  }

  /* Small grids */
  .attr-grid,
  .result-methods,
  .dist-grid      { grid-template-columns: 1fr 1fr !important; }
  .guide-grid,
  .backends-grid  { grid-template-columns: 1fr !important; }

  /* Footer */
  .footer-links { flex-wrap: wrap !important; gap: 1rem !important; justify-content: center !important; }
}

@media (max-width: 420px) {
  .attr-grid, .result-methods, .dist-grid { grid-template-columns: 1fr !important; }
  .hero h1    { font-size: 2.8rem !important; }
  .page-title { font-size: 1.75rem !important; }
  .init-block { font-size: 0.7rem !important; }
}
`;

  /* ─── 2. Inject CSS ──────────────────────────────────────────── */
  function injectCSS() {
    if (document.getElementById("slx-mobile-css")) return;
    const style = document.createElement("style");
    style.id = "slx-mobile-css";
    style.textContent = CSS;
    document.head.appendChild(style);
  }

  /* ─── 3. Build hamburger button ─────────────────────────────── */
  function buildHamburger() {
    if (document.getElementById("slx-hamburger")) return;
    const btn = document.createElement("button");
    btn.id = "slx-hamburger";
    btn.setAttribute("aria-label", "Menú");
    btn.innerHTML = "<span></span><span></span><span></span>";
    btn.addEventListener("click", toggleMenu);

    const navInner = document.querySelector(".nav-inner");
    if (navInner) navInner.appendChild(btn);
  }

  /* ─── 4. Build mobile menu ──────────────────────────────────── */
  function buildMobileMenu() {
    if (document.getElementById("slx-mob-nav")) return;

    const pages = [
      { href: "https://ghostanalyst30.github.io/StatsLibX/Documentation_Page/index.html",          label: "Inicio" },
      { href: "https://ghostanalyst30.github.io/StatsLibX/Documentation_Page/descriptive.html",    label: "Descriptive" },
      { href: "https://ghostanalyst30.github.io/StatsLibX/Documentation_Page/inferential.html",    label: "Inferential" },
      { href: "https://ghostanalyst30.github.io/StatsLibX/Documentation_Page/computational.html",  label: "Computational" },
      { href: "https://ghostanalyst30.github.io/StatsLibX/Documentation_Page/utils.html",          label: "Utils" },
      { href: "https://ghostanalyst30.github.io/StatsLibX/Documentation_Page/preprocessing.html",  label: "Preprocessing" },
      { href: "https://ghostanalyst30.github.io/StatsLibX/Documentation_Page/datasets.html",       label: "Datasets" },
    ];

    // Detect current page
    const current = window.location.pathname.split("/").pop() || "index.html";

    const menu = document.createElement("div");
    menu.id = "slx-mob-nav";
    menu.setAttribute("role", "navigation");
    menu.setAttribute("aria-label", "Menú móvil");

    pages.forEach(({ href, label }) => {
      const a = document.createElement("a");
      a.href = href;
      a.textContent = label;
      if (href === current) a.classList.add("active");
      a.addEventListener("click", closeMenu);
      menu.appendChild(a);
    });

    // Divider
    const div = document.createElement("div");
    div.className = "mob-divider";
    menu.appendChild(div);

    // pip install button
    const pip = document.createElement("div");
    pip.id = "slx-mob-pip";
    pip.textContent = "$ pip install statslibx  ⎘";
    pip.addEventListener("click", () => {
      navigator.clipboard.writeText("pip install statslibx").then(showToast);
    });
    menu.appendChild(pip);

    // Insert right after <nav>
    const nav = document.querySelector("nav");
    if (nav) nav.insertAdjacentElement("afterend", menu);
  }

  /* ─── 5. Toast notification ─────────────────────────────────── */
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
    setTimeout(() => t.classList.remove("show"), 2200);
  }

  /* ─── 6. Toggle / close logic ───────────────────────────────── */
  function toggleMenu() {
    const btn  = document.getElementById("slx-hamburger");
    const menu = document.getElementById("slx-mob-nav");
    if (!btn || !menu) return;
    const isOpen = menu.classList.toggle("open");
    btn.classList.toggle("open", isOpen);
    btn.setAttribute("aria-expanded", isOpen);
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

  // Close on Escape key
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeMenu();
  });

  // Close when clicking outside the menu
  document.addEventListener("click", (e) => {
    const menu = document.getElementById("slx-mob-nav");
    const btn  = document.getElementById("slx-hamburger");
    if (!menu || !menu.classList.contains("open")) return;
    if (!menu.contains(e.target) && !btn.contains(e.target)) closeMenu();
  });

  /* ─── 7. Init ────────────────────────────────────────────────── */
  function init() {
    injectCSS();
    buildHamburger();
    buildMobileMenu();
    buildToast();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

})();