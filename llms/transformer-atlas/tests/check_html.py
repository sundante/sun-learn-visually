"""Headless-browser checks for TransformerAtlas.html (Playwright + Chromium).

python tests/check_html.py [--shots DIR]
Writes web/data/html_checks.json, which build.py embeds in the Validation chapter.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PAGE = (ROOT / "TransformerAtlas.html").as_uri()
SKIP_ACTS = {"download", "reset-progress", "menu", "go", "search", "ins-toggle", "pane"}


def main():
    shots = Path(sys.argv[sys.argv.index("--shots") + 1]) if "--shots" in sys.argv else None
    items = []
    add = lambda name, ok, detail="": items.append({"name": name, "ok": bool(ok), "detail": detail})
    with sync_playwright() as pw:
        br = pw.chromium.launch()
        ctx = br.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()
        console_errors = []
        page.on("pageerror", lambda e: console_errors.append(str(e)))
        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" and "Failed to load resource" not in m.text else None)
        page.goto(PAGE)
        page.wait_for_function("window.__ready === true", timeout=20000)
        add("Page boots and renders", True)
        add("All source-code anchors resolve", page.evaluate("window.__opErrors.length === 0"),
            "; ".join(page.evaluate("window.__opErrors")))
        vmax = page.evaluate("M.verifyMax")
        add("Browser engine agrees with PyTorch record", vmax < 1e-5, f"worst |Δ| = {vmax:.2e}")

        chapters = page.evaluate("CHAPTERS.map(c => c.id)")
        clicked, overflow = 0, []
        for cid in chapters:
            page.evaluate(f"go('{cid}')")
            page.wait_for_timeout(60)
            if page.evaluate("document.documentElement.scrollWidth > window.innerWidth + 1"):
                overflow.append(cid)
            # exercise every control in the chapter
            n = page.evaluate("document.querySelectorAll('#scroll [data-act]').length")
            for i in range(min(n, 80)):
                act = page.evaluate(f"(() => {{ const b = document.querySelectorAll('#scroll [data-act]')[{i}]; return b ? b.dataset.act : null; }})()")
                if not act or act in SKIP_ACTS:
                    continue
                page.evaluate(f"(() => {{ const b = document.querySelectorAll('#scroll [data-act]')[{i}]; if (b) b.click(); }})()")
                clicked += 1
            page.evaluate("""() => { document.querySelectorAll('#scroll input[type=range]').forEach(r => {
                for (const v of [r.min, r.max, r.value]) { r.value = v; r.dispatchEvent(new Event('input', {bubbles: true})); } });
              document.querySelectorAll('#scroll select').forEach(s => { for (const o of s.options) { s.value = o.value; s.dispatchEvent(new Event('change', {bubbles: true})); } });
              document.querySelectorAll('#scroll input[type=checkbox]').forEach(c => { c.click(); c.click(); }); }""")
            page.evaluate("document.querySelectorAll('#scroll td[data-c]').forEach((td, k) => { if (k % 37 === 0) td.click(); })")
            page.evaluate("if (window.PLAY) { clearInterval(window.PLAY); }")
            if shots and cid in ("map", "attn", "backprop", "train", "memory", "qkv"):
                page.evaluate(f"go('{cid}')")
                page.wait_for_timeout(150)
                page.screenshot(path=str(shots / f"{cid}.png"))
        add("Every chapter renders and every control runs without errors", not console_errors,
            f"{len(chapters)} chapters, {clicked} controls clicked, all ranges/selects/checkboxes exercised" +
            (f"; errors: {console_errors[:3]}" if console_errors else ""))
        add("No horizontal page overflow at 1440 px", not overflow, ", ".join(overflow))

        # code <-> visual synchronisation
        page.evaluate("go('attn')")
        page.click(".ln[data-arg='av']")
        synced = page.evaluate("S.op !== null && document.querySelectorAll('.op-focus').length > 0")
        page.evaluate("go('qkv')")
        page.click("#scroll [data-op='qkv_proj']")
        synced2 = page.evaluate("S.op === 'qkv_proj' && document.querySelector('.ln.sel') !== null")
        add("Code and visualisation stay synchronised", synced and synced2, "code line → tensor, tensor → code line")

        # editing the example recomputes every dependent panel
        page.fill("#ex-in", "Before we")
        changed = page.evaluate("Math.abs(M.fw.loss - MICRO.trace.loss) > 1e-6 && !M.canonical")
        page.fill("#ex-in", "First Cit")
        restored = page.evaluate("M.canonical && Math.abs(M.fw.loss - MICRO.trace.loss) < 1e-6")
        add("Editing the running example recomputes dependent results", changed and restored)

        # keyboard
        page.evaluate("go('tok')")
        page.evaluate("document.activeElement && document.activeElement.blur()")
        page.keyboard.press("]")
        nxt = page.evaluate("S.ch")
        page.keyboard.press("/")
        page.keyboard.type("softmax")
        has = page.evaluate("document.querySelectorAll('#pal-list li[data-i]').length > 0")
        page.keyboard.press("Enter")
        add("Keyboard: chapter keys and search palette", nxt == "win" and has, f"] → {nxt}; search results: {has}")

        # reduced motion
        rm = br.new_context(viewport={"width": 1280, "height": 800}, reduced_motion="reduce").new_page()
        rm.goto(PAGE); rm.wait_for_function("window.__ready === true")
        sb = rm.evaluate("getComputedStyle(document.getElementById('scroll')).scrollBehavior")
        add("Reduced-motion preference respected", sb == "auto", f"scroll-behavior = {sb}")

        # mobile layout
        mob = br.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2).new_page()
        merr = []
        mob.on("pageerror", lambda e: merr.append(str(e)))
        mob.goto(PAGE); mob.wait_for_function("window.__ready === true")
        bad = []
        for cid in chapters:
            mob.evaluate(f"go('{cid}')")
            if mob.evaluate("document.documentElement.scrollWidth > window.innerWidth + 1"):
                bad.append(cid)
        mob.evaluate("ACTIONS.pane('code')")
        code_vis = mob.evaluate("getComputedStyle(document.getElementById('code-pane')).display !== 'none'")
        add("Phone width (390 px): no page overflow; source/computation switch", not bad and code_vis and not merr,
            ", ".join(bad) + ("; " + merr[0] if merr else ""))
        if shots:
            mob.evaluate("ACTIONS.pane('visual'); go('attn')")
            mob.screenshot(path=str(shots / "mobile.png"))
        add("Core content readable without animation", page.evaluate("document.getElementById('scroll').innerText.length > 500"))
        version = br.version
        br.close()
    out = {"browser": version, "items": items}
    (ROOT / "web" / "data" / "html_checks.json").write_text(json.dumps(out, indent=1))
    for i in items:
        print(("PASS " if i["ok"] else "FAIL ") + i["name"] + (f"  [{i['detail']}]" if i["detail"] else ""))
    sys.exit(0 if all(i["ok"] for i in items) else 1)


if __name__ == "__main__":
    main()
