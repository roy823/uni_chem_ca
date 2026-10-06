"""Exercise the local 3D viewer against exported, computed trajectories."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765/")
    parser.add_argument("--output", default="outputs/water_clusters_v04")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            channel="msedge", headless=True, args=["--enable-unsafe-swiftshader"]
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1080}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "console",
            lambda message: errors.append(message.text) if message.type == "error" else None,
        )
        page.goto(args.url)
        page.wait_for_function(
            "document.querySelector('#frame-number').textContent.includes('201')"
        )
        assert not page.locator("#load-error").is_visible()
        assert page.locator("#viewport canvas").count() == 1
        assert page.locator("#atom-count").inner_text() == "7"
        assert page.locator("#performance-body tr").count() == 5
        page.get_by_role("button", name="播放轨迹").click()
        page.wait_for_timeout(800)
        page.get_by_role("button", name="暂停轨迹").click()
        assert float(page.locator("#current-time").inner_text().split()[0]) > 0
        page.get_by_role("slider", name="轨迹帧").press("End")
        assert page.locator("#frame-number").inner_text() == "201 / 201"
        page.get_by_role("slider", name="轨迹帧").press("Home")
        page.get_by_role("checkbox", name="电荷着色").check()
        page.get_by_role("checkbox", name="电荷着色").uncheck()
        page.get_by_role("combobox", name="跟踪氢原子").select_option("1")
        page.get_by_role("combobox", name="跟踪氢原子").select_option("6")
        page.get_by_role("button", name="重置视角").click()
        page.screenshot(path=str(output / "viewer_desktop.png"), full_page=True)
        # Select an actual four-water trajectory and inspect the matching diagnostics.
        options = page.locator("#experiment option").evaluate_all(
            "elements => elements.map(e => ({value:e.value, text:e.textContent}))"
        )
        cluster = next(o["value"] for o in options if "neutral_300k_n_4" in o["value"])
        page.get_by_role("combobox", name="选择实验").select_option(cluster)
        page.wait_for_function("document.querySelector('#atom-count').textContent === '12'")
        assert page.locator("#coordination").inner_text() == "2 / 2 / 2 / 2"
        page.screenshot(path=str(output / "viewer_four_water.png"), full_page=True)
        shared = next(
            o["value"] for o in options if "shared_proton" in o["value"] and "0.1" in o["value"]
        )
        page.get_by_role("combobox", name="选择实验").select_option(shared)
        page.wait_for_function("document.querySelector('#atom-count').textContent === '7'")
        ionic = next(
            o["value"]
            for o in options
            if "oriented_ion_pair" in o["value"] and o["value"].endswith("0.1")
        )
        page.get_by_role("combobox", name="选择实验").select_option(ionic)
        page.wait_for_function("document.querySelector('#atom-count').textContent === '6'")
        slider = page.get_by_role("slider", name="轨迹帧")
        slider.press("Home")
        for _ in range(140):
            slider.press("ArrowRight")
        assert page.locator("#reaction-state").inner_text() == "电子支持的离子候选"
        page.screenshot(path=str(output / "viewer_ionic_pair.png"), full_page=True)
        page.get_by_role("combobox", name="选择实验").select_option(shared)
        page.wait_for_function("document.querySelector('#atom-count').textContent === '7'")
        canvas = page.locator("#viewport canvas").bounding_box()
        page.mouse.move(canvas["x"] + canvas["width"] * 0.45, canvas["y"] + canvas["height"] * 0.5)
        page.mouse.down()
        page.mouse.move(
            canvas["x"] + canvas["width"] * 0.62, canvas["y"] + canvas["height"] * 0.4, steps=12
        )
        page.mouse.up()
        page.get_by_role("button", name="重置视角").click()
        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(300)
        page.screenshot(path=str(output / "viewer_mobile.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        browser.close()
    assert not errors, errors
    report = {
        "browser": "Microsoft Edge / Chromium, headless WebGL2",
        "tested": [
            "real-data load",
            "play/pause",
            "frame scrub",
            "proton selection",
            "charge coloring",
            "four-water scenario",
            "electron-supported ionic observation",
            "camera rotation/reset",
            "mobile layout",
        ],
        "console_errors": errors,
    }
    (output / "viewer_check.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
