"""Exercise the browser workflow in simulation; --url inspects a running service read-only."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import httpx
from playwright.sync_api import expect, sync_playwright

from piper_capture.validate import validate_dataset

ROOT = Path(__file__).resolve().parents[1]


def inspect_layout(page):
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Page overflows horizontally"
    overflow = page.locator("button:visible, .joint-value:visible, .stat:visible").evaluate_all(
        "els => els.filter(e => e.scrollWidth > e.clientWidth + 2).map(e => e.outerHTML)"
    )
    assert not overflow, overflow


def camera_pixels(page):
    return page.locator(".camera-frame img").evaluate_all("""imgs => imgs.map(img => {
      const c = document.createElement('canvas'); c.width = 64; c.height = 48;
      const ctx = c.getContext('2d'); ctx.drawImage(img,0,0,64,48);
      const pixels = ctx.getImageData(0,0,64,48).data;
      const colors = new Set(); let sum = 0;
      for(let i=0;i<pixels.length;i+=4) {
        colors.add(`${pixels[i]},${pixels[i+1]},${pixels[i+2]}`);
        sum += pixels[i]+pixels[i+1]+pixels[i+2];
      }
      return {colors:colors.size,sum,width:img.naturalWidth,height:img.naturalHeight};
    })""")


def wait_js(page, expression, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if page.evaluate(expression):
            return
        page.wait_for_timeout(50)
    raise AssertionError(f"Browser condition timed out: {expression}")


def wait_capture(page, predicate):
    wait_js(
        page,
        f"async () => {{ const s = await (await fetch('/api/status')).json(); return {predicate}; }}",
    )


def exercise(page, output):
    wait_capture(page, "s.ready")
    page.locator("#task-input").fill("Retrieve the blue book")
    page.locator("#record-button").click()
    wait_capture(page, "s.frames >= 30")
    page.reload()
    wait_capture(page, "s.state === 'RECORDING' && s.frames >= 50")
    expect(page.locator("#record-button")).to_contain_text("停止并保存")
    page.screenshot(path=str(output / "desktop-recording.png"), full_page=True)
    page.locator("#stop-review-button").click()
    expect(page.locator(".pending-item")).to_have_count(1)
    page.locator("[data-review][data-kind=pending]").click()
    expect(page.locator("#review-content")).to_be_visible()
    wait_js(page, "[...document.querySelectorAll('video')].every(v => v.readyState >= 2)")
    expect(page.locator("#accept-button")).to_be_enabled()
    page.locator("#play-button").click()
    wait_js(page, "document.querySelector('video').currentTime > 0.25")
    assert page.locator("video").evaluate_all("vs => Math.abs(vs[0].currentTime-vs[1].currentTime) < 0.15")
    page.locator("#play-button").click()
    page.locator("#seek-input").fill("5")
    page.locator("#seek-input").dispatch_event("change")
    page.locator("#step-forward").click()
    wait_js(page, "Math.abs(document.querySelector('video').currentTime - 6/30) < 0.01")
    page.locator("#step-back").click()
    wait_js(page, "Math.abs(document.querySelector('video').currentTime - 5/30) < 0.01")
    page.locator("#playback-speed").select_option("2")
    assert page.locator("video").evaluate_all("vs => vs.every(v => v.playbackRate === 2)")
    page.locator("#loop-toggle").check()
    last = int(page.locator("#seek-input").get_attribute("max"))
    page.locator("#seek-input").fill(str(last - 4))
    page.locator("#seek-input").dispatch_event("change")
    page.locator("#play-button").click()
    wait_js(
        page, "document.querySelector('video').currentTime < 0.5 && !document.querySelector('video').paused"
    )
    page.locator("#play-button").click()
    page.locator("#joint-select").select_option("6")
    assert page.locator("#trajectory-chart").evaluate("""c => {
      const p = c.getContext('2d').getImageData(0,0,c.width,c.height).data;
      return p.filter((v,i) => i%4 === 3 && v>0).length > 500;
    }"""), "Blank trajectory chart"
    inspect_layout(page)
    page.screenshot(path=str(output / "desktop-review.png"), full_page=True)
    page.set_viewport_size({"width": 390, "height": 844})
    inspect_layout(page)
    page.screenshot(path=str(output / "mobile-review.png"), full_page=True)
    page.locator("#accept-button").click()
    expect(page.locator("#review-dialog")).not_to_be_visible()
    wait_capture(page, "s.history.length === 1 && s.pending.length === 0")
    page.locator(".tab[data-view=history]").click()
    expect(page.locator("#history-body tr")).to_have_count(1)
    page.locator("#history-search").fill("unmatched")
    expect(page.locator("#history-body tr")).to_have_count(0)
    page.locator("#history-search").fill("blue")
    expect(page.locator("#history-body tr")).to_have_count(1)
    page.locator("[data-review][data-kind=saved]").click()
    expect(page.locator("#review-content")).to_be_visible()
    expect(page.locator("#review-actions")).not_to_be_visible()
    page.locator("#close-review").click()
    page.locator(".tab[data-view=capture]").click()
    expect(page.locator("#record-button")).to_be_enabled()
    page.locator("#task-input").fill("Discard this test episode")
    page.locator("#record-button").click()
    wait_capture(page, "s.frames >= 35")
    page.locator("#stop-review-button").click()
    page.locator("[data-review][data-kind=pending]").click()
    expect(page.locator("#discard-button")).to_be_enabled()
    page.locator("#discard-button").click()
    page.locator("#cancel-discard").click()
    expect(page.locator("#review-dialog")).to_be_visible()
    page.locator("#discard-button").click()
    page.locator("#confirm-discard").click()
    expect(page.locator("#review-dialog")).not_to_be_visible()
    wait_capture(page, "s.history.length === 1 && s.pending.length === 0")
    exercise_keyboard_and_selection(page, output)


def exercise_keyboard_and_selection(page, output):
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.locator("#task-input").fill("Keyboard demonstration")
    page.locator("#task-input").press("Space")
    wait_capture(page, "s.state === 'READY'")
    assert page.locator("#task-input").input_value().endswith(" ")
    page.locator("#view-title").click()
    page.keyboard.down("Space")
    page.keyboard.down("Space")
    page.keyboard.up("Space")
    wait_capture(page, "s.state === 'RECORDING' && s.frames >= 35")
    page.keyboard.press("Space")
    wait_capture(page, "s.success_count === 2 && s.pending.length === 0")
    expect(page.locator("#success-count")).to_have_text("2")
    expect(page.locator("#record-button")).to_be_enabled()
    page.keyboard.press("Space")
    wait_capture(page, "s.frames >= 35")
    page.keyboard.press("l")
    wait_capture(page, "s.failure_count === 1 && s.pending.length === 0")
    expect(page.locator("#failure-count")).to_have_text("1")
    inspect_layout(page)
    page.screenshot(path=str(output / "desktop-outcomes.png"), full_page=True)
    page.set_viewport_size({"width": 390, "height": 844})
    inspect_layout(page)
    page.screenshot(path=str(output / "mobile-outcomes.png"), full_page=True)
    page.locator(".tab[data-view=history]").click()
    page.locator("#history-search").fill("")
    expect(page.locator("#history-body tr")).to_have_count(3)
    expect(page.locator(".export-episode")).to_have_text(["--"] * 3)
    page.locator('[data-select="saved:1"]').check()
    expect(page.locator('[data-export-preview="saved:1"] .export-episode')).to_have_text("episode_000000")
    page.locator('[data-select="saved:0"]').check()
    expect(page.locator('[data-export-preview="saved:1"] .export-episode')).to_have_text("episode_000001")
    page.locator("#history-search").fill("000001")
    expect(page.locator("#history-body tr")).to_have_count(1)
    expect(page.locator('[data-export-preview="saved:1"] .export-episode')).to_have_text("episode_000001")
    page.locator("#select-all").uncheck()
    expect(page.locator('[data-export-preview="saved:1"] .export-episode')).to_have_text("--")
    page.locator("#history-search").fill("")
    expect(page.locator('[data-export-preview="saved:0"] .export-episode')).to_have_text("episode_000000")
    page.locator('[data-select="saved:1"]').check()
    page.locator('[data-select="saved:0"]').uncheck()
    expect(page.locator('[data-export-preview="saved:1"] .export-episode')).to_have_text("episode_000000")
    page.locator("#selection-limit").fill("2")
    page.locator("#select-number").click()
    expect(page.locator("#selection-count")).to_have_text("已选 2 条")
    page.locator("#select-all").check()
    expect(page.locator("#selection-count")).to_have_text("已选 3 条")
    preview = page.locator("[data-export-preview]").evaluate_all("""els => Object.fromEntries(els.map(el => [
        el.dataset.exportPreview,
        el.querySelector('.export-directory').textContent + el.querySelector('.export-episode').textContent
    ]))""")
    inspect_layout(page)
    page.screenshot(path=str(output / "mobile-selection.png"), full_page=True)
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.screenshot(path=str(output / "desktop-selection.png"), full_page=True)
    with page.expect_download() as downloading:
        page.locator("#export-selected").click()
        page.locator("#download-export").click()
    archive = output / "selection.zip"
    downloading.value.save_as(archive)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(output / "exported")
    mapping = json.loads((output / "exported/selection.json").read_text())["episodes"]
    assert preview == {f"{row['kind']}:{row['source_id']}": row["export_name"] for row in mapping}
    assert validate_dataset(output / "exported/success")["episodes"] == 2
    assert len(list((output / "exported/failed").glob("*/manifest.json"))) == 1
    page.locator("#select-all").uncheck()
    expect(page.locator(".export-episode")).to_have_text(["--"] * 3)
    page.locator("#outcome-filter").select_option("failure")
    expect(page.locator("#history-body tr")).to_have_count(1)
    page.locator("[data-review][data-kind=failed]").click()
    expect(page.locator("#review-content")).to_be_visible()
    page.locator("#review-task").click()
    page.keyboard.press("Space")
    wait_capture(page, "s.state === 'READY'")
    page.locator("#close-review").click()
    page.locator("#select-all").check()
    page.locator("#delete-selected").click()
    page.locator("#cancel-delete").click()
    expect(page.locator("#history-body tr")).to_have_count(1)
    page.locator("#delete-selected").click()
    page.locator("#confirm-delete").click()
    wait_capture(page, "s.failure_count === 0 && s.success_count === 2")
    page.locator("#outcome-filter").select_option("all")
    expect(page.locator("#history-body tr")).to_have_count(2)
    page.locator("#history-body [data-select]").first.check()
    page.locator("#delete-selected").click()
    page.locator("#confirm-delete").click()
    wait_capture(page, "s.history.length === 1 && s.success_count === 1")
    expect(page.locator("#selection-count")).to_have_text("已选 0 条")
    page.locator(".tab[data-view=capture]").click()
    exercise_batch_and_robot(page, output)


def exercise_batch_and_robot(page, output):
    page.locator("#batch-mode").check()
    for index in range(3):
        page.locator("#task-input").fill(f"Batch episode {index}")
        page.locator("#record-button").click()
        wait_capture(page, "s.frames >= 35")
        page.locator("#record-button").click()
        expect(page.locator(".pending-item")).to_have_count(index + 1)
        expect(page.locator("#record-button")).to_be_enabled()
    expect(page.locator("#success-count")).to_have_text("1")
    page.locator("#pending-select-all").check()
    page.locator("#batch-accept").click()
    page.locator("#cancel-batch").click()
    page.locator("[data-pending-select]").last.uncheck()
    page.locator("#batch-accept").click()
    page.locator("#confirm-batch").click()
    expect(page.locator("#success-count")).to_have_text("3")
    expect(page.locator(".pending-item")).to_have_count(1)
    page.locator("#pending-select-all").check()
    page.locator("#batch-failure").click()
    page.locator("#confirm-batch").click()
    expect(page.locator("#failure-count")).to_have_text("1")
    page.reload()
    expect(page.locator("#success-count")).to_have_text("3")
    expect(page.locator("#failure-count")).to_have_text("1")
    # Simulate an old, delayed poll response arriving after a successful save.
    page.evaluate("applyStatus({...status,sequence:status.sequence-1,success_count:0}); render()")
    expect(page.locator("#success-count")).to_have_text("3")
    page.locator(".tab[data-view=history]").click()
    page.locator("#select-all").check()
    page.locator("#export-selected").click()
    directory = (output / "local_export").resolve()
    page.locator("#export-destination").fill(str(directory))
    page.screenshot(path=str(output / "export-dialog.png"), full_page=True)
    page.locator("#confirm-export").click()
    expect(page.locator("#export-result")).to_contain_text(str(directory))
    assert validate_dataset(directory / "success")["episodes"] == 3
    assert (directory / "failed/episode_000000/episode_000000.parquet").is_file()
    page.locator("[data-review][data-kind=saved]").first.click()
    expect(page.locator("#review-content")).to_be_visible()
    page.locator("#robot-replay-button").click()
    expect(page.locator("#robot-speed")).to_have_value("1")
    expect(page.locator("#robot-start")).to_be_disabled()
    page.locator("#replay-check").click()
    expect(page.locator("#robot-plan")).to_contain_text("预计")
    expect(page.locator("#robot-plan")).to_contain_text("原值回放")
    expect(page.locator("#robot-plan")).to_contain_text("无软件位置限位")
    expect(page.locator("#robot-plan")).to_contain_text("1 倍（采集原速）")
    expect(page.locator("#robot-plan")).to_contain_text("终点保持 0.5 秒")
    page.locator("#robot-speed").select_option("0.5")
    expect(page.locator("#robot-poses table")).to_have_count(0)
    page.locator("#replay-check").click()
    expect(page.locator("#robot-plan")).to_contain_text("0.5 倍")
    page.locator("#robot-speed").select_option("1")
    page.locator("#replay-check").click()
    expect(page.locator("#robot-plan")).to_contain_text("1 倍（采集原速）")
    expect(page.locator("#robot-poses th").nth(3)).to_have_text("单位")
    page.locator("#robot-field").select_option("observation.state")
    expect(page.locator("#robot-poses table")).to_have_count(0)
    expect(page.locator("#robot-start")).to_be_disabled()
    page.locator("#replay-check").click()
    expect(page.locator("#robot-plan")).to_contain_text("原值回放")
    page.locator("#robot-field").select_option("action")
    page.locator("#robot-gripper").check()
    page.locator("#replay-check").click()
    expect(page.locator("#robot-plan")).to_contain_text("原值回放")
    expect(page.locator("#robot-plan")).to_contain_text("六轴与夹爪")
    page.locator("#leader-disconnected").check()
    page.locator("#workspace-clear").check()
    expect(page.locator("#robot-start")).to_be_enabled()
    inspect_layout(page)
    page.screenshot(path=str(output / "robot-replay.png"), full_page=True)
    page.set_viewport_size({"width": 390, "height": 844})
    inspect_layout(page)
    page.screenshot(path=str(output / "robot-replay-mobile.png"), full_page=True)
    page.locator("#robot-start").click()
    expect(page.locator("#robot-progress")).to_contain_text("正在回起点", timeout=10000)
    expect(page.locator("#robot-stop")).to_be_enabled()
    expect(page.locator("#robot-stop")).to_be_in_viewport(ratio=1)
    expect(page.locator("#robot-progress")).to_be_in_viewport(ratio=1)
    inspect_layout(page)
    page.screenshot(path=str(output / "robot-returning-mobile.png"), full_page=True)
    expect(page.locator("#robot-progress")).to_contain_text("回放中", timeout=10000)
    page.locator("#robot-stop").click()
    expect(page.locator("#robot-progress")).to_contain_text("回放已停止")
    page.locator("#close-robot").click()
    page.locator("#close-review").click()
    page.locator(".tab[data-view=capture]").click()
    expect(page.locator("#record-button")).to_be_disabled()
    page.locator("#devices-button").click()
    page.locator("#reconnect-button").click()
    page.locator("#close-devices").click()
    expect(page.locator("#record-button")).to_be_enabled()


def verify(url, output, *, mutate):
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        page.goto(url)
        expect(page.locator(".camera-frame.live")).to_have_count(2, timeout=20000)
        before = camera_pixels(page)
        assert all(c["colors"] > 30 and c["width"] == 640 and c["height"] == 480 for c in before), before
        inspect_layout(page)
        page.screenshot(path=str(output / "desktop-capture.png"), full_page=True)
        if mutate:
            page.wait_for_timeout(250)
            assert before != camera_pixels(page), "Camera images do not update"
            exercise(page, output)
        page.set_viewport_size({"width": 390, "height": 844})
        inspect_layout(page)
        page.screenshot(path=str(output / "mobile-capture.png"), full_page=True)
        page.locator("#devices-button").click()
        expect(page.locator(".device-row")).to_have_count(10)
        inspect_layout(page)
        page.screenshot(path=str(output / "mobile-devices.png"), full_page=True)
        if mutate:
            page.locator("#reconnect-button").click()
            wait_capture(page, "s.ready")
        page.locator("#close-devices").click()
        page.set_viewport_size({"width": 320, "height": 740})
        inspect_layout(page)
        page.set_viewport_size({"width": 1920, "height": 1080})
        inspect_layout(page)
        browser.close()
    assert not errors, errors
    return {"browser_errors": errors, "cameras": before, "screenshots": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Inspect this running service without recording or reconnecting")
    parser.add_argument("--output", type=Path, default=ROOT / "work_dirs" / f"web_check_{int(time.time())}")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    process = None
    try:
        url = args.url
        if not url:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            with (args.output / "server.log").open("w") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "piper_capture.cli",
                        "web",
                        "--mock",
                        "--config",
                        str(ROOT / "configs/right_arm.yaml"),
                        "--output",
                        str(args.output / "dataset"),
                        "--port",
                        str(port),
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            url = f"http://127.0.0.1:{port}"
        with httpx.Client(trust_env=False) as client:
            for _ in range(100):
                try:
                    if client.get(f"{url}/api/status").is_success:
                        break
                except httpx.TransportError:
                    pass
                if process and process.poll() is not None:
                    raise RuntimeError((args.output / "server.log").read_text())
                time.sleep(0.1)
            else:
                raise RuntimeError("Web service did not start")
        result = verify(url, args.output, mutate=not args.url)
    finally:
        if process:
            process.terminate()
            process.wait(timeout=20)
    if not args.url:
        result["dataset"] = validate_dataset(args.output / "dataset")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
