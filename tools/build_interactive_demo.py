#!/usr/bin/env python3
"""Build a local interactive webpage showcasing a VLM-R3 reasoning run."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import textwrap
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import TCPServer
from typing import Any, Dict

PALETTE = [
    "#19F7FF", "#FF6B9A", "#8CFF55", "#FDBF2D",
    "#A855F7", "#2FD3C9", "#FF7847", "#63A4FF",
]


def load_payload(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_base_image(payload_path: Path, payload: Dict[str, Any]) -> Path:
    base_image_entry = payload.get("base_image", {})
    base_path = Path(base_image_entry.get("path", ""))
    if not base_path.is_absolute():
        base_path = (payload_path.parent / base_path).resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Base image not found: {base_path}")
    return base_path


def build_html(data: Dict[str, Any], json_blob: str) -> str:
    question = data.get("question", "Interactive reasoning demo")
    answer = data.get("answer") or "(answer unavailable)"
    media_kind = data.get("media_type", "image")

    styles = textwrap.dedent(
        """
        body {
            background: radial-gradient(circle at 10% 20%, #0b1220, #04050a 65%);
            font-family: 'Inter', 'Segoe UI', sans-serif;
            color: #E4EEFF;
            margin: 0;
            min-height: 100vh;
        }
        a { color: #7BE6FF; }
        .layout {
            max-width: 1280px;
            margin: 0 auto;
            padding: 32px 24px 64px;
        }
        header {
            margin-bottom: 24px;
        }
        .tagline {
            display: inline-flex;
            align-items: center;
            gap: 12px;
            padding: 10px 16px;
            border-radius: 999px;
            background: rgba(25, 247, 255, 0.08);
            border: 1px solid rgba(123, 230, 255, 0.2);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-size: 0.75rem;
            color: #7BE6FF;
        }
        h1 {
            font-size: 2rem;
            margin: 18px 0 8px;
            color: #FFFFFF;
        }
        .answer {
            color: #7CFFB2;
            font-weight: 600;
            letter-spacing: 0.03em;
        }
        .content {
            display: grid;
            gap: 24px;
            grid-template-columns: minmax(360px, 1.35fr) minmax(320px, 1fr);
        }
        .image-stage {
            position: relative;
            border-radius: 22px;
            overflow: hidden;
            background: linear-gradient(135deg, rgba(25,247,255,0.12), rgba(72,56,255,0.08));
            border: 1px solid rgba(123, 230, 255, 0.2);
            box-shadow: 0 18px 60px rgba(12, 20, 40, 0.55);
        }
        .image-stage img {
            width: 100%;
            display: block;
        }
        .image-stage svg {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }
        .timeline {
            background: rgba(6, 11, 22, 0.92);
            border-radius: 18px;
            border: 1px solid rgba(123, 230, 255, 0.18);
            padding: 18px 20px;
            box-shadow: 0 14px 40px rgba(5, 12, 28, 0.45);
        }
        .status-line {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            font-size: 0.9rem;
            color: #B7C9FF;
        }
        .steps {
            max-height: 520px;
            overflow-y: auto;
            padding-right: 6px;
        }
        .step-item {
            padding: 12px 14px 12px 16px;
            margin-bottom: 12px;
            border-radius: 14px;
            border-left: 4px solid transparent;
            background: rgba(9, 15, 28, 0.65);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border 0.2s ease;
            cursor: pointer;
        }
        .step-item:hover {
            transform: translateX(4px);
            box-shadow: 0 8px 20px rgba(25, 247, 255, 0.08);
        }
        .step-item.active {
            border-left-color: var(--step-color);
            background: linear-gradient(90deg, rgba(25,247,255,0.22), rgba(12,20,36,0.72));
            box-shadow: 0 12px 35px rgba(25, 247, 255, 0.18);
        }
        .step-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
            font-weight: 600;
        }
        .badge {
            font-size: 0.78rem;
            background: rgba(255,255,255,0.1);
            padding: 2px 10px;
            border-radius: 999px;
        }
        .step-body {
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 0.85rem;
            line-height: 1.45;
            white-space: normal;
        }
        .controls {
            margin: 16px 0 8px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .controls button {
            padding: 10px 18px;
            border-radius: 999px;
            border: none;
            background: linear-gradient(135deg, #19F7FF, #7B61FF);
            color: #02131e;
            font-weight: 600;
            cursor: pointer;
            letter-spacing: 0.05em;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .controls button.playing {
            background: linear-gradient(135deg, #FF6B9A, #FF8847);
            color: #04050a;
        }
        .controls button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 22px rgba(25, 247, 255, 0.25);
        }
        .controls input[type=range] {
            flex: 1;
        }
        details {
            margin-top: 16px;
            background: rgba(9, 13, 24, 0.85);
            border-radius: 12px;
            border: 1px solid rgba(123, 230, 255, 0.12);
            padding: 12px 16px;
        }
        details summary {
            cursor: pointer;
            font-weight: 600;
            color: #7BE6FF;
        }
        details pre {
            max-height: 280px;
            overflow-y: auto;
            white-space: pre-wrap;
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 0.8rem;
            color: #DEE8FF;
        }
        .footer-note {
            margin-top: 32px;
            text-align: center;
            font-size: 0.8rem;
            color: #7084B0;
        }
        @media (max-width: 992px) {
            .content {
                grid-template-columns: 1fr;
            }
            .steps {
                max-height: 360px;
            }
        }
        """
    )

    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>VLM-R3 Interactive Demo</title>
  <style>{styles}</style>
</head>
<body>
  <div class=\"layout\">
    <header>
      <span class=\"tagline\">VLM-R3 reasoning · {media_kind}</span>
      <h1 id=\"question\">{question}</h1>
      <div class=\"answer\">Predicted answer: <span id=\"answer\">{answer}</span></div>
    </header>
    <main class=\"content\">
      <section>
        <div class=\"image-stage\">
          <img id=\"base-image\" alt=\"Reasoning canvas\" />
          <svg id=\"overlay\"></svg>
        </div>
      </section>
      <aside class=\"timeline\">
        <div class=\"status-line\">
          <span id=\"status\"></span>
          <span id=\"media-kind\"></span>
        </div>
        <div class=\"steps\" id=\"step-list\"></div>
        <div class=\"controls\">
          <button id=\"play-toggle\">Auto-play</button>
          <input id=\"step-range\" type=\"range\" min=\"1\" value=\"1\" />
        </div>
        <details>
          <summary>Full reasoning transcript</summary>
          <pre id=\"full-response\"></pre>
        </details>
      </aside>
    </main>
    <div class=\"footer-note\">
      Generated with the VLM-R3 interactive demo builder.
    </div>
  </div>
  <script id=\"payload-data\" type=\"application/json\">{json_blob}</script>
  <script>
    const palette = {json.dumps(data.get('palette', PALETTE))};
    const payload = JSON.parse(document.getElementById('payload-data').textContent);
    const steps = payload.steps || [];
    const overlay = document.getElementById('overlay');
    const baseImage = document.getElementById('base-image');
    const stepList = document.getElementById('step-list');
    const status = document.getElementById('status');
    const mediaKind = document.getElementById('media-kind');
    const slider = document.getElementById('step-range');
    const playToggle = document.getElementById('play-toggle');
    const fullResponse = document.getElementById('full-response');

    baseImage.src = payload.base_image.path;
    overlay.setAttribute('viewBox', `0 0 ${payload.base_image.width} ${payload.base_image.height}`);
    slider.max = Math.max(steps.length, 1);
    mediaKind.textContent = `${payload.media_type || 'image'} · ${steps.length} steps`;
    fullResponse.textContent = payload.full_response || '';

    const stepNodes = [];
    let activeIndex = 0;
    let timer = null;

    const escapeHtml = (str) => str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    const formatStepText = (text) => escapeHtml(text || '').replace(/\n/g, '<br />');

    const renderStep = (index) => {
      if (!steps.length) {
        status.textContent = 'No reasoning steps recorded.';
        overlay.innerHTML = '';
        return;
      }
      activeIndex = (index + steps.length) % steps.length;
      const step = steps[activeIndex];
      slider.value = activeIndex + 1;
      status.textContent = `Step ${activeIndex + 1} / ${steps.length}`;

      stepNodes.forEach((node, idx) => {
        node.classList.toggle('active', idx === activeIndex);
      });

      overlay.innerHTML = '';
      if (Array.isArray(step.bbox) && step.bbox.length === 4) {
        const [x1, y1, x2, y2] = step.bbox;
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', x1);
        rect.setAttribute('y', y1);
        rect.setAttribute('width', x2 - x1);
        rect.setAttribute('height', y2 - y1);
        rect.setAttribute('rx', 8);
        rect.setAttribute('ry', 8);
        rect.setAttribute('fill', 'rgba(25, 247, 255, 0.12)');
        rect.setAttribute('stroke', palette[activeIndex % palette.length]);
        rect.setAttribute('stroke-width', 3);
        overlay.appendChild(rect);

        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', x1 + 10);
        label.setAttribute('y', Math.max(y1 - 10, 18));
        label.setAttribute('fill', palette[activeIndex % palette.length]);
        label.setAttribute('font-size', '18');
        label.setAttribute('font-weight', '700');
        label.textContent = `#${activeIndex + 1}`;
        overlay.appendChild(label);
      }
    };

    const pause = () => {
      if (timer) {
        clearInterval(timer);
        timer = null;
        playToggle.classList.remove('playing');
        playToggle.textContent = 'Auto-play';
      }
    };

    const play = () => {
      if (timer || !steps.length) {
        pause();
        return;
      }
      playToggle.classList.add('playing');
      playToggle.textContent = 'Pause';
      timer = setInterval(() => {
        renderStep(activeIndex + 1);
      }, 2600);
    };

    steps.forEach((step, idx) => {
      const container = document.createElement('div');
      container.className = 'step-item';
      container.style.setProperty('--step-color', palette[idx % palette.length]);
      container.innerHTML = `
        <div class="step-header">
          <span class="badge">#${idx + 1}</span>
          <span>${step.bbox ? 'crop' : 'think'}</span>
        </div>
        <div class="step-body">${formatStepText(step.text)}</div>
      `;
      container.addEventListener('mouseenter', () => { renderStep(idx); pause(); });
      container.addEventListener('click', () => { renderStep(idx); pause(); });
      stepList.appendChild(container);
      stepNodes.push(container);
    });

    slider.addEventListener('input', (event) => {
      renderStep(Number(event.target.value) - 1);
      pause();
    });

    playToggle.addEventListener('click', () => {
      if (timer) {
        pause();
      } else {
        play();
      }
    });

    document.getElementById('question').textContent = payload.question || '{question}';
    document.getElementById('answer').textContent = payload.answer || '{answer}';

    if (steps.length) {
      renderStep(0);
    } else {
      renderStep(-1);
    }
  </script>
</body>
</html>
"""
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True, help="Path to the JSON payload exported by the notebook.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where the web demo will be written.")
    parser.add_argument("--serve", action="store_true", help="Start a local HTTP server after generating the files.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the optional HTTP server.")
    args = parser.parse_args()

    payload_path = args.payload.resolve()
    if not payload_path.exists():
        raise FileNotFoundError(f"Payload file not found: {payload_path}")

    payload = load_payload(payload_path)
    base_image_path = resolve_base_image(payload_path, payload)

    output_dir = args.output_dir.resolve()
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    payload_copy_path = assets_dir / "payload.json"
    with payload_copy_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    base_dest = assets_dir / f"base_image{base_image_path.suffix}"
    shutil.copy2(base_image_path, base_dest)

    web_data = {
        "question": payload.get("question"),
        "answer": payload.get("answer"),
        "media_type": payload.get("media_type"),
        "palette": payload.get("palette", PALETTE),
        "base_image": {
            "path": str(base_dest.relative_to(output_dir)),
            "width": payload.get("base_image", {}).get("width"),
            "height": payload.get("base_image", {}).get("height"),
        },
        "steps": payload.get("steps", []),
        "full_response": payload.get("full_response"),
    }

    html = build_html(web_data, json.dumps(web_data, ensure_ascii=False, indent=2))
    html_path = output_dir / "index.html"
    html_path.write_text(html, encoding="utf-8")

    print(f"Interactive demo written to {html_path}")
    if args.serve:
        os.chdir(output_dir)
        handler = SimpleHTTPRequestHandler
        with TCPServer(("0.0.0.0", args.port), handler) as httpd:
            print(f"Serving on http://localhost:{args.port}/ (press Ctrl+C to stop)")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nServer stopped.")


if __name__ == "__main__":
    main()
