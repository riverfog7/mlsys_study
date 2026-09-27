"""Print an executed notebook, expanding every Plotly menu/frame into SVG figures.

Uses the uv-managed Jupyter/websocket libraries and an installed Chromium. It
does not execute experiments or replace missing outputs with another run.
"""
from __future__ import annotations

import argparse
import base64
import copy
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request
import uuid

import nbformat
from nbconvert import HTMLExporter
import websocket

from study02.runtime import STUDY_DIR, digest, write_json


def variants(figure):
    """Expand the control types emitted by visuals.py, retaining the actual data."""
    frames = figure.get("frames", [])
    menus = figure.get("layout", {}).get("updatemenus", [])
    if frames:
        result = []
        for i, frame in enumerate(frames):
            current = copy.deepcopy(figure)
            current["layout"].update(frame.get("layout", {}))
            if frame.get("data"):
                for target, data in zip(frame.get("traces", range(len(frame["data"]))), frame["data"]):
                    current["data"][target].update(data)
            result.append({"label": f"명령 단계 {i + 1}/{len(frames)}", "figure": current})
        return result
    if menus:
        if len(menus) != 1:
            raise ValueError("Multiple control menus need an explicit expansion policy")
        result = []
        for button in menus[0]["buttons"]:
            if button["method"] != "update":
                raise ValueError(f"Unsupported PDF menu action: {button['method']}")
            current = copy.deepcopy(figure)
            args = button["args"]
            for key, values in args[0].items():
                if key != "visible":
                    raise ValueError(f"Unrecognized trace update: {key}")
                for trace, value in zip(current["data"], values):
                    trace[key] = value
            if len(args) > 1:
                current["layout"].update(args[1])
            result.append({"label": str(button["label"]), "figure": current})
        return result
    return [{"label": "", "figure": copy.deepcopy(figure)}]


class DevTools:
    def __init__(self, url):
        self.ws = websocket.create_connection(url, origin="http://127.0.0.1", timeout=60)
        self.index = 0

    def call(self, method, params=None):
        self.index += 1
        self.ws.send(json.dumps({"id": self.index, "method": method, "params": params or {}}))
        while True:
            response = json.loads(self.ws.recv())
            if response.get("id") == self.index:
                if "error" in response:
                    raise RuntimeError(response["error"])
                return response.get("result", {})

    def evaluate(self, expression):
        result = self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True})
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result.get("result", {}).get("value")


PRINT_CSS = r"""
@page { size: A4; margin: 14mm 13mm 18mm; }
@media print {
  body { font-size: 10pt; color: #172033; background: white; }
  .jp-Notebook { padding: 0 !important; width: 100% !important; }
  .jp-Cell { display: block !important; break-inside: auto; margin: 0 0 12px; }
  .jp-InputPrompt, .jp-OutputPrompt, .anchor-link, .modebar { display: none !important; }
  .jp-Cell-inputWrapper, .jp-Cell-outputWrapper, .jp-InputArea, .jp-OutputArea-child { display: block !important; }
  .jp-OutputArea-output, .jp-RenderedHTMLCommon, .jp-InputArea-editor { overflow: visible !important; max-height: none !important; }
  pre, details pre { max-height: none !important; overflow: visible !important; white-space: pre-wrap !important;
                     word-break: break-word; font-size: 8pt; line-height: 1.5; }
  code { overflow-wrap: anywhere; }
  h1, h2, h3, summary { break-after: avoid; }
  h2 { border-top: 2px solid #2563eb; padding-top: 10px; margin-top: 22px; }
  table { width: 100%; border-collapse: collapse; font-size: 8.5pt; }
  th, td { overflow-wrap: anywhere; padding: 5px; }
  tr { break-inside: avoid; }
  details { margin: 8px 0; }
  summary { font-weight: bold; color: #334155; }
  .pdf-figure { break-inside: avoid; margin: 10px 0 14px; display: block; }
  .pdf-figure img { width: 100%; height: auto; display: block; }
  .pdf-figure figcaption { font-size: 8pt; color: #475569; margin-bottom: 3px; }
  .pdf-kernel-table { font-size: 7pt; }
  .pdf-kernel-table { table-layout: fixed; }
  .pdf-kernel-table th:first-child, .pdf-kernel-table td:first-child { width: 76%; }
  .pdf-kernel-table th:nth-child(2), .pdf-kernel-table td:nth-child(2) { width: 10%; }
  .pdf-kernel-table th:nth-child(3), .pdf-kernel-table td:nth-child(3) { width: 14%; }
  .pdf-kernel-table td:not(:first-child) { white-space: nowrap; }
  .pdf-kernel-table td:first-child { font-family: monospace; word-break: break-all; }
  .print-notebook { width: 100%; padding: 0; }
  .print-cell, .print-output { display: block; overflow: visible; height: auto; break-inside: auto; }
  .print-input { background: #f8fafc; border: 1px solid #e2e8f0; padding: 7px 9px; margin: 8px 0; }
  .print-input pre { margin: 0; }
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=STUDY_DIR / "main.ipynb")
    parser.add_argument("--output", type=Path, default=STUDY_DIR / "main.pdf")
    parser.add_argument("--browser", type=Path)
    args = parser.parse_args()
    notebook = nbformat.read(args.notebook, as_version=4)
    nbformat.validate(notebook)
    if any(c.execution_count is None for c in notebook.cells if c.cell_type == "code"):
        raise ValueError("Execute every code cell before creating the PDF")
    if any(o.output_type == "error" for c in notebook.cells for o in c.get("outputs", [])):
        raise ValueError("The notebook contains an error output")
    figures = [o["data"]["application/vnd.plotly.v1+json"] for c in notebook.cells
               for o in c.get("outputs", []) if "application/vnd.plotly.v1+json" in o.get("data", {})]
    expanded = [variants(figure) for figure in figures]
    if not figures:
        raise ValueError("No executed Plotly figures found")
    candidates = ([args.browser] if args.browser else []) + [
        Path(p) for p in [shutil.which("chromium"), shutil.which("google-chrome")] if p
    ] + sorted((Path.home() / ".cache/ms-playwright").glob("chromium-*/chrome-linux*/chrome"), reverse=True)
    browser = next((p for p in candidates if p and p.is_file()), None)
    if browser is None:
        raise RuntimeError("An installed Chromium is required; pass --browser /path/to/chrome")
    work = STUDY_DIR / "artifacts" / f"pdf-export-{uuid.uuid4().hex[:12]}"
    work.mkdir(parents=True)
    html, _ = HTMLExporter(template_name="lab").from_notebook_node(notebook)
    html_path = work / "executed.html"
    html_path.write_text(html, encoding="utf-8")
    profile = work / "chromium"
    log = (work / "chromium.log").open("w")
    process = subprocess.Popen([
        str(browser), "--headless", "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
        "--remote-debugging-address=127.0.0.1", "--remote-debugging-port=0",
        "--remote-allow-origins=http://127.0.0.1", f"--user-data-dir={profile}", html_path.as_uri(),
    ], stdout=log, stderr=log)
    devtools = None
    try:
        port_file = profile / "DevToolsActivePort"
        deadline = time.monotonic() + 30
        while not port_file.exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"Chromium did not start; see {work / 'chromium.log'}")
            time.sleep(.2)
        port = port_file.read_text().splitlines()[0]
        pages = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=10))
        page = next(p for p in pages if p["type"] == "page")
        devtools = DevTools(page["webSocketDebuggerUrl"])
        devtools.call("Emulation.setDeviceMetricsOverride", {"width": 1200, "height": 1000, "deviceScaleFactor": 1, "mobile": False})
        deadline = time.monotonic() + 50
        while devtools.evaluate("document.querySelectorAll('.js-plotly-plot').length") != len(figures):
            if time.monotonic() > deadline:
                raise RuntimeError("Not all executed figures rendered; refusing an incomplete PDF")
            time.sleep(.3)
        devtools.evaluate("document.fonts.ready.then(() => true)")
        devtools.evaluate("window.pdfOriginalPlots = [...document.querySelectorAll('.js-plotly-plot')]; window.pdfFigureCount=0; true")
        for index, group in enumerate(expanded):
            # Work in bounded batches so a failed figure identifies its original.
            payload = json.dumps({"index": index, "variants": group}, ensure_ascii=False, allow_nan=False)
            expression = r"""(async () => {
              const spec = PAYLOAD;
              const original = window.pdfOriginalPlots[spec.index];
              const fragment = document.createDocumentFragment();
              const scratch = document.createElement('div');
              scratch.style.cssText='position:fixed;left:-12000px;top:0;width:900px';
              document.body.append(scratch);
              for (const item of spec.variants) {
                const f=item.figure;
                const layout={...f.layout,width:900,height:Math.min(f.layout.height||520,1050),
                              updatemenus:[],sliders:[],autosize:false};
                layout.font={...layout.font,size:14};
                // Four narrow performance panels fit the notebook screen, but
                // A4 needs a 2x2 arrangement to keep units clear of nearby bars.
                if ((layout.title?.text||'').includes('GPU 시간 · 처리량 · launch 비교')) {
                  layout.height=850;
                  const xs=[[0,.42],[.58,1],[0,.42],[.58,1]];
                  const ys=[[.60,1],[.60,1],[0,.40],[0,.40]];
                  for(let axis=0;axis<4;axis++) {
                    const suffix=axis===0?'':String(axis+1);
                    layout['xaxis'+suffix]={...layout['xaxis'+suffix],domain:xs[axis]};
                    layout['yaxis'+suffix]={...layout['yaxis'+suffix],domain:ys[axis]};
                    if(layout.annotations?.[axis]) {
                      layout.annotations[axis]={...layout.annotations[axis],x:(xs[axis][0]+xs[axis][1])/2,
                                                y:axis<2?1:.40,yanchor:'bottom'};
                    }
                  }
                }
                const traces=f.data.map(t=>({...t,...(t.textfont?{textfont:{...t.textfont,size:14}}:{})}));
                await Plotly.newPlot(scratch,traces,layout,{staticPlot:true,responsive:false});
                const src=await Plotly.toImage(scratch,{format:'svg',width:900,height:layout.height});
                const figure=document.createElement('figure'); figure.className='pdf-figure';
                const caption=document.createElement('figcaption');
                caption.textContent=`그림 ${spec.index+1}${item.label?' · '+item.label:''}`;
                const img=document.createElement('img'); img.src=src;
                img.alt=(layout.title?.text||'figure').replace(/<[^>]*>/g,' ')+(item.label?' / '+item.label:'');
                await img.decode(); figure.append(caption,img); fragment.append(figure);
                window.pdfFigureCount++;
              }
              // Hover is unavailable in PDF. Retain exact kernel names, counts and
              // durations as a compact table for each actual profiler timeline.
              if ((original.layout.title?.text||'').includes('실제 CUDA timeline')) {
                const totals=new Map();
                for (const t of original.data) {
                  if (!(t.name||'').startsWith('GPU stream')) continue;
                  for(let i=0;i<t.x.length;i++) {
                    const name=t.customdata[i]; const old=totals.get(name)||{count:0,us:0};
                    old.count++; old.us+=t.x[i]; totals.set(name,old);
                  }
                }
                const table=document.createElement('table'); table.className='pdf-kernel-table';
                const head=document.createElement('tr');
                for(const value of ['실제 kernel 이름','profile 구간 호출 수','duration 합계 (μs)']) {
                  const th=document.createElement('th');th.textContent=value;head.append(th);
                } table.append(head);
                for(const [name,value] of totals) {
                  const tr=document.createElement('tr');
                  for(const text of [name,value.count,value.us.toFixed(3)]) {
                    const td=document.createElement('td');td.textContent=text;tr.append(td);
                  } table.append(tr);
                } fragment.append(table);
              }
              Plotly.purge(scratch); scratch.remove(); original.replaceWith(fragment);
              return window.pdfFigureCount;
            })()""".replace("PAYLOAD", payload)
            count = devtools.evaluate(expression)
            if index % 5 == 0 or index == len(expanded) - 1:
                print(f"PDF figures: {index + 1}/{len(expanded)} originals → {count} static views", flush=True)
        devtools.evaluate("""(() => {
          document.title='nanoGPT CUDA Compiler Lab';
          document.querySelectorAll('details').forEach(d=>d.open=true);
          document.querySelectorAll('script').forEach(s=>s.remove());
          // Notebook output wrappers use grid/flex sizing that Chromium can clip
          // across printed pages. Rebuild a simple block-flow print document.
          const main=document.createElement('main');main.className='print-notebook';
          for(const cell of document.querySelectorAll('.jp-Cell')) {
            const section=document.createElement('section');section.className='print-cell';
            const markdown=cell.querySelector('.jp-RenderedMarkdown');
            if(markdown) {const content=markdown.cloneNode(true);content.className='';section.append(content);}
            const input=cell.querySelector('.jp-InputArea .highlight');
            if(input) {const content=input.cloneNode(true);content.classList.add('print-input');section.append(content);}
            for(const output of cell.querySelectorAll('.jp-OutputArea-output')) {
              const content=output.cloneNode(true);content.className='print-output';content.removeAttribute('style');
              for(const div of content.querySelectorAll('div')) {div.className='';div.removeAttribute('style');}
              section.append(content);
            }
            main.append(section);
          }
          const mathDefs=document.querySelector('#MathJax_SVG_Hidden')?.cloneNode(true);
          document.body.replaceChildren(main);
          if(mathDefs)document.body.append(mathDefs);
          const note=document.createElement('p'); note.textContent='실행 결과 PDF · Dropdown/slider의 모든 구간과 단계를 펼쳤습니다. 원본 interactive 출력은 main.ipynb에서 확인할 수 있습니다.';
          document.querySelector('h1').after(note); return true;
        })()""")
        devtools.evaluate("(() => {const s=document.createElement('style');s.textContent=" + json.dumps(PRINT_CSS) + ";document.head.append(s);return true;})()")
        static_html = devtools.evaluate("document.documentElement.outerHTML")
        (work / "print.html").write_text("<!doctype html>\n" + static_html, encoding="utf-8")
        devtools.call("Emulation.setEmulatedMedia", {"media": "print"})
        printed = devtools.call("Page.printToPDF", {
            "printBackground": True, "preferCSSPageSize": True, "displayHeaderFooter": True,
            "headerTemplate": "<div></div>",
            "footerTemplate": "<div style='font-size:8px;width:100%;text-align:right;padding:0 14mm;color:#64748b'>study02 · CUDA compiler lab · <span class='pageNumber'></span> / <span class='totalPages'></span></div>",
            "generateDocumentOutline": True, "generateTaggedPDF": True,
        })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(base64.b64decode(printed["data"]))
        report = {"notebook": str(args.notebook.resolve()), "notebook_sha256": digest(args.notebook),
                  "output": str(args.output.resolve()), "pdf_sha256": digest(args.output),
                  "original_plotly_figures": len(figures), "static_views": sum(map(len, expanded)),
                  "variants_per_figure": [len(group) for group in expanded],
                  "print_html": str(work / "print.html")}
        write_json(work / "pdf-export.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    finally:
        if devtools:
            devtools.ws.close()
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
        log.close()


if __name__ == "__main__":
    main()
