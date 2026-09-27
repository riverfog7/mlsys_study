"""Plotly views of current-run data. No pre-rendered figures or stored captures."""
from __future__ import annotations

from collections import defaultdict, deque
import html
import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BLUE, TEAL, ORANGE, PURPLE, GRAY = "#2563eb", "#0f9d92", "#d97706", "#8659c5", "#64748b"


def style(fig, title, source="현재 실행에서 생성한 데이터", height=520):
    fig.update_layout(template="plotly_white", height=height,
                      title=dict(text=title + "<br><sup>" + html.escape(source) + "</sup>", x=0.02, font=dict(size=20)),
                      font=dict(family="Noto Sans CJK KR, Malgun Gothic, Arial, sans-serif", size=12),
                      margin=dict(l=65, r=35, t=100, b=115),
                      hoverlabel=dict(bgcolor="white", font_size=12),
                      legend=dict(orientation="h", y=-0.23),
                      meta=dict(provenance=source))
    return fig


def array(value):
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def heatmap(value, title, *, x=None, y=None, colorscale="Viridis", source="현재 CUDA tensor", text=False):
    z = array(value)
    visible = [[float(v) if np.isfinite(v) else None for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=visible, x=x, y=y, colorscale=colorscale,
                             text=np.round(z, 3).tolist() if text else None,
                             texttemplate="%{text}" if text else None,
                             hovertemplate="column=%{x}<br>row=%{y}<br>value=%{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return style(fig, title, source, height=max(340, min(680, z.shape[0] * 34 + 220)))


def heatmap_selector(values, labels, title, *, xaxis="key 위치", yaxis="query 위치"):
    fig = go.Figure()
    for i, value in enumerate(values):
        z = array(value)
        z = [[float(v) if np.isfinite(v) else None for v in row] for row in z]
        fig.add_trace(go.Heatmap(z=z, visible=i == 0, colorscale="Viridis",
                                hovertemplate="column=%{x}<br>row=%{y}<br>%{z:.5f}<extra></extra>"))
    fig.update_layout(updatemenus=[dict(x=1, y=1.16, buttons=[
        dict(label=label, method="update", args=[{"visible": [j == i for j in range(len(values))]}])
        for i, label in enumerate(labels)])])
    fig.update_xaxes(title=xaxis)
    fig.update_yaxes(title=yaxis, autorange="reversed")
    return style(fig, title)


def _dag_page(nodes, edges, title, source, highlights=()):
    by_id = {n["id"]: n for n in nodes}
    incoming, children = defaultdict(int), defaultdict(list)
    for edge in edges:
        if edge["source"] in by_id and edge["target"] in by_id:
            incoming[edge["target"]] += 1
            children[edge["source"]].append(edge["target"])
    queue = deque(n for n in by_id if not incoming[n])
    depth = {n: 0 for n in queue}
    visited = 0
    while queue:
        current = queue.popleft()
        visited += 1
        for target in children[current]:
            depth[target] = max(depth.get(target, 0), depth[current] + 1)
            incoming[target] -= 1
            if incoming[target] == 0:
                queue.append(target)
    if visited != len(nodes):
        raise ValueError("Expected a DAG; refusing to invent a layout for cyclic dependencies")
    layers = defaultdict(list)
    for node in nodes:
        layers[depth[node["id"]]].append(node["id"])
    positions, cursor = {}, 0.0
    for level in sorted(layers):
        names = layers[level]
        rows = [names[i:i + 5] for i in range(0, len(names), 5)]
        for row, chunk in enumerate(rows):
            for i, name in enumerate(chunk):
                positions[name] = (i - (len(chunk) - 1) / 2, cursor - row * .75)
        cursor -= .75 * (len(rows) - 1) + 1.35
    fig = go.Figure()
    bypass = 0
    for edge in edges:
        if edge["source"] not in positions or edge["target"] not in positions:
            continue
        x0, y0 = positions[edge["source"]]
        x1, y1 = positions[edge["target"]]
        ax, ay = x0, y0
        dx, dy = x1 - x0, y1 - y0
        crosses = False
        for name, (px, py) in positions.items():
            if name in {edge['source'], edge['target']}:
                continue
            fraction = ((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy)
            if 0 < fraction < 1 and abs(px - (x0 + fraction * dx)) < .23 and abs(py - (y0 + fraction * dy)) < .23:
                crosses = True
                break
        if depth[edge['target']] - depth[edge['source']] > 1 or crosses:
            # Route long dependencies outside other nodes; a straight shortcut can
            # otherwise appear to connect to intermediate operations it bypasses.
            bend = min(p[0] for p in positions.values()) - .35 - .12 * bypass
            bypass += 1
            fig.add_shape(type="path", path=f"M {x0},{y0} L {bend},{y0-.25} L {bend},{y1+.28}",
                          line=dict(color="#94a3b8", width=1.3), layer="below")
            ax, ay = bend, y1 + .28
        fig.add_annotation(x=x1, y=y1, ax=ax, ay=ay, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.3,
                           arrowcolor="#94a3b8", standoff=13, startstandoff=13)
    groups = defaultdict(list)
    for node in nodes:
        groups[node.get("op", "operation")].append(node)
    for op, group in groups.items():
        color = BLUE if op == "placeholder" else PURPLE if op == "output" else ORANGE if "Extern" in op else TEAL
        if op == "boundary":
            color = GRAY
        hover, labels = [], []
        for n in group:
            label = n.get("label", n["id"])
            label = label if len(label) <= 23 else label[:15] + '…' + label[-6:]
            labels.append(html.escape(label))
            meta = json.dumps(n.get("metadata", {}), ensure_ascii=False)
            hover.append("<b>" + html.escape(n["id"]) + "</b><br>" + html.escape(n.get("target", "")) +
                         "<br>" + html.escape(meta) + "<br>" + html.escape(n.get("args", "")[:350]))
        fig.add_trace(go.Scatter(x=[positions[n["id"]][0] for n in group],
                                 y=[positions[n["id"]][1] for n in group], mode="markers+text",
                                 name=op, text=labels, textposition="middle right", textfont=dict(size=12),
                                 marker=dict(size=22, color=[PURPLE if n["id"] in highlights else color for n in group],
                                             line=dict(color="white", width=2)),
                                 customdata=hover, hovertemplate="%{customdata}<extra></extra>"))
    extent = max((abs(p[0]) for p in positions.values()), default=0)
    fig.update_xaxes(visible=False, range=[-extent - .6 - .12 * bypass, extent + 1.8])
    fig.update_yaxes(visible=False, range=[min(p[1] for p in positions.values()) - .7, .8])
    return style(fig, title, source, height=max(460, min(1050, int(-cursor * 55 + 200))))


def dag(graph, title=None, *, page_size=26, highlights=()):
    """Page a large graph, retaining incoming boundary nodes and all source data."""
    nodes, edges = graph["nodes"], graph["edges"]
    if not nodes:
        raise ValueError("No captured nodes to display")
    title = title or graph.get("stage", "Graph")
    if len(nodes) <= page_size:
        return _dag_page(nodes, edges, title, "현재 capture · 화살표=값의 의존성, 시간축 아님", highlights)
    pages, by_id = [], {n["id"]: n for n in nodes}
    for start in range(0, len(nodes), page_size):
        selected = nodes[start:start + page_size]
        selected_ids = {n["id"] for n in selected}
        boundary = {e["source"] for e in edges if e["target"] in selected_ids and e["source"] not in selected_ids}
        extra = [{**by_id[name], "op": "boundary", "label": "이전: " + name} for name in by_id if name in boundary]
        page_edges = [e for e in edges if e["target"] in selected_ids and e["source"] in selected_ids | boundary]
        pages.append(_dag_page(extra + selected, page_edges, title,
                               f"현재 capture · node {start + 1}–{min(start + page_size, len(nodes))}/{len(nodes)} · 회색=이전 구간 입력",
                               highlights))
    combined = go.Figure()
    offsets = []
    for i, page in enumerate(pages):
        first = len(combined.data)
        for trace in page.data:
            trace.visible = i == 0
            combined.add_trace(trace)
        offsets.append((first, len(combined.data)))
    combined.update_layout(pages[0].layout)
    buttons = []
    for i, page in enumerate(pages):
        lo, hi = offsets[i]
        buttons.append(dict(label=f"구간 {i + 1}/{len(pages)}", method="update",
                            args=[{"visible": [lo <= j < hi for j in range(len(combined.data))]},
                                  {"annotations": list(page.layout.annotations), "shapes": list(page.layout.shapes),
                                   "xaxis": page.layout.xaxis.to_plotly_json(),
                                   "yaxis": page.layout.yaxis.to_plotly_json(), "height": page.layout.height,
                                   "title": page.layout.title.to_plotly_json()}]))
    combined.update_layout(updatemenus=[dict(x=1, y=1.13, buttons=buttons)])
    return combined


def concept(labels, title, edges=None):
    nodes = [{"id": str(i), "label": label, "op": "개념", "target": label} for i, label in enumerate(labels)]
    edges = edges or [(i, i + 1) for i in range(len(labels) - 1)]
    return _dag_page(nodes, [{"source": str(a), "target": str(b)} for a, b in edges], title,
                     "개념도 · 실행 시 작성 · 실제 compiler capture와 구별")


def autograd_graph(loss):
    queue, seen, nodes, edges, keepalive = deque([loss.grad_fn]), {}, [], [], []
    while queue:
        fn = queue.popleft()
        if fn is None or id(fn) in seen:
            continue
        keepalive.append(fn)
        seen[id(fn)] = str(len(seen))
        queue.extend(parent for parent, _ in fn.next_functions if parent is not None)
    for fn in keepalive:
        variable = getattr(fn, "variable", None)
        meta = {"shape": list(variable.shape), "device": str(variable.device)} if variable is not None else {}
        nodes.append({"id": seen[id(fn)], "label": type(fn).__name__, "op": "backward node",
                      "target": type(fn).__name__, "metadata": meta})
        edges.extend({"source": seen[id(fn)], "target": seen[id(parent)]}
                     for parent, _ in fn.next_functions if parent is not None)
    return {"stage": "Eager autograd: backward 연결", "nodes": nodes, "edges": edges}


def bytecode_steps(result):
    frames = []
    for i, step in enumerate(result["steps"]):
        shapes, annotations = [], []
        for j, value in enumerate(step["stack"]):
            shapes.append(dict(type="rect", x0=0, x1=1.8, y0=j, y1=j + .75,
                               fillcolor="#dbeafe", line=dict(color=BLUE)))
            annotations.append(dict(x=.9, y=j + .375, text=str(value), showarrow=False, font=dict(size=20)))
        annotations.extend([dict(x=.9, y=3.5, text="operand stack", showarrow=False),
                            dict(x=3.6, y=3.5, text="local variables", showarrow=False),
                            dict(x=2.3, y=4.6, text=html.escape(step["instruction"]), showarrow=False,
                                 font=dict(size=16, color=BLUE)),
                            dict(x=3.6, y=2, text="<br>".join(f"{k} = {v}" for k, v in step["locals"].items()),
                                 showarrow=False, font=dict(size=20))])
        frames.append(go.Frame(name=str(i), layout=dict(shapes=shapes, annotations=annotations)))
    fig = go.Figure(frames=frames)
    fig.update_layout(shapes=frames[0].layout.shapes, annotations=frames[0].layout.annotations,
                      sliders=[dict(active=0, currentvalue=dict(prefix="명령 단계: "), steps=[
                          dict(label=str(i + 1), method="animate", args=[[str(i)],
                               {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}])
                          for i in range(len(frames))])])
    fig.update_xaxes(visible=False, range=[-.4, 5.4])
    fig.update_yaxes(visible=False, range=[-.5, 5.2])
    style(fig, "실제 dis 명령을 따라가는 작은 stack 모델", result["stack_note"], 500)
    return fig


def fusion_groups(capture):
    pre, post = capture["pre"]["nodes"], capture["post"]["nodes"]
    labels = ["pre " + n["id"] for n in pre] + ["post " + n["id"] for n in post]
    indices = {n["id"]: i for i, n in enumerate(pre)}
    sources, targets = [], []
    for i, node in enumerate(post):
        for member in node["members"]:
            if member in indices:
                sources.append(indices[member]); targets.append(len(pre) + i)
    if not sources:
        raise ValueError("No verified pre/post node membership in this capture")
    fig = go.Figure(go.Sankey(node=dict(label=labels, color=[BLUE] * len(pre) + [TEAL] * len(post)),
                              link=dict(source=sources, target=targets, value=[1] * len(sources))))
    return style(fig, "Scheduler 전후: 실제 node 이름으로 확인한 묶음", "현재 IR · 선의 폭=포함된 pre node 수 · GPU 시간/메모리 양 아님")


def lifetimes(wrapper):
    fig = go.Figure()
    groups = defaultdict(list)
    for buffer in wrapper["buffers"]:
        groups[buffer["storage"]].append(buffer)
    for storage, buffers in groups.items():
        fig.add_trace(go.Bar(name="storage " + storage, orientation="h",
                             y=[b["name"] for b in buffers], base=[b["first_line"] for b in buffers],
                             x=[max(1, b["last_line"] - b["first_line"]) for b in buffers],
                             customdata=[f"{b['relation']} · shape={b['shape']} · logical bytes={b['logical_bytes']}" for b in buffers],
                             hovertemplate="%{y}<br>%{customdata}<extra>%{fullData.name}</extra>"))
    fig.update_layout(barmode="overlay", showlegend=len(groups) <= 10)
    fig.update_xaxes(title="generated wrapper의 소스 줄 번호")
    fig.update_yaxes(title="논리 buffer", autorange="reversed")
    return style(fig, "Buffer lifetime / storage reuse", "현재 wrapper · 정적 참조 구간이며 GPU wall time이 아님",
                 max(460, min(1000, len(wrapper["buffers"]) * 24 + 180)))


def timeline(result):
    fig = go.Figure()
    groups = defaultdict(list)
    origin = min(e['start_us'] for e in result['timeline'] if e['category'] == 'kernel')
    for event in result["timeline"]:
        if event["category"] == "kernel":
            groups[event["lane"]].append(event)
    for lane, events in groups.items():
        fig.add_trace(go.Bar(name=lane, orientation="h", y=[lane] * len(events),
                             x=[e["duration_us"] for e in events], base=[e["start_us"] - origin for e in events],
                             marker_color=TEAL,
                             customdata=[e["name"] for e in events],
                             hovertemplate="%{customdata}<br>start=%{base:.2f} μs<br>duration=%{x:.2f} μs<extra></extra>"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="첫 GPU kernel 시작으로부터 경과 시간 (μs)")
    return style(fig, f"{result['workload']} · {result['mode']} · 실제 CUDA timeline",
                 f"{result['run_id']} · GPU kernel만 표시 · profiler {result['profile_iterations']}회", 360)


def performance(results):
    fig = make_subplots(rows=1, cols=4, subplot_titles=("CUDA event 중앙값", "입력 처리량", "CUDA launches / call", "Kernel duration 합 / call"))
    names = [f"{r['attention']} / {r['mode']}" if r['workload'] == 'block1' else r['mode'] for r in results]
    for col, values, unit in [(1, [r['steady_cuda_event_ms'] for r in results], 'ms'),
                               (2, [r['throughput_million_elements_s'] for r in results], 'M input elements/s'),
                               (3, [r['cuda_launches_per_call'] for r in results], 'launches'),
                               (4, [r['cuda_kernel_duration_ms_per_call'] for r in results], 'ms (profile)')]:
        errors = dict(type='data', symmetric=False,
                      array=[r['cuda_event_q75_ms'] - r['steady_cuda_event_ms'] for r in results],
                      arrayminus=[r['steady_cuda_event_ms'] - r['cuda_event_q25_ms'] for r in results]) if col == 1 else None
        fig.add_trace(go.Bar(x=names, y=values, marker_color=[BLUE, TEAL, ORANGE, PURPLE][:len(results)],
                             error_y=errors,
                             showlegend=False, hovertemplate=f"%{{x}}<br>%{{y:.5f}} {unit}<extra></extra>"), row=1, col=col)
        fig.update_yaxes(title=unit, row=1, col=col)
    fig.update_xaxes(tickangle=-22)
    return style(fig, f"{results[0]['workload']}: GPU 시간 · 처리량 · launch 비교",
                 "현재 CUDA 실측 · event=반복 측정 중앙값/IQR · kernel duration은 별도 profiler 측정", 530)


def performance_sweep(pairs):
    fig = make_subplots(rows=1, cols=2, subplot_titles=('입력 크기에 따른 CUDA event latency', '실제 속도비: eager / compiled'))
    for mode, color in [('eager', BLUE), ('compiled', TEAL)]:
        rows = [pair[0 if mode == 'eager' else 1] for pair in pairs]
        median = [r['steady_cuda_event_ms'] for r in rows]
        fig.add_trace(go.Scatter(x=[r['input_elements'] for r in rows], y=median, mode='lines+markers', name=mode,
                                marker_color=color,
                                error_y=dict(type='data', symmetric=False,
                                             array=[r['cuda_event_q75_ms']-m for r,m in zip(rows,median)],
                                             arrayminus=[m-r['cuda_event_q25_ms'] for r,m in zip(rows,median)]),
                                customdata=[str(r['input_shape']) for r in rows],
                                hovertemplate='shape=%{customdata}<br>N=%{x}<br>%{y:.5f} ms<extra>%{fullData.name}</extra>'),row=1,col=1)
    fig.add_trace(go.Scatter(x=[a['input_elements'] for a,b in pairs],
                             y=[a['steady_cuda_event_ms']/b['steady_cuda_event_ms'] for a,b in pairs],
                             mode='lines+markers',name='속도비',marker_color=PURPLE,
                             hovertemplate='N=%{x}<br>%{y:.3f}×<extra></extra>'),row=1,col=2)
    fig.add_hline(y=1, line_dash='dot',line_color=GRAY,row=1,col=2)
    fig.update_xaxes(type='log',title='입력 원소 수 (log scale)')
    fig.update_yaxes(title='ms',row=1,col=1)
    fig.update_yaxes(title='1보다 크면 compiled가 빠름',row=1,col=2)
    return style(fig,'작은 입력부터 큰 입력까지: 유리한 경우와 이득이 작은 경우',
                 '현재 실측 · shape마다 별도 eager/compiled 프로세스 · 오차 막대=반복 측정 IQR',480)


def compilation_cost(eager, compiled):
    saving = eager['steady_wall_ms'] - compiled['steady_wall_ms']
    extra = compiled['first_call_ms'] - eager['first_call_ms']
    break_even = max(1, int(np.ceil(extra / saving)) + 1) if saving > 0 else None
    end = max(100, min(10_000_000, (break_even or 10000) * 4))
    counts = np.unique(np.geomspace(1, end, 70).astype(int))
    fig = make_subplots(rows=1,cols=2,subplot_titles=('첫 호출 비용: 실제 측정','반복할 때의 누적시간 모델'))
    fig.add_trace(go.Bar(x=['eager','compiled'],y=[eager['first_call_ms'],compiled['first_call_ms']],
                        marker_color=[BLUE,TEAL],showlegend=False),row=1,col=1)
    for r,color in [(eager,BLUE),(compiled,TEAL)]:
        predicted = (r['first_call_ms'] + (counts-1)*r['steady_wall_ms']) / 1000
        fig.add_trace(go.Scatter(x=counts.tolist(),y=predicted.tolist(),name=r['mode'],line=dict(color=color)),row=1,col=2)
    fig.update_yaxes(title='ms',row=1,col=1)
    fig.update_xaxes(type='log',title='호출 횟수',row=1,col=2)
    fig.update_yaxes(title='예상 누적 초',row=1,col=2)
    if break_even is not None and break_even <= end:
        fig.add_vline(x=break_even,line_dash='dot',line_color=GRAY,row=1,col=2)
    return style(fig,f"컴파일 비용 회수: {'약 '+str(break_even)+'회' if break_even else '반복 latency 이득 없음'}",
                 '왼쪽=실측 · 오른쪽=first call + (N-1)×steady wall 중앙값으로 계산한 모델',480)


def triton_mask(elements, block):
    programs = (elements + block - 1) // block
    offsets = np.arange(programs * block).reshape(programs, block)
    fig = go.Figure(go.Heatmap(z=(offsets < elements).astype(int).tolist(), customdata=offsets.tolist(),
                             colorscale=[[0, "#e2e8f0"], [1, TEAL]], showscale=False,
                             hovertemplate="program=%{y}<br>local offset=%{x}<br>global index=%{customdata}<br>valid=%{z}<extra></extra>"))
    fig.update_xaxes(title="program 내부의 element offset")
    fig.update_yaxes(title="program_id(0)", dtick=1, autorange="reversed")
    return style(fig, f"{elements} elements / BLOCK={block} → {programs} programs",
                 "현재 launch 인자로 계산 · program 하나는 CUDA thread 하나와 다름", 350)
