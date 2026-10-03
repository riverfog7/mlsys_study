"""Collect and render the notebook's GPT-2 KV-cache demonstration.

Collection observes the supplied model and copies explanatory matrices to CPU.
Rendering uses only those records and their copied metadata. Importing this
module does not load a model, run a forward pass, or display a figure.
"""

import html
import math

import numpy as np
import plotly.graph_objects as go
import torch
from IPython.display import HTML
from plotly.subplots import make_subplots


def collect_kv_cache(model, tokenizer, *, prompt, decode_steps, layer, head):
    """Return prefill/decode records, copied display metadata, and a summary.

    The supplied GPT-2 must expose eager attention probabilities, as configured
    in the notebook. Past Q rows are retained only for the explanation; the
    actual model cache remains K/V. Hooks and training state are restored even
    when a forward pass or a verification assertion fails.
    """
    kv_transformer = model.transformer
    kv_device = next(kv_transformer.parameters()).device
    kv_head_dim = kv_transformer.config.n_embd // kv_transformer.config.n_head
    kv_input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(kv_device)
    assert kv_input_ids.shape[1] + decode_steps <= kv_transformer.config.n_positions
    assert 0 <= layer < len(kv_transformer.h)
    assert 0 <= head < kv_transformer.config.n_head
    assert decode_steps >= 0
    metadata = {
        "prompt": prompt, "decode_steps": decode_steps,
        "layer": layer, "head": head, "head_dim": kv_head_dim,
        "scale_attn_weights": bool(kv_transformer.config.scale_attn_weights),
        "scale_attn_by_inverse_layer_idx": bool(kv_transformer.config.scale_attn_by_inverse_layer_idx),
    }


    def kv_pair(cache, layer_index):
        """Read one layer's actual cache (Transformers Cache or older tuple format)."""
        if hasattr(cache, "layers"):
            layer = cache.layers[layer_index]
            return layer.keys, layer.values
        return cache[layer_index]


    def kv_token_label(token_id):
        return tokenizer.decode([token_id]).replace(" ", "·").replace("\n", "↵")


    kv_projection = {}


    def kv_capture_projection(module, args, output):
        # The normal forward pass computes Q, K, V together. Observe it, without
        # replacing the model's attention implementation or doing another projection.
        query, _, _ = output.split(kv_transformer.config.n_embd, dim=-1)
        query = query.reshape(1, -1, kv_transformer.config.n_head, kv_head_dim).transpose(1, 2)
        kv_projection["query"] = query[0, head].detach().float().cpu().clone()


    kv_records = []
    kv_query_history = []  # Visualization only: Q is not part of the model's KV cache.
    kv_cache = None
    kv_feed_ids = kv_input_ids
    kv_previous_keys = kv_previous_values = None
    kv_was_training = kv_transformer.training
    kv_hook = kv_transformer.h[layer].attn.c_attn.register_forward_hook(kv_capture_projection)

    try:
        kv_transformer.eval()
        with torch.inference_mode():
            for step in range(decode_steps + 1):
                output = kv_transformer(
                    kv_feed_ids, past_key_values=kv_cache, use_cache=True,
                    output_attentions=True, return_dict=True,
                )
                kv_cache = output.past_key_values
                queries = kv_projection["query"]
                layer_keys, layer_values = kv_pair(kv_cache, layer)
                keys = layer_keys[0, head].detach().float().cpu().clone()
                values = layer_values[0, head].detach().float().cpu().clone()
                total_length, query_length = keys.shape[0], queries.shape[0]
                past_length = total_length - query_length

                # Reconstruct this head's actual attention probabilities for display.
                # All head dimensions participate; there is no padding in this demo.
                scores = queries @ keys.T
                if kv_transformer.config.scale_attn_weights:
                    scores /= math.sqrt(kv_head_dim)
                if kv_transformer.config.scale_attn_by_inverse_layer_idx:
                    scores /= layer + 1
                query_positions = torch.arange(past_length, total_length)
                causal_mask = torch.arange(total_length)[None, :] <= query_positions[:, None]
                probabilities = scores.masked_fill(~causal_mask, -torch.inf).softmax(dim=-1)
                actual_probabilities = output.attentions[layer][0, head].detach().float().cpu()
                torch.testing.assert_close(probabilities, actual_probabilities, rtol=1e-4, atol=1e-5)
                probabilities = actual_probabilities
                torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(query_length))
                assert torch.count_nonzero(probabilities[~causal_mask]).item() == 0

                # Retain past Q only for the full-matrix explanation. The cached
                # forward above still computes only this step's query rows.
                kv_query_history.append(queries)
                all_queries = torch.cat(kv_query_history, dim=0)
                full_raw_scores = all_queries @ keys.T
                full_scaled_scores = full_raw_scores.clone()
                if kv_transformer.config.scale_attn_weights:
                    full_scaled_scores /= math.sqrt(kv_head_dim)
                if kv_transformer.config.scale_attn_by_inverse_layer_idx:
                    full_scaled_scores /= layer + 1
                full_causal_mask = torch.ones(total_length, total_length, dtype=torch.bool).tril()
                full_attention = full_scaled_scores.masked_fill(~full_causal_mask, -torch.inf).softmax(dim=-1)
                torch.testing.assert_close(full_attention[past_length:], probabilities, rtol=1e-4, atol=1e-5)
                full_attention[past_length:] = probabilities
                full_head_output = full_attention @ values

                # Verify that the previously stored K/V really stayed unchanged.
                previous_delta = None
                if kv_previous_keys is not None:
                    assert torch.equal(keys[:past_length], kv_previous_keys)
                    assert torch.equal(values[:past_length], kv_previous_values)
                    previous_delta = max(
                        (keys[:past_length] - kv_previous_keys).abs().max().item(),
                        (values[:past_length] - kv_previous_values).abs().max().item(),
                    )

                # Reference: run the entire prefix again, with no cache at all.
                reference = kv_transformer(kv_input_ids, use_cache=False, return_dict=True)
                cached_hidden = output.last_hidden_state[:, -1, :]
                reference_hidden = reference.last_hidden_state[:, -1, :]
                torch.testing.assert_close(cached_hidden, reference_hidden, rtol=1e-4, atol=1e-4)
                hidden_error = (cached_hidden - reference_hidden).abs().max().item()

                # Reuse the language-model head of the model already loaded above.
                logits = model.lm_head(cached_hidden)
                next_id = logits.argmax(dim=-1, keepdim=True)
                cache_bytes = sum(
                    tensor.numel() * tensor.element_size()
                    for layer_index in range(len(kv_transformer.h))
                    for tensor in kv_pair(kv_cache, layer_index)
                )
                kv_token_ids = kv_input_ids[0].tolist()
                kv_records.append({
                    "step": step,
                    "past_length": past_length,
                    "query_length": query_length,
                    "total_length": total_length,
                    "tokens": [kv_token_label(token_id) for token_id in kv_token_ids],
                    "text": tokenizer.decode(kv_token_ids),
                    "next_token": kv_token_label(next_id.item()),
                    "queries": queries.numpy().copy(),
                    "all_queries": all_queries.numpy().copy(),
                    "keys": keys.numpy().copy(),
                    "values": values.numpy().copy(),
                    "attention": probabilities.numpy().copy(),
                    "full_raw_scores": full_raw_scores.numpy().copy(),
                    "full_scaled_scores": full_scaled_scores.numpy().copy(),
                    "full_causal_mask": full_causal_mask.numpy().copy(),
                    "full_attention": full_attention.numpy().copy(),
                    "full_head_output": full_head_output.numpy().copy(),
                    "cache_bytes": cache_bytes,
                    "previous_delta": previous_delta,
                    "hidden_error": hidden_error,
                })
                kv_previous_keys, kv_previous_values = keys, values
                if step < decode_steps:
                    kv_input_ids = torch.cat([kv_input_ids, next_id], dim=1)
                    kv_feed_ids = next_id  # Only this new token enters the next forward pass.
    finally:
        kv_hook.remove()
        kv_transformer.train(kv_was_training)

    summary = (
        f"프롬프트: {prompt!r} → 처리한 전체 문장: {kv_records[-1]['text']!r}\n"
        f"{len(kv_records)}단계 확인 완료 · 기존 K/V 변화 0 · 캐시 유무 최대 오차 "
        f"{max(record['hidden_error'] for record in kv_records):.2e}"
    )
    return {"records": kv_records, "metadata": metadata, "summary": summary}


def _build_figure(demo):
    """Build all frames with fixed overlay slots for backward navigation."""
    kv_records = demo["records"]
    metadata = demo["metadata"]
    kv_head_dim = metadata["head_dim"]
    kv_layer, kv_head = metadata["layer"], metadata["head"]
    if not kv_records:
        raise ValueError("The KV demo must contain a prefill record")

    kv_colors = {
        "cached": "#087F8C", "new": "#CC8900", "query": "#7856C7",
        "ink": "#19313C", "reference": "#64748B", "masked": "#E0E5EC",
    }
    kv_signed_scale = [[0, "#C66758"], [0.5, "#FAFAF8"], [1, "#4786AA"]]
    kv_probability_scale = [[0, "#F2F6F5"], [0.3, "#8ACAC4"], [1, "#126A72"]]
    kv_limits = {
        field: max(float(np.abs(record[field]).max()) for record in kv_records) or 1.0
        for field in ("all_queries", "keys", "full_raw_scores", "full_scaled_scores", "values", "full_head_output")
    }


    def kv_matrix_trace(values, hover, *, limit=None, probability=False, text=None):
        values = np.asarray(values)
        return go.Heatmap(
            z=values.tolist(), x=list(range(values.shape[1])), y=list(range(values.shape[0])),
            customdata=hover, hovertemplate="%{customdata}<br>값: %{z:.5f}<extra></extra>",
            colorscale=kv_probability_scale if probability else kv_signed_scale,
            zmin=0 if probability else -limit, zmax=1 if probability else limit,
            showscale=False, hoverongaps=False, xgap=1, ygap=1,
            text=text, texttemplate="%{text}" if text is not None else "",
            textfont=dict(size=11, color=kv_colors["ink"]),
        )


    def kv_step_figure(record):
        total, past, count = record["total_length"], record["past_length"], record["query_length"]
        positions = list(range(total))
        dimensions = list(range(kv_head_dim))
        tokens = [html.escape(token) for token in record["tokens"]]
        mode = "Prefill" if record["step"] == 0 else f"Decode {record['step']}"
        mask = np.asarray(record["full_causal_mask"], dtype=bool)
        scaled = np.asarray(record["full_scaled_scores"])
        probability = np.asarray(record["full_attention"])
        show_numbers = total <= 12

        scale_label = f" / √{kv_head_dim}" if metadata["scale_attn_weights"] else ""
        if metadata["scale_attn_by_inverse_layer_idx"]:
            scale_label += f" / {kv_layer + 1}"
        titles = [
            f"<b>① Q · 전체 위치</b> [{total}, {kv_head_dim}]",
            f"<b>② Kᵀ · 캐시 전치</b> [{kv_head_dim}, {total}]",
            f"<b>③ Q × Kᵀ · 마스크 전</b> [{total}, {total}]",
            f"<b>④ Causal mask M</b> [{total}, {total}]",
            f"<b>⑤ S = QKᵀ{scale_label} + M</b>",
            f"<b>⑥ P = softmax(S)</b> [{total}, {total}]",
            f"<b>⑦ V · 캐시</b> [{total}, {kv_head_dim}]",
            f"<b>⑧ O = P × V</b> [{total}, {kv_head_dim}]",
        ]
        figure = make_subplots(
            rows=3, cols=3, specs=[[{}, {}, {}], [{}, {}, {}], [{}, {}, None]],
            horizontal_spacing=0.12, vertical_spacing=0.13,
            row_heights=[0.36, 0.36, 0.28], subplot_titles=titles,
        )

        def query_role(index):
            return "이번 호출에서 계산" if index >= past else "이전 Q의 설명용 이력 · 이번 호출에서는 계산하지 않음 · KV 캐시 아님"

        def cache_role(index):
            return "이번 호출에서 계산해 캐시에 추가" if index >= past else "이전부터 저장된 캐시를 그대로 읽음"

        def score_hover(row, col, stage):
            state = "이번 호출에서 계산하는 행" if row >= past else "설명용 참조 행 · 이번 decode에서 계산하지 않음"
            causal = "허용: key 위치 ≤ query 위치" if mask[row, col] else "미래 key: causal mask로 차단 → 확률 0"
            return f"{stage}<br>q{row} ({tokens[row]}) → k{col} ({tokens[col]})<br>{state}<br>{causal}"

        def numbers(matrix, decimals=2):
            return [[f"{value:.{decimals}f}" for value in row] for row in matrix] if show_numbers else None

        figure.add_trace(kv_matrix_trace(
            record["all_queries"],
            [[f"Q[{i}, {d}] · {tokens[i]}<br>{query_role(i)}" for d in dimensions] for i in positions],
            limit=kv_limits["all_queries"],
        ), row=1, col=1)
        figure.add_trace(kv_matrix_trace(
            np.asarray(record["keys"]).T,
            [[f"Kᵀ[{d}, {j}] = K[{j}, {d}] · {tokens[j]}<br>{cache_role(j)}" for j in positions] for d in dimensions],
            limit=kv_limits["keys"],
        ), row=1, col=2)
        figure.add_trace(kv_matrix_trace(
            record["full_raw_scores"],
            [[score_hover(i, j, f"QKᵀ · {kv_head_dim}개 성분의 내적") for j in positions] for i in positions],
            limit=kv_limits["full_raw_scores"], text=numbers(record["full_raw_scores"], 1),
        ), row=1, col=3)

        mask_text = [["0" if mask[i, j] else "−∞" for j in positions] for i in positions]
        mask_hover = [[score_hover(i, j, "가산 causal mask M") for j in positions] for i in positions]
        figure.add_trace(go.Heatmap(
            z=(~mask).astype(int).tolist(), x=positions, y=positions,
            zmin=0, zmax=1, colorscale=[[0, "#F4F7FA"], [1, kv_colors["masked"]]],
            text=mask_text, texttemplate="%{text}" if show_numbers else "",
            textfont=dict(size=12, color=kv_colors["ink"]), customdata=mask_hover,
            hovertemplate="%{customdata}<br>M = %{text}<extra></extra>",
            showscale=False, xgap=2, ygap=2,
        ), row=2, col=1)

        masked_values = np.where(mask, scaled, np.nan)
        masked_text = [[f"{scaled[i, j]:.2f}" if mask[i, j] else "" for j in positions] for i in positions]
        figure.add_trace(kv_matrix_trace(
            masked_values,
            [[score_hover(i, j, "스케일 후 + M") for j in positions] for i in positions],
            limit=kv_limits["full_scaled_scores"], text=masked_text if show_numbers else None,
        ), row=2, col=2)
        # Keep the upper triangle visible, with explicit masked cells.
        figure.add_trace(go.Heatmap(
            z=np.where(mask, np.nan, 1).tolist(), x=positions, y=positions,
            colorscale=[[0, kv_colors["masked"]], [1, kv_colors["masked"]]],
            zmin=0, zmax=1, text=mask_text,
            texttemplate="%{text}" if show_numbers else "", textfont=dict(size=12, color="#536273"),
            customdata=mask_hover, hovertemplate="%{customdata}<br>마스크 후: −∞<extra></extra>",
            showscale=False, hoverongaps=False, xgap=2, ygap=2,
        ), row=2, col=2)
        figure.add_trace(kv_matrix_trace(
            probability,
            [[score_hover(i, j, "P · 행별 softmax") for j in positions] for i in positions],
            probability=True, text=numbers(probability),
        ), row=2, col=3)
        figure.add_trace(kv_matrix_trace(
            record["values"],
            [[f"V[{i}, {d}] · {tokens[i]}<br>{cache_role(i)}" for d in dimensions] for i in positions],
            limit=kv_limits["values"],
        ), row=3, col=1)
        figure.add_trace(kv_matrix_trace(
            record["full_head_output"],
            [[f"O[{i}, {d}] = Σⱼ P[{i}, j] V[j, {d}]<br>"
              + ("이번 호출에 필요한 출력" if i >= past else "이전 query의 설명용 출력 · 이번 decode에서 재계산하지 않음")
              + "<br>head 하나의 출력 · 출력 projection 전" for d in dimensions] for i in positions],
            limit=kv_limits["full_head_output"],
        ), row=3, col=2)

        feature_ticks = sorted(set([0, kv_head_dim // 4, kv_head_dim // 2, 3 * kv_head_dim // 4, kv_head_dim - 1]))
        for row, col in ((1, 1), (3, 1), (3, 2)):
            figure.update_xaxes(range=[-0.5, kv_head_dim - 0.5], tickvals=feature_ticks, title_text="head의 성분 d", row=row, col=col)
            figure.update_yaxes(range=[total - 0.5, -0.5], tickvals=positions, ticktext=[str(i) for i in positions], title_text="토큰 위치", row=row, col=col)
        figure.update_xaxes(range=[-0.5, total - 0.5], tickvals=positions, title_text="key 위치 j", row=1, col=2)
        figure.update_yaxes(range=[kv_head_dim - 0.5, -0.5], tickvals=feature_ticks, title_text="성분 d", row=1, col=2)
        for row, col in ((1, 3), (2, 1), (2, 2), (2, 3)):
            figure.update_xaxes(range=[-0.5, total - 0.5], tickvals=positions, title_text="key 위치 j", row=row, col=col)
            figure.update_yaxes(range=[total - 0.5, -0.5], tickvals=positions, title_text="query 위치 i", row=row, col=col)
        figure.update_xaxes(showgrid=False, zeroline=False, fixedrange=True, tickfont=dict(size=10), title_font=dict(size=11))
        figure.update_yaxes(showgrid=False, zeroline=False, fixedrange=True, tickfont=dict(size=10), title_font=dict(size=11))

        # Fixed shape slots prevent stale overlays when stepping backwards.
        for row, col, width in ((1, 1, kv_head_dim), (1, 3, total), (2, 1, total), (2, 2, total), (2, 3, total), (3, 2, kv_head_dim)):
            figure.add_shape(
                type="rect", x0=-0.5, x1=width - 0.5, y0=-0.5, y1=max(past - 0.5, -0.5),
                fillcolor="rgba(248,250,252,0.46)", line=dict(color="#A0ACBA", width=1, dash="dot"),
                visible=past > 0, layer="above", row=row, col=col,
            )
            figure.add_shape(
                type="rect", x0=-0.5, x1=width - 0.5, y0=past - 0.5, y1=total - 0.5,
                fillcolor="rgba(0,0,0,0)", line=dict(color=kv_colors["query"], width=3),
                layer="above", row=row, col=col,
            )
        for row, col, transposed in ((1, 2, True), (3, 1, False)):
            for start, end, color, visible in ((0, past, kv_colors["cached"], past > 0), (past, total, kv_colors["new"], True)):
                figure.add_shape(
                    type="rect", x0=start - 0.5 if transposed else -0.5,
                    x1=end - 0.5 if transposed else kv_head_dim - 0.5,
                    y0=-0.5 if transposed else start - 0.5,
                    y1=kv_head_dim - 0.5 if transposed else end - 0.5,
                    fillcolor="rgba(0,0,0,0)", line=dict(color=color, width=3),
                    visible=visible, layer="above", row=row, col=col,
                )

        q_positions = f"0…{total - 1}" if past == 0 else str(total - 1)
        row_note = "모든 query 행을 계산" if past == 0 else f"맨 아래 q{total - 1} 행만 계산"
        token_line = "　".join(f"{i}: {token}" for i, token in enumerate(tokens))
        if len(token_line) > 145:
            token_line = token_line[:142] + "…"
        notes = [
            dict(x=0, y=1.155, text=f"<b>이번 입력 {count}개</b> · Q [{count}, {kv_head_dim}] × Kᵀ [{kv_head_dim}, {total}] → 실제 점수 [{count}, {total}] · {row_note}"),
            dict(x=0, y=1.105, text=(
                f"<span style='color:{kv_colors['query']}'>━ 이번 Q·점수·출력</span>　<span style='color:{kv_colors['cached']}'>━ 저장된 K·V</span>　<span style='color:{kv_colors['new']}'>━ 새 K·V</span>　<span style='color:{kv_colors['reference']}'>┄ 흐린 행: 설명용 과거 값</span>"
                "<br><span style='font-size:10px'>셀 색: 음수는 붉은색, 양수는 푸른색 · 행렬별 색 범위는 단계 간 고정 · P는 진한 청록색일수록 높은 확률</span>"
            )),
            dict(x=0, y=1.05, text=token_line, font=dict(size=11)),
            dict(x=0.77, y=0.22, text=(
                f"<b>이번 스텝에서 필요한 영역</b><br><br>"
                f"Q: 위치 {q_positions}<br>Kᵀ: 모든 {total}개 열<br>V: 모든 {total}개 행<br><br>"
                f"점수: {count} × {total}개<br>"
                + ("Prefill: 위쪽 삼각형을 가림" if past == 0 else "Decode: 마지막 query에는<br>현재 prefix의 모든 key가 허용됨")
            ), font=dict(size=12)),
            dict(x=0, y=-0.095, text="QKᵀ의 위쪽 삼각형도 원래 점수는 존재합니다. M에서 미래 key를 −∞로 가리고, P에서 0으로 만듭니다.", font=dict(size=12)),
            dict(x=0, y=-0.14, text="전체 Q·과거 점수·과거 출력은 설명용입니다. 실제 KV 캐시는 K·V만 보관하며, decode는 보라색 행만 계산합니다.", font=dict(size=12)),
            dict(x=0, y=-0.185, text=f"Layer {kv_layer} / Head {kv_head} · 모든 {kv_head_dim}개 성분 표시 · 칸에 마우스를 올려 값 확인 · 다음 토큰: {html.escape(record['next_token'])}", font=dict(size=11)),
            dict(x=0, y=-0.23, text=f"전체 층 K·V: {record['cache_bytes'] / 1024:.0f} KiB · 기존 K·V 변화: {'해당 없음' if past == 0 else '0'} · 캐시 없는 결과와 최대 차이: {record['hidden_error']:.1e}", font=dict(size=11)),
        ]
        for note in notes:
            figure.add_annotation(xref="paper", yref="paper", xanchor="left", yanchor="top", align="left", showarrow=False, **note)
        for annotation in figure.layout.annotations[:len(titles)]:
            annotation.font = dict(size=13, color=kv_colors["ink"])
        figure.update_layout(
            template="plotly_white", width=1180, height=1370,
            margin=dict(l=72, r=45, t=260, b=225),
            paper_bgcolor="#FAFBFC", plot_bgcolor="#FAFBFC",
            font=dict(family="Arial, Apple SD Gothic Neo, sans-serif", size=12, color=kv_colors["ink"]),
            title=dict(
                text=f"<b>KV cache · {mode} · 전체 행렬에서 이번 계산 찾기</b><br><sup>Q × Kᵀ → scale + causal mask → softmax → P × V</sup>",
                x=0.02, y=0.99, xanchor="left", yanchor="top", font=dict(size=23),
            ),
            hoverlabel=dict(font_size=12),
        )
        return figure


    kv_figures = [kv_step_figure(record) for record in kv_records]
    kv_figure = go.Figure(kv_figures[0])
    kv_figure.frames = [go.Frame(name=str(index), data=figure.data, layout=figure.layout) for index, figure in enumerate(kv_figures)]
    kv_figure.update_layout(sliders=[dict(
        active=0, x=0, y=1.205, len=1, pad=dict(t=0, b=0),
        currentvalue=dict(visible=False),
        steps=[dict(
            method="animate", label="Prefill" if index == 0 else f"Decode {index}",
            args=[[str(index)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        ) for index in range(len(kv_records))],
    )])
    return kv_figure


def render_kv_cache(demo):
    """Return self-contained notebook HTML without displaying or using a model.

    Use ``display(render_kv_cache(demo))`` in the notebook. Plotly is embedded
    so the saved output remains interactive without a running kernel or CDN.
    """
    figure = _build_figure(demo)
    content = figure.to_html(
        full_html=False, include_plotlyjs=True, auto_play=False,
        config={"displayModeBar": False, "responsive": False},
    )
    return HTML("<div style='width:100%;overflow-x:auto'>" + content + "</div>")
