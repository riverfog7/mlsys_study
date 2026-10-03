"""Plotly charts for study03. Importing this module does not run profiling."""

from collections import defaultdict
from textwrap import dedent
import html as _html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, HTML, Markdown

_PROFILE_GROUPS = {
    'QKV projection': 'QKV / 출력 projection', 'Output projection': 'QKV / 출력 projection',
    'QKᵀ': 'QKᵀ / PV', 'PV': 'QKᵀ / PV',
    'Scale / causal mask': '스케일·마스크·softmax', 'Softmax': '스케일·마스크·softmax',
    'Attention copies': '기타 / 복사 / norm', 'Other / norms': '기타 / 복사 / norm',
    'MLP': 'MLP', 'LM head': 'LM head',
}
_PROFILE_ORDER = ['QKV / 출력 projection', 'QKᵀ / PV', '스케일·마스크·softmax', 'MLP', 'LM head', '기타 / 복사 / norm']
_PROFILE_COLORS = ['#5b8fb9', '#3b68af', '#de7735', '#a1aab6', '#c2c8d0', '#dfe3e8']


def profile_overview(runs):
    lengths = [run['tokens'] for run in runs]
    counts = []
    for run in runs:
        totals = defaultdict(float)
        for kernel in run['kernels']:
            totals[_PROFILE_GROUPS[kernel['stage']]] += kernel['duration_us'] / 1000
        counts.append(totals)

    fig = make_subplots(rows=1, cols=2, subplot_titles=['일반 forward · CUDA events', 'GPU 커널 시간 구성 · torch.profiler'], horizontal_spacing=0.14)
    fig.add_trace(go.Bar(
        x=[str(n) for n in lengths], y=[r['median_ms'] for r in runs], name='전체 forward',
        marker_color='#606d7e', showlegend=False,
        error_y=dict(type='data', symmetric=False,
                     array=[max(r['elapsed_ms'])-r['median_ms'] for r in runs],
                     arrayminus=[r['median_ms']-min(r['elapsed_ms']) for r in runs]),
        text=[f'{r["median_ms"]:.1f} ms' for r in runs], textposition='outside',
        hovertemplate='%{x} tokens<br>중앙값 %{y:.2f} ms<extra></extra>',
    ), row=1, col=1)
    for group, color in zip(_PROFILE_ORDER, _PROFILE_COLORS):
        fig.add_trace(go.Bar(x=[str(n) for n in lengths], y=[c[group] for c in counts],
                             name=group, marker_color=color,
                             hovertemplate='%{x} tokens<br>%{fullData.name}<br>%{y:.3f} ms<extra></extra>'), row=1, col=2)
    fig.update_layout(template='plotly_white', barmode='stack', height=450,
                      title='GPT-2 전체 forward: 입력이 길어지면 어디의 시간이 늘어날까?',
                      legend=dict(orientation='h', y=-0.24), margin=dict(t=90, b=125))
    fig.update_xaxes(title_text='입력 길이 (tokens)')
    fig.update_yaxes(title_text='시간 (ms)', rangemode='tozero')
    return fig


def attention_flow(config):
    D, heads = config.n_embd, config.n_head
    d = D // heads
    flow = go.Figure()

    def flow_box(x, y, width, height, text, fill):
        flow.add_shape(type='rect', x0=x-width/2, x1=x+width/2,
                       y0=y-height/2, y1=y+height/2,
                       fillcolor=fill, line=dict(color='#c3ccd7', width=1), layer='below')
        flow.add_annotation(x=x, y=y, text=text, showarrow=False,
                            font=dict(size=14, color='#25384d'), align='center')

    def flow_arrow(x0, y0, x1, y1):
        flow.add_annotation(x=x1, y=y1, ax=x0, ay=y0,
                            xref='x', yref='y', axref='x', ayref='y',
                            text='', showarrow=True, arrowhead=2,
                            arrowsize=1, arrowwidth=1.7, arrowcolor='#62758a')

    flow_box(50, 94, 60, 10, '<b>Attention 입력 X</b><br>N × D', '#f0f3f7')
    flow_box(18, 75, 27, 11, '<b>Q = XW_Q</b><br>query · N × d', '#e8f0f8')
    flow_box(50, 75, 27, 11, '<b>K = XW_K</b><br>key · N × d', '#e8f0f8')
    flow_box(82, 75, 27, 11, '<b>V = XW_V</b><br>전달할 내용 · N × d', '#e8f0f8')
    flow_box(34, 57, 50, 11, '<b>① 점수표 S = QKᵀ</b><br>N × N · query별 key 점수', '#e8f0f8')
    flow_box(34, 39, 50, 11, '<b>② scale + causal mask</b><br>√d로 나누고 미래 토큰 가리기', '#fbeadd')
    flow_box(34, 21, 50, 11, '<b>③ 행마다 softmax → P</b><br>N × N · 각 행의 가중치 합 = 1', '#fbeadd')
    flow_box(50, 3, 64, 11, '<b>④ head 출력 O = PV</b><br>N × d · P의 비율로 V 벡터들을 가중합', '#e8f0f8')
    flow_box(50, -16, 84, 10,
             f'<b>{heads}개 head 출력 합치기 → 출력 projection W_O</b><br>N × ({heads} · {d}) → N × {D}',
             '#f0f3f7')

    for edge in [(26,89,18,80.5), (50,89,50,80.5), (74,89,82,80.5),
                 (18,69.5,22,62.5), (50,69.5,46,62.5),
                 (34,51.5,34,44.5), (34,33.5,34,26.5),
                 (34,15.5,42,8.5), (50,-2.5,50,-11)]:
        flow_arrow(*edge)
    # V는 점수표 처리 경로를 지나지 않고 마지막 가중합의 입력으로 전달됩니다.
    flow.add_shape(type='line', x0=82, y0=69.5, x1=82, y1=12,
                   line=dict(color='#62758a', width=1.7))
    flow_arrow(82,12,74,8.5)
    flow.update_layout(
        template='plotly_white', height=760, showlegend=False,
        title=dict(text=f'Attention: 점수 → 관심 비율 → 내용의 가중합<br><sup>N = 토큰 수 · D = {D} · head 차원 d = {d}</sup>', x=0.03),
        margin=dict(l=15, r=15, t=85, b=15),
        xaxis=dict(range=[0,100], visible=False, fixedrange=True),
        yaxis=dict(range=[-27,103], visible=False, fixedrange=True),
    )
    return flow


def attention_timeline(run, block=0):
    attention = [k for k in run['kernels'] if k['block'] == block and k['scope'].split('/')[-1] in ('attention', 'qkv', 'out')]
    origin = attention[0]['start_us']
    labels = [f'{i+1:02d}  {k["stage"]} · {k["operator"].removeprefix("aten::")}' for i, k in enumerate(attention)]
    stage_color = {stage: _PROFILE_COLORS[_PROFILE_ORDER.index(group)] for stage, group in _PROFILE_GROUPS.items()}
    timeline = go.Figure(go.Bar(
        y=labels, x=[k['duration_us']/1000 for k in attention],
        base=[(k['start_us']-origin)/1000 for k in attention], orientation='h',
        marker_color=[stage_color[k['stage']] for k in attention],
        customdata=[[k['operator'], k['name'], k['duration_us']] for k in attention],
        hovertemplate='%{customdata[0]}<br>%{customdata[2]:.2f} µs<br>%{customdata[1]}<extra></extra>',
    ))
    timeline.update_layout(template='plotly_white', title=f'{block}번 블록 attention · 실제 GPU 타임라인',
                           height=530, margin=dict(l=260, r=35, t=70, b=55),
                           xaxis_title='첫 attention 커널 시작 이후 (ms)',
                           yaxis=dict(autorange='reversed'), showlegend=False)
    return timeline


def hardware_counters(hardware):
    records = hardware['kernels']
    labels = [f'{i+1:02d}  {k["stage"]} · {k["operator"].removeprefix("aten::")}' for i,k in enumerate(records)]
    read_mib = [k['metrics']['dram__bytes_read.sum']/2**20 for k in records]
    write_mib = [k['metrics']['dram__bytes_write.sum']/2**20 for k in records]
    dram_util = [k['metrics']['dram__throughput.avg.pct_of_peak_sustained_elapsed'] for k in records]
    sm_util = [k['metrics']['sm__throughput.avg.pct_of_peak_sustained_elapsed'] for k in records]

    counter_fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.08,
                               subplot_titles=['DRAM 데이터 이동량', '자원 처리율 · 최대치 대비'])
    for values, name, color in [(read_mib, 'DRAM 읽기', '#5b8fb9'), (write_mib, 'DRAM 쓰기', '#de7735')]:
        counter_fig.add_trace(go.Bar(y=labels, x=values, orientation='h', name=name, marker_color=color, offsetgroup='traffic', alignmentgroup='traffic',
                                    hovertemplate='%{y}<br>%{fullData.name}: %{x:.2f} MiB<extra></extra>'), row=1, col=1)
    for values, name, color in [(dram_util, 'DRAM 대역폭', '#de7735'), (sm_util, 'SM 처리율', '#3b68af')]:
        # 별도 offsetgroup으로 커널마다 두 막대를 나란히 배치합니다. 처리율을 합산하지 않습니다.
        counter_fig.add_trace(go.Bar(y=labels, x=values, orientation='h', name=name,
                                    offsetgroup=name, alignmentgroup='utilization', marker_color=color,
                                    text=[f'{value:.1f}%' for value in values], textposition='outside',
                                    cliponaxis=False, textfont=dict(size=11), constraintext='none',
                                    hovertemplate='%{y}<br>%{fullData.name}: %{x:.1f}%<extra></extra>'), row=1, col=2)
    counter_fig.update_layout(template='plotly_white', title='0번 블록 attention · 실제 하드웨어 카운터',
                             barmode='stack', bargap=0.2, bargroupgap=0.1, height=680, margin=dict(l=255, r=30, t=90, b=110),
                             legend=dict(orientation='h', y=-0.16, traceorder='normal'))
    counter_fig.update_yaxes(autorange='reversed')
    counter_fig.update_xaxes(title_text='읽기 + 쓰기 (MiB)', rangemode='tozero', row=1, col=1)
    counter_fig.update_xaxes(title_text='최대치 대비 처리율 (%)', range=[0, 115], tickvals=[0, 50, 100], row=1, col=2)
    return counter_fig


def profile_summary(run, hardware):
    records = hardware['kernels']
    middle_stages = {'Scale / causal mask', 'Softmax'}
    all_kernel_ms = sum(k['duration_us'] for k in run['kernels']) / 1000
    middle_ms = sum(k['duration_us'] for k in run['kernels'] if k['stage'] in middle_stages) / 1000
    core_ms = sum(k['duration_us'] for k in run['kernels'] if k['stage'] in middle_stages | {'QKᵀ', 'PV'}) / 1000
    middle_records = [k for k in records if k['stage'] in middle_stages and k['operator'] != 'aten::fill_']
    traffic = sum(k['metrics']['dram__bytes_read.sum'] + k['metrics']['dram__bytes_write.sum'] for k in middle_records) / 2**20
    bandwidths = [k['metrics']['dram__throughput.avg.pct_of_peak_sustained_elapsed'] for k in middle_records]
    compute = [k['metrics']['sm__throughput.avg.pct_of_peak_sustained_elapsed'] for k in middle_records]
    summary = f"""
    - **실제 forward:** {run['tokens']:,}토큰, {run['median_ms']:.2f} ms (프로파일러 없이 20회 중앙값).
    - **스케일·마스크·softmax:** 12개 블록 합계 **{middle_ms:.2f} ms**, 프로파일된 전체 GPU 커널 시간의 **{middle_ms/all_kernel_ms:.1%}**.
    - **attention 핵심 연산 안에서의 비중:** 위 중간 처리가 QKᵀ → 스케일·마스크·softmax → PV 커널 시간의 **{middle_ms/core_ms:.1%}**. QKV/출력 projection은 이 비율의 분모에서 제외했습니다.
    - **0번 블록의 중간 처리:** 실제 DRAM 읽기+쓰기 **{traffic:.1f} MiB**. 해당 큰 커널들의 DRAM 대역폭 지표 **{min(bandwidths):.1f}~{max(bandwidths):.1f}%**, SM 처리율 **{min(compute):.1f}~{max(compute):.1f}%**.
    """
    return dedent(summary)


def show_attention_comparison(comparison):
    meta, runs = comparison['metadata'], comparison['lengths']
    labels = {key: entry['label'] for key, entry in comparison['backends'].items()}
    colors = {'eager': '#7d8795', 'turing': '#dc8b27', 'efficient': '#2189b6', 'flash': '#8166b3'}
    methods = [key for key in ('eager', 'turing', 'efficient', 'flash') if any(
        run['attention_core'].get(key, {}).get('status') == 'measured' for run in runs)]
    rows = []
    for key, entry in comparison['backends'].items():
        reason = entry.get('reason', '')
        reason = next((part.strip() for part in reason.split('|')
                       if 'only supports gpu architectures' in part), reason.split('|')[0])
        rows.append(f"<tr><td>{_html.escape(entry['label'])}</td>"
                    f"<td>{'실측 완료' if entry['status']=='measured' else '실행 불가'}</td>"
                    f"<td>{_html.escape(reason)}</td></tr>")
    display(Markdown(f"**{meta['gpu']} · FP16 · batch 1 · causal prefill**  \n"
                     f"측정: {meta['measured_at_utc']} · PyTorch {meta['torch']} · Transformers {meta['transformers']}  \n"
                     f"각 경로 warmup {meta['warmup']}회 후 {meta['repeats']}회 측정했습니다. "
                     "Eager도 FP16이며 앞의 1절 FP32 수치와 섞지 않습니다."))
    display(HTML('<table><thead><tr><th>실행 경로</th><th>상태</th><th>이유</th></tr></thead><tbody>'
                 + ''.join(rows) + '</tbody></table>'))
    if 'turing' in methods:
        turing = meta['turing']
        revision = turing.get('source_revision')
        source = turing['source_url'] + ('/tree/'+revision if revision else '')
        display(Markdown('**주황색 막대는 실제로 실행한 [FlashAttention Turing (ssiu)]('
                         + source + ')입니다.** Turing용 FA2 구현이며 공식 FA1과 구분합니다. '
                         'PyTorch 내장 FlashAttention의 미지원 여부는 위 표에 별도로 표시합니다.  \n'
                         f"빌드: `{turing.get('package_version')}` · GPT-2에서 쓰는 head 크기 64의 causal 순전파. "
                         '커널 소스는 원본이며, 입력 재배열과 복사 비용도 측정에 포함했습니다.'))
    elif comparison['backends']['flash']['status'] != 'measured':
        display(Markdown('**이번 실행에는 FlashAttention 실측값이 없습니다.** '
                         'SDPA efficient는 별도 구현이며, 실행 불가인 경로에는 막대를 그리지 않습니다.'))

    fig = make_subplots(rows=2, cols=2, vertical_spacing=.18, horizontal_spacing=.12,
                        subplot_titles=['Attention core (block 0)', 'Full GPT-2 forward',
                                        'Attention core (block 0)', 'Full GPT-2 forward'])
    lengths = [str(run['tokens']) for run in runs]
    for column, scope in enumerate(('attention_core', 'model_forward'), start=1):
        for row, (field, unit, scale) in enumerate([
            ('median_ms', 'Latency (ms)', 1),
            ('peak_extra_bytes', 'Extra allocated memory (MiB)', 2**20),
        ], start=1):
            highest = 0.0
            shared_labels = {}
            if row == 2 and column == 2:
                for index, run in enumerate(runs):
                    values = [run[scope][key].get(field) for key in methods]
                    if all(value is not None for value in values) and max(values) == min(values):
                        shared_labels[index] = values[0] / scale
            for key in methods:
                records = [run[scope].get(key, {}) for run in runs]
                values = [r[field]/scale if r.get('status')=='measured' else None for r in records]
                error_y = None
                if row == 1:
                    upper = [r['max_ms']-r['median_ms'] if r.get('status')=='measured' else 0 for r in records]
                    lower = [r['median_ms']-r['min_ms'] if r.get('status')=='measured' else 0 for r in records]
                    error_y = dict(type='data', symmetric=False, array=upper, arrayminus=lower, thickness=1, width=3)
                    highest = max(highest, *(r.get('max_ms', 0) for r in records))
                else:
                    highest = max(highest, *(v or 0 for v in values))
                fig.add_trace(go.Bar(
                    x=lengths, y=values, name=labels[key], legendgroup=key, offsetgroup=key,
                    showlegend=row==1 and column==1, marker_color=colors[key], error_y=error_y,
                    text=['' if v is None or i in shared_labels else f'{v:.3f}' if row==1 else f'{v:.2f}' for i,v in enumerate(values)],
                    textposition='outside', textfont=dict(size=10), cliponaxis=False,
                    hovertemplate='<b>%{fullData.name}</b><br>%{x} tokens<br>'+unit+': %{y:.4f}<extra></extra>',
                ), row=row, col=column)
            for index, value in shared_labels.items():
                fig.add_annotation(x=index, y=value, text=f'{value:.2f}',
                                   showarrow=False, yshift=12, row=row, col=column)
            fig.update_yaxes(title_text=unit, range=[0, highest*1.25], row=row, col=column)
            fig.update_xaxes(title_text='Input tokens', type='category', row=row, col=column)
    fig.update_layout(template='plotly_white', barmode='group', height=760,
                      margin=dict(t=105, b=55, l=65, r=25),
                      legend=dict(orientation='h', x=0, y=1.15, xanchor='left', yanchor='bottom'),
                      font=dict(size=12))
    # Native Plotly MIME keeps the saved notebook small and interactive in JupyterLab.
    display({'application/vnd.plotly.v1+json': fig.to_plotly_json()}, raw=True)
    display(Markdown('위쪽은 실행 시간, 아래쪽은 실행 중 **추가로 할당된 GPU 메모리의 최대량**입니다. '
                     '범례를 누르면 경로를 숨기거나 다시 볼 수 있고, 막대에 마우스를 올리면 수치가 나옵니다. '
                     '시간 막대는 중앙값, 오차 막대는 최솟값~최댓값입니다. '
                     '메모리 값은 모델 가중치 전체 크기나 DRAM 전송 바이트가 아닙니다.'))

    longest = max(runs, key=lambda r: r['tokens'])
    summary = []
    for key in methods:
        if key == 'eager':
            continue
        core, full = longest['attention_core'][key], longest['model_forward'][key]
        if core.get('status')!='measured' or full.get('status')!='measured':
            continue
        base_core, base_full = longest['attention_core']['eager'], longest['model_forward']['eager']
        summary.append(f"**{labels[key]} · {longest['tokens']}토큰:** eager 대비 속도는 attention "
                       f"{base_core['median_ms']/core['median_ms']:.2f}배, 모델 전체 "
                       f"{base_full['median_ms']/full['median_ms']:.2f}배입니다. "
                       f"Attention의 추가 할당은 {base_core['peak_extra_bytes']/2**20:.2f} → "
                       f"{core['peak_extra_bytes']/2**20:.2f} MiB, 모델 전체는 "
                       f"{base_full['peak_extra_bytes']/2**20:.2f} → {full['peak_extra_bytes']/2**20:.2f} MiB입니다.")
    display(Markdown('\n\n'.join(summary)))
    display(Markdown('전체 모델에는 projection·MLP·모든 위치의 logits 계산도 들어갑니다. '
                     'Attention의 속도 향상률이 전체 속도에 그대로 적용되지는 않으며, '
                     '전체 메모리 최대량도 attention 밖의 큰 출력에 의해 결정될 수 있습니다.'))

    accuracy = ['| 토큰 | 경로 | Attention 최대 오차 | 예측 1위 일치율 | 최대 확률 차이 |',
                '| ---: | --- | ---: | ---: | ---: |']
    for run in runs:
        for key in methods:
            if key=='eager' or run['model_forward'][key].get('status')!='measured':
                continue
            core, full = run['attention_core'][key]['correctness'], run['model_forward'][key]['correctness']
            accuracy.append(f"| {run['tokens']} | {labels[key]} | {core['max_abs']:.6f} | "
                            f"{full['argmax_agreement']*100:.2f}% | {full['max_probability_difference']*100:.3f}%p |")
    display(Markdown('**출력도 비교했습니다.** Attention 오차는 같은 Q·K·V를 FP32로 계산한 결과가 기준입니다. '
                     '예측 1위와 확률 차이는 전체 모델의 FP16 eager가 기준입니다. '
                     '일치율은 정답률이 아니라 두 경로가 같은 다음 토큰을 1위로 골랐는지입니다. '
                     'FP16 출력과 예측은 완전히 같지는 않습니다.\n\n'+'\n'.join(accuracy)))
    kernel_rows = []
    for key in methods:
        record = longest['attention_core'][key]
        if record.get('status')!='measured':
            continue
        profile = record['profile']
        names = '<br>'.join(_html.escape(k['name']) for k in profile['kernels'])
        kernel_rows.append(f"<tr><td>{_html.escape(labels[key])}</td><td>{profile['kernel_count']}</td>"
                           f"<td style='overflow-wrap:anywhere'>{names}</td></tr>")
    display(HTML(f'<details><summary>실제로 실행된 attention 커널 · {longest["tokens"]}토큰</summary>'
                 '<p>시간 측정과 별도로 기록했습니다. Turing 경로는 실제 flash_fwd_kernel을 확인하고 SDPA 호출이 없는지도 검사합니다.</p>'
                 '<table style="table-layout:fixed;width:100%"><thead><tr><th>경로</th><th>커널 수</th><th>CUDA 커널 이름</th></tr></thead><tbody>'
                 + ''.join(kernel_rows) + '</tbody></table></details>'))
