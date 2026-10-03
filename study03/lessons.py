"""Self-contained notebook lessons. Importing this module renders no examples."""

import html
import json
import math
import uuid

import numpy as np
from IPython.display import HTML, IFrame

def _softmax_direct(scores):
    scores=np.asarray(scores,dtype=np.float32)
    with np.errstate(over="ignore",invalid="ignore",divide="ignore",under="ignore"):
        exponentials=np.exp(scores)
        denominator=np.sum(exponentials,axis=-1,dtype=np.float32)
        probabilities=exponentials/(denominator[...,None] if scores.ndim>1 else denominator)
    return exponentials,denominator,probabilities


def _softmax_stable(scores):
    scores=np.asarray(scores,dtype=np.float32)
    maximum=np.max(scores,axis=-1)
    shifted=scores-(maximum[...,None] if scores.ndim>1 else maximum)
    exponentials,denominator,probabilities=_softmax_direct(shifted)
    return maximum,shifted,exponentials,denominator,probabilities


def render_softmax(q, k, v, *, mode="normal", query_index=0,
                   shifts=(0, 85.5, 1000)):
    """Return an interactive HTML lesson using the current 4×2 Q, K and V.

    Q represents the already-scaled query; scores are recomputed in FP32 on every
    call. Modes are "normal", "stable" and "online". Normal preserves the supplied
    offset order; stable visits offsets in descending order. Online uses unshifted
    scores and two K/V blocks. Non-finite normal-softmax results are intentional.
    The caller displays the returned HTML as the notebook cell's final expression.
    """
    if mode not in ("normal", "stable", "online"):
        raise ValueError("mode must be 'normal', 'stable' or 'online'")
    SM_Q, SM_K, SM_VALUES = (np.asarray(value, dtype=np.float32) for value in (q, k, v))
    if any(value.shape != (4, 2) for value in (SM_Q, SM_K, SM_VALUES)):
        raise ValueError("This lesson shows complete 4×2 Q, K and V matrices.")
    if not all(np.isfinite(value).all() for value in (SM_Q, SM_K, SM_VALUES)):
        raise ValueError("Q, K and V must contain finite values.")
    if not isinstance(query_index, (int, np.integer)) or not 0 <= query_index < 4:
        raise ValueError("query_index must be an integer from 0 through 3")
    SM_QUERY_INDEX = int(query_index)
    SM_SHIFTS = tuple(shifts)
    if not SM_SHIFTS and mode != "online":
        raise ValueError("normal and stable modes require at least one offset")
    SM_SCORE_MATRIX = SM_Q @ SM_K.T

    sm_palette = {
        "ink": "#17313F", "muted": "#657487", "line": "#D6DEE8",
        "blue": "#317BA0", "teal": "#087F8C", "purple": "#7856C7",
        "amber": "#C78615", "red": "#C34F4F", "pale": "#F3F6FA",
    }


    def sm_escape(value):
        return html.escape(str(value))


    def sm_number(value, digits=4):
        value = float(value)
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "−∞" if value < 0 else "+∞"
        if value == 0:
            return "0"
        if abs(value) >= 10000 or abs(value) < 0.0001:
            return f"{value:.{max(digits - 1, 1)}e}"
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")


    def sm_math_markup(value, size):
        import re
        escaped = sm_escape(value)
        return re.sub(r"e\^\(([^()]*)\)",
                      lambda match: 'e<tspan baseline-shift="super" font-size="'
                      + str(max(11, size * 0.72)) + '">' + match.group(1) + '</tspan>', escaped)


    def sm_text(x, y, value, size=15, color=None, anchor="start", weight=400):
        return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color or sm_palette["ink"]}" '
                f'text-anchor="{anchor}" font-weight="{weight}">{sm_math_markup(value,size)}</text>')


    def sm_rect(x, y, width, height, fill="white", stroke=None, radius=10, dash=None):
        dashed = f' stroke-dasharray="{dash}"' if dash else ""
        return (f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" '
                f'fill="{fill}" stroke="{stroke or sm_palette["line"]}" stroke-width="1.5"{dashed}/>')


    def sm_arrow(x1, y1, x2, y2, label=None, color=None):
        color = color or sm_palette["muted"]
        angle = math.atan2(y2 - y1, x2 - x1)
        left = (x2 - 9 * math.cos(angle) + 4 * math.sin(angle), y2 - 9 * math.sin(angle) - 4 * math.cos(angle))
        right = (x2 - 9 * math.cos(angle) - 4 * math.sin(angle), y2 - 9 * math.sin(angle) + 4 * math.cos(angle))
        result = (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2"/>'
                  f'<polygon points="{x2},{y2} {left[0]},{left[1]} {right[0]},{right[1]}" fill="{color}"/>')
        if label:
            result += sm_text((x1 + x2) / 2, (y1 + y2) / 2 - 10, label, 12, color, "middle")
        return result


    def sm_svg(parts, height, title):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {height}" '
                f'role="img" aria-label="{sm_escape(title)}" style="display:block;width:100%;min-width:900px;height:auto;'
                'font-family:Arial,Apple SD Gothic Neo,Malgun Gothic,sans-serif;background:#FCFDFE">'
                f'<title>{sm_escape(title)}</title>' + "".join(parts) + '</svg>')


    def sm_vector(x, y, width, title, values, color, probability=False, indices=None):
        values = list(values)
        indices = list(range(len(values))) if indices is None else list(indices)
        result = sm_rect(x, y, width, 66 + 32 * len(values), stroke=color)
        result += sm_text(x + 16, y + 29, title, 17, color, weight=700)
        for offset, (index, value) in enumerate(zip(indices, values)):
            row_y = y + 46 + 32 * offset
            finite = np.isfinite(value)
            result += sm_rect(x + 12, row_y, width - 24, 27, fill=sm_palette["pale"], stroke=sm_palette["pale"], radius=4)
            result += sm_text(x + 22, row_y + 18, f"{index}", 11, sm_palette["muted"])
            result += sm_text(x + 51, row_y + 19, sm_number(value), 14, color if finite else sm_palette["red"], weight=600)
            if probability and finite:
                bar_width = (width - 164) * min(max(float(value), 0), 1)
                result += sm_rect(x + 144, row_y + 8, max(bar_width, 0), 10, fill=color, stroke=color, radius=2)
            if not finite:
                result += sm_text(x + width - 20, row_y + 18, "!", 14, sm_palette["red"], "end", 700)
        return result


    def sm_state(x, y, width, title, lines, color):
        result = sm_rect(x, y, width, 58 + 31 * len(lines), stroke=color)
        result += sm_text(x + 16, y + 29, title, 17, color, weight=700)
        for index, line in enumerate(lines):
            result += sm_text(x + 16, y + 62 + 31 * index, line, 15)
        return result


    def sm_deck(title, frames, note, show=True, context=None):
        # Native radio inputs and CSS make the steps interactive without JavaScript,
        # external packages, a CDN, or a running kernel after the output is rendered.
        deck_id = "sm_" + uuid.uuid4().hex
        rules = []
        choices, labels, scenes = [], [], []
        for index, frame in enumerate(frames):
            choice_id = f"{deck_id}_choice_{index}"
            rules.extend([
                f'#{choice_id}:checked ~ .sm-scenes > .sm-scene[data-step="{index}"] {{display:block;}}',
                f'#{choice_id}:checked ~ .sm-toolbar label[for="{choice_id}"] {{background:#7856C7;color:white;border-color:#7856C7;}}',
                f'#{choice_id}:focus-visible ~ .sm-toolbar label[for="{choice_id}"] {{outline:3px solid #B4A2E7;}}',
            ])
            checked = " checked" if index == 0 else ""
            choices.append(f'<input class="sm-choice" type="radio" id="{choice_id}" name="{deck_id}" aria-label="{sm_escape(frame["label"])}"{checked}>')
            labels.append(f'<label for="{choice_id}">{sm_escape(frame["label"])}</label>')
            nav = []
            if index > 0:
                nav.append(f'<label for="{deck_id}_choice_{index - 1}">← 이전 단계</label>')
            if index + 1 < len(frames):
                nav.append(f'<label for="{deck_id}_choice_{index + 1}">다음 단계 →</label>')
            body = frame["html"] if "html" in frame else f'<div class="sm-scroll">{frame["svg"]}</div>'
            if frame.get("context"):
                body = f'<div class="sm-scroll">{frame["context"]}</div>' + body
            scenes.append(f'<div class="sm-scene" data-step="{index}">{body}'
                          f'<p class="sm-caption">{sm_escape(frame["caption"])}</p><div class="sm-nav">{"".join(nav)}</div></div>')
        css = f'''
    #{deck_id} {{position:relative;border:1px solid #D6DEE8;border-radius:12px;background:#FCFDFE;color:#17313F;padding:18px;font-family:Arial,Apple SD Gothic Neo,Malgun Gothic,sans-serif;}}
    #{deck_id} .sm-choice {{position:absolute;width:1px;height:1px;opacity:0;}}
    #{deck_id} .sm-toolbar, #{deck_id} .sm-nav {{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0;}}
    #{deck_id} label {{display:inline-block;border:1px solid #D6DEE8;border-radius:7px;padding:8px 12px;background:white;color:#17313F;cursor:pointer;font-size:13px;}}
    #{deck_id} > .sm-scenes > .sm-scene {{display:none;}}
    #{deck_id} .sm-scroll {{overflow-x:auto;}}
    #{deck_id} .sm-caption {{font-size:14px;line-height:1.7;margin:12px 0;}}
    #{deck_id} .sm-note {{font-size:12px;line-height:1.6;color:#657487;}}
    #{deck_id} h4 {{margin:0;font-size:21px;color:#17313F;}}
    '''
        content = (f'<div id="{deck_id}"><style>{css}{"".join(rules)}</style>{"".join(choices)}'
                   f'<h4>{sm_escape(title)}</h4><div class="sm-toolbar">{"".join(labels)}</div>'
                   + (f'<div class="sm-scroll">{context}</div>' if context else '')
                   + f'<div class="sm-scenes">{"".join(scenes)}</div><p class="sm-note">{sm_escape(note)}</p></div>')
        if show:
            return HTML(content)
        return content


    sm_direct = _softmax_direct


    sm_stable = _softmax_stable


    def sm_tuple(values, digits=4):
        return "[" + ", ".join(sm_number(x, digits) for x in np.asarray(values).reshape(-1)) + "]"


    def sm_matrix(x, y, width, title, values, row_labels=None, col_labels=None,
                  color=None, *, row_focus=None, col_focus=None):
        """Show all cells. A focus shades assigned rows/columns without slicing Q or K."""
        array = np.atleast_2d(values)
        rows, cols = array.shape
        color = color or sm_palette["ink"]
        cell_width = (width - 48) / cols
        parts = [sm_text(x, y, f"{title}  ({rows}×{cols})", 17, color, weight=700)]
        for col in range(cols):
            parts.append(sm_text(x + 48 + (col + 0.5) * cell_width, y + 26,
                                 col_labels[col] if col_labels else str(col), 11, sm_palette["muted"], "middle"))
        for row in range(rows):
            parts.append(sm_text(x + 36, y + 52 + row * 30,
                                 row_labels[row] if row_labels else str(row), 11, sm_palette["muted"], "end"))
            for col in range(cols):
                selected = ((row_focus is None or row in row_focus)
                            and (col_focus is None or col in col_focus))
                fill = "#E7F3F4" if selected and (row_focus is not None or col_focus is not None) else "white"
                parts.append(sm_rect(x + 48 + col * cell_width, y + 35 + row * 30,
                                     cell_width - 4, 27, fill=fill,
                                     stroke=color if selected else sm_palette["line"], radius=4))
                value = array[row, col]
                value_text = sm_number(value) if isinstance(value, (float, int, np.number)) else str(value)
                finite = bool(np.isfinite(value)) if isinstance(value, (float, int, np.number)) else True
                ink = (color if selected else sm_palette["muted"]) if finite else sm_palette["red"]
                parts.append(sm_text(x + 48 + (col + 0.5) * cell_width - 2, y + 54 + row * 30,
                                     value_text, 15, ink, "middle", 600))
        return parts


    def sm_e(exponent):
        return "e^(" + sm_number(exponent).replace("-","−") + ")"


    def sm_esum(exponents, coefficients=None):
        coefficients = np.ones(len(exponents)) if coefficients is None else coefficients
        terms = []
        for exponent, coefficient in zip(exponents, coefficients):
            if coefficient == 0:
                continue
            coefficient_text = "" if coefficient == 1 else sm_number(coefficient) + "×"
            terms.append(coefficient_text + sm_e(exponent))
        return " + ".join(terms) if terms else "0"


    def sm_expr_tuple(expressions):
        return "[" + ", ".join(expressions) + "]"


    def sm_symbolic_u(exponents, values):
        return [sm_esum(exponents, values[:,dimension]) for dimension in range(values.shape[1])]


    def sm_read_matrix(x, y, width, title, values, row_labels=None, col_labels=None,
                       color=None, *, row_focus=None, col_focus=None):
        parts = sm_matrix(x,y,width,title,values,row_labels,col_labels,color,
                          row_focus=row_focus,col_focus=col_focus)
        rows,cols=np.atleast_2d(values).shape
        selected_rows=list(range(rows)) if row_focus is None else list(row_focus)
        selected_cols=list(range(cols)) if col_focus is None else list(col_focus)
        if selected_rows and selected_cols:
            cell_width=(width-48)/cols
            left=x+48+min(selected_cols)*cell_width-2
            top=y+35+min(selected_rows)*30-2
            rectangle=sm_rect(left,top,(max(selected_cols)-min(selected_cols)+1)*cell_width,
                              (max(selected_rows)-min(selected_rows)+1)*30-1,
                              fill="none",stroke=color or sm_palette["blue"],radius=3)
            parts.append(rectangle.replace('stroke-width="1.5"','stroke-width="3.2"'))
        return parts


    def sm_range(name, indices):
        indices=list(indices)
        return f"{name}[{indices[0]}:{indices[-1]+1}]" if indices else "없음"


    def sm_full_context(offset=0, *, key_now=(), value_now=(), seen_keys=None,
                        seen_values=None, action="전체 점수 행을 따라가는 계산"):
        seen_keys=list(range(4)) if seen_keys is None else list(seen_keys)
        seen_values=list(range(4)) if seen_values is None else list(seen_values)
        full_scores=(SM_SCORE_MATRIX+np.float32(offset)).astype(object)
        for key in range(4):
            if key not in seen_keys:
                full_scores[:,key]="?"
        q_labels=[f"q{i}" for i in range(4)]
        k_labels=[f"k{i}" for i in range(4)]
        current=[]
        if key_now:current.append(sm_range("K",key_now))
        if value_now:current.append(sm_range("V",value_now))
        read_text=" · ".join(current) if current else "새 K/V 읽기 없음"
        parts=[sm_text(20,28,f"전체 네 query · 설명 강조 q{SM_QUERY_INDEX} · {action}",18,weight=700),
               sm_text(20,53,f"지금 읽는 입력: {read_text}",16,sm_palette["blue"],weight=600)]
        parts += sm_read_matrix(20,95,260,"전체 Q · 네 행",SM_Q,q_labels,["d0","d1"],
                                sm_palette["purple"],row_focus=None)
        parts += sm_read_matrix(310,95,260,"전체 Kᵀ · 이번 열",SM_K.T,["d0","d1"],k_labels,
                                sm_palette["blue"],col_focus=key_now)
        parts += sm_read_matrix(600,95,260,"전체 V · 이번 행",SM_VALUES,[f"v{i}" for i in range(4)],["d0","d1"],
                                sm_palette["amber"],row_focus=value_now)
        parts += sm_read_matrix(890,95,290,"전체 S · 읽은 key",full_scores,q_labels,k_labels,
                                sm_palette["purple"],row_focus=None,col_focus=seen_keys)
        parts += [sm_text(20,273,"보라: 전체 query/점수 · 파랑: 이번 K 읽기 · 황금: 이번 V 읽기 · ?: 아직 읽지 않은 key의 점수",14),
                  sm_text(20,295,"회색 입력은 전체 원본의 참조입니다. 테두리는 현재 단계에서 사용하는 행·열 범위입니다.",14,sm_palette["muted"])]
        return sm_svg(parts,315,"전체 원본과 이번 읽기 범위를 표시한 Q K V 점수 행렬")


    def sm_heading(question, instruction):
        return [sm_text(30, 35, question, 23, weight=700), sm_text(30, 70, instruction, 15, sm_palette["muted"])]


    def sm_exp_grid(exponents, known_columns=None):
        exponents=np.asarray(exponents)
        assert exponents.shape==(4,4)
        known_columns=list(range(4)) if known_columns is None else list(known_columns)
        result=np.full((4,4),"?",dtype=object)
        for row in range(4):
            for col in known_columns:result[row,col]=sm_e(exponents[row,col])
        return result


    def sm_contribution_grid(exponents, active_columns):
        result=np.full((4,4),"0",dtype=object)
        for row in range(4):
            for col in active_columns:result[row,col]=sm_e(exponents[row,col])
        return result


    def sm_symbolic_rows(exponents, values):
        return np.asarray([sm_symbolic_u(row,values) for row in exponents],dtype=object)


    def sm_l_grid(exponents):
        return np.asarray([[sm_esum(row)] for row in exponents],dtype=object)


    def sm_score_scene(scores, offset):
        parts=sm_heading("① 네 query와 네 key의 점수표 만들기", "Q·Kᵀ·S를 모두 전체 크기로 표시합니다. 아래 식만 강조한 q행의 계산을 풀어 씁니다.")
        parts += sm_read_matrix(30,130,260,"전체 Q",SM_Q,[f"q{i}" for i in range(4)],["d0","d1"],sm_palette["purple"])
        parts += sm_read_matrix(380,130,340,"전체 Kᵀ",SM_K.T,["d0","d1"],[f"k{i}" for i in range(4)],sm_palette["blue"])
        parts += sm_matrix(835,130,340,"전체 S",scores,[f"q{i}" for i in range(4)],[f"k{i}" for i in range(4)],sm_palette["purple"])
        parts += [sm_arrow(305,215,365,215,"×"),sm_arrow(735,215,820,215,"=")]
        q=SM_Q[SM_QUERY_INDEX]
        for j in range(4):
            expression=" + ".join(f"({sm_number(q[d])})×({sm_number(SM_K[j,d])})" for d in range(2))
            if offset:expression+=" + "+sm_number(offset)
            parts.append(sm_text(40,370+j*42,f"S[q{SM_QUERY_INDEX},k{j}] = {expression} = {sm_number(scores[SM_QUERY_INDEX,j])}",20))
        parts.append(sm_text(40,568,"각 행은 서로 다른 query이고, 네 열은 같은 네 key입니다.",18))
        return sm_svg(parts,615,"Q4×2 K전치2×4 S4×4 전체 행렬")


    def sm_exp_scene(inputs, exponentials):
        parts=sm_heading("지수값도 네 query × 네 key 전체로 보기", "같은 위치의 점수 → e의 지수 항 → FP32 검산 값을 나란히 비교합니다.")
        q=[f"q{i}" for i in range(4)];k=[f"k{i}" for i in range(4)]
        parts += sm_matrix(30,165,330,"전체 지수 입력",inputs,q,k)
        parts += sm_matrix(435,165,330,"전체 E · 지수 항",sm_exp_grid(inputs),q,k,sm_palette["teal"])
        parts += sm_matrix(840,165,330,"전체 E · FP32 검산",exponentials,q,k)
        parts += [sm_arrow(370,250,425,250,"exp"),sm_arrow(775,250,830,250,"≈"),
                  sm_text(40,395,"예: e^(0)=1, e^(−1)≈0.3679. 어느 점수 차이에서 나온 값인지 위첨자로 읽습니다.",20),
                  sm_text(40,454,"수학적 지수 항은 그대로 표시하고, FP32의 overflow는 검산 행렬에 +∞로 표시합니다.",18)]
        return sm_svg(parts,520,"전체4×4 지수 입력과 e 지수값 및 실제FP32 결과")


    def sm_sum_scene(inputs, exponentials, denominator):
        parts=sm_heading("각 query 행의 네 항을 더해 자기 분모 만들기", "ℓ은 하나의 전역 값이 아니라 query마다 하나씩, 네 행짜리 상태입니다.")
        q=[f"q{i}" for i in range(4)];k=[f"k{i}" for i in range(4)]
        parts += sm_matrix(30,145,530,"전체 E",sm_exp_grid(inputs),q,k,sm_palette["teal"])
        parts += sm_matrix(670,145,500,"전체 ℓ · 지수 항의 합",sm_l_grid(inputs),q,["sum"],sm_palette["teal"])
        for i in range(4):parts.append(sm_arrow(570,197+i*30,660,197+i*30,"행 합"))
        parts += sm_matrix(30,397,500,"전체 ℓ · FP32 검산",denominator[:,None],q,["sum"])
        row=SM_QUERY_INDEX
        parts += [sm_text(670,425,f"예: ℓ{row} = "+sm_esum(inputs[row]),20,sm_palette["teal"],weight=600),
                  sm_text(670,485,"다른 query 행은 자기 네 항을",19),
                  sm_text(670,522,"자기 분모로 따로 합합니다.",19)]
        return sm_svg(parts,605,"전체4×4 지수 행렬과4×1 행별 분모")


    def sm_probability_scene(inputs, exponentials, denominator, probabilities, reference=None):
        q=[f"q{i}" for i in range(4)];k=[f"k{i}" for i in range(4)]
        symbolic=np.asarray([[sm_e(inputs[i,j])+f" / ℓ{i}" for j in range(4)] for i in range(4)],dtype=object)
        parts=sm_heading("확률표도 전체 4×4: 각 행은 자기 ℓ로 나누기", "각 분자의 지수 항과 분모의 query 번호를 확인하세요.")
        parts += sm_matrix(30,145,530,"전체 P · 지수 항 / ℓ",symbolic,q,k,sm_palette["purple"])
        parts += sm_matrix(670,145,500,"전체 P · FP32 검산",probabilities,q,k)
        parts += sm_matrix(30,380,530,"전체 ℓ",sm_l_grid(inputs),q,["sum"],sm_palette["teal"])
        with np.errstate(invalid="ignore"):
            sums=probabilities.sum(axis=1,dtype=np.float32)
        parts += sm_matrix(670,380,500,"각 행 확률 합 · FP32",sums[:,None],q,["sum"])
        if reference is not None:
            parts.append(sm_text(40,585,"offset 0과 전체 확률표 최대 차이: "+sm_number(np.max(np.abs(probabilities-reference))),18))
        else:
            parts.append(sm_text(40,585,"유효한 각 행의 확률 합은 1입니다. overflow로 실패한 행은 검산 값에 그대로 나타납니다.",18))
        return sm_svg(parts,625,"전체4×4 확률표와4×1 각행의 분모 및 확률합")


    def sm_pv_scene(inputs, probabilities):
        q=[f"q{i}" for i in range(4)];k=[f"k{i}" for i in range(4)]
        symbolic=np.asarray([[sm_e(inputs[i,j])+f" / ℓ{i}" for j in range(4)] for i in range(4)],dtype=object)
        with np.errstate(invalid="ignore"):output=probabilities @ SM_VALUES
        parts=sm_heading("전체 P × 전체 V = 전체 O", "V는 원래 4×2 방향 그대로입니다. key 열 j의 비율로 V의 같은 위치 행 j를 가중합합니다.")
        parts += sm_matrix(30,145,430,"전체 P",symbolic,q,k,sm_palette["purple"])
        parts += sm_read_matrix(550,145,235,"전체 V",SM_VALUES,[f"v{i}" for i in range(4)],["d0","d1"],sm_palette["amber"])
        parts += sm_matrix(935,145,235,"전체 O · 검산",output,q,["d0","d1"],sm_palette["teal"])
        parts += [sm_arrow(470,230,535,230,"×"),sm_arrow(795,230,920,230,"=")]
        row=SM_QUERY_INDEX
        symbolic_u=sm_symbolic_u(inputs[row],SM_VALUES)
        parts += [sm_text(40,390,f"설명용 q{row}의 성분 0: ( "+symbolic_u[0]+f" ) / ℓ{row}",22),
                  sm_text(40,445,f"설명용 q{row}의 성분 1: ( "+symbolic_u[1]+f" ) / ℓ{row}",22),
                  sm_text(40,530,"위 그림의 O는 q0~q3 네 행 모두입니다. 위의 두 식만 한 행의 성분을 풀어 쓴 것입니다.",18)]
        return sm_svg(parts,600,"P4×4 곱하기V4×2는O4×2 전체 출력")


    def sm_shift_scene(scores, maximum, shifted):
        q=[f"q{i}" for i in range(4)];k=[f"k{i}" for i in range(4)]
        parts=sm_heading("전체 점수표에서 행마다 다른 최대값을 빼기", "m도 네 query의 4×1 벡터입니다. 한 행의 모든 key 점수에서 그 행의 m을 뺍니다.")
        parts += sm_matrix(30,145,430,"전체 S",scores,q,k)
        parts += sm_matrix(540,145,150,"행별 m",maximum[:,None],q,["max"],sm_palette["teal"])
        parts += sm_matrix(780,145,390,"전체 S−m",shifted,q,k,sm_palette["teal"])
        parts += [sm_arrow(470,230,525,230,"행 max"),sm_arrow(700,230,765,230,"빼기"),
                  sm_text(40,390,"다음 단계의 지수는 각 칸의 S_ij−m_i입니다.",22),
                  sm_text(40,450,"각 행에서 공통 인자 e^(m_i)가 약분되므로 같은 확률을 얻습니다.",20),
                  sm_text(40,518,"네 행 모두 최대 지수 입력이 0이 되고, 최대 지수값은 e^(0)=1입니다.",18)]
        return sm_svg(parts,580,"전체4×4 점수표와행별4×1 최대값으로계산한전체차이행렬")


    def sm_show_softmax(stable):
        cases=[]
        offsets=sorted(SM_SHIFTS,reverse=True) if stable else SM_SHIFTS
        reference=sm_stable(SM_SCORE_MATRIX)[-1]
        keys=list(range(4))
        for offset in offsets:
            scores=SM_SCORE_MATRIX+np.float32(offset)
            if stable:
                maximum,shifted,exponentials,denominator,probabilities=sm_stable(scores)
            else:
                exponentials,denominator,probabilities=sm_direct(scores)
            inputs=shifted if stable else scores
            def context(action,key_now=(),value_now=()):
                return sm_full_context(offset,key_now=key_now,value_now=value_now,
                                       seen_keys=keys,seen_values=keys if value_now else [],action=action)
            frames=[{"label":"① 전체 점수","svg":sm_score_scene(scores,offset),
                     "context":context("Q[0:4] × K[0:4]로 전체 점수 생성",key_now=keys),
                     "caption":"Q 네 행과 K 네 행을 모두 표시합니다. K만 실제 곱셈 방향인 Kᵀ로 표시합니다."}]
            if stable:
                frames.append({"label":"② 행별 최대값","svg":sm_shift_scene(scores,maximum,shifted),
                               "context":context("전체 S의 각 행에서 자기 최대값 빼기"),
                               "caption":"4×4 점수표와 네 query의 최대값을 모두 표시합니다."})
            frames.extend([
                {"label":"전체 지수값","svg":sm_exp_scene(inputs,exponentials),"context":context("네 행의 모든 지수값 계산"),"caption":"E는 전체 4×4입니다. 위첨자는 수학적 지수 항, FP32 행렬은 검산 결과입니다."},
                {"label":"행별 분모","svg":sm_sum_scene(inputs,exponentials,denominator),"context":context("E의 각 행을 합해 네 분모 생성"),"caption":"ℓ도 한 query만 표시하지 않고 네 행 전부 표시합니다."},
                {"label":"전체 확률표","svg":sm_probability_scene(inputs,exponentials,denominator,probabilities,reference if stable else None),"context":context("각 행을 자기 ℓ로 나누기"),"caption":"P의 행·열을 모두 표시합니다. overflow도 해당 행의 검산 값에 표시합니다."},
                {"label":"전체 V 가중합","svg":sm_pv_scene(inputs,probabilities),"context":context("V[0:4]의 모든 행을 읽고 전체 출력 생성",value_now=keys),"caption":"P 4×4, V 4×2, O 4×2를 그대로 곱합니다. V는 전치하지 않습니다."},
            ])
            case_html=sm_deck("전체 네 query · offset +"+sm_number(offset),frames,
                              "행렬은 항상 전체 크기로 표시합니다. 지수 식은 위첨자, 소수는 FP32 검산용입니다.",show=False)
            cases.append({"label":"입력 offset +"+sm_number(offset),"html":case_html,"caption":""})
        return sm_deck("Stable softmax: 네 행 전체의 같은 확률" if stable else "Normal softmax: 네 행 전체의 점수에서 출력까지",cases,
                "모든 Q/K/V와 중간 행렬은 전체 크기로 표시합니다. 행별 분모와 key 열·V 행의 대응을 확인하세요.")


    def render_online():
        # 네 query 모두의 Online 상태를 계산합니다. 화면에서는 행렬을 잘라내지 않습니다.
        full_scores=SM_SCORE_MATRIX
        s0,s1=full_scores[:,:2],full_scores[:,2:]
        v0,v1=SM_VALUES[:2],SM_VALUES[2:]
        m0,m_block=np.max(s0,axis=1),np.max(s1,axis=1)
        m1=np.maximum(m0,m_block)
        old_exponents=full_scores-m0[:,None]
        corrected_exponents=full_scores-m1[:,None]
        e0=np.exp(s0-m0[:,None])
        l0=e0.sum(axis=1,dtype=np.float32)
        u0=e0 @ v0
        alpha=np.exp(m0-m1)
        e0_corrected=alpha[:,None]*e0
        l0_corrected=alpha*l0
        u0_corrected=alpha[:,None]*u0
        e1=np.exp(s1-m1[:,None])
        l_block=e1.sum(axis=1,dtype=np.float32)
        u_block=e1 @ v1
        l1=l0_corrected+l_block
        u1=u0_corrected+u_block
        online_output=u1/l1[:,None]
        direct_probabilities=sm_stable(full_scores)[-1]
        direct_output=direct_probabilities @ SM_VALUES
        np.testing.assert_allclose(online_output,direct_output,rtol=1e-6,atol=1e-7)
        old_contribution=np.zeros_like(full_scores)
        old_contribution[:,:2]=e0
        new_contribution=np.zeros_like(full_scores)
        new_contribution[:,2:]=e1
        np.testing.assert_allclose(old_contribution @ SM_VALUES,u0,rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(new_contribution @ SM_VALUES,u_block,rtol=1e-6,atol=1e-7)
        q_labels=[f"q{i}" for i in range(4)]
        k_labels=[f"k{i}" for i in range(4)]
        d_labels=["d0","d1"]
        old_l_expr=np.asarray([[sm_esum(row)] for row in old_exponents[:,:2]],dtype=object)
        corrected_l_expr=np.asarray([[sm_esum(row)] for row in corrected_exponents[:,:2]],dtype=object)
        final_l_expr=sm_l_grid(corrected_exponents)
        old_u_expr=sm_symbolic_rows(old_exponents[:,:2],v0)
        corrected_u_expr=sm_symbolic_rows(corrected_exponents[:,:2],v0)
        block_u_expr=sm_symbolic_rows(corrected_exponents[:,2:],v1)
        final_u_expr=sm_symbolic_rows(corrected_exponents,SM_VALUES)
        alpha_expr=np.asarray([[sm_e(x)] for x in m0-m1],dtype=object)
        sm_online_frames=[]

        parts=sm_heading("① 네 query와 첫 K/V 블록으로 상태 만들기", "행렬의 원래 4×4 위치를 유지합니다. 아직 읽지 않은 점수는 ?, 아직 누적하지 않은 기여는 0입니다.")
        parts += sm_read_matrix(30,145,430,"읽은 key의 E 기여",sm_contribution_grid(old_exponents,[0,1]),q_labels,k_labels,sm_palette["teal"],col_focus=[0,1])
        parts += sm_read_matrix(550,145,235,"전체 V",SM_VALUES,[f"v{i}" for i in range(4)],d_labels,sm_palette["amber"],row_focus=[0,1])
        parts += sm_matrix(935,145,235,"전체 u",old_u_expr,q_labels,d_labels,sm_palette["purple"])
        parts += [sm_arrow(470,230,535,230,"×"),sm_arrow(795,230,920,230,"=")]
        parts += sm_matrix(30,397,235,"전체 m",m0[:,None],q_labels,["max"])
        parts += sm_matrix(365,397,430,"전체 ℓ",old_l_expr,q_labels,["sum"],sm_palette["teal"])
        parts += sm_matrix(935,397,235,"u · 검산",u0,q_labels,d_labels)
        sm_online_frames.append({"label":"① K/V[0:2] 읽기","svg":sm_svg(parts,610,"전체4×4 E와4×2 V u 및4×1 상태"),
            "context":sm_full_context(key_now=[0,1],value_now=[0,1],seen_keys=[0,1],seen_values=[0,1],action="첫 K/V 두 행 읽기"),
            "caption":"V는 4×2 전체를 표시합니다. E 기여의 k2·k3 열은 현재 누적에 참여하지 않아 0이며, 해당 점수는 위의 전체 S에서 아직 ?입니다."})

        parts=sm_heading("② 새 K[2:4]를 읽고 각 query의 기준 바꾸기", "m_old·m_block·m_new·α 모두 네 행입니다. query마다 보정 계수가 다를 수 있습니다.")
        for x,title,array,color in ((30,"이전 m",m0[:,None],None),(335,"새 블록 m",m_block[:,None],sm_palette["blue"]),(640,"새 m",m1[:,None],sm_palette["purple"]),(945,"α · 지수 항",alpha_expr,sm_palette["teal"])):
            parts += sm_matrix(x,165,225,title,array,q_labels,["value"],color)
        row=SM_QUERY_INDEX
        parts += [sm_text(40,397,f"예: q{row}의 α = e^({sm_number(m0[row])}−{sm_number(m1[row])}) = {sm_e(m0[row]-m1[row])}",26,sm_palette["purple"],weight=600),
                  sm_text(40,463,"새 V의 기여는 아직 더하지 않았습니다. 다음 단계는 입력을 다시 읽지 않고 상태만 보정합니다.",18)]
        sm_online_frames.append({"label":"② K[2:4] 읽기","svg":sm_svg(parts,530,"네query의전체최대값과e지수보정계수"),
            "context":sm_full_context(key_now=[2,3],value_now=[],seen_keys=list(range(4)),seen_values=[0,1],action="새 K 두 행으로 나머지 점수 생성"),
            "caption":"최대값과 α는 네 query 모두 표시합니다. 이번 읽기 범위는 파란 K의 두 열이며 V는 새로 읽지 않습니다."})

        parts=sm_heading("③ 새 입력 없이 이전 네 행의 상태만 보정", "E·ℓ·u를 모두 전체 모양으로 비교합니다. 기존 key 열의 지수만 새 기준에 맞춥니다.")
        parts += sm_matrix(30,145,520,"이전 E 기여 · 전체 위치",sm_contribution_grid(old_exponents,[0,1]),q_labels,k_labels)
        parts += sm_matrix(650,145,520,"보정 후 E 기여 · 전체 위치",sm_contribution_grid(corrected_exponents,[0,1]),q_labels,k_labels,sm_palette["teal"])
        for row in range(4):parts.append(sm_arrow(560,197+row*30,640,197+row*30,"× "+sm_e(m0[row]-m1[row])))
        parts += sm_matrix(30,362,520,"이전 전체 ℓ",old_l_expr,q_labels,["sum"])
        parts += sm_matrix(650,362,520,"보정 후 전체 ℓ",corrected_l_expr,q_labels,["sum"],sm_palette["teal"])
        parts += sm_matrix(30,579,520,"이전 전체 u",old_u_expr,q_labels,d_labels)
        parts += sm_matrix(650,579,520,"보정 후 전체 u",corrected_u_expr,q_labels,d_labels,sm_palette["purple"])
        parts.append(sm_text(40,794,"지수의 기준을 바꾸는 것만으로, 이전 분모와 V 기여가 함께 보정됩니다.",19))
        sm_online_frames.append({"label":"③ 상태만 보정","svg":sm_svg(parts,835,"전체행렬의이전지수값분모분자와보정결과"),
            "context":sm_full_context(key_now=[],value_now=[],seen_keys=list(range(4)),seen_values=[0,1],action="새 K/V 읽기 없음 · 이전 ℓ·u만 갱신"),
            "caption":"오른쪽 두 열의 0은 현재 기여에 아직 포함하지 않았다는 뜻입니다. 실제 지수값이 0이라는 뜻은 아닙니다. 이전 V를 다시 읽지 않습니다."})

        parts=sm_heading("④ 새 V[2:4]를 읽고 네 query의 새 기여 추가", "전체 E와 전체 V를 유지합니다. 이번 V 읽기는 v2·v3 두 행이고, 다른 행의 기여는 이전 u에 있습니다.")
        parts += sm_matrix(30,145,430,"새 블록의 E 기여",sm_contribution_grid(corrected_exponents,[2,3]),q_labels,k_labels,sm_palette["teal"])
        parts += sm_read_matrix(550,145,235,"전체 V",SM_VALUES,[f"v{i}" for i in range(4)],d_labels,sm_palette["amber"],row_focus=[2,3])
        parts += sm_matrix(935,145,235,"새 기여 u",block_u_expr,q_labels,d_labels,sm_palette["purple"])
        parts += [sm_arrow(470,230,535,230,"×"),sm_arrow(795,230,920,230,"기여")]
        parts += sm_matrix(30,397,340,"보정된 이전 u",corrected_u_expr,q_labels,d_labels)
        parts += sm_matrix(435,397,340,"새 블록의 u",block_u_expr,q_labels,d_labels)
        parts += sm_matrix(840,397,340,"합친 전체 u",final_u_expr,q_labels,d_labels,sm_palette["purple"])
        parts += [sm_arrow(380,480,425,480,"+"),sm_arrow(785,480,830,480,"=")]
        parts += sm_matrix(30,640,650,"전체 ℓ",final_l_expr,q_labels,["sum"],sm_palette["teal"])
        parts += sm_matrix(820,640,350,"u · FP32 검산",u1,q_labels,d_labels)
        sm_online_frames.append({"label":"④ V[2:4] 읽기","svg":sm_svg(parts,840,"전체E4×4 V4×2와새Value기여및전체분자"),
            "context":sm_full_context(key_now=[],value_now=[2,3],seen_keys=list(range(4)),seen_values=list(range(4)),action="새 V 두 행의 기여 추가"),
            "caption":"새 블록 E 기여의 k0·k1 열은 0입니다. 그 기여는 이미 이전 u에 있습니다. 따라서 전체 4×4 E 기여 × 전체 4×2 V가 새 블록의 u를 만듭니다."})

        parts=sm_heading("⑤ 전체 네 행을 자기 분모로 나누기", "새 입력을 읽지 않고 누적한 u와 ℓ로 전체 O 4×2를 완성합니다.")
        parts += sm_matrix(30,145,520,"전체 u · 지수 항",final_u_expr,q_labels,d_labels,sm_palette["purple"])
        parts += sm_matrix(650,145,520,"전체 ℓ · 지수 항의 합",final_l_expr,q_labels,["sum"],sm_palette["teal"])
        parts += sm_matrix(30,397,520,"Online 전체 O",online_output,q_labels,d_labels,sm_palette["purple"])
        parts += sm_matrix(650,397,520,"Stable 전체 O · 비교",direct_output,q_labels,d_labels,sm_palette["teal"])
        parts.append(sm_text(40,602,"전체 네 행 최대 차이: "+sm_number(np.max(np.abs(online_output-direct_output))),20,weight=600))
        sm_online_frames.append({"label":"⑤ 전체 출력","svg":sm_svg(parts,650,"전체4×2 Online Stable출력과전체상태"),
            "context":sm_full_context(key_now=[],value_now=[],seen_keys=list(range(4)),seen_values=list(range(4)),action="새 입력 읽기 없음 · 전체 u/ℓ"),
            "caption":"마지막에도 하나의 query 행을 잘라내지 않습니다. 두 O 행렬의 네 행 전체가 일치합니다."})
        return sm_deck("Online softmax: 전체 행렬과 e 지수 항 유지",sm_online_frames,
                "모든 Q/K/V와 중간 행렬은 전체 모양을 유지합니다. 굵은 파랑/황금 테두리로 이번 K/V 읽기를 확인합니다. ?는 아직 계산하지 않은 지수 또는 점수 위치입니다.")

    return render_online() if mode == "online" else sm_show_softmax(mode == "stable")


# FlashAttention operation scenes

def flash_trace(q, k, v, *, mode="FA1", query_tile=2, key_tile=2):
    """Logical forward frames; every row has its own m, l and accumulator.

    FA2 CTAs are displayed at equal key progress for comparison, not as a
    synchronized GPU schedule. No physical allocation or timing is simulated.
    """
    q, k, v = (np.asarray(x, dtype=np.float32) for x in (q, k, v))
    if mode not in ("FA1", "FA2"):
        raise ValueError("mode must be FA1 or FA2")
    if any(x.ndim != 2 or not x.size or not np.isfinite(x).all() for x in (q, k, v)):
        raise ValueError("Q/K/V must be finite nonempty matrices")
    if q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0]:
        raise ValueError("Q/K dimensions and K/V token counts must match")
    if query_tile < 1 or key_tile < 1:
        raise ValueError("tile sizes must be positive")
    nq, nk, dv = q.shape[0], k.shape[0], v.shape[1]
    queries, keys = list(range(0, nq, query_tile)), list(range(0, nk, key_tile))
    visits = ([[[qs, ks]] for ks in keys for qs in queries] if mode == "FA1"
              else [[[qs, ks] for qs in queries] for ks in keys])
    m = np.full(nq, -np.inf, dtype=np.float32)
    l = np.zeros(nq, dtype=np.float32)
    acc = np.zeros((nq, dv), dtype=np.float32)
    output = np.zeros_like(acc)
    seen = np.zeros(nq, dtype=int)
    frames = []
    records = []

    def frame(phase, visit, tiles):
        frames.append({"phase": phase, "visit": visit, "tiles": tiles,
                       "acc": acc.copy(), "output": output.copy(),
                       "m": m.copy(), "l": l.copy(), "seen": seen.copy()})

    for visit, positions in enumerate(visits):
        tiles = []
        for qs, ks in positions:
            qe, ke = min(qs+query_tile, nq), min(ks+key_tile, nk)
            qr, kr = slice(qs, qe), slice(ks, ke)
            scores = q[qr] @ k[kr].T
            maximum = np.maximum(m[qr], scores.max(axis=1))
            alpha = np.exp(m[qr]-maximum)
            e = np.exp(scores-maximum[:, None])
            contribution = e @ v[kr]
            old_u = l[qr, None]*acc[qr] if mode == "FA1" else acc[qr].copy()
            denominator = alpha*l[qr]+e.sum(axis=1, dtype=np.float32)
            numerator = alpha[:, None]*old_u+contribution
            o = numerator/denominator[:, None]
            tiles.append({"qs":qs, "qe":qe, "ks":ks, "ke":ke,
                          "s":scores, "e":e, "contribution":contribution,
                          "old_m":m[qr].copy(), "old_l":l[qr].copy(), "old_u":old_u.copy(),
                          "alpha":alpha, "m":maximum, "l":denominator,
                          "u":numerator, "o":o})
        frame("read", visit, tiles)
        frame("score", visit, tiles)
        frame("value", visit, tiles)
        for tile in tiles:
            qr = slice(tile["qs"], tile["qe"])
            m[qr], l[qr] = tile["m"], tile["l"]
            acc[qr] = tile["o"] if mode == "FA1" else tile["u"]
            if mode == "FA1" or tile["ke"] == nk:
                output[qr] = tile["o"]
            seen[qr] = tile["ke"]
            records.append(tile)
        frame("update", visit, tiles)
    if mode == "FA2":
        frame("divide", len(visits), [])
    frame("done", len(visits), [])
    return {"mode":mode, "queryTile":query_tile, "keyTile":key_tile,
            "q":q, "k":k, "v":v, "visits":visits,
            "frames":frames, "records":records, "output":output}

def flash_reference(q, k, v):
    scores = np.asarray(q, dtype=np.float32) @ np.asarray(k, dtype=np.float32).T
    weights = np.exp(scores-scores.max(axis=1, keepdims=True))
    return (weights/weights.sum(axis=1, keepdims=True)) @ np.asarray(v, dtype=np.float32)

def _flash_json(value):
    """JSON-safe frames: unknown initial maxima are null, not Infinity."""
    if isinstance(value, np.ndarray):
        return _flash_json(value.tolist())
    if isinstance(value, dict):
        return {k:_flash_json(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_flash_json(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value

FLASH_DOCUMENT = r"""<!doctype html>
<html lang="ko">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{color-scheme:light dark;--bg:#fff;--fg:#20242b;--muted:#66717e;--line:#d7dde5;--quiet:#f3f5f8;--blue:#247bb8;--orange:#c37027}
@media(prefers-color-scheme:dark){:root{--bg:#202328;--fg:#eff2f6;--muted:#adb8c6;--line:#454d58;--quiet:#2b313b;--blue:#68b7ed;--orange:#e9a368}}
*{box-sizing:border-box}body{margin:0;padding:12px;background:var(--bg);color:var(--fg);font:14px/1.6 Arial,'Malgun Gothic',sans-serif}
button{font:inherit;color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:5px;padding:5px 12px;cursor:pointer;min-height:34px}button[aria-pressed=true],#next{background:var(--fg);color:var(--bg)}button:disabled{opacity:.35;cursor:default}
.toolbar{display:flex;gap:6px;align-items:center;flex-wrap:wrap}.count{margin-left:auto;font-size:12px;color:var(--muted)}
h3{font-size:18px;font-weight:600;margin:16px 0 6px}.schedule{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}.route{border-left:3px solid var(--blue);background:var(--quiet);padding:5px 10px}.route.second{border-color:var(--orange)}
.stages{display:flex;align-items:center;gap:6px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin:12px 0}.stage.active{color:var(--fg);font-weight:600;border-bottom:2px solid var(--blue)}
.narration{border-left:3px solid var(--blue);background:var(--quiet);padding:10px 13px;margin:12px 0 18px;line-height:1.75;font-size:15px}.detail{font-size:13px;margin-top:5px}.detail:empty{display:none}
.flow{display:grid;align-items:center;gap:10px;margin:16px 0 10px;grid-template-columns:var(--columns)}
.card{min-width:0;align-self:center}.heading{display:flex;justify-content:space-between;align-items:baseline;gap:5px;margin-bottom:8px}.label{font-size:17px;font-weight:600}.shape{font-size:11px;color:var(--muted)}
.matrix{display:grid;gap:4px;font-variant-numeric:tabular-nums}.axis{display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:12px;min-height:25px}.cell{display:flex;align-items:center;justify-content:center;min-height:35px;padding:3px 1px;border:1px solid var(--line);font-size:14px}.cell.active{background:color-mix(in srgb,var(--blue) 17%,var(--bg));border-color:color-mix(in srgb,var(--blue) 35%,var(--line))}.cell.second.active{background:color-mix(in srgb,var(--orange) 17%,var(--bg));border-color:color-mix(in srgb,var(--orange) 40%,var(--line))}.cell.inactive{color:var(--muted)}.cell.empty{color:var(--muted);background:var(--quiet)}
.card.result .label{color:var(--blue)}.card.result .cell.active{border-color:var(--blue)}.operator{text-align:center;min-width:0;line-height:1.3}.symbol{font-size:30px;display:block}.op-label{font-size:11px;color:var(--muted);display:block;margin-top:5px;word-break:keep-all}.operator.arrow .symbol{color:var(--blue)}
.connector{display:flex;align-items:center;gap:9px;color:var(--blue);font-size:13px;margin:12px 0}.connector:before{content:'↓';font-size:24px}.connector:after{content:'';height:1px;flex:1;background:var(--line)}
.foot{font-size:12px;color:var(--muted);margin-top:16px}.group-label{font-size:13px;font-weight:600;margin-top:18px}.group-label.second{color:var(--orange)}
@media(max-width:700px){.flow{grid-template-columns:1fr!important;gap:8px;max-width:440px;margin:16px auto 22px}.operator.arrow .symbol{font-size:0}.operator.arrow .symbol:after{content:'↓';font-size:28px}.operator .op-label{display:inline;margin-left:8px}.operator .symbol{display:inline;font-size:25px}.card{width:100%}.count{margin-left:0}}
@media(pointer:coarse){button{min-height:44px}}
</style>
<main id="flash-lesson">
<div class="toolbar" role="group" aria-label="설명 선택">
 <button type="button" data-lesson="0" aria-pressed="true">한 행부터</button>
 <button type="button" data-lesson="1" aria-pressed="false">FA1</button>
 <button type="button" data-lesson="2" aria-pressed="false">FA2</button>
 <button type="button" data-lesson="3" aria-pressed="false">팀 안의 분담</button>
 <span id="count" class="count"></span><button type="button" id="prev">이전</button><button type="button" id="next">다음</button>
</div>
<h3 id="question"></h3><div id="schedule" class="schedule"></div><div id="stages" class="stages"></div>
<div class="narration"><div id="caption" aria-live="polite"></div><div id="detail" class="detail"></div></div>
<div id="calculation"></div>
<div class="foot">행·열 번호는 1부터입니다. 모든 행렬은 전체 크기로 보여 주며, 색칠한 부분끼리 계산합니다. ‘·’는 이번 연산에 값이 없는 자리입니다. 이 전체 틀을 실제 메모리에 만드는 것은 아닙니다.</div>
</main>
<script type="application/json" id="flash-data">__DATA__</script>
<script>
(() => {
const root=document.getElementById('flash-lesson'),lessons=JSON.parse(document.getElementById('flash-data').textContent);
const $=id=>document.getElementById(id),fmt=v=>v==null?'·':Number(v.toFixed(3)).toString();
const blank=(rows,cols)=>Array.from({length:rows},()=>Array(cols).fill(null));
const transpose=a=>a[0].map((_,c)=>a.map(row=>row[c]));
let lesson=0,step=0,serial=0;
function matrix(name,values,active,role='input',owner=()=>false){
 const rows=values.length,cols=values[0].length;
 let cells='<span class="axis"></span>'+Array.from({length:cols},(_,j)=>`<span class="axis">${j+1}</span>`).join('');
 for(let i=0;i<rows;i++){
  cells+=`<span class="axis">${i+1}</span>`;
  for(let j=0;j<cols;j++)cells+=`<span class="cell ${active(i,j)?'active':'inactive'} ${owner(i,j)?'second':''} ${values[i][j]==null?'empty':''}" data-row="${i}" data-col="${j}">${fmt(values[i][j])}</span>`;
 }
 return `<section class="card ${role}" data-name="${name}" data-values='${JSON.stringify(values)}'><div class="heading"><span class="label">${name}</span><span class="shape">${rows} × ${cols}</span></div><div class="matrix" style="grid-template-columns:25px repeat(${cols},minmax(0,1fr))">${cells}</div></section>`;
}
const op=(symbol,label='',arrow=false)=>`<div class="operator ${arrow?'arrow':''}"><span class="symbol">${symbol}</span><span class="op-label">${label}</span></div>`;
function flow(operation,nodes,symbols,labels=[],extra={}){
 const columns=nodes.map((_,i)=>i===nodes.length-1?'minmax(0,1fr)':'minmax(0,1fr) 58px').join(' ');
 return `<div class="flow" data-operation="${operation}" data-context='${JSON.stringify(extra)}' style="--columns:${columns}">`+nodes.map((node,i)=>node+(i<symbols.length?op(symbols[i],labels[i]||'',symbols[i]==='→'):'')).join('')+'</div>';
}
const continuation=text=>`<div class="connector">${text}</div>`;
function range(a,b){return a+1===b?`${a+1}`:`${a+1}~${b}`;}
function render(){
 const l=lessons[lesson],f=l.frames[step],p=f.phase,tiles=f.tiles||[];
 root.dataset.lesson=lesson;root.dataset.step=step;root.dataset.phase=p;
 root.querySelectorAll('[data-lesson]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.lesson)===lesson)));
 $('count').textContent=`${step+1} / ${l.frames.length}`;$('prev').disabled=lesson===0&&step===0;$('next').disabled=lesson===lessons.length-1&&step===l.frames.length-1;
 $('next').textContent=step===l.frames.length-1&&lesson<lessons.length-1?'다음 설명':'다음';
 $('question').textContent=l.question;$('detail').textContent='';
 const activeQ=i=>tiles.some(t=>i>=t.qs&&i<t.qe),activeK=i=>tiles.some(t=>i>=t.ks&&i<t.ke);
 const activeS=(i,j)=>tiles.some(t=>i>=t.qs&&i<t.qe&&j>=t.ks&&j<t.ke);
 const owner=i=>l.kind==='fa2'&&i>=2;
 const rowCard=(name,data,result=false)=>matrix(name,data,(i,j)=>data[i][j]!=null,result?'result':'input',owner);
 const tileRows=field=>{
  const out=blank(4,2);for(const t of tiles)for(let i=t.qs;i<t.qe;i++)out[i]=t[field][i-t.qs].slice();return out;
 };
 const tileVector=get=>{
  const out=blank(4,1);for(const t of tiles)for(let i=t.qs;i<t.qe;i++)out[i][0]=get(t,i-t.qs);return out;
 };
 const tileGrid=field=>{
  const out=blank(4,4);for(const t of tiles)for(let i=t.qs;i<t.qe;i++)for(let j=t.ks;j<t.ke;j++)out[i][j]=t[field][i-t.qs][j-t.ks];return out;
 };
 const scores=tileGrid('s'),weights=tileGrid('e'),newMix=tileRows('contribution'),previousMix=tileRows('old_u');
 const scale=tileVector((t,i)=>t.alpha[i]),previousSum=tileVector((t,i)=>t.old_l[i]),newSum=tileVector((t,i)=>t.e[i].reduce((a,b)=>a+b,0)),totalSum=tileVector((t,i)=>t.l[i]);
 const adjusted=previousMix.map((row,i)=>row.map(x=>x==null?null:x*scale[i][0]));
 const hadPrevious=tiles.some(t=>t.old_l.some(x=>x>0));
 const scoreCard=()=>matrix('S',scores,activeS,false,owner),weightCard=()=>matrix('Weights',weights,activeS,'input',owner);
 const valueCard=()=>matrix('V',l.v,(i)=>activeK(i));
 const stages=[['score','점수'],['weights','비중'],['mix','V 곱하기'],['restore','이전 나눗셈 되돌리기'],['rescale','보정'],['add','더하기'],['totals','비중 합계'],['divide','나누기']];
 $('stages').innerHTML=p.startsWith('split')?'':stages.filter(([key])=>key!=='restore'||l.mode==='FA1').map(([key,label])=>`<span class="stage ${p===key||p==='divide_final'&&key==='divide'?'active':''}">${label}</span>`).join('<span>→</span>');
 $('schedule').innerHTML=tiles.map((t,i)=>`<span class="route ${owner(t.qs)?'second':''}">${l.kind==='fa2'?`팀 ${i+1} · `:''}Q ${range(t.qs,t.qe)}행 · K·V ${range(t.ks,t.ke)}행</span>`).join('');
 let html='';
 if(p==='score'){
  $('caption').textContent='Q의 색칠한 행과 Kᵀ의 색칠한 열을 곱하면, S의 색칠한 칸이 만들어집니다.';
  if(l.kind==='fa1'&&f.visit>0)$('detail').textContent=tiles[0].qs===0?'이번에는 K·V의 다음 묶음을 가져왔습니다. Q의 앞 행부터 다시 계산합니다.':'K·V는 그대로 두었습니다. Q의 다음 행 묶음으로 바꿔 계산합니다.';
  if(l.kind==='fa2')$('detail').textContent=f.visit===0?'두 팀이 서로 다른 Q의 행을 맡습니다. 각 팀은 자기 S의 일부만 만들며 독립적으로 진행합니다.':'각 팀이 맡은 Q의 행은 그대로입니다. K·V의 다음 묶음을 가져와 S의 다음 열들을 계산합니다.';
  html=flow('matmul',[matrix('Q',l.q,(i)=>activeQ(i),'input',owner),matrix('Kᵀ',transpose(l.k),(_,j)=>activeK(j)),matrix('S',scores,activeS,'result',owner)],['×','→'],['행렬곱','결과']);
 }else if(p==='weights'){
  $('caption').textContent='방금 만든 S가 이번 연산의 입력입니다. Weights는 소프트맥스에서 마지막에 비중 합계로 나누기 전의 숫자입니다.';
  $('detail').textContent='앞에서 배운 온라인 소프트맥스처럼 지금까지의 가장 큰 점수를 기준으로 크기를 맞춥니다. 이전 내용의 보정은 잠시 뒤에 따로 보여 줍니다.';
  html=flow('weights',[scoreCard(),matrix('Weights',weights,activeS,'result',owner)],['→'],['임시 비중 계산'],{maximum:tileVector((t,i)=>t.m[i])});
 }else if(p==='mix'){
  $('caption').textContent='방금 만든 Weights를 V의 같은 토큰 행과 곱합니다. New mix는 이번에 가져온 토큰들의 내용을 섞은 결과입니다.';
  $('detail').textContent='아직 다른 K·V 묶음의 점수는 없어도, 지금 가진 비중과 내용끼리는 곱할 수 있습니다.';
  html=flow('matmul',[weightCard(),valueCard(),rowCard('New mix',newMix,true)],['×','→'],['행렬곱','결과']);
 }else if(p==='restore'){
  $('caption').textContent='FA1은 이미 나눈 결과를 저장했습니다. Previous O에 그때 썼던 Previous sum을 다시 곱해, 나누기 전의 Previous mix로 되돌립니다.';
  const oldOutput=previousMix.map((row,i)=>row.map(x=>x==null?null:x/previousSum[i][0]));
  html=flow('row-multiply',[rowCard('Previous O',oldOutput),rowCard('Previous sum',previousSum),rowCard('Previous mix',previousMix,true)],['×','→'],['행마다 곱하기','나누기 전 내용']);
 }else if(p==='rescale'){
  $('caption').textContent='Previous mix는 이전에 섞어 둔 내용입니다. Scale은 행마다 곱할 보정 비율이고, Adjusted mix는 그 비율로 크기를 맞춘 내용입니다.';
  const t=tiles[0];$('detail').textContent=`${t.qs+1}행의 기준 점수는 ${fmt(t.old_m[0])}에서 ${fmt(t.m[0])}이 됐습니다. 그 행의 이전 비중들이 모두 ${fmt(t.alpha[0])}배가 되므로, 이미 섞은 두 숫자도 똑같이 ${fmt(t.alpha[0])}배 합니다. 이것은 행렬곱이 아니라 행마다 같은 수를 곱하는 단계입니다.`;
  html=flow('row-multiply',[rowCard('Previous mix',previousMix),rowCard('Scale',scale),rowCard('Adjusted mix',adjusted,true)],['×','→'],['행마다 곱하기','보정한 내용']);
 }else if(p==='add'){
  $('caption').textContent=hadPrevious?'방금 보정한 Adjusted mix에 이번에 만든 New mix를 더합니다. 같은 자리끼리 더한 결과가 Updated mix입니다.':'처음 계산하는 행에는 이전 내용이 없으므로 Previous mix가 0입니다. 이번 New mix를 더한 결과를 Updated mix라고 부릅니다.';
  $('detail').textContent=l.mode==='FA2'?'FA2는 이 Updated mix를 다음 K·V 묶음을 처리할 때까지 보관합니다. 아직 비중 합계로 나누지 않습니다.':'FA1은 이 합계를 잠시 뒤 비중 합계로 나눈 다음 저장합니다.';
  html=flow('add',[rowCard(hadPrevious?'Adjusted mix':'Previous mix',adjusted),rowCard('New mix',newMix),rowCard('Updated mix',tileRows('u'),true)],['+','→'],['같은 자리끼리','새로 모은 내용']);
 }else if(p==='totals'){
  $('caption').textContent='마지막에 나눌 숫자도 함께 계산합니다. 비중은 행마다 따로 더하며, Total sum은 지금까지 본 모든 비중의 합계입니다.';
  if(hadPrevious){
   $('detail').textContent='New sum은 이번 Weights의 행별 합계입니다. Previous sum은 이전 합계이며, 내용에 썼던 Scale을 똑같이 곱한 뒤 New sum을 더합니다.';
   html=flow('row-sum',[weightCard(),rowCard('New sum',newSum,true)],['→'],['각 행의 비중 더하기']);
   html+=continuation('방금 구한 New sum이 아래 덧셈의 입력이 됩니다.');
   html+=flow('sum-update',[rowCard('Previous sum',previousSum),rowCard('Scale',scale),rowCard('New sum',newSum),rowCard('Total sum',totalSum,true)],['×','+','→'],['행마다 곱하기','같은 행끼리','새 비중 합계']);
  }else{
   $('detail').textContent='처음 묶음에서는 이전 비중이 없으므로 이번 Weights의 행별 합계가 곧 Total sum입니다.';
   html=flow('row-sum',[weightCard(),rowCard('Total sum',totalSum,true)],['→'],['각 행의 비중 더하기']);
  }
 }else if(p==='divide'||p==='divide_final'){
  const final=p==='divide_final',mix=final?blank(4,2):tileRows('u'),sum=final?blank(4,1):totalSum,out=final?blank(4,2):tileRows('o');
  if(final)for(let i=0;i<l.activeQueries;i++){mix[i]=f.acc[i];sum[i][0]=f.l[i];out[i]=f.output[i];}
  $('caption').textContent='Updated mix의 각 행을 그 행의 Total sum으로 나눕니다. 비중에 맞춰 섞인 결과가 O입니다.';
  $('detail').textContent=final?'FA2는 모든 K·V 묶음을 반영한 뒤 이 나눗셈을 한 번 합니다.':tiles[0].ke<4?'FA1은 이번 묶음을 처리한 뒤에도 나눕니다. 지금의 O는 중간 결과이며 다음 묶음을 반영할 때 다시 갱신합니다.':'이 행들은 모든 K·V 묶음을 반영했으므로 O가 완성됐습니다.';
  html=flow('row-divide',[rowCard('Updated mix',mix),rowCard('Total sum',sum),rowCard('O',out,true)],['÷','→'],['행마다 나누기','비율을 맞춘 결과']);
 }else if(p==='done'){
  const output=blank(4,2),reference=blank(4,2);for(let i=0;i<l.activeQueries;i++){output[i]=f.output[i];reference[i]=l.reference[i];}
  $('caption').textContent='계산이 끝났습니다. Reference는 모든 점수와 비중을 한꺼번에 계산한 일반 어텐션의 결과입니다.';
  $('detail').textContent='조금씩 곱하고, 이전 내용을 보정해서 더해도 최종 숫자는 같습니다.';
  html=flow('equal',[rowCard('O',output),rowCard('Reference',reference,true)],['='],['같은 결과']);
 }else if(p.startsWith('split')){
  const t=tiles[0],splitK=p==='split_k',mixes=[];
  $('question').textContent=splitK?'FA1 · 두 작업조의 몫을 더해야 한다':'FA2 · 작업조마다 다른 결과 행을 만든다';
  $('schedule').innerHTML='';
  $('caption').textContent=splitK?'Weights의 열을 두 작업조가 나눠 맡습니다. 두 작업조 모두 같은 결과 행의 일부를 만들기 때문에 마지막에 더해야 합니다.':'Weights의 행을 두 작업조가 나눠 맡습니다. 각 작업조가 서로 다른 결과 행을 만들므로 두 몫을 더하는 단계가 없습니다.';
  $('detail').textContent='작업조는 한 팀 안에서 일을 나누는 작은 단위입니다. 아래에서는 두 작업조로 단순화했습니다. Mix 1과 Mix 2는 각 작업조가 만든 내용입니다.';
  for(let g=0;g<2;g++){
   const w=blank(4,4),mix=blank(4,2),mask=(i,j)=>i>=t.qs&&i<t.qe&&j>=t.ks&&j<t.ke&&(splitK?j===t.ks+g:i===t.qs+g);
   for(let i=t.qs;i<t.qe;i++){
    if(!splitK&&i!==t.qs+g)continue;
    mix[i]=[0,0];
    for(let j=t.ks;j<t.ke;j++)if(mask(i,j)){w[i][j]=t.e[i-t.qs][j-t.ks];for(let d=0;d<2;d++)mix[i][d]+=w[i][j]*l.v[j][d];}
   }
   mixes.push(mix);html+=`<div class="group-label ${g?'second':''}">작업조 ${g+1}</div>`;
   html+=flow('matmul',[matrix(`Weights ${g+1}`,w,mask,'input',()=>g===1),matrix('V',l.v,(i)=>splitK?i===t.ks+g:i>=t.ks&&i<t.ke,'input',()=>g===1),matrix(`Mix ${g+1}`,mix,(i,j)=>mix[i][j]!=null,'result',()=>g===1)],['×','→'],['행렬곱',splitK?'같은 결과 행의 일부':'자기가 맡은 결과 행']);
  }
  if(splitK){html+=continuation('위에서 만든 Mix 1과 Mix 2를 같은 자리끼리 더합니다.');html+=flow('add',[rowCard('Mix 1',mixes[0]),rowCard('Mix 2',mixes[1]),rowCard('New mix',newMix,true)],['+','→'],['작업조의 몫 합치기','두 작업조가 만든 내용']);}
 }
 $('calculation').innerHTML=html;
 fit();
}
function fit(){try{if(window.frameElement)window.frameElement.style.height=Math.ceil(root.getBoundingClientRect().height+28)+'px';}catch(e){}}
root.querySelectorAll('[data-lesson]').forEach(b=>b.addEventListener('click',()=>{lesson=Number(b.dataset.lesson);step=0;render();}));
$('next').addEventListener('click',()=>{if(step<lessons[lesson].frames.length-1)step++;else if(lesson<lessons.length-1){lesson++;step=0;}render();});
$('prev').addEventListener('click',()=>{if(step>0)step--;else if(lesson>0){lesson--;step=lessons[lesson].frames.length-1;}render();});
root.addEventListener('keydown',e=>{if(e.key==='ArrowRight'){e.preventDefault();$('next').click();}else if(e.key==='ArrowLeft'){e.preventDefault();$('prev').click();}});
try{const s=window.parent.getComputedStyle(window.parent.document.body);[['--bg','--jp-layout-color0'],['--fg','--jp-ui-font-color1'],['--muted','--jp-ui-font-color2'],['--line','--jp-border-color1']].forEach(([a,b])=>{const v=s.getPropertyValue(b).trim();if(v)document.documentElement.style.setProperty(a,v);});}catch(e){}
new ResizeObserver(fit).observe(root);render();
})();
</script>
</html>
"""

def flash_lesson(q, k, v):
    q, k, v = (np.asarray(x, dtype=np.float32) for x in (q, k, v))
    if any(x.shape != (4, 2) for x in (q, k, v)):
        raise ValueError("이 설명은 토큰 4개, 토큰당 숫자 2개의 예제를 사용합니다.")
    lessons = []
    for kind, question, rows, mode in [
        ("row", "Q의 1행부터: 무엇을 곱하고 무엇을 더하는가", 1, "FA2"),
        ("fa1", "FA1: K·V를 두고 Q의 행 묶음을 바꾼다", 4, "FA1"),
        ("fa2", "FA2: 각 팀이 Q의 행을 맡고 K·V를 가져온다", 4, "FA2"),
    ]:
        trace = flash_trace(q[:rows], k, v, mode=mode)
        reference = flash_reference(q[:rows], k, v)
        np.testing.assert_allclose(trace["output"], reference, rtol=1e-6, atol=1e-7)
        frames = []
        for frame in trace["frames"]:
            if frame["phase"] != "read":
                continue
            phases = ["score", "weights", "mix"]
            if any(np.any(tile["old_l"] > 0) for tile in frame["tiles"]):
                if mode == "FA1":
                    phases.append("restore")
                phases.append("rescale")
            phases.extend(["add", "totals"])
            if mode == "FA1":
                phases.append("divide")
            frames.extend({**frame, "phase": phase} for phase in phases)
        final = trace["frames"][-1]
        if mode == "FA2":
            frames.append({**final, "phase": "divide_final"})
        frames.append(final)
        trace.pop("records")
        trace.update(frames=frames, kind=kind, question=question, activeQueries=rows,
                     q=q, reference=reference)
        lessons.append(trace)
    split = flash_trace(q, k, v, mode="FA1")
    first = split["frames"][0]
    split.pop("records")
    split.update(kind="split", question="팀 안에서 연산을 나누는 방법", activeQueries=4,
                 frames=[{**first, "phase": "split_k"}, {**first, "phase": "split_q"}])
    lessons.append(split)
    data = json.dumps(_flash_json(lessons), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    document = FLASH_DOCUMENT.replace("__DATA__", data.replace("</", r"<\/"))
    return IFrame("about:blank", width="100%", height=680, extras=[
        'title="FlashAttention 연산 흐름"',
        'sandbox="allow-scripts allow-same-origin"',
        'style="display:block;width:100%;height:680px;border:0"',
        'srcdoc="' + html.escape(document, quote=True) + '"',
    ])


# Memory movement and causal masking

def attention_context(kind):
    if kind not in ('memory', 'causal'):
        raise ValueError('Unknown lesson')
    q = np.array([[1, 1], [1, -.5], [.5, 1], [-.5, 1]], dtype=np.float64)
    k = np.array([[.5, 1.5], [1.5, -.5], [1.5, 1.5], [.5, -.5]], dtype=np.float64)
    v = np.array([[1, 0], [0, 1], [2, 1], [1, 2]], dtype=np.float64)
    scores = q @ k.T
    def probabilities(values):
        weights = np.exp(values-values.max(axis=1, keepdims=True))
        return weights / weights.sum(axis=1, keepdims=True)
    p = probabilities(scores)
    mask = np.tri(4, dtype=bool)
    masked = np.where(mask, scores, -np.inf)
    causal_p = probabilities(masked)
    causal_o = causal_p @ v
    for row in range(4):
        np.testing.assert_allclose(causal_o[row], (probabilities(scores[row:row+1, :row+1]) @ v[:row+1])[0], atol=1e-12)
    def pad(values, rows, cols):
        result = [[None]*cols for _ in range(rows)]
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                result[i][j] = float(values[i,j])
        return result
    weights = np.exp(scores[:2,:2]-scores[:2,:2].max(axis=1, keepdims=True))
    mixed = weights @ v[:2]
    data = dict(q=q.tolist(), k=k.tolist(), v=v.tolist(), s=scores.tolist(), p=p.tolist(), o=(p@v).tolist(),
                mask=mask.astype(int).tolist(), causal_p=causal_p.tolist(), causal_o=causal_o.tolist(),
                masked_s=[[float(scores[i,j]) if mask[i,j] else '×' for j in range(4)] for i in range(4)],
                memory=dict(partial_k=pad(k[:2],4,2), partial_v=pad(v[:2],4,2), partial_s=pad(scores[:2,:2],4,4),
                            partial_weights=pad(weights,4,4), partial_mix=pad(mixed,4,2),
                            partial_output=pad(mixed/weights.sum(axis=1,keepdims=True),4,2),
                            final_output=pad((p@v)[:2],4,2)))
    document = CONTEXT_DOCUMENT.replace('__KIND__', kind).replace('__DATA__', json.dumps(data, ensure_ascii=False, allow_nan=False).replace('</', r'<\/'))
    return IFrame('about:blank', width='100%', height=740, extras=[
        'title="Attention '+kind+' lesson"', 'sandbox="allow-scripts allow-same-origin"',
        'style="display:block;width:100%;height:740px;border:0"', 'srcdoc="'+html.escape(document,quote=True)+'"'])

CONTEXT_DOCUMENT = r"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:light dark;--bg:#fff;--fg:#20242b;--muted:#647181;--line:#d5dee8;--quiet:#f3f6f9;--blue:#237db4;--red:#b45b52}*{box-sizing:border-box}body{margin:0;padding:12px;background:var(--bg);color:var(--fg);font:14px/1.6 Arial,'Malgun Gothic',sans-serif}
@media(prefers-color-scheme:dark){:root{--bg:#202328;--fg:#eef2f7;--muted:#b0bac7;--line:#455160;--quiet:#2b333e;--blue:#72b8ea;--red:#e3978f}}
button,select{font:inherit;color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:4px;padding:5px 10px;min-height:34px}button{cursor:pointer}button[aria-pressed=true],#next{color:var(--bg);background:var(--fg)}button:disabled{opacity:.35;cursor:default}.toolbar{display:flex;gap:6px;align-items:center;flex-wrap:wrap}.count{margin-left:auto;color:var(--muted);font-size:12px}
h3{font-size:18px;margin:14px 0 8px}.caption{background:var(--quiet);border-left:3px solid var(--blue);padding:10px 12px;margin:12px 0 18px;line-height:1.8}.detail{font-size:12px;color:var(--muted);margin:6px 0 14px}
.compare{display:grid;grid-template-columns:1fr 1fr;gap:20px}.panel{min-width:0}.panel h4{font-size:17px;margin:0 0 8px}.transfer{display:grid;grid-template-columns:minmax(0,1fr) 32px minmax(0,1fr);gap:6px;align-items:center}.area{align-self:stretch;border:1px solid var(--line);padding:8px;min-width:0}.place{font-size:12px;font-weight:600;border-bottom:1px solid var(--line);padding-bottom:5px;margin-bottom:8px}.area.local{border-top:3px solid var(--blue)}.area.global{border-top:3px solid var(--red)}
.card{min-width:0;margin:8px 0}.heading{display:flex;justify-content:space-between;align-items:baseline;gap:5px;margin-bottom:5px}.label{font-weight:600;font-size:16px}.shape{font-size:10px;color:var(--muted)}.matrix{display:grid;gap:3px;font-variant-numeric:tabular-nums}.axis{min-height:22px;display:flex;justify-content:center;align-items:center;font-size:10px;color:var(--muted)}.cell{min-height:29px;border:1px solid var(--line);display:flex;align-items:center;justify-content:center;font-size:12px;padding:2px 0}.cell.active{background:color-mix(in srgb,var(--blue) 17%,var(--bg));border-color:color-mix(in srgb,var(--blue) 45%,var(--line))}.cell.empty{color:var(--muted);background:var(--quiet)}.cell.blocked{background:color-mix(in srgb,var(--red) 13%,var(--bg));color:var(--red)}.cell.tile-top{border-top:2px solid var(--fg)}.cell.tile-left{border-left:2px solid var(--fg)}
.operator{text-align:center;line-height:1.4;font-size:27px;color:var(--blue)}.operator small{display:block;font-size:10px;color:var(--muted);word-break:keep-all}.panel-note{margin-top:8px;font-size:12px;min-height:45px}.flow{display:grid;grid-template-columns:var(--cols);gap:12px;align-items:center}.flow .cell{min-height:35px;font-size:14px}.flow .axis{font-size:12px}.flow .label{font-size:18px}.foot{font-size:11px;color:var(--muted);margin-top:14px}
@media(max-width:740px){.compare{grid-template-columns:1fr;gap:22px}.flow{grid-template-columns:1fr!important;max-width:440px;margin:auto}.flow .operator.arrow{font-size:0}.flow .operator.arrow:before{content:'↓';font-size:26px}.count{margin-left:0}.transfer{grid-template-columns:minmax(0,1fr) 30px minmax(0,1fr)}}
</style>
<main id="attention-context"><div class="toolbar" id="toolbar"></div><h3 id="title"></h3><div id="caption" class="caption"></div><div id="detail" class="detail"></div><div id="scene"></div><div id="foot" class="foot"></div></main>
<script type="application/json" id="context-data">__DATA__</script>
<script>
(()=>{
const data=JSON.parse(document.getElementById('context-data').textContent),kind='__KIND__',root=document.getElementById('attention-context'),$=id=>document.getElementById(id);
let step=0,algorithm='FA2',causal=true,row=1;
const fmt=v=>v==null?'·':typeof v==='number'?Number(v.toFixed(3)).toString():v;
function matrix(name,values,active=()=>true,blocked=()=>false,tiles=false,axes=null){
 const rows=values.length,cols=values[0].length;let h='<span class="axis"></span>'+Array.from({length:cols},(_,j)=>`<span class="axis">${axes?axes[j]:j+1}</span>`).join('');
 for(let i=0;i<rows;i++){h+=`<span class="axis">${axes?axes[i]:i+1}</span>`;for(let j=0;j<cols;j++)h+=`<span class="cell ${active(i,j)?'active':''} ${blocked(i,j)?'blocked':''} ${values[i][j]==null?'empty':''} ${tiles&&i===2?'tile-top':''} ${tiles&&j===2?'tile-left':''}">${fmt(values[i][j])}</span>`;}
 return `<section class="card" data-name="${name}" data-values='${JSON.stringify(values)}'><div class="heading"><span class="label">${name}</span><span class="shape">${rows} × ${cols}</span></div><div class="matrix" style="grid-template-columns:18px repeat(${cols},minmax(0,1fr))">${h}</div></section>`;
}
const arrow=label=>`<div class="operator arrow">→<small>${label}</small></div>`;
function area(place,values,name){return `<div class="area ${place==='VRAM'?'global':'local'}"><div class="place">${place}</div>`+values.map((v,i)=>matrix(Array.isArray(name)?name[i]:name,v,(_,j)=>true)).join('')+'</div>';}
function panel(title,from,to,values,name,label,note,target=null){return `<section class="panel"><h4>${title}</h4><div class="transfer">${area(from,values,name)}${arrow(label)}${area(to,target||values,name)}</div><div class="panel-note">${note}</div></section>`;}
function flow(cards,labels,operation){const cols=cards.map((_,i)=>i===cards.length-1?'minmax(0,1fr)':'minmax(0,1fr) 65px').join(' ');return `<div class="flow" data-operation="${operation}" style="--cols:${cols}">`+cards.map((c,i)=>c+(i<cards.length-1?(labels[i]==='×'?'<div class="operator">×<small>행렬곱</small></div>':arrow(labels[i])):'')).join('')+'</div>';}
function render(){
 const count=kind==='memory'?8:5;root.dataset.kind=kind;root.dataset.step=step;root.dataset.algorithm=algorithm;root.dataset.causal=causal;root.dataset.row=row;
 $('toolbar').innerHTML=(kind==='memory'?`<button data-alg="FA1" aria-pressed="${algorithm==='FA1'}">Eager / FA1</button><button data-alg="FA2" aria-pressed="${algorithm==='FA2'}">Eager / FA2</button>`:`<button id="mask-on" aria-pressed="${causal}">Causal on</button><button id="mask-off" aria-pressed="${!causal}">Causal off</button><label>Q 행 <select id="query-row">${[0,1,2,3].map(i=>`<option value="${i}" ${i===row?'selected':''}>${i+1}</option>`).join('')}</select></label>`)+`<span class="count">${step+1} / ${count}</span><button id="prev" ${step===0?'disabled':''}>이전</button><button id="next" ${step===count-1?'disabled':''}>다음</button>`;
 $('detail').textContent='';
 if(kind==='memory'){
  const f=data.memory,partial=algorithm==='FA1'?f.partial_output:f.partial_mix;
  const stages=[
   {title:'K·V는 두 방식 모두 VRAM에서 읽는다',caption:'FlashAttention도 K·V를 가져와야 합니다. 작은 묶음을 읽고 계산한 뒤 다음 묶음을 가져옵니다. Q를 맡은 팀이 여러 개이면 각 팀도 필요한 K·V를 읽습니다.',e:['VRAM','GPU 연산',[data.k,data.v],['K','V'],'읽기','Eager가 사용하는 전체 입력 범위입니다.'],a:['VRAM','SM 작업공간',[data.k,data.v],['K','V'],'읽기','이번 팀은 K·V의 1~2행부터 읽습니다.',[f.partial_k,f.partial_v]]},
   {title:'S를 만든 직후: 어디에 두는가?',caption:'Eager는 전체 S를 큰 텐서로 남깁니다. FlashAttention은 이번 묶음의 S를 작업공간에서 바로 다음 계산에 넘깁니다.',e:['GPU 연산','VRAM',[data.s],'S','쓰기','전체 점수표를 저장합니다.'],a:['SM 작업공간','SM 작업공간',[f.partial_s],'S','유지','이 작은 점수 조각으로 곧바로 비중을 계산합니다.']},
   {title:'Softmax가 점수를 읽는다',caption:'Eager의 다음 커널은 저장해 둔 S를 읽습니다. FlashAttention은 같은 작업공간에 있는 점수 조각을 사용합니다.',e:['VRAM','GPU 연산',[data.s],'S','읽기','앞 단계에서 저장한 전체 S를 다시 읽습니다.'],a:['SM 작업공간','SM 작업공간',[f.partial_s],'S','사용','S를 큰 텐서로 저장했다가 다시 가져오는 단계가 없습니다.']},
   {title:'비중을 만든 직후에도 차이가 생긴다',caption:'P는 비중 합계로 나눈 소프트맥스 결과입니다. Weights는 아직 마지막 나눗셈을 하기 전의 비중입니다. Eager는 P를 저장하고, FlashAttention은 Weights를 바로 씁니다.',e:['GPU 연산','VRAM',[data.p],'P','쓰기','다음 행렬곱에서 사용할 전체 P를 저장합니다.'],a:['SM 작업공간','SM 작업공간',[f.partial_weights],'Weights','유지','V와 곱할 때까지 같은 작업공간에 둡니다.']},
   {title:'V와 곱할 비중을 가져온다',caption:'Eager는 저장한 P를 다시 읽어 V와 곱합니다. FlashAttention은 작업공간에 남아 있는 Weights로 V를 바로 섞습니다.',e:['VRAM','GPU 연산',[data.p],'P','읽기','큰 중간 텐서를 읽는 단계가 한 번 더 생깁니다.'],a:['SM 작업공간','SM 작업공간',[f.partial_weights],'Weights','사용','4절에서 본 Weights와 V의 곱셈으로 이어집니다.']},
   {title:'이번 묶음의 결과를 어디에 남기는가?',caption:algorithm==='FA1'?'FA1은 Q의 다른 행 묶음으로 옮겨 가기 위해 현재까지의 결과를 저장합니다. 기준 점수와 비중 합계도 함께 남깁니다.':'FA2는 같은 팀이 Q의 행을 끝까지 맡습니다. 다음 K·V 묶음을 처리할 때 쓸 누적 내용을 곁에 남깁니다.',e:['GPU 연산','VRAM',[data.o],'O','쓰기','Eager는 이 예제의 전체 결과를 저장합니다.'],a:['SM 작업공간',algorithm==='FA1'?'VRAM':'SM 작업공간',[partial],algorithm==='FA1'?'O (partial)':'Updated mix',algorithm==='FA1'?'쓰기':'유지',algorithm==='FA1'?'다른 Q 묶음을 계산하러 갑니다.':'아직 나누기 전 내용입니다. 다음 K·V를 가져옵니다.']},
   {title:'다음 K·V 묶음에 이전 결과를 이어 붙인다',caption:algorithm==='FA1'?'FA1이 이 Q 묶음으로 돌아오면 저장한 중간 결과를 다시 가져옵니다. 4절의 나눗셈 되돌리기·보정·덧셈을 이어 합니다.':'FA2는 남겨 둔 내용에 곧바로 보정과 덧셈을 이어 합니다. 다른 Q 묶음의 팀을 기다리지 않습니다.',e:['VRAM','VRAM',[data.o],'O','보관','Eager는 이미 모든 K·V를 반영했습니다.'],a:[algorithm==='FA1'?'VRAM':'SM 작업공간','SM 작업공간',[partial],algorithm==='FA1'?'O (partial)':'Updated mix',algorithm==='FA1'?'읽기':'사용',algorithm==='FA1'?'같은 행의 기준 점수와 비중 합계도 복원합니다.':'같은 행의 기준 점수와 비중 합계도 계속 유지합니다.']},
   {title:'담당 행이 모든 K·V를 반영하면 최종 O를 저장한다',caption:'이 팀이 계산한 O의 1~2행을 저장합니다. 다른 행도 저장되면 전체 O가 완성됩니다. 큰 S와 P를 중간에 저장하고 읽는 일이 줄어드는 것이 핵심입니다.',e:['VRAM','VRAM',[data.o],'O','보관','전체 결과가 이미 저장돼 있습니다.'],a:['SM 작업공간','VRAM',[f.final_output],'O','쓰기',algorithm==='FA2'?'FA2는 마지막 나눗셈 후 자기 결과 행만 저장합니다. 다른 팀은 3~4행을 맡습니다.':'FA1은 이 Q 묶음의 마지막 갱신을 저장합니다. 3~4행도 같은 방식으로 마칩니다.']}
  ];
  const s=stages[step];$('title').textContent=s.title;$('caption').textContent=s.caption;
  $('scene').innerHTML='<div class="compare">'+panel('Eager',...s.e)+panel(algorithm,...s.a)+'</div>';
  $('detail').textContent='SM은 GPU의 계산 구역입니다. 여기서 작업공간은 SM의 레지스터와 shared memory를 뜻합니다. 그림은 한 head의 논리적 데이터 이동입니다.';
  $('foot').textContent='전체 행렬 틀은 위치를 보여 줍니다. ·는 이 팀이 현재 보관하지 않는 자리입니다. Eager의 행렬곱도 내부적으로 타일을 사용합니다. VRAM 화살표는 global 텐서 읽기·쓰기를 뜻하며 실제 DRAM 전송량은 캐시에 따라 달라집니다.';
 }else{
  const mask=causal?data.mask:data.mask.map(r=>r.map(()=>1)),p=causal?data.causal_p:data.p,o=causal?data.causal_o:data.o,s=causal?data.masked_s:data.s;
  const active=i=>i===row,blocked=(i,j)=>causal&&j>i;
  const titles=['어느 토큰까지 볼 수 있는가?','미래 토큰의 점수를 가린다','가린 칸은 비중이 0이 된다','이 비중으로 V를 섞는다','FlashAttention도 같은 규칙을 작은 묶음에 적용한다'];
  $('title').textContent=causal?titles[step]:['마스크를 끄면 모든 토큰을 볼 수 있다','마스크를 끄면 S를 그대로 사용한다','모든 토큰 사이에서 비중을 나눈다','미래 토큰의 V도 결과에 들어온다','마스크가 없으면 모든 묶음을 계산한다'][step];
  if(step===0){$('caption').textContent=`Mask의 1은 참고 가능, 0은 가릴 칸입니다. ${row+1}행은 토큰 ${row+1}의 결과를 만드는 행입니다. ${causal?'자기 자신과 앞쪽 토큰만 볼 수 있습니다.':'지금은 마스크를 꺼서 뒤쪽 토큰도 볼 수 있습니다.'}`;$('scene').innerHTML='<div style="max-width:480px;margin:auto">'+matrix('Mask',mask,active,blocked)+'</div>';}
  if(step===1){$('caption').textContent=causal?'S의 미래 토큰 칸을 가립니다. Masked S의 ×는 이후 비중 계산에서 제외하는 자리라는 뜻입니다. 행과 열의 위치는 그대로입니다.':'마스크를 껐으므로 어떤 칸도 가리지 않습니다. 오른쪽 S는 왼쪽 S와 같으며 뒤쪽 토큰의 점수도 그대로 들어갑니다.';$('scene').innerHTML=flow([matrix('S',data.s,active),matrix(causal?'Masked S':'S',s,active,blocked)],[causal?'미래 칸 가리기':'그대로 사용'],'mask');}
  if(step===2){$('caption').textContent=causal?'Masked S에 소프트맥스를 적용해 P를 만듭니다. 가린 칸의 비중은 0이고, 볼 수 있는 칸들끼리 합이 1이 됩니다.':'모든 점수에 소프트맥스를 적용합니다. 미래 토큰에도 비중이 생기고, 네 토큰의 비중을 모두 더해야 1이 됩니다.';$('scene').innerHTML=flow([matrix(causal?'Masked S':'S',s,active,blocked),matrix('P',p,active,blocked)],['소프트맥스'],'softmax');}
  if(step===3){$('caption').textContent=causal?'P와 V를 곱해 O를 만듭니다. 미래 토큰의 비중이 0이므로 그 토큰의 V는 현재 행의 결과에 영향을 주지 않습니다.':'P와 V를 곱해 O를 만듭니다. 미래 토큰에도 비중이 있으므로 그 토큰의 V까지 섞입니다. Causal on으로 바꾸면 이 기여가 사라집니다.';$('scene').innerHTML=flow([matrix('P',p,active,blocked),matrix('V',data.v,i=>!causal||i<=row),matrix('O',o,active)],['×','결과'],'matmul');}
  if(step===4){const decisions=causal?[['Mask','Skip'],['Keep','Mask']]:[['Keep','Keep'],['Keep','Keep']];$('caption').textContent='Mask를 2행·2열씩 묶어서 봅니다. Keep은 전부 허용, Mask는 일부만 가림, Skip은 전부 미래라 건너뛸 수 있다는 뜻입니다.';$('detail').textContent='Q의 1~2행과 K의 3~4행이 만나는 오른쪽 위 묶음은 전부 미래입니다. 대각선 묶음은 일부 칸만 가려야 합니다.';$('scene').innerHTML=flow([matrix('Mask',mask,()=>true,blocked,true),matrix('Tile decisions',decisions,()=>true,(_,j)=>false,false,['1–2','3–4'])],['2 × 2 묶음으로 보기'],'tiles');}
  if(!causal)$('detail').textContent='Causal off는 차이를 보기 위한 비교입니다. 이 노트북의 실제 GPT-2 측정은 causal mask를 켭니다.';
  if(row===3&&step<4)$('detail').textContent+=' 4행은 마지막 토큰이라 가릴 미래 토큰이 없습니다. 이 행의 결과는 on/off가 같습니다.';
  $('foot').textContent='표의 모든 행과 열을 표시했습니다. Q 행을 바꾸면 그 토큰이 어떤 V를 참고하는지 따라갈 수 있습니다. ×는 0점이라는 뜻이 아니라 비중 계산에서 제외한다는 뜻입니다.';
 }
 $('prev').onclick=()=>{step--;render()};$('next').onclick=()=>{step++;render()};
 root.querySelectorAll('[data-alg]').forEach(b=>b.onclick=()=>{algorithm=b.dataset.alg;render()});
 if(kind==='causal'){$('mask-on').onclick=()=>{causal=true;render()};$('mask-off').onclick=()=>{causal=false;render()};$('query-row').onchange=e=>{row=Number(e.target.value);render()};}
 fit();
}
function fit(){try{if(window.frameElement)window.frameElement.style.height=Math.ceil(root.getBoundingClientRect().height+28)+'px'}catch(e){}}
try{const s=window.parent.getComputedStyle(window.parent.document.body);[['--bg','--jp-layout-color0'],['--fg','--jp-ui-font-color1'],['--muted','--jp-ui-font-color2'],['--line','--jp-border-color1']].forEach(([a,b])=>{const v=s.getPropertyValue(b).trim();if(v)document.documentElement.style.setProperty(a,v)});}catch(e){}
new ResizeObserver(fit).observe(root);render();
})();
</script></html>
"""
