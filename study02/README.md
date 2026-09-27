# nanoGPT CUDA compiler study

`main.ipynb` 하나에서 모든 실습과 실행 결과를 읽을 수 있습니다. PyTorch 공개 API만 사용해 본 독자를 위한 한국어 학습 자료로, 작은 GELU를 Python → Dynamo FX → AOT → Inductor IR → Triton → CUDA 실행까지 따라간 뒤 LayerNorm, MLP, GPT block으로 확장합니다. Plotly 그림이 기본 출력이고, 원본 graph/IR/code는 펼쳐서 읽습니다. 독립 Python 파일은 이 노트북의 실행을 지원합니다.

## 실행

저장소 루트에서 실행합니다. 의존성은 루트 `pyproject.toml` / `uv.lock`으로 통합 관리합니다.

```bash
uv sync --locked
uv run --locked jupyter lab
```

Jupyter에서 `study02/main.ipynb`를 열고 이 저장소의 uv Python 커널로 **Restart Kernel and Run All**을 실행합니다. 첫 설정 셀의 `executable`이 이 저장소 `.venv/bin/python`인지 확인하세요. 노트북을 저장소 루트나 `study02`에서 시작할 수 있습니다.

현재 lock과 `.python-version`의 기준은 Python 3.9 / PyTorch 2.8.0이며 NVIDIA CUDA가 필요합니다. CPU로 자동 전환하지 않습니다. RTX 2080에서는 FP32를 기본으로 사용합니다. BF16은 native 지원 여부를 확인합니다. SDPA API 선택을 FlashAttention 선택과 동일시하지 않습니다.

## 학습 구성

| 노트북 절 | 주제 | 원문 |
|---|---|---|
| 00–01 | Pipeline 지도, tensor/storage/stride, 그래프 문법, autograd와 module 상태 | §1–3 |
| 02 | 비동기 GPU 실행, memory 계층, pointwise/reduction/GEMM, tile | §4 |
| 03–04 | Embedding·LayerNorm·attention·MLP, eager와 GELU | §5–6 |
| 05–06 | CPython bytecode/frame, symbolic 실행, FakeTensor/SymInt, Dynamo/guards | §7 |
| 07 | Functionalization, joint/FW/BW graph, saved values·tangents | §8 |
| 08–09 | Lowering, index/loop/buffer, scheduler grouping, memory reuse | §9–11 |
| 10 | Generated Triton, TTIR/TTGIR/LLVM IR/PTX/cubin/SASS | §12 |
| 11–13 | Block의 전체 계보, manual/SDPA/training, 실제 GPU profile, fusion·megakernel | §13–15 |
| 14 | 복습, 실행 파일, 산출물·재현 안내 | §16 |

원전은 사용자가 제공한 `gpt-2/compiler_lab/build/nanogpt-compiler.pdf`(85쪽)입니다. 원전의 CPU 캡처를 가져오지 않고 CUDA로 직접 생성합니다. Source 코드 정의만 필요한 범위에서 재사용했습니다. 출처: Andrej Karpathy의 nanoGPT와 제공된 compiler_lab의 교육용 GELU/capture 방식. `model.py`는 parameter 이름, weight tying, causal attention, inference/training 의미를 유지한 작은 구성입니다. 원본 nanoGPT MIT 라이선스는 `LICENSE.nanogpt`에 있습니다.

## 모든 결과는 실행할 때 생성

- 각 실습은 `sys.executable`로 독립 `.py` 모듈을 새 프로세스에서 실행합니다.
- Torch import 전에 log/debug/Inductor/Triton cache 위치를 고유 run 디렉터리로 설정합니다.
- 실행이 성공한 뒤 그 run의 `result.json`과 raw 파일로 그림을 만듭니다. 이전 run, 압축파일, 원문의 동봉 결과를 찾는 경로는 없습니다.
- `study02/artifacts/<experiment>-<unique-id>/`에는 manifest, source/raw SHA-256, 결과, 원본 로그·IR·코드·trace가 남습니다. 이 디렉터리는 Git에서 제외됩니다.
- 기존 출력 디렉터리를 `--output`으로 다시 지정하면 오류를 냅니다. 실패한 실행은 진단을 남기고 이전 결과로 대체하지 않습니다.
- 루트 `tmp/`의 확인용 결과 복사본도 notebook 입력으로 사용하지 않습니다.
- 최종 `main.ipynb`에는 전체를 실제 실행해 얻은 output을 저장합니다. 이 output은 복습용 표시이며, 재실행의 입력으로 사용하지 않습니다. 재실행하면 새 실험과 그림을 생성합니다.
- `main.pdf`는 완성된 노트북을 전체 실행한 뒤 변환한 결과입니다. Interactive graph의 구간·단계도 PDF에서 읽을 수 있도록 펼칩니다.

## 노트북 밖에서 실험 실행

```bash
uv run --locked python -m study02.experiments.python_runtime
uv run --locked python -m study02.experiments.dynamo
uv run --locked python -m study02.experiments.aot_autograd
uv run --locked python -m study02.experiments.inductor --workload gelu
uv run --locked python -m study02.experiments.inductor --workload layernorm
uv run --locked python -m study02.experiments.inductor --workload residual_layernorm
uv run --locked python -m study02.experiments.inductor --workload mlp
uv run --locked python -m study02.experiments.inductor --workload block1 --attention manual
uv run --locked python -m study02.experiments.inductor --workload block1 --attention sdpa
uv run --locked python -m study02.experiments.inductor --workload block1 --training
uv run --locked python -m study02.experiments.profiling --workload gelu --mode eager
uv run --locked python -m study02.experiments.profiling --workload gelu --mode compiled
uv run --locked python -m study02.experiments.profiling --workload vocab_projection --vocab-size 50257
uv run --locked python -m study02.experiments.profiling --workload vocab_projection --vocab-size 50304
uv run --locked python -m study02.experiments.triton_codegen
```

`--dtype float16`로 별도 dtype 실험을 할 수 있습니다. 모델·입력·비교 대상을 같은 조건으로 맞춰 결과를 읽으세요. Shape와 compile 시간은 환경에 따라 달라지며 수치 검증을 통과해야 비교 결과를 사용합니다.

### SASS와 Nsight

Triton 실행은 실제 `CompiledKernel.asm`의 각 stage를 저장합니다. SASS는 `nvdisasm` 또는 `cuobjdump`가 있으면 생성합니다. 도구가 없거나 지원되지 않는 경우 해당 단계의 이유를 표시합니다.

```bash
uv run --locked python -m study02.experiments.triton_codegen --nsight
```

이 옵션은 ncu로 별도 kernel 실행을 수행해 occupancy와 DRAM traffic counter를 요청합니다. NVIDIA performance-counter 접근 권한이 필요하며 실패한 측정은 `hardware_counters.reason`과 raw log에 남습니다. 최종 노트북은 `RUN_NCU=True`로 이 단계도 시도했습니다. 이 호스트에서는 `ERR_NVGPUCTRPERM`을 확인했으며, 측정되지 않은 hardware counter를 숫자로 채우지 않습니다. Register/shared-memory compile metadata, CUDA profiler와 SASS는 실제로 수집했습니다.

## 컴파일의 GPU 성능 효과

GELU의 작은/중간/큰 입력, 큰 residual+LayerNorm, GEMM 중심 MLP, GPT block을 비교합니다. 유리한 사례와 이득이 작거나 느린 사례를 모두 표시합니다.

- 이미 GPU에 있는 같은 입력·weight·dtype를 eager/compiled 별도 프로세스에서 실행합니다.
- Warm-up 후 100회 호출을 7번 측정하고 CUDA event 중앙값과 IQR을 표시합니다.
- GPU 시간·입력 처리량·실제 launch 수·kernel duration 합을 함께 읽습니다. Memory는 별도 그래프입니다.
- Timeline의 기본 표시는 실제 GPU kernel입니다. CPU host event도 raw trace에 보존됩니다.
- 첫 compile 비용과 비용 회수 모델은 별도 그림으로 표시합니다. 누적시간 모델은 실측값을 이용한 계산임을 구분합니다.
- Block의 IR와 성능 실험은 동일한 `B=4,T=128,C=256,H=4,V=1024`로 연결합니다.

## 그림 읽기

- FX/AOT의 화살표는 값의 의존성입니다. Graph의 화면 위치를 GPU 실행 시각으로 읽지 않습니다.
- 큰 DAG는 구간 메뉴로 이동합니다. 회색은 다른 구간에서 들어오는 node이며 원본 node/edge는 모두 저장합니다.
- Scheduling Sankey의 폭은 포함된 pre node 수입니다. Time 또는 byte 수가 아닙니다.
- Buffer lifetime은 generated wrapper의 정적 소스 줄 범위입니다. 실제 GPU 시간은 별도의 profiler timeline에서 봅니다.
- Kernel timeline은 현재 CUDA profiler 결과입니다. CPU event와 GPU event는 중첩될 수 있습니다.
- 실제 캡처가 아닌 구조·비용 설명은 `개념도`/`이론 계산`으로 표시합니다. Graph→IR→kernel 관계를 확인하지 못한 부분을 추측으로 연결하지 않습니다.

## 검증 재실행

```bash
uv run --locked python -m unittest discover -s study02/tests -v
uv run --locked python -m study02.verify
uv run --locked python -m study02.verify --cwd study02
uv run --locked python -m study02.export_pdf
```

검증기는 해당 uv Python을 가리키는 임시 kernelspec으로 새 커널을 띄워 노트북 전체를 실행합니다. 사용자 전역 Jupyter 등록은 변경하지 않습니다. 결과는 `artifacts/validation-*/main.executed.ipynb`, `main.html`, `verification.json`에 저장됩니다. Notebook/source는 검증 중 덮어쓰지 않습니다.

PDF 변환기는 실행이 끝난 노트북만 받습니다. 설치된 Chromium을 사용하고 필요하면 `--browser /path/to/chrome`을 지정합니다. Plotly의 모든 dropdown/slider 구간을 SVG로 펼쳐 담고, 실제 CUDA timeline의 시간 단위와 kernel 이름을 보존합니다. `--notebook`과 `--output`으로 검증 산출물을 직접 지정할 수도 있습니다.

Source syntax만 확인할 때는 생성된 IR의 읽기용 `.py`를 제외합니다. Compiler의 `fx_graph_readable.py`는 `class <lambda>` 같은 표시를 포함할 수 있어 일반 Python 소스가 아닙니다.

```bash
uv run --locked python -m compileall -q -x '/artifacts/' study02
```

## 이 호스트에서 완료한 검증

- Python 3.9.25, PyTorch 2.8.0+cu128, Triton 3.4.0, RTX 2080 8GB에서 실행했습니다.
- 새 커널 전체 실행: 코드 44개, CUDA 실험 26개, Plotly 그림 68개.
- 수치·causality·gradient·mutation·fresh process·오류 처리·실제 IR parser 테스트 7개 통과.
- 원본 nanoGPT와 동일 weight를 넣어 CUDA logits, masked loss, parameter gradient의 일치를 확인했습니다.
- 최종 PDF: 108쪽, 모든 interactive 상태를 펼친 정적 그림 102개.
- Nsight hardware counter는 실제 권한 오류를 기록했습니다. 다른 CUDA/Triton/SASS 실습은 실행됐습니다.
