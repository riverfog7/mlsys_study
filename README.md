# MLSys study

- `study01/main.ipynb`: GPT-2 token/embedding 탐색 노트.
- `study02/main.ipynb`: nanoGPT를 Python에서 실제 CUDA kernel까지 추적하는 한국어 compiler lab.
- `study02/main.pdf`: 전체 실행 결과와 interactive 그래프의 모든 구간을 펼친 PDF.

```bash
uv sync --locked
uv run --locked jupyter lab
```

`study02`는 NVIDIA CUDA가 필요하며 Plotly로 실행 결과를 시각화합니다. Graph·IR·kernel·profile은 각 실습에서 새로 생성합니다. 상세한 실행·검증 안내는 [study02/README.md](study02/README.md)를 참고하세요.
