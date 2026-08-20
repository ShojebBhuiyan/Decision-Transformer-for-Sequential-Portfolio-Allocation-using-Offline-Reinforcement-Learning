# Portfolio Decision Transformer

Offline RL for sequential portfolio allocation using a Decision Transformer.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu118
python scripts/cuda_smoke_test.py
pytest tests/ -q
python main.py --stage prepare
python main.py --stage all
```

See [project_plan.md](project_plan.md) for full scope and [project_context.md](project_context.md) for the mathematical formulation.
