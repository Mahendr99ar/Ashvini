# Ashvini backend

Pipeline, API and admin tools. See the repository README for the full picture.

```bash
pip install -e ".[dev,model]"
python -m ashvini.pipeline                 # synthetic catalogue by default
uvicorn ashvini.api.main:app --reload      # http://localhost:8000/docs
pytest -q
```
