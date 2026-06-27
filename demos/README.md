# Demo assets

The README GIFs are **emulated** (rendered with Pillow — no terminal recorder or
browser needed), so they regenerate deterministically anywhere:

```bash
pip install pillow
python demos/generate_demos.py
```

This writes:

- `docs/assets/demo-terminal.gif` — the `quickstart → indexer → server → frontend`
  command walkthrough.
- `docs/assets/demo-frontend.gif` — the chat UI answering a question (drawn to
  match the real `vetosh frontend` page).

Edit the `SCRIPT` / `QUESTION` / `ANSWER` constants in `generate_demos.py` to
change the demo content.
