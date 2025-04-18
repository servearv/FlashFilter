# FlashFilter (compact edition)

```
# install deps
git clone <repo>
cd flashfilter
python -m venv .venv && source .venv/bin/activate
pip install opencv-python numpy scipy

# run filter
python -m flashfilter.cli input.mp4 output.mp4
```

All heavy‑lifting sits in **src/flashfilter/core.py** – tweak parameters at the top or extend with new in‑painting methods later.

