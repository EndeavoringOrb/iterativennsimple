@if not exist .venv (
    uv venv
)
uv pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu128
uv pip install pandas numpy requests pyarrow matplotlib tqdm