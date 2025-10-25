\
    # Person-Follow (Windows, No ROS)
    Quick start (Windows, Python 3.10/3.11)
    1) `python -m venv .venv && .venv\Scripts\activate`
    2) `pip install --upgrade pip wheel`
       - GPU (CUDA 12.x): `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
       - CPU only: `pip install torch torchvision`
    3) `pip install -r requirements.txt`
    4) Put your YOLO weights (e.g., `yolo11n.pt`) into the project root or set path in `config.yaml`
    5) Edit `config.yaml` (camera index, LiDAR, serial, payload mode)
    6) Run detection-only demo: `start_detect_demo.bat` (or `python detect_demo.py`)
       Run full follow: `start_all.bat` (or `python main.py`)

    Serial payload:
    - `signed_pct` (default): `SPD {v_pct} TRN {w_pct}\n` with -100..+100
    - `pct_plus_dir`: `SPD {v_abs} DIR {F|B} TRN {t_abs} TDIR {L|R}\n` with 0..100 + direction flags
