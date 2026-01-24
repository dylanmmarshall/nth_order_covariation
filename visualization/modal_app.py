"""Modal app for tensor reduction visualization.

Usage:
    # Upload local results to Modal volume first
    modal run visualization/modal_app.py --upload

    # Start JupyterLab (CPU-only, good for fields/Jacobians)
    modal run visualization/modal_app.py

    # With GPU (for loading Hessians into memory)
    modal run visualization/modal_app.py --gpu a100    # 40 GB - single Hessian
    modal run visualization/modal_app.py --gpu h100    # 80 GB - multiple Hessians

    # Upload and start in one command
    modal run visualization/modal_app.py --upload --gpu a100

    # Longer timeout
    modal run visualization/modal_app.py --timeout-minutes 120

Available GPUs (by VRAM):
    none (CPU only), t4 (16GB), l4 (24GB), a10g (24GB), l40s (48GB),
    a100 (40GB), a100-80gb (80GB), h100 (80GB), h200 (141GB), b200 (192GB)
"""
import secrets
import time
import urllib.request
from pathlib import Path
import modal

app = modal.App("HOR-viz")

# GPU configurations
GPU_CONFIGS = {
    "none": None,             # CPU only
    "t4": "T4",               # 16 GB - testing
    "l4": "L4",               # 24 GB - light workloads
    "a10g": "A10G",           # 24 GB - medium workloads
    "l40s": "L40S",           # 48 GB - Ada Lovelace
    "a100": "A100",           # 40 GB default
    "a100-40gb": "A100-40GB", # 40 GB explicit
    "a100-80gb": "A100-80GB", # 80 GB explicit
    "h100": "H100",           # 80 GB Hopper
    "h200": "H200",           # 141 GB HBM3e
    "b200": "B200",           # 192 GB Blackwell
}

# Paths relative to this file
SCRIPT_DIR = Path(__file__).parent
NOTEBOOK_PATH = SCRIPT_DIR / "tensor_reduction_full.ipynb"

# Image with visualization dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Scientific computing
        "numpy",
        "scipy",
        # Visualization
        "matplotlib",
        "seaborn",
        "plotly",
        # Jupyter
        "jupyterlab",
        "ipywidgets",
        # Utilities
        "tqdm",
    )
    .add_local_file(str(NOTEBOOK_PATH), "/root/tensor_reduction_full.ipynb")
)

# Volume for data (shared with training app)
volume = modal.Volume.from_name("HOR-data", create_if_missing=True)
JUPYTER_PORT = 8888


@app.local_entrypoint()
def main(timeout_minutes: int = 60, gpu: str = "none", upload: bool = False):
    """Start JupyterLab for tensor visualization via Modal Sandbox.

    Args:
        timeout_minutes: Session timeout (default 1 hour)
        gpu: GPU type - none for CPU, or t4/a100/h100 etc for GPU
        upload: Upload local results to Modal volume first
    """
    if upload:
        print("Uploading results to Modal volume...")
        results_dir = Path(__file__).parent.parent / "results"
        with volume.batch_upload(force=True) as batch:
            # Fields (small)
            for f in results_dir.glob("fields_*.npy"):
                print(f"  Uploading {f.name}...")
                batch.put_file(str(f), f.name)
            # Jacobians (medium)
            for f in results_dir.glob("jac_*.npy"):
                print(f"  Uploading {f.name}...")
                batch.put_file(str(f), f.name)
            # Hessian - only mrf1 (large, ~31 GB)
            hess_file = results_dir / "hess_mrf1.npy"
            if hess_file.exists():
                print(f"  Uploading {hess_file.name} (~31 GB, this may take a while)...")
                batch.put_file(str(hess_file), hess_file.name)
        print("Upload complete!\n")

    # Validate GPU choice
    gpu_lower = gpu.lower()
    if gpu_lower not in GPU_CONFIGS:
        print(f"Unknown GPU '{gpu}'. Available: {', '.join(GPU_CONFIGS.keys())}")
        return
    gpu_type = GPU_CONFIGS[gpu_lower]

    # Create authentication token
    token = secrets.token_urlsafe(13)
    token_secret = modal.Secret.from_dict({"JUPYTER_TOKEN": token})

    gpu_str = gpu_type if gpu_type else "CPU"
    print(f"Starting JupyterLab sandbox on {gpu_str}...")
    print(f"Timeout: {timeout_minutes} minutes ({timeout_minutes/60:.1f} hours)")

    sandbox_kwargs = dict(
        encrypted_ports=[JUPYTER_PORT],
        secrets=[token_secret],
        timeout=timeout_minutes * 60,
        image=image,
        app=app,
        volumes={"/data": volume},
    )
    if gpu_type:
        sandbox_kwargs["gpu"] = gpu_type

    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            "jupyter",
            "lab",
            "--no-browser",
            "--allow-root",
            "--ip=0.0.0.0",
            f"--port={JUPYTER_PORT}",
            "--ServerApp.allow_origin=*",
            "--ServerApp.allow_remote_access=1",
            "--notebook-dir=/root",
            **sandbox_kwargs,
        )

    print(f"Sandbox ID: {sandbox.object_id}")

    # Get the tunnel URL
    tunnel = sandbox.tunnels()[JUPYTER_PORT]
    url = f"{tunnel.url}/?token={token}"

    print(f"\n{'='*60}")
    print(f"JupyterLab running on {gpu_str}")
    print(f"{'='*60}")
    print(f"  URL: {url}")
    print(f"  Timeout: {timeout_minutes} min ({timeout_minutes/60:.1f} hr)")
    print(f"  Data: /data/ (fields, Jacobians, Hessians)")
    print(f"{'='*60}")
    print("\nPress Ctrl+C to stop early.\n")

    # Wait for Jupyter to be ready
    def is_jupyter_up():
        try:
            response = urllib.request.urlopen(f"{tunnel.url}/api/status?token={token}", timeout=5)
            if response.getcode() == 200:
                return True
        except Exception:
            return False
        return False

    print("Waiting for JupyterLab to start...")
    start_time = time.time()
    while time.time() - start_time < 120:
        if is_jupyter_up():
            print("JupyterLab is ready!")
            break
        time.sleep(2)
    else:
        print("Warning: Timed out waiting for JupyterLab health check (may still work)")

    # Keep alive until timeout or Ctrl+C
    try:
        print(f"\nJupyterLab running. Open the URL above in your browser.")
        print(f"The notebook 'tensor_reduction_full.ipynb' is in /root/")
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nShutting down sandbox...")
        sandbox.terminate()
        print("Done.")
