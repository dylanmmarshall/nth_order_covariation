"""Modal app for GPU-accelerated sequence saliency analysis.

Usage:
    # Start JupyterLab on GPU (interactive notebook)
    modal run modal_app.py

    # With specific GPU
    modal run modal_app.py --gpu h200    # 141 GB - recommended for L~100
    modal run modal_app.py --gpu b200    # 192 GB - largest workloads
    modal run modal_app.py --gpu h100    # 80 GB - L<=70

    # Longer timeout for Hessian computation
    modal run modal_app.py --gpu h200 --timeout-minutes 480

    # Re-upload data
    modal run modal_app.py --upload

Available GPUs (by VRAM):
    t4 (16GB), l4 (24GB), a10g (24GB), l40s (48GB),
    a100 (40GB), a100-80gb (80GB), h100 (80GB),
    h200 (141GB), b200 (192GB)
"""
import secrets
import time
import urllib.request
import json
import modal

app = modal.App("HOR")

# GPU configurations
GPU_CONFIGS = {
    "t4": "T4",               # 16 GB - testing, ~$0.59/hr
    "l4": "L4",               # 24 GB - light workloads, ~$0.80/hr
    "a10g": "A10G",           # 24 GB - medium workloads, ~$1.10/hr
    "l40s": "L40S",           # 48 GB - Ada Lovelace, ~$1.70/hr
    "a100": "A100",           # 40 GB default, ~$2.10/hr
    "a100-40gb": "A100-40GB", # 40 GB explicit
    "a100-80gb": "A100-80GB", # 80 GB explicit, ~$2.50/hr
    "h100": "H100",           # 80 GB Hopper, ~$3.95/hr
    "h200": "H200",           # 141 GB HBM3e - large Hessians, ~$4.54/hr
    "b200": "B200",           # 192 GB Blackwell - largest workloads, ~$6.25/hr
}

# GPU image with JAX CUDA support + analysis tools
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Core ML
        "jax[cuda12]",
        "optax",
        "flax",
        # Scientific computing
        "scipy",
        "numpy",
        "pandas",
        # Visualization
        "matplotlib",
        "seaborn",
        # Jupyter
        "jupyterlab",
        "ipywidgets",
        # Utilities
        "tqdm",
        "h5py",  # For large tensor storage
    )
    .add_local_file("models.py", "/root/models.py")
    .add_local_file("autoencoders_modal.ipynb", "/root/autoencoders_modal.ipynb")
)

# Volume for persisting data and results
volume = modal.Volume.from_name("HOR-data", create_if_missing=True)
JUPYTER_PORT = 8888

@app.local_entrypoint()
def main(upload: bool = False, timeout_minutes: int = 240, gpu: str = "t4"):
    """Start JupyterLab on GPU via Modal Sandbox.

    Args:
        upload: Re-upload data to Modal volume
        timeout_minutes: Session timeout (default 4 hours for Hessian computation)
        gpu: GPU type - t4, a10g, a100, a100-80gb, h100
    """
    # Validate GPU choice
    gpu_lower = gpu.lower()
    if gpu_lower not in GPU_CONFIGS:
        print(f"Unknown GPU '{gpu}'. Available: {', '.join(GPU_CONFIGS.keys())}")
        return
    gpu_type = GPU_CONFIGS[gpu_lower]
    if upload:
        print("Uploading data to Modal volume...")
        with volume.batch_upload(force=True) as batch:
            # DeepSequence data (original)
            batch.put_file("deepseq_data.npz", "deepseq_data.npz")
            batch.put_file("deepseq_data_sub_190_250.npz", "deepseq_data_sub_190_250.npz")
            # New MSA data
            batch.put_file("data/AF-P0AA25-F1-msa_v6.npz", "AF-P0AA25-F1-msa_v6.npz")
            batch.put_file("data/AF-P0AA25-F1-model_v6_contacts.npz", "AF-P0AA25-F1-model_v6_contacts.npz")
        print("Data uploaded to /data/:")
        print("  - deepseq_data.npz (L=252)")
        print("  - deepseq_data_sub_190_250.npz (L=60)")
        print("  - AF-P0AA25-F1-msa_v6.npz (L=101)")
        print("  - AF-P0AA25-F1-model_v6_contacts.npz")
    # Create authentication token
    token = secrets.token_urlsafe(13)
    token_secret = modal.Secret.from_dict({"JUPYTER_TOKEN": token})
    print(f"Starting JupyterLab sandbox on {gpu_type} GPU...")
    print(f"Timeout: {timeout_minutes} minutes ({timeout_minutes/60:.1f} hours)")
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
            encrypted_ports=[JUPYTER_PORT],
            secrets=[token_secret],
            timeout=timeout_minutes * 60,
            image=image,
            app=app,
            gpu=gpu_type,
            volumes={"/data": volume},
        )
    print(f"Sandbox ID: {sandbox.object_id}")
    # Get the tunnel URL
    tunnel = sandbox.tunnels()[JUPYTER_PORT]
    url = f"{tunnel.url}/?token={token}"
    print(f"\n{'='*60}")
    print(f"JupyterLab running on {gpu_type}")
    print(f"{'='*60}")
    print(f"  URL: {url}")
    print(f"  Timeout: {timeout_minutes} min ({timeout_minutes/60:.1f} hr)")
    print(f"  Data: /data/deepseq_data.npz")
    print(f"  Results: /data/ (persisted)")
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
    while time.time() - start_time < 120:  # 2 min startup timeout
        if is_jupyter_up():
            print("JupyterLab is ready!")
            break
        time.sleep(2)
    else:
        print("Warning: Timed out waiting for JupyterLab health check (may still work)")
    # Keep alive until timeout or Ctrl+C
    try:
        print(f"\nJupyterLab running. Open the URL above in your browser.")
        print(f"The notebook 'autoencoders_modal.ipynb' is in /root/")
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nShutting down sandbox...")
        sandbox.terminate()
        print("Done.")
