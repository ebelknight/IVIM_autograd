import os
import subprocess

def install_jaxlib():
    try:
        import jaxlib
    except ImportError:
        print("Installing jaxlib...")
        # Check for GPU
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if "NVIDIA-SMI" in result.stdout:
            # Install jaxlib with GPU support
            subprocess.check_call([sys.executable, "-m", "pip", "install", "jaxlib[cuda]"])
        else:
            # Install jaxlib without GPU support
            subprocess.check_call([sys.executable, "-m", "pip", "install", "jaxlib"])

if __name__ == "__main__":
    install_jaxlib()
