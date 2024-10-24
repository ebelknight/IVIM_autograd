from setuptools import setup, find_packages
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

install_jaxlib()

setup(
    name='my_ivim_dki_project',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'nibabel',
        'Pillow',
        'jax',
        'tkinter'
    ],
    entry_points={
        'console_scripts': [
            'run_ivim_dki = ui_stuff:main',  # Assuming you have a main() function in ui_stuff.py
        ],
    },
)
