IVIM-DKI Analysis using JAX!

This project implements an IVIM-DKI (Intravoxel Incoherent Motion - Diffusion Kurtosis Imaging) analysis that's been optimized by using Google's JAX library. Below are the instructions and dependencies required to set up the environment and run the analysis.


###To run this project, you will need the following libraries:

NumPy: A fundamental package for scientific computing in Python.
Nibabel: A library to read and write neuroimaging data formats.
JAX: A library for high-performance numerical computing.
Tkinter: A standard Python interface to the Tk GUI toolkit.

###Installation
Installing Dependencies
You can install the required libraries using pip:

bash

Copy code

pip install numpy nibabel jax tkinter

###Installing JAX

JAX is a key library for this project, and it must be installed correctly. Follow these steps to ensure JAX and its associated libraries are installed with matching versions.

###Install build and wheel:

bash
Copy code
pip install build wheel

###Install numpy and scipy:

bash
Copy code
pip install numpy scipy


###Building JAX from Source
To build JAX from source, follow these steps:

Clone the JAX repository from GitHub:

bash
Copy code
git clone https://github.com/google/jax.git
Navigate to the JAX directory:

bash
Copy code
cd jax
Build JAX using the build script:

bash
Copy code
python build/build.py --target_cpu_features=default

###Usage
After installing the dependencies and building JAX, you can run the IVIM-DKI analysis by executing the main script. Ensure your input data is correctly formatted and placed in the appropriate directory.

bash
Copy code
python main.py
Modify the main.py file or other configuration files as necessary to suit your specific data and requirements.

###Acknowledgements
Google JAX - The high-performance numerical computing library used in this project.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Built by Hisham Hanif and Himansu Maurya