import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import nibabel as nib
from ivim_dki import IVIMDKIAnalysis  # Import the class
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import time
import pandas as pd

# Initialize the IVIMDKIAnalysis instance
analysis = IVIMDKIAnalysis()
current_angle = 0

def select_data():
    global img_data, slices, input_file_path, current_angle
    file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii")])
    if file_path:
        img = nib.load(file_path)
        img_data = img.get_fdata()
        slices = img_data.shape[2]  # Assuming 3D data
        analysis.inp = img_data  # Store the input data in the analysis object
        current_angle = 0  # Reset the rotation angle
        input_file_path = os.path.basename(file_path)
        update_preview(0)  # Show the first slice
        slice_slider.config(to=slices-1, from_=0)
        slice_slider.set(0)

def update_preview(val):
    global current_angle
    slice_idx = int(float(val))
    slice_data = img_data[:, :, slice_idx, 0]  # First 3D scan
    slice_data_np = np.array(slice_data)  # Ensure compatibility with Matplotlib
    
    # Rotate the image by the current angle
    slice_data_np = np.rot90(slice_data_np, k=current_angle // 90)

    # Normalize the data to range 0-1 for better visualization
    slice_data_np = (slice_data_np - np.min(slice_data_np)) / (np.max(slice_data_np) - np.min(slice_data_np))
    
    ax.clear()
    ax.imshow(slice_data_np, cmap='viridis')
    ax.axis('off')
    canvas.draw()
    
    # Update slice number label
    slice_number_label.config(text=f"Slice: {slice_idx + 1}/{slices}")

def flip_preview():
    global current_angle
    current_angle = (current_angle + 90) % 360
    update_preview(slice_slider.get())

def store_parameters():
    analysis.b_values = np.fromstring(b_values_entry.get(), sep=',')
    analysis.initial_guess = np.fromstring(initial_guess_entry.get(), sep=',')
    analysis.lower_bounds = np.fromstring(lower_bounds_entry.get(), sep=',')
    analysis.upper_bounds = np.fromstring(upper_bounds_entry.get(), sep=',')
    analysis.learning_rate = float(learning_rate_entry.get())
    analysis.alpha = float(alpha_entry.get())
    analysis.iterations = int(iterations_entry.get())
    analysis.tv_iterations = int(tv_iterations_entry.get())

def run_analysis():
    store_parameters()  # Store the parameters before running the analysis
    result = []
    progress_bar['value'] = 0
    start_time = time.time()

    def update_progress_bar():
        progress = 0
        while progress < 100:
            progress += 1
            progress_bar['value'] = progress
            root.update_idletasks()
            time.sleep(0.1)  # Simulate processing time
        return time.time() - start_time

    time_taken = update_progress_bar()
    
    result = analysis.run_ivim_dki_analysis()
    analysis.Para = result  # Store the result in the analysis object
    progress_bar['value'] = 100
    time_label.config(text=f"Time taken: {time_taken:.2f} seconds")
    print("Analysis Complete")

def update_output_preview(val):
    global current_angle
    slice_idx = int(float(val))
    selected_map = output_map_var.get()
    if selected_map == 'D':
        map_data = analysis.Para[:, :, :, 0]
    elif selected_map == 'D*':
        map_data = analysis.Para[:, :, :, 1]
    elif selected_map == 'f':
        map_data = analysis.Para[:, :, :, 2]
    elif selected_map == 'k':
        map_data = analysis.Para[:, :, :, 3]
    else:
        return

    slice_data = map_data[:, :, slice_idx]
    slice_data_np = np.array(slice_data)  # Convert JAX array to NumPy array
    
    # Rotate the image by the current angle
    slice_data_np = np.rot90(slice_data_np, k=current_angle // 90)

    # Normalize the data to range 0-1 for better visualization
    slice_data_np = (slice_data_np - np.min(slice_data_np)) / (np.max(slice_data_np) - np.min(slice_data_np))
    
    ax_output.clear()
    ax_output.imshow(slice_data_np, cmap='viridis')
    ax_output.axis('off')
    canvas_output.draw()
    
    # Update slice number label
    output_slice_number_label.config(text=f"Slice: {slice_idx + 1}/{slices}")

def on_input_slider_move(val):
    update_preview(val)

def on_output_slider_move(val):
    update_output_preview(val)

def display_map(map_type):
    if not hasattr(analysis, 'Para'):
        file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii")])
        if file_path:
            img = nib.load(file_path)
            map_data = img.get_fdata()
            if map_type == 'D':
                analysis.Para = np.zeros_like(map_data)
                analysis.Para[:, :, :, 0] = map_data
            elif map_type == 'D*':
                analysis.Para = np.zeros_like(map_data)
                analysis.Para[:, :, :, 1] = map_data
            elif map_type == 'f':
                analysis.Para = np.zeros_like(map_data)
                analysis.Para[:, :, :, 2] = map_data
            elif map_type == 'k':
                analysis.Para = np.zeros_like(map_data)
                analysis.Para[:, :, :, 3] = map_data
        else:
            return

    output_map_var.set(map_type)
    slices = analysis.Para.shape[2]  # Assuming 3D data
    output_slice_slider.config(to=slices-1, from_=0)
    update_output_preview(0)

def save_maps():
    if hasattr(analysis, 'Para'):
        base_name = os.path.splitext(input_file_path)[0]
        param_names = ['D', 'D_star', 'f', 'k']
        output_dir = "Output_Parameter_Maps"  # Corrected directory name

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, param in enumerate(param_names):
            map_data = analysis.Para[:, :, :, i]
            nii_img = nib.Nifti1Image(map_data, np.eye(4))
            nib.save(nii_img, os.path.join(output_dir, f"{base_name}_{param}.nii"))
        print("Maps saved successfully")
    else:
        print("No parameter maps to save")

# New functions for statistics
def select_parameter_map():
    global parameter_map, parameter_slices
    file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii")], initialdir="Output_Parameter_Maps")
    if file_path:
        img = nib.load(file_path)
        parameter_map = img.get_fdata()
        parameter_slices = parameter_map.shape[2]  # Assuming 3D data
        update_statistics_preview(0)  # Show the first slice
        statistics_slice_slider.config(to=parameter_slices-1, from_=0)
        statistics_slice_slider.set(0)

def select_roi():
    global roi_mask
    file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii")])
    if file_path:
        img = nib.load(file_path)
        roi_mask = img.get_fdata()
        update_statistics_preview(statistics_slice_slider.get())  # Update preview with ROI mask applied

def update_statistics_preview(val):
    slice_idx = int(float(val))
    if 'parameter_map' in globals() and 'roi_mask' in globals():
        slice_data = parameter_map[:, :, slice_idx] * roi_mask[:, :, slice_idx]  # Apply mask
    elif 'parameter_map' in globals():
        slice_data = parameter_map[:, :, slice_idx]
    else:
        return
    slice_data_np = np.array(slice_data)  # Ensure compatibility with Matplotlib
    
    # Normalize the data to range 0-1 for better visualization
    slice_data_np = (slice_data_np - np.min(slice_data_np)) / (np.max(slice_data_np) - np.min(slice_data_np))
    
    ax_statistics.clear()
    ax_statistics.imshow(slice_data_np, cmap='viridis')
    ax_statistics.axis('off')
    canvas_statistics.draw()
    
    # Update slice number label
    statistics_slice_number_label.config(text=f"Slice: {slice_idx + 1}/{parameter_slices}")

def calculate_and_save_statistics():
    if 'parameter_map' in globals() and 'roi_mask' in globals():
        masked_data = parameter_map[roi_mask > 0]
        mean_value = np.mean(masked_data)
        std_value = np.std(masked_data)
        statistics_data = {
            "Mean": [mean_value],
            "Standard Deviation": [std_value]
        }
        df = pd.DataFrame(statistics_data)
        output_dir = "Output_Statistics"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        base_name = os.path.splitext(input_file_path)[0]
        output_file = os.path.join(output_dir, f"{base_name}_statistics.xlsx")
        df.to_excel(output_file, index=False)
        print(f"Statistics saved to {output_file}")
    else:
        print("Parameter map or ROI mask not selected")

root = tk.Tk()
root.title("IVIM-DKI Analysis")
root.geometry("1000x700")  # Increased window size for better layout

# Create a notebook (tabs)
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill='both')

# Create frames for each tab
input_frame = ttk.Frame(notebook)
output_frame = ttk.Frame(notebook)
statistics_frame = ttk.Frame(notebook)
about_frame = ttk.Frame(notebook)

# Add frames to notebook
notebook.add(input_frame, text="Input")
notebook.add(output_frame, text="Output")
notebook.add(statistics_frame, text="Statistics")
notebook.add(about_frame, text="About")

# ------------------ Input Tab Content ------------------
input_label_frame = tk.LabelFrame(input_frame, text="Input data and Analysis")
input_label_frame.pack(fill='both', expand=True, padx=10, pady=10)

preview_frame = tk.Frame(input_label_frame)
preview_frame.pack(side=tk.LEFT, padx=10, pady=10, fill='both', expand=True)

controls_frame = tk.Frame(input_label_frame)
controls_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill='y')  # Changed to fill 'y' to prevent expansion horizontally

select_data_button = tk.Button(controls_frame, text="Select IVIM-DKI data", command=select_data)
select_data_button.pack(pady=5, fill='x')

# Set up Matplotlib figure and canvas for input preview
fig = Figure(figsize=(4, 4), dpi=100)
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=preview_frame)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

slice_slider = ttk.Scale(controls_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=on_input_slider_move)
slice_slider.pack(pady=5, fill='x')

# Label to display the current slice number for input preview
slice_number_label = tk.Label(preview_frame, text="Slice: 0/0")
slice_number_label.pack(pady=5)

# Flip button
flip_button = tk.Button(preview_frame, text="Flip 90°", command=flip_preview)
flip_button.pack(pady=5)

# Input fields for parameters
b_values_label = tk.Label(controls_frame, text="B-values (comma separated):")
b_values_label.pack(anchor='w')
b_values_entry = tk.Entry(controls_frame)
b_values_entry.pack(fill='x')
b_values_entry.insert(0, "0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000")

initial_guess_label = tk.Label(controls_frame, text="Initial guesses (comma separated):")
initial_guess_label.pack(anchor='w')
initial_guess_entry = tk.Entry(controls_frame)
initial_guess_entry.pack(fill='x')
initial_guess_entry.insert(0, "0.0008,0.00913,0.12,0.9")

lower_bounds_label = tk.Label(controls_frame, text="Lower bounds (comma separated):")
lower_bounds_label.pack(anchor='w')
lower_bounds_entry = tk.Entry(controls_frame)
lower_bounds_entry.pack(fill='x')
lower_bounds_entry.insert(0, "0.0001, 0.0001, 0.001, 0.01")

upper_bounds_label = tk.Label(controls_frame, text="Upper bounds (comma separated):")
upper_bounds_label.pack(anchor='w')
upper_bounds_entry = tk.Entry(controls_frame)
upper_bounds_entry.pack(fill='x')
upper_bounds_entry.insert(0, "0.05, 0.5, 1, 3")

learning_rate_label = tk.Label(controls_frame, text="Learning rate:")
learning_rate_label.pack(anchor='w')
learning_rate_entry = tk.Entry(controls_frame)
learning_rate_entry.pack(fill='x')
learning_rate_entry.insert(0, "0.01")

alpha_label = tk.Label(controls_frame, text="Alpha:")
alpha_label.pack(anchor='w')
alpha_entry = tk.Entry(controls_frame)
alpha_entry.pack(fill='x')
alpha_entry.insert(0, "0.001")

iterations_label = tk.Label(controls_frame, text="Iterations:")
iterations_label.pack(anchor='w')
iterations_entry = tk.Entry(controls_frame)
iterations_entry.pack(fill='x')
iterations_entry.insert(0, "200")

tv_iterations_label = tk.Label(controls_frame, text="TV Iterations:")
tv_iterations_label.pack(anchor='w')
tv_iterations_entry = tk.Entry(controls_frame)
tv_iterations_entry.pack(fill='x')
tv_iterations_entry.insert(0, "2")

run_button = tk.Button(controls_frame, text="Run", command=run_analysis)
run_button.pack(pady=10, fill='x')

# Progress bar
progress_bar = ttk.Progressbar(controls_frame, orient="horizontal", length=200, mode="determinate")
progress_bar.pack(pady=5, fill='x')

# Time taken label
time_label = tk.Label(controls_frame, text="Time taken: 0.00 seconds")
time_label.pack(pady=5)

# ------------------ Output Tab Content ------------------
output_label_frame = tk.LabelFrame(output_frame, text="IVIM-DKI with TV maps")
output_label_frame.pack(fill='both', expand=True, padx=10, pady=10)

output_note_label = tk.Label(output_label_frame, text="Note: You can either view parameter map after analysis is complete or select the desired parameter map from Output_ParameterMaps folder by clicking any buttons below.")
output_note_label.pack(pady=5)

output_controls_frame = tk.Frame(output_label_frame)
output_controls_frame.pack(pady=5)

output_map_var = tk.StringVar()

d_map_button = tk.Button(output_controls_frame, text="D map", command=lambda: display_map('D'))
d_map_button.grid(row=0, column=0, padx=5, pady=5, sticky='ew')

d_star_map_button = tk.Button(output_controls_frame, text="D* map", command=lambda: display_map('D*'))
d_star_map_button.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

f_map_button = tk.Button(output_controls_frame, text="f map", command=lambda: display_map('f'))
f_map_button.grid(row=1, column=0, padx=5, pady=5, sticky='ew')

k_map_button = tk.Button(output_controls_frame, text="k map", command=lambda: display_map('k'))
k_map_button.grid(row=1, column=1, padx=5, pady=5, sticky='ew')

# Make buttons expand equally
output_controls_frame.columnconfigure(0, weight=1)
output_controls_frame.columnconfigure(1, weight=1)

save_maps_button = tk.Button(output_label_frame, text="Save parameter maps", command=save_maps)
save_maps_button.pack(pady=10, fill='x')

output_preview_frame = tk.Frame(output_label_frame)
output_preview_frame.pack(pady=5, fill='both', expand=True)

# Set up Matplotlib figure and canvas for output preview
fig_output = Figure(figsize=(4, 4), dpi=100)
ax_output = fig_output.add_subplot(111)
canvas_output = FigureCanvasTkAgg(fig_output, master=output_preview_frame)
canvas_output.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

output_slice_slider = ttk.Scale(output_label_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=on_output_slider_move)
output_slice_slider.pack(pady=5, fill='x')

# Label to display the current slice number for output preview
output_slice_number_label = tk.Label(output_preview_frame, text="Slice: 0/0")
output_slice_number_label.pack(pady=5)

# ------------------ Statistics Tab Content ------------------
# Create a separate frame for selections
statistics_selection_frame = tk.Frame(statistics_frame)
statistics_selection_frame.pack(fill='both', expand=True, padx=10, pady=10)

# Statistics Label Frame
statistics_label_frame = tk.LabelFrame(statistics_selection_frame, text="Select Parameter Maps")
statistics_label_frame.pack(fill='x', pady=5)

select_parameter_map_button = tk.Button(statistics_label_frame, text="Select Parameter Map", command=select_parameter_map)
select_parameter_map_button.pack(pady=5, padx=5, fill='x')

# ROI Label Frame
roi_label_frame = tk.LabelFrame(statistics_selection_frame, text="Select ROI Masks")
roi_label_frame.pack(fill='x', pady=5)

select_roi_button = tk.Button(roi_label_frame, text="Select ROI", command=select_roi)
select_roi_button.pack(pady=5, padx=5, fill='x')

# Set up Matplotlib figure and canvas for statistics preview
fig_statistics = Figure(figsize=(4, 4), dpi=100)
ax_statistics = fig_statistics.add_subplot(111)
canvas_statistics = FigureCanvasTkAgg(fig_statistics, master=roi_label_frame)
canvas_statistics.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

statistics_slice_slider = ttk.Scale(roi_label_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=update_statistics_preview)
statistics_slice_slider.pack(pady=5, padx=5, fill='x')

# Label to display the current slice number for statistics preview
statistics_slice_number_label = tk.Label(roi_label_frame, text="Slice: 0/0")
statistics_slice_number_label.pack(pady=5)

# Create a separate frame for the calculate button to ensure it's always visible
statistics_button_frame = tk.Frame(statistics_frame)
statistics_button_frame.pack(fill='x', padx=10, pady=10)

calculate_stats_button = tk.Button(statistics_button_frame, text="Calculate ROI Stats & Save Statistics", command=calculate_and_save_statistics)
calculate_stats_button.pack(pady=10, fill='x')

# ------------------ About Tab Content ------------------
about_label = tk.Label(about_frame, text="IVIM-DKI Analysis Tool\nVersion 1.0\nDeveloped by Your Name", justify='center')
about_label.pack(expand=True)

root.mainloop()
