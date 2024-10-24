# main.py

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import nibabel as nib
from ivim_dki import IVIMDKIAnalysis  # Import the actual class
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import time
import pandas as pd
import threading

# Initialize the IVIMDKIAnalysis instance
analysis = IVIMDKIAnalysis()
current_angle = 0

def select_data():
    global img_data, slices, input_file_path, current_angle
    file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii *.nii.gz")])
    if file_path:
        try:
            img = nib.load(file_path)
            img_data = img.get_fdata()
            if img_data.ndim != 4:
                messagebox.showerror("Data Error", "Selected NIfTI file does not have 4 dimensions (X, Y, Z, Time).")
                return
            slices = img_data.shape[2]  # Assuming 3D data
            analysis.inp = img_data  # Store the input data in the analysis object
            current_angle = 0  # Reset the rotation angle
            input_file_path = os.path.basename(file_path)
            update_preview(0)  # Show the first slice
            slice_slider.config(to=slices-1, from_=0)
            slice_slider.set(0)
            messagebox.showinfo("Data Loaded", f"Loaded data from {input_file_path}")
        except Exception as e:
            messagebox.showerror("Loading Error", f"Failed to load NIfTI file: {e}")

def update_preview(val):
    global current_angle
    try:
        slice_idx = int(float(val))
        slice_data = img_data[:, :, slice_idx, 0]  # First time point
        slice_data_np = np.array(slice_data)  # Ensure compatibility with Matplotlib
        
        # Rotate the image by the current angle
        slice_data_np = np.rot90(slice_data_np, k=current_angle // 90)
    
        # Normalize the data to range 0-1 for better visualization
        min_val = np.min(slice_data_np)
        max_val = np.max(slice_data_np)
        if max_val - min_val != 0:
            slice_data_np = (slice_data_np - min_val) / (max_val - min_val)
        else:
            slice_data_np = np.zeros_like(slice_data_np)
        
        ax.clear()
        ax.imshow(slice_data_np, cmap='viridis')
        ax.axis('off')
        canvas.draw()
        
        # Update slice number label
        slice_number_label.config(text=f"Slice: {slice_idx + 1}/{slices}")
    except Exception as e:
        messagebox.showerror("Display Error", f"Failed to update preview: {e}")

def flip_preview():
    global current_angle
    current_angle = (current_angle + 90) % 360
    update_preview(slice_slider.get())

def store_parameters():
    try:
        analysis.b_values = np.fromstring(b_values_entry.get(), sep=',')
        analysis.initial_guess = np.fromstring(initial_guess_entry.get(), sep=',')
        analysis.lower_bounds = np.fromstring(lower_bounds_entry.get(), sep=',')
        analysis.upper_bounds = np.fromstring(upper_bounds_entry.get(), sep=',')
        analysis.learning_rate = float(learning_rate_entry.get())
        analysis.alpha = float(alpha_entry.get())
        analysis.iterations = int(iterations_entry.get())
        analysis.tv_iterations = int(tv_iterations_entry.get())
        
        # Validate parameter lengths
        if not (len(analysis.b_values) == len(analysis.initial_guess) == 
                len(analysis.lower_bounds) == len(analysis.upper_bounds)):
            raise ValueError("All parameter lists must have the same number of elements.")
    except ValueError as e:
        messagebox.showerror("Parameter Error", f"Invalid parameter input: {e}")
        return False
    return True

def run_analysis_thread():
    run_analysis()

def run_analysis():
    if not store_parameters():
        return  # Exit if parameters are invalid
    
    if analysis.inp is None:
        messagebox.showwarning("Input Missing", "Please load input data before running the analysis.")
        return
    
    # Disable the run button to prevent multiple analyses
    run_button.config(state=tk.DISABLED)
    
    # Reset progress bar and time label
    progress_bar['value'] = 0
    time_label.config(text="Time taken: 0.00 seconds")
    
    def analysis_task():
        try:
            start_time = time.time()
            print("Starting IVIM-DKI analysis...")
            
            # Run the analysis
            Para = analysis.run_ivim_dki_analysis()
            
            # Convert JAX DeviceArray to NumPy array for visualization and saving
            Para_np = np.array(Para)
            analysis.Para = Para_np  # Store the NumPy array version for consistency
            
            end_time = time.time()
            time_taken = end_time - start_time
            
            print("Analysis Complete")
            messagebox.showinfo("Analysis Complete", f"Analysis completed in {time_taken:.2f} seconds.")
            
            # Update progress bar to 100%
            progress_bar['value'] = 100
            time_label.config(text=f"Time taken: {time_taken:.2f} seconds")
        except Exception as e:
            messagebox.showerror("Analysis Error", f"An error occurred during analysis: {e}")
        finally:
            # Re-enable the run button
            run_button.config(state=tk.NORMAL)
    
    # Run the analysis in a separate thread to keep the GUI responsive
    thread = threading.Thread(target=analysis_task)
    thread.start()
    
    # Optionally, implement a simple progress update (since actual progress tracking is complex)
    def simulate_progress():
        for i in range(1, 101):
            if progress_bar['value'] >= 100:
                break
            progress_bar['value'] = i
            root.update_idletasks()
            time.sleep(0.01)  # Simulate processing time
    
    # Start simulating progress
    threading.Thread(target=simulate_progress).start()

def update_output_preview(val):
    global current_angle
    try:
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
        slice_data_np = np.array(slice_data)  # Convert array (NumPy)
    
        # Rotate the image by the current angle
        slice_data_np = np.rot90(slice_data_np, k=current_angle // 90)
    
        # Normalize the data to range 0-1 for better visualization
        min_val = np.min(slice_data_np)
        max_val = np.max(slice_data_np)
        if max_val - min_val != 0:
            slice_data_np = (slice_data_np - min_val) / (max_val - min_val)
        else:
            slice_data_np = np.zeros_like(slice_data_np)
        
        ax_output.clear()
        ax_output.imshow(slice_data_np, cmap='viridis')
        ax_output.axis('off')
        canvas_output.draw()
        
        # Update slice number label
        output_slice_number_label.config(text=f"Slice: {slice_idx + 1}/{slices}")
    except Exception as e:
        messagebox.showerror("Display Error", f"Failed to update output preview: {e}")

def on_input_slider_move(val):
    update_preview(val)

def on_output_slider_move(val):
    update_output_preview(val)

def display_map(map_type):
    if not hasattr(analysis, 'Para') or analysis.Para is None:
        messagebox.showwarning("Parameter Map Missing", "Please run the analysis before viewing parameter maps.")
        return

    output_map_var.set(map_type)
    slices = analysis.Para.shape[2]  # Assuming 3D data
    output_slice_slider.config(to=slices-1, from_=0)
    output_slice_slider.set(0)
    update_output_preview(0)

def save_maps():
    if hasattr(analysis, 'Para') and analysis.Para is not None:
        try:
            base_name = os.path.splitext(input_file_path)[0]
            param_names = ['D', 'D_star', 'f', 'k']
            output_dir = "Output_Parameter_Maps"
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
    
            for i, param in enumerate(param_names):
                map_data = analysis.Para[:, :, :, i]
                nii_img = nib.Nifti1Image(map_data, affine=np.eye(4))
                nib.save(nii_img, os.path.join(output_dir, f"{base_name}_{param}.nii"))
            messagebox.showinfo("Save Successful", f"Parameter maps saved to '{output_dir}' directory.")
            print("Maps saved successfully")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save parameter maps: {e}")
    else:
        messagebox.showwarning("No Data", "No parameter maps available to save. Please run the analysis first.")

# New functions for statistics
def select_parameter_map():
    global parameter_map, parameter_slices
    if not hasattr(analysis, 'Para') or analysis.Para is None:
        messagebox.showwarning("Parameter Map Missing", "Please run the analysis before selecting a parameter map.")
        return

    map_types = ['D', 'D_star', 'f', 'k']
    map_type = map_types[0]  # Default selection

    map_selection = tk.simpledialog.askstring("Select Parameter Map", f"Enter parameter map type ({', '.join(map_types)}):")
    if map_selection is None:
        return  # User cancelled

    map_selection = map_selection.strip()
    if map_selection not in map_types:
        messagebox.showerror("Invalid Selection", f"Invalid map type selected. Choose from {', '.join(map_types)}.")
        return

    selected_map = map_selection
    parameter_map = analysis.Para[:, :, :, map_types.index(selected_map)]
    parameter_slices = parameter_map.shape[2]  # Assuming 3D data

    update_statistics_preview(0)  # Show the first slice
    statistics_slice_slider.config(to=parameter_slices-1, from_=0)
    statistics_slice_slider.set(0)
    messagebox.showinfo("Map Selected", f"Selected '{selected_map}' parameter map for statistics.")

def select_roi():
    global roi_mask
    file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii *.nii.gz")])
    if file_path:
        try:
            img = nib.load(file_path)
            roi_mask = img.get_fdata()
            if roi_mask.shape != analysis.Para.shape[:3]:
                messagebox.showerror("ROI Mask Error", "ROI mask dimensions do not match the parameter map dimensions.")
                return
            update_statistics_preview(statistics_slice_slider.get())  # Update preview with ROI mask applied
            messagebox.showinfo("ROI Loaded", f"Loaded ROI mask from {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Loading Error", f"Failed to load ROI mask: {e}")

def update_statistics_preview(val):
    slice_idx = int(float(val))
    try:
        if 'parameter_map' in globals() and 'roi_mask' in globals():
            slice_data = parameter_map[:, :, slice_idx] * roi_mask[:, :, slice_idx]  # Apply mask
        elif 'parameter_map' in globals():
            slice_data = parameter_map[:, :, slice_idx]
        else:
            return
        slice_data_np = np.array(slice_data)  # Ensure compatibility with Matplotlib

        # Normalize the data to range 0-1 for better visualization
        min_val = np.min(slice_data_np)
        max_val = np.max(slice_data_np)
        if max_val - min_val != 0:
            slice_data_np = (slice_data_np - min_val) / (max_val - min_val)
        else:
            slice_data_np = np.zeros_like(slice_data_np)
        
        ax_statistics.clear()
        ax_statistics.imshow(slice_data_np, cmap='viridis')
        ax_statistics.axis('off')
        canvas_statistics.draw()
        
        # Update slice number label
        statistics_slice_number_label.config(text=f"Slice: {slice_idx + 1}/{parameter_slices}")
    except Exception as e:
        messagebox.showerror("Statistics Error", f"Failed to update statistics preview: {e}")

def calculate_and_save_statistics():
    try:
        if 'parameter_map' in globals() and 'roi_mask' in globals():
            masked_data = parameter_map[roi_mask > 0]
            if masked_data.size == 0:
                messagebox.showwarning("No Data", "ROI mask does not cover any data.")
                return
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
            messagebox.showinfo("Statistics Saved", f"Statistics saved to '{output_file}'.")
            print(f"Statistics saved to {output_file}")
        else:
            messagebox.showwarning("Missing Data", "Please select a parameter map and ROI mask first.")
    except Exception as e:
        messagebox.showerror("Statistics Error", f"Failed to calculate/save statistics: {e}")

# Initialize the main window
root = tk.Tk()
root.title("IVIM-DKI Analysis")

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

# Input tab content
input_label_frame = tk.LabelFrame(input_frame, text="Input Data and Analysis")
input_label_frame.pack(fill='both', expand=True, padx=10, pady=10)

preview_frame = tk.Frame(input_label_frame)
preview_frame.pack(side=tk.LEFT, padx=10, pady=10)

controls_frame = tk.Frame(input_label_frame)
controls_frame.pack(side=tk.RIGHT, padx=10, pady=10)

select_data_button = tk.Button(controls_frame, text="Select IVIM-DKI Data", command=select_data)
select_data_button.pack(pady=5)

# Set up Matplotlib figure and canvas for input preview
fig = Figure(figsize=(4, 4), dpi=100)
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=preview_frame)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

slice_slider = ttk.Scale(controls_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=on_input_slider_move, length=200)
slice_slider.pack(pady=5)

# Label to display the current slice number for input preview
slice_number_label = tk.Label(preview_frame, text="Slice: 0/0")
slice_number_label.pack(pady=5)

# Flip button
flip_button = tk.Button(preview_frame, text="Flip 90°", command=flip_preview)
flip_button.pack(pady=5)

# Input fields for parameters
b_values_label = tk.Label(controls_frame, text="B-values (comma separated):")
b_values_label.pack(anchor='w')
b_values_entry = tk.Entry(controls_frame, width=40)
b_values_entry.pack(pady=2)
b_values_entry.insert(0, "0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000")

initial_guess_label = tk.Label(controls_frame, text="Initial guesses (comma separated):")
initial_guess_label.pack(anchor='w')
initial_guess_entry = tk.Entry(controls_frame, width=40)
initial_guess_entry.pack(pady=2)
initial_guess_entry.insert(0, "0.0013,0.013,0.23,1.1")

lower_bounds_label = tk.Label(controls_frame, text="Lower bounds (comma separated):")
lower_bounds_label.pack(anchor='w')
lower_bounds_entry = tk.Entry(controls_frame, width=40)
lower_bounds_entry.pack(pady=2)
lower_bounds_entry.insert(0, "0.0001, 0.001, 0.0, 0.5")

upper_bounds_label = tk.Label(controls_frame, text="Upper bounds (comma separated):")
upper_bounds_label.pack(anchor='w')
upper_bounds_entry = tk.Entry(controls_frame, width=40)
upper_bounds_entry.pack(pady=2)
upper_bounds_entry.insert(0, "0.005, 0.05, 1.0, 2.0")

learning_rate_label = tk.Label(controls_frame, text="Learning rate:")
learning_rate_label.pack(anchor='w')
learning_rate_entry = tk.Entry(controls_frame, width=40)
learning_rate_entry.pack(pady=2)
learning_rate_entry.insert(0, "0.01")

alpha_label = tk.Label(controls_frame, text="Alpha:")
alpha_label.pack(anchor='w')
alpha_entry = tk.Entry(controls_frame, width=40)
alpha_entry.pack(pady=2)
alpha_entry.insert(0, "0.1")

iterations_label = tk.Label(controls_frame, text="Iterations:")
iterations_label.pack(anchor='w')
iterations_entry = tk.Entry(controls_frame, width=40)
iterations_entry.pack(pady=2)
iterations_entry.insert(0, "100")

tv_iterations_label = tk.Label(controls_frame, text="TV Iterations:")
tv_iterations_label.pack(anchor='w')
tv_iterations_entry = tk.Entry(controls_frame, width=40)
tv_iterations_entry.pack(pady=2)
tv_iterations_entry.insert(0, "10")

run_button = tk.Button(controls_frame, text="Run Analysis", command=run_analysis_thread)
run_button.pack(pady=10)

# Progress bar
progress_bar = ttk.Progressbar(controls_frame, orient="horizontal", length=200, mode="determinate")
progress_bar.pack(pady=5)

# Time taken label
time_label = tk.Label(controls_frame, text="Time taken: 0.00 seconds")
time_label.pack(pady=5)

# Output tab content
output_label_frame = tk.LabelFrame(output_frame, text="IVIM-DKI Parameter Maps")
output_label_frame.pack(fill='both', expand=True, padx=10, pady=10)

output_note_label = tk.Label(output_label_frame, text="View and save parameter maps after analysis.")
output_note_label.pack(pady=5)

output_controls_frame = tk.Frame(output_label_frame)
output_controls_frame.pack(pady=5)

output_map_var = tk.StringVar()

# Buttons to select parameter maps
d_map_button = tk.Button(output_controls_frame, text="D map", command=lambda: display_map('D'))
d_map_button.grid(row=0, column=0, padx=5, pady=5)

d_star_map_button = tk.Button(output_controls_frame, text="D* map", command=lambda: display_map('D_star'))
d_star_map_button.grid(row=0, column=1, padx=5, pady=5)

f_map_button = tk.Button(output_controls_frame, text="f map", command=lambda: display_map('f'))
f_map_button.grid(row=1, column=0, padx=5, pady=5)

k_map_button = tk.Button(output_controls_frame, text="k map", command=lambda: display_map('k'))
k_map_button.grid(row=1, column=1, padx=5, pady=5)

save_maps_button = tk.Button(output_label_frame, text="Save Parameter Maps", command=save_maps)
save_maps_button.pack(pady=10)

output_preview_frame = tk.Frame(output_label_frame)
output_preview_frame.pack(pady=5)

# Set up Matplotlib figure and canvas for output preview
fig_output = Figure(figsize=(4, 4), dpi=100)
ax_output = fig_output.add_subplot(111)
canvas_output = FigureCanvasTkAgg(fig_output, master=output_preview_frame)
canvas_output.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

output_slice_slider = ttk.Scale(output_label_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=on_output_slider_move, length=200)
output_slice_slider.pack(pady=5)

# Label to display the current slice number for output preview
output_slice_number_label = tk.Label(output_preview_frame, text="Slice: 0/0")
output_slice_number_label.pack(pady=5)

# Statistics tab content
statistics_label_frame = tk.LabelFrame(statistics_frame, text="Statistics Analysis")
statistics_label_frame.pack(fill='both', expand=True, padx=10, pady=10)

select_parameter_map_button = tk.Button(statistics_label_frame, text="Select Parameter Map", command=select_parameter_map)
select_parameter_map_button.pack(pady=5)

roi_label_frame = tk.LabelFrame(statistics_frame, text="ROI Selection")
roi_label_frame.pack(fill='both', expand=True, padx=10, pady=10)

select_roi_button = tk.Button(roi_label_frame, text="Select ROI Mask", command=select_roi)
select_roi_button.pack(pady=5)

# Set up Matplotlib figure and canvas for statistics preview
fig_statistics = Figure(figsize=(4, 4), dpi=100)
ax_statistics = fig_statistics.add_subplot(111)
canvas_statistics = FigureCanvasTkAgg(fig_statistics, master=roi_label_frame)
canvas_statistics.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

statistics_slice_slider = ttk.Scale(roi_label_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=update_statistics_preview, length=200)
statistics_slice_slider.pack(pady=5)

# Label to display the current slice number for statistics preview
statistics_slice_number_label = tk.Label(roi_label_frame, text="Slice: 0/0")
statistics_slice_number_label.pack(pady=5)

calculate_stats_button = tk.Button(statistics_frame, text="Calculate ROI Stats & Save Statistics", command=calculate_and_save_statistics)
calculate_stats_button.pack(pady=10)

# About tab content
about_text = """
IVIM-DKI Analysis Tool

Developed using Python, Tkinter, JAX, and other libraries.

This tool allows users to perform Intravoxel Incoherent Motion (IVIM) and Diffusion Kurtosis Imaging (DKI) analysis on MRI data.

Developed by: Hisham Hanif and Himansu Maurya, IIT Delhi, Batch of 2024
Date: 1st May 2024
"""

about_label = tk.Label(about_frame, text=about_text, justify=tk.LEFT, padx=10, pady=10)
about_label.pack()

root.mainloop()


###
##PLACEHOLDER VERSION, IN CASE JAX ISN'T WORKING, FOR TESTING UI ONLY
# main.py

# import tkinter as tk
# from tkinter import filedialog, ttk
# import numpy as np
# import nibabel as nib
# from ivim_dki import IVIMDKIAnalysis  # Import the placeholder class
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import os
# import time
# import pandas as pd

# # Initialize the IVIMDKIAnalysis instance
# analysis = IVIMDKIAnalysis()
# current_angle = 0

# def select_data():
#     global img_data, slices, input_file_path, current_angle
#     file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii")])
#     if file_path:
#         img = nib.load(file_path)
#         img_data = img.get_fdata()
#         slices = img_data.shape[2]  # Assuming 3D data
#         analysis.inp = img_data  # Store the input data in the analysis object
#         current_angle = 0  # Reset the rotation angle
#         input_file_path = os.path.basename(file_path)
#         update_preview(0)  # Show the first slice
#         slice_slider.config(to=slices-1, from_=0)
#         slice_slider.set(0)

# def update_preview(val):
#     global current_angle
#     slice_idx = int(float(val))
#     slice_data = img_data[:, :, slice_idx, 0]  # First 3D scan
#     slice_data_np = np.array(slice_data)  # Ensure compatibility with Matplotlib
    
#     # Rotate the image by the current angle
#     slice_data_np = np.rot90(slice_data_np, k=current_angle // 90)

#     # Normalize the data to range 0-1 for better visualization
#     min_val = np.min(slice_data_np)
#     max_val = np.max(slice_data_np)
#     if max_val - min_val != 0:
#         slice_data_np = (slice_data_np - min_val) / (max_val - min_val)
#     else:
#         slice_data_np = np.zeros_like(slice_data_np)
    
#     ax.clear()
#     ax.imshow(slice_data_np, cmap='viridis')
#     ax.axis('off')
#     canvas.draw()
    
#     # Update slice number label
#     slice_number_label.config(text=f"Slice: {slice_idx + 1}/{slices}")

# def flip_preview():
#     global current_angle
#     current_angle = (current_angle + 90) % 360
#     update_preview(slice_slider.get())

# def store_parameters():
#     try:
#         analysis.b_values = np.fromstring(b_values_entry.get(), sep=',')
#         analysis.initial_guess = np.fromstring(initial_guess_entry.get(), sep=',')
#         analysis.lower_bounds = np.fromstring(lower_bounds_entry.get(), sep=',')
#         analysis.upper_bounds = np.fromstring(upper_bounds_entry.get(), sep=',')
#         analysis.learning_rate = float(learning_rate_entry.get())
#         analysis.alpha = float(alpha_entry.get())
#         analysis.iterations = int(iterations_entry.get())
#         analysis.tv_iterations = int(tv_iterations_entry.get())
#     except ValueError as e:
#         print(f"Error parsing parameters: {e}")
#         tk.messagebox.showerror("Parameter Error", f"Invalid parameter input: {e}")

# def run_analysis():
#     store_parameters()  # Store the parameters before running the analysis
#     result = []
#     progress_bar['value'] = 0
#     start_time = time.time()

#     def update_progress_bar():
#         progress = 0
#         while progress < 100:
#             progress += 1
#             progress_bar['value'] = progress
#             root.update_idletasks()
#             time.sleep(0.01)  # Simulate processing time
#         return time.time() - start_time

#     time_taken = update_progress_bar()
    
#     result = analysis.run_ivim_dki_analysis()
#     analysis.Para = result  # Store the result in the analysis object
#     progress_bar['value'] = 100
#     time_label.config(text=f"Time taken: {time_taken:.2f} seconds")
#     print("Analysis Complete")

# def update_output_preview(val):
#     global current_angle
#     slice_idx = int(float(val))
#     selected_map = output_map_var.get()
#     if selected_map == 'D':
#         map_data = analysis.Para[:, :, :, 0]
#     elif selected_map == 'D*':
#         map_data = analysis.Para[:, :, :, 1]
#     elif selected_map == 'f':
#         map_data = analysis.Para[:, :, :, 2]
#     elif selected_map == 'k':
#         map_data = analysis.Para[:, :, :, 3]
#     else:
#         return

#     slice_data = map_data[:, :, slice_idx]
#     slice_data_np = np.array(slice_data)  # Convert array (NumPy)

#     # Rotate the image by the current angle
#     slice_data_np = np.rot90(slice_data_np, k=current_angle // 90)

#     # Normalize the data to range 0-1 for better visualization
#     min_val = np.min(slice_data_np)
#     max_val = np.max(slice_data_np)
#     if max_val - min_val != 0:
#         slice_data_np = (slice_data_np - min_val) / (max_val - min_val)
#     else:
#         slice_data_np = np.zeros_like(slice_data_np)
    
#     ax_output.clear()
#     ax_output.imshow(slice_data_np, cmap='viridis')
#     ax_output.axis('off')
#     canvas_output.draw()
    
#     # Update slice number label
#     output_slice_number_label.config(text=f"Slice: {slice_idx + 1}/{slices}")

# def on_input_slider_move(val):
#     update_preview(val)

# def on_output_slider_move(val):
#     update_output_preview(val)

# def display_map(map_type):
#     if not hasattr(analysis, 'Para'):
#         file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii")])
#         if file_path:
#             img = nib.load(file_path)
#             map_data = img.get_fdata()
#             if map_type == 'D':
#                 analysis.Para = np.zeros_like(map_data)
#                 analysis.Para[:, :, :, 0] = map_data
#             elif map_type == 'D*':
#                 analysis.Para = np.zeros_like(map_data)
#                 analysis.Para[:, :, :, 1] = map_data
#             elif map_type == 'f':
#                 analysis.Para = np.zeros_like(map_data)
#                 analysis.Para[:, :, :, 2] = map_data
#             elif map_type == 'k':
#                 analysis.Para = np.zeros_like(map_data)
#                 analysis.Para[:, :, :, 3] = map_data
#         else:
#             return

#     output_map_var.set(map_type)
#     slices = analysis.Para.shape[2]  # Assuming 3D data
#     output_slice_slider.config(to=slices-1, from_=0)
#     update_output_preview(0)

# def save_maps():
#     if hasattr(analysis, 'Para'):
#         base_name = os.path.splitext(input_file_path)[0]
#         param_names = ['D', 'D_star', 'f', 'k']
#         output_dir = "Output_Parameter_Maps"
        
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)

#         for i, param in enumerate(param_names):
#             map_data = analysis.Para[:, :, :, i]
#             nii_img = nib.Nifti1Image(map_data, np.eye(4))
#             nib.save(nii_img, os.path.join(output_dir, f"{base_name}_{param}.nii"))
#         print("Maps saved successfully")
#     else:
#         print("No parameter maps to save")

# # New functions for statistics
# def select_parameter_map():
#     global parameter_map, parameter_slices
#     file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii")], initialdir="Output_Parameter_Maps")
#     if file_path:
#         img = nib.load(file_path)
#         parameter_map = img.get_fdata()
#         parameter_slices = parameter_map.shape[2]  # Assuming 3D data
#         update_statistics_preview(0)  # Show the first slice
#         statistics_slice_slider.config(to=parameter_slices-1, from_=0)
#         statistics_slice_slider.set(0)

# def select_roi():
#     global roi_mask
#     file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii")])
#     if file_path:
#         img = nib.load(file_path)
#         roi_mask = img.get_fdata()
#         update_statistics_preview(statistics_slice_slider.get())  # Update preview with ROI mask applied

# def update_statistics_preview(val):
#     slice_idx = int(float(val))
#     if 'parameter_map' in globals() and 'roi_mask' in globals():
#         slice_data = parameter_map[:, :, slice_idx] * roi_mask[:, :, slice_idx]  # Apply mask
#     elif 'parameter_map' in globals():
#         slice_data = parameter_map[:, :, slice_idx]
#     else:
#         return
#     slice_data_np = np.array(slice_data)  # Ensure compatibility with Matplotlib

#     # Normalize the data to range 0-1 for better visualization
#     min_val = np.min(slice_data_np)
#     max_val = np.max(slice_data_np)
#     if max_val - min_val != 0:
#         slice_data_np = (slice_data_np - min_val) / (max_val - min_val)
#     else:
#         slice_data_np = np.zeros_like(slice_data_np)
    
#     ax_statistics.clear()
#     ax_statistics.imshow(slice_data_np, cmap='viridis')
#     ax_statistics.axis('off')
#     canvas_statistics.draw()
    
#     # Update slice number label
#     statistics_slice_number_label.config(text=f"Slice: {slice_idx + 1}/{parameter_slices}")

# def calculate_and_save_statistics():
#     if 'parameter_map' in globals() and 'roi_mask' in globals():
#         masked_data = parameter_map[roi_mask > 0]
#         if masked_data.size == 0:
#             print("No data within ROI mask.")
#             tk.messagebox.showwarning("No Data", "ROI mask does not cover any data.")
#             return
#         mean_value = np.mean(masked_data)
#         std_value = np.std(masked_data)
#         statistics_data = {
#             "Mean": [mean_value],
#             "Standard Deviation": [std_value]
#         }
#         df = pd.DataFrame(statistics_data)
#         output_dir = "Output_Statistics"
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         base_name = os.path.splitext(input_file_path)[0]
#         output_file = os.path.join(output_dir, f"{base_name}_statistics.xlsx")
#         df.to_excel(output_file, index=False)
#         print(f"Statistics saved to {output_file}")
#         tk.messagebox.showinfo("Success", f"Statistics saved to {output_file}")
#     else:
#         print("Parameter map or ROI mask not selected")
#         tk.messagebox.showwarning("Missing Data", "Please select a parameter map and ROI mask first.")

# root = tk.Tk()
# root.title("IVIM-DKI Analysis")

# # Create a notebook (tabs)
# notebook = ttk.Notebook(root)
# notebook.pack(expand=True, fill='both')

# # Create frames for each tab
# input_frame = ttk.Frame(notebook)
# output_frame = ttk.Frame(notebook)
# statistics_frame = ttk.Frame(notebook)
# about_frame = ttk.Frame(notebook)

# # Add frames to notebook
# notebook.add(input_frame, text="Input")
# notebook.add(output_frame, text="Output")
# notebook.add(statistics_frame, text="Statistics")
# notebook.add(about_frame, text="About")

# # Input tab content
# input_label_frame = tk.LabelFrame(input_frame, text="Input data and Analysis")
# input_label_frame.pack(fill='both', expand=True, padx=10, pady=10)

# preview_frame = tk.Frame(input_label_frame)
# preview_frame.pack(side=tk.LEFT, padx=10, pady=10)

# controls_frame = tk.Frame(input_label_frame)
# controls_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# select_data_button = tk.Button(controls_frame, text="Select IVIM-DKI data", command=select_data)
# select_data_button.pack(pady=5)

# # Set up Matplotlib figure and canvas for input preview
# fig = Figure(figsize=(4, 4), dpi=100)
# ax = fig.add_subplot(111)
# canvas = FigureCanvasTkAgg(fig, master=preview_frame)
# canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# slice_slider = ttk.Scale(controls_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=on_input_slider_move)
# slice_slider.pack(pady=5)

# # Label to display the current slice number for input preview
# slice_number_label = tk.Label(preview_frame, text="Slice: 0/0")
# slice_number_label.pack(pady=5)

# # Flip button
# flip_button = tk.Button(preview_frame, text="Flip 90°", command=flip_preview)
# flip_button.pack(pady=5)

# # Input fields for parameters
# b_values_label = tk.Label(controls_frame, text="B-values (comma separated):")
# b_values_label.pack()
# b_values_entry = tk.Entry(controls_frame, width=30)
# b_values_entry.pack()
# b_values_entry.insert(0, "0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000")

# initial_guess_label = tk.Label(controls_frame, text="Initial guesses (comma separated):")
# initial_guess_label.pack()
# initial_guess_entry = tk.Entry(controls_frame, width=30)
# initial_guess_entry.pack()
# initial_guess_entry.insert(0, "0.0013,0.013,0.23,1.1")

# lower_bounds_label = tk.Label(controls_frame, text="Lower bounds (comma separated):")
# lower_bounds_label.pack()
# lower_bounds_entry = tk.Entry(controls_frame, width=30)
# lower_bounds_entry.pack()
# lower_bounds_entry.insert(0, "0.0001, 0.001, 0.001, 0.5")

# upper_bounds_label = tk.Label(controls_frame, text="Upper bounds (comma separated):")
# upper_bounds_label.pack()
# upper_bounds_entry = tk.Entry(controls_frame, width=30)
# upper_bounds_entry.pack()
# upper_bounds_entry.insert(0, "0.05, 0.05, 1.0, 2.0")

# learning_rate_label = tk.Label(controls_frame, text="Learning rate:")
# learning_rate_label.pack()
# learning_rate_entry = tk.Entry(controls_frame, width=30)
# learning_rate_entry.pack()
# learning_rate_entry.insert(0, "0.01")

# alpha_label = tk.Label(controls_frame, text="Alpha:")
# alpha_label.pack()
# alpha_entry = tk.Entry(controls_frame, width=30)
# alpha_entry.pack()
# alpha_entry.insert(0, "0.1")

# iterations_label = tk.Label(controls_frame, text="Iterations:")
# iterations_label.pack()
# iterations_entry = tk.Entry(controls_frame, width=30)
# iterations_entry.pack()
# iterations_entry.insert(0, "100")

# tv_iterations_label = tk.Label(controls_frame, text="TV Iterations:")
# tv_iterations_label.pack()
# tv_iterations_entry = tk.Entry(controls_frame, width=30)
# tv_iterations_entry.pack()
# tv_iterations_entry.insert(0, "10")

# run_button = tk.Button(controls_frame, text="Run Analysis", command=run_analysis)
# run_button.pack(pady=10)

# # Progress bar
# progress_bar = ttk.Progressbar(controls_frame, orient="horizontal", length=200, mode="determinate")
# progress_bar.pack(pady=5)

# # Time taken label
# time_label = tk.Label(controls_frame, text="Time taken: 0.00 seconds")
# time_label.pack(pady=5)

# # Output tab content
# output_label_frame = tk.LabelFrame(output_frame, text="IVIM-DKI with TV maps")
# output_label_frame.pack(fill='both', expand=True, padx=10, pady=10)

# output_note_label = tk.Label(output_label_frame, text="Note: You can either view parameter map after analysis is complete or select the desired parameter map from Output_ParameterMaps folder by clicking any buttons below.")
# output_note_label.pack(pady=5)

# output_controls_frame = tk.Frame(output_label_frame)
# output_controls_frame.pack(pady=5)

# output_map_var = tk.StringVar()

# d_map_button = tk.Button(output_controls_frame, text="D map", command=lambda: display_map('D'))
# d_map_button.grid(row=0, column=0, padx=5, pady=5)

# d_star_map_button = tk.Button(output_controls_frame, text="D* map", command=lambda: display_map('D*'))
# d_star_map_button.grid(row=0, column=1, padx=5, pady=5)

# f_map_button = tk.Button(output_controls_frame, text="f map", command=lambda: display_map('f'))
# f_map_button.grid(row=1, column=0, padx=5, pady=5)

# k_map_button = tk.Button(output_controls_frame, text="k map", command=lambda: display_map('k'))
# k_map_button.grid(row=1, column=1, padx=5, pady=5)

# save_maps_button = tk.Button(output_label_frame, text="Save parameter maps", command=save_maps)
# save_maps_button.pack(pady=10)

# output_preview_frame = tk.Frame(output_label_frame)
# output_preview_frame.pack(pady=5)

# # Set up Matplotlib figure and canvas for output preview
# fig_output = Figure(figsize=(4, 4), dpi=100)
# ax_output = fig_output.add_subplot(111)
# canvas_output = FigureCanvasTkAgg(fig_output, master=output_preview_frame)
# canvas_output.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# output_slice_slider = ttk.Scale(output_label_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=on_output_slider_move)
# output_slice_slider.pack(pady=5)

# # Label to display the current slice number for output preview
# output_slice_number_label = tk.Label(output_preview_frame, text="Slice: 0/0")
# output_slice_number_label.pack(pady=5)

# # Statistics tab content
# statistics_label_frame = tk.LabelFrame(statistics_frame, text="Select parameter maps")
# statistics_label_frame.pack(fill='both', expand=True, padx=10, pady=10)

# select_parameter_map_button = tk.Button(statistics_label_frame, text="Select parameter map", command=select_parameter_map)
# select_parameter_map_button.pack(pady=5)

# roi_label_frame = tk.LabelFrame(statistics_frame, text="Select ROI masks")
# roi_label_frame.pack(fill='both', expand=True, padx=10, pady=10)

# select_roi_button = tk.Button(roi_label_frame, text="Select ROI", command=select_roi)
# select_roi_button.pack(pady=5)

# # Set up Matplotlib figure and canvas for statistics preview
# fig_statistics = Figure(figsize=(4, 4), dpi=100)
# ax_statistics = fig_statistics.add_subplot(111)
# canvas_statistics = FigureCanvasTkAgg(fig_statistics, master=roi_label_frame)
# canvas_statistics.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# statistics_slice_slider = ttk.Scale(roi_label_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=update_statistics_preview)
# statistics_slice_slider.pack(pady=5)

# # Label to display the current slice number for statistics preview
# statistics_slice_number_label = tk.Label(roi_label_frame, text="Slice: 0/0")
# statistics_slice_number_label.pack(pady=5)

# calculate_stats_button = tk.Button(statistics_frame, text="Calculate ROI stats & Save statistics", command=calculate_and_save_statistics)
# calculate_stats_button.pack(pady=10)

# root.mainloop()
