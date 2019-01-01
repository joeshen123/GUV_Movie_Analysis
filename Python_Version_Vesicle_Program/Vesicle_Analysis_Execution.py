import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from GUV_Analysis_Module import *

# Use tkinter to interactively select files to import
root = tk.Tk()
root.withdraw()

my_filetypes = [('all files', '.*'),('Image files', '.tif')]

Original_Image_path = filedialog.askopenfilename(title='Please Select a File', filetypes = my_filetypes)

root = tk.Tk()
root.withdraw()

Intensity_Image_path = filedialog.askopenfilename(title='Please Select a File', filetypes = my_filetypes)

image = io.imread(Original_Image_path)
intensity_image = io.imread(Intensity_Image_path)

# Running Image Analysis Pipelines
Analysis_Stack = Image_Stacks(image,intensity_image)

Analysis_Stack.set_parameter()

Analysis_Stack.stack_enhance_blur()

Analysis_Stack.set_points()

Analysis_Stack.tracking_single_circle()

Analysis_Stack.displaying_circle_movies()

display_image_sequence(Analysis_Stack.Rendering_Image_Stack,'Display Original Images')
display_image_sequence(Analysis_Stack.Rendering_Intensity_Image,'Display Intensity Images')

Analysis_Stack.live_plotting_intensity()

answer = messagebox.askyesnocancel("Question","Do you want to save plot as movie?")

if answer==True:
    Analysis_Stack.line_ani.save('Dynamic_Plotting.mp4',fps=10)
