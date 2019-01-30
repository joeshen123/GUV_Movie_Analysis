from nd2reader.reader import ND2Reader
import numpy as np
import time
from tqdm import tqdm
from tkinter import simpledialog
from tkinter import filedialog
import tkinter as tk
from skimage.external import tifffile
import matplotlib.pyplot as plt
from skimage import io


# Use tkinter to interactively select files to import
root = tk.Tk()
root.withdraw()

my_filetypes = [('all files', '.*'),('Movie files', '.nd2')]

Image_Stack_Path = filedialog.askopenfilename(title='Please Select a Movie', filetypes = my_filetypes)


# Define a function to convert time series of ND2 images to a numpy list of Max Intensity Projection
# images.

def Z_Stack_Images_Extractor(address, fields_of_view):
   Image_Sequence = ND2Reader(address)
   time_series = Image_Sequence.sizes['t']
   z_stack = Image_Sequence.sizes['z']
   
   Intensity_MI_Slice = []
   MI_Slice = []
   for time in tqdm(range(time_series)):
     z_stack_images = []
     z_stack_Intensity_images = []
     for z_slice in range(z_stack):
        slice = Image_Sequence.get_frame_2D(c=1, t=time, z=z_slice, v=fields_of_view)
        Intensity_slice = Image_Sequence.get_frame_2D(c=0, t=time, z=z_slice, v=fields_of_view)
        z_stack_images.append(slice)
        z_stack_Intensity_images.append(Intensity_slice)

     z_stack_images = np.array(z_stack_images)
     z_stack_Intensity_images = np.array(z_stack_Intensity_images)

     MI = np.max(z_stack_images, axis = 0)
     Intensity_MI = np.max(z_stack_Intensity_images, axis = 0)

     MI_Slice.append(MI)
     Intensity_MI_Slice.append(Intensity_MI)

   MI_Slice = np.array(MI_Slice)
   Intensity_MI_Slice = np.array(Intensity_MI_Slice)

   return (MI_Slice, Intensity_MI_Slice)

FOV_num = simpledialog.askinteger("Input", "Which fields of view number you want to put ?",
                                parent=root, minvalue = 0, maxvalue = 100)


MI_Images, MI_Images_Intensity = Z_Stack_Images_Extractor(Image_Stack_Path,fields_of_view=FOV_num)


#Save Max Intensity Images to tiff hyperstack for furthur analysis

File_save_names = filedialog.asksaveasfilename(parent=root,title="Please select a file name for saving:",filetypes=[('Image Files', '.tif')])
File_save_names_Intensity = File_save_names.replace(".tif", "_Intensity.tif")

tifffile.imsave(File_save_names,MI_Images.astype('uint16'),bigtiff=True,metadata={'axes': 'TYX'})
tifffile.imsave(File_save_names_Intensity,MI_Images_Intensity.astype('uint16'),bigtiff=True,metadata={'axes': 'TYX'})

