from nd2reader.reader import ND2Reader
import numpy as np
import time
from tqdm import tqdm
from tkinter import simpledialog
from tkinter import filedialog
import tkinter as tk
from skimage.external import tifffile
import matplotlib.pyplot as plt
# Use tkinter to interactively select files to import
root = tk.Tk()
root.withdraw()

my_filetypes = [('all files', '.*'),('Movie files', '.nd2')]

Image_Stack_Path = filedialog.askopenfilename(title='Please Select a Movie', filetypes = my_filetypes)


# Define a function to convert time series of ND2 images to a numpy list of Max Intensity Projection
# images.

def Z_Stack_Images_Extractor(address, fields_of_view, channel):
   Image_Sequence = ND2Reader(address)
   time_series = Image_Sequence.sizes['t']
   z_stack = Image_Sequence.sizes['z']

   MI_Slice = []
   for time in tqdm(range(time_series)):
     z_stack_images = []
     for z_slice in range(z_stack):
        slice = Image_Sequence.get_frame_2D(c=channel, t=time, z=z_slice, v=fields_of_view)
        z_stack_images.append(slice)
     z_stack_images = np.array(z_stack_images)

     MI = np.max(z_stack_images, axis = 0)

     MI_Slice.append(MI)

   MI_Slice = np.array(MI_Slice)

   return MI_Slice

FOV_num = simpledialog.askinteger("Input", "Which fields of view number you want to put ?",
                                parent=root, minvalue = 0, maxvalue = 100)


MI_Images = Z_Stack_Images_Extractor(Image_Stack_Path,fields_of_view=FOV_num, channel=1)
MI_Images_Intensity = Z_Stack_Images_Extractor(Image_Stack_Path,fields_of_view=FOV_num, channel=0)

#Save Max Intensity Images to tiff hyperstack for furthur analysis

File_save_names = filedialog.asksaveasfilename(parent=root,title="Please select a file name for saving:",filetypes=[('Image Files', '.tif')])

tifffile.imsave(File_save_names,MI_Images.astype('uint16'),bigtiff=True,metadata={'axes': 'TYX'})
tifffile.imsave('%s_Intensity' % File_save_names,MI_Images_Intensity.astype('uint16'),bigtiff=True,metadata={'axes': 'TYX'})
