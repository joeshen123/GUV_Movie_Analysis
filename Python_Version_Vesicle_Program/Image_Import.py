from nd2reader.reader import ND2Reader
import numpy as np
import time
from tkinter import ttk
from tkinter import simpledialog
from tkinter import filedialog
import tkinter as tk
from skimage.external import tifffile
import matplotlib.pyplot as plt
from skimage import io
import cv2
from skimage import img_as_ubyte
import warnings
import pickle

#Ignore warnings issued by skimage through conversion to uint8
warnings.simplefilter("ignore",UserWarning)
warnings.simplefilter("ignore",RuntimeWarning)

# Use tkinter to interactively select files to import
root = tk.Tk()
root.withdraw()

my_filetypes = [('all files', '.*'),('Movie files', '.nd2')]

Image_Stack_Path = filedialog.askopenfilename(title='Please Select a Movie', filetypes = my_filetypes)


# Define a function to return the best plane from a stack of planes

def find_perfect_plane(img_stack):
   stack_len = img_stack.shape[0]
   max_num = 0
   best_n = 0
   circle_num_list = []
   for n in range(stack_len):
      img = img_as_ubyte(img_stack[n,:,:])
      img = cv2.normalize(img,None,alpha=0, beta=255,norm_type=cv2.NORM_MINMAX)
      img = cv2.equalizeHist(img)
      img = cv2.GaussianBlur(img,(11,11),0,0)
   
      circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,0.1,20,
                                 param1=200,param2=30,minRadius=0,maxRadius=80)
   
      if not (circles is None):
        circle = np.uint16(np.around(circles))
        circle_num = circle.shape[1]
        circle_num_list.append((n,circle_num))
      
      else:
         circle_num_list.append((n,0))
   

   circle_num_list = sorted(circle_num_list, key = lambda element: element[1], reverse = True)
   
   top_4 = circle_num_list[:4]

      
   slice_list = [x[0] for x in top_4]
      
    
   return slice_list

# Define a function to convert time series of ND2 images to a numpy list of Max Intensity Projection
# images.

def Z_Stack_Images_Extractor(address, fields_of_view):
   Image_Sequence = ND2Reader(address)
   
   time_series = Image_Sequence.sizes['t']
   z_stack = Image_Sequence.sizes['z']
   
   Intensity_best_Slice = []
   MI_Slice = []

   n = 0

   # create progress bar
   windows = tk.Tk()
   windows.title("Extracting Best Fitting Plane")
   s = ttk.Style(windows)

   s.layout("LabeledProgressbar",
         [('LabeledProgressbar.trough',
           {'children': [('LabeledProgressbar.pbar',
                          {'side': 'left', 'sticky': 'ns'}),
                         ("LabeledProgressbar.label",
                          {"sticky": ""})],
           'sticky': 'nswe'})])
   

   progress = ttk.Progressbar(windows, orient = 'horizontal', length = 1000, mode = 'determinate', style = "LabeledProgressbar",)
   s.configure("LabeledProgressbar", text="0 / %d     ", foreground = red, background = reds)

   progress.grid()
   progress.pack(side=tk.TOP)
   progress['maximum'] = time_series - 1

   progress['value'] = n
   
   progress.update_idletasks()

   for time in range(time_series):
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
     MI_Slice.append(MI)
     
     if n <= 5:
        best_list = find_perfect_plane(z_stack_images)

     
     best_intensity = np.array([z_stack_Intensity_images[n,:,:] for n in best_list])

     Intensity_best_Slice.append(np.max(best_intensity, axis = 0))
     
     n+=1
     progress['value'] = n

     s.configure("LabeledProgressbar", text='%d / %d   ' %(n, time_series-1))
     progress.update()
   progress.destroy()


   MI_Slice = np.array(MI_Slice)
   Intensity_best_Slice = np.array(Intensity_best_Slice)

   return (MI_Slice, Intensity_best_Slice)

FOV_num = simpledialog.askinteger("Input", "Which fields of view number you want to put ?",
                                parent=root, minvalue = 0, maxvalue = 100)

FOV_num = FOV_num - 1

MI_Images, best_Image_Intensity = Z_Stack_Images_Extractor(Image_Stack_Path,fields_of_view=FOV_num)


#Save Max Intensity Images to tiff hyperstack for furthur analysis

File_save_names = filedialog.asksaveasfilename(parent=root,title="Please select a file name for saving:",filetypes=[('Image Files', '.tif')])
File_save_names_Intensity = File_save_names.replace(".tif", "_Intensity.tif")

tifffile.imsave(File_save_names,MI_Images.astype('uint16'),bigtiff=True,metadata={'axes': 'TYX'})
tifffile.imsave(File_save_names_Intensity,best_Image_Intensity.astype('uint16'),bigtiff=True,metadata={'axes': 'TYX'})

