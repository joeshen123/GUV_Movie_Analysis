import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from GUV_Analysis_Module import *
import pickle
import warnings
from tkinter import simpledialog

#Ignore warnings issued by skimage through conversion to uint8
warnings.simplefilter("ignore",UserWarning)
warnings.simplefilter("ignore",RuntimeWarning)

# Use tkinter to interactively select files to import
root = tk.Tk()
root.withdraw()

GUV_Post_Analysis_df_list = []

my_filetypes = [('all files', '.*'),('Image files', '.tif'),('Attribute files', '.pkl')]

filez = filedialog.askopenfilenames(parent = root, title='Please Select a File', filetypes = my_filetypes)

file_list= root.tk.splitlist(filez)

for file in file_list:
  if 'Intensity' in file:
    Intensity_Image_path = file
  else:
    Original_Image_path = file
   

image = io.imread(Original_Image_path)
intensity_image = io.imread(Intensity_Image_path)

# Running Image Analysis Pipelines
Analysis_Stack = Image_Stacks(image,intensity_image)
   
Analysis_Stack.set_parameter()    
Analysis_Stack.stack_enhance_blur()
Analysis_Stack.set_points()
Analysis_Stack.tracking_multiple_circles()

Analysis_Stack.displaying_circle_movies()

display_image_sequence(Analysis_Stack.Rendering_Image_Stack,'Display Original Images')
display_image_sequence(Analysis_Stack.Rendering_Intensity_Image,'Display Intensity Images')

# Ask the user whether to save the dataframe
save_df_answer = messagebox.askyesnocancel("Question","Do you want to save this analysis result?")
   
# if user wants to save, add circle patch to current measured GUV
if save_df_answer == True:
  GUV_Post_Analysis_df_list += Analysis_Stack.stats_df_list
  Pandas_list_plotting(GUV_Post_Analysis_df_list, 'Intensity')
  Pandas_list_plotting(GUV_Post_Analysis_df_list,'Radius')
  
del_answer = messagebox.askyesnocancel("Question","Do you want to delete some measurements?")

while del_answer == True:
  delete_answer = simpledialog.askinteger("Input", "What number do you want to delete? ",
                                 parent=root,
                                 minvalue=0, maxvalue=100)


  GUV_Post_Analysis_df_list.pop(delete_answer)
  Pandas_list_plotting(GUV_Post_Analysis_df_list, 'Intensity')
  Pandas_list_plotting(GUV_Post_Analysis_df_list,'Radius')
  del_answer = messagebox.askyesnocancel("Question","Do you want to delete more measurements?")
      

  
   

   

list_save_name = filedialog.asksaveasfilename(parent=root,title="Please select a name for saving datasheet:",filetypes=[('Data files', '.pkl')])

if list_save_name != '':
  with open(list_save_name, 'wb') as f:
      pickle.dump(GUV_Post_Analysis_df_list, f)

