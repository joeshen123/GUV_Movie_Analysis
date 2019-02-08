import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from GUV_Analysis_Module import *
import pickle
import warnings

#Ignore warnings issued by skimage through conversion to uint8
warnings.simplefilter("ignore",UserWarning)
warnings.simplefilter("ignore",RuntimeWarning)

# Use tkinter to interactively select files to import
root = tk.Tk()
root.withdraw()

GUV_Post_Analysis_df_list = []

my_filetypes = [('all files', '.*'),('Image files', '.tif')]

Original_Image_path = filedialog.askopenfilename(title='Please Select a File', filetypes = my_filetypes)

root = tk.Tk()
root.withdraw()

Intensity_Image_path = filedialog.askopenfilename(title='Please Select a File', filetypes = my_filetypes)

image = io.imread(Original_Image_path)
intensity_image = io.imread(Intensity_Image_path)

button = True

n = 1

while button == True:
    
   # Running Image Analysis Pipelines
    Analysis_Stack = Image_Stacks(image,intensity_image)

    #If this is the first in the loop, apply filter to the image. If not reuse previously preprocessed images
    if n == 1:
      Analysis_Stack.set_parameter()
      Analysis_Stack.stack_enhance_blur()
      image = Analysis_Stack.Image_stack_median
      intensity_image = Analysis_Stack.Intensity_stack_med

    else:
      Analysis_Stack.reuse_preprocessed_stack()

    
    Analysis_Stack.set_points()
    Analysis_Stack.tracking_single_circle()

    Analysis_Stack.displaying_circle_movies()

    display_image_sequence(Analysis_Stack.Rendering_Image_Stack,'Display Original Images')
    #display_image_sequence(Analysis_Stack.Rendering_Intensity_Image,'Display Intensity Images')

   #Ask the User whether to play the time lapse plotting animation

    #answer = messagebox.askyesnocancel("Question","Do you want to play the dynamic plotting animations?")


    #if answer ==True:
      #Analysis_Stack.live_plotting_intensity()


      #second_answer = messagebox.askyesnocancel("Question","Do you want to save plot as movie?")

      #if second_answer==True:
        #movie_save_name = filedialog.asksaveasfilename(parent=root,title="Please select a movie name for saving:",filetypes=[('Movie files', '.mp4')])
        #Analysis_Stack.line_ani.save(movie_save_name,fps=10)

     #if second_answer == False:
         #Analysis_Stack.line_ani = None

    
   # Ask the user whether to save the dataframe
    save_df_answer = messagebox.askyesnocancel("Question","Do you want to save this analysis result?")

    if save_df_answer == True:
       GUV_Post_Analysis_df_list.append(Analysis_Stack.stats_df)


   # Ask users to choose another Vesicles for analysis
    button = messagebox.askyesnocancel("Question", "Do you want to start another round of analysis?")

    n+=1


list_save_name = filedialog.asksaveasfilename(parent=root,title="Please select a name for saving datasheet:",filetypes=[('Data files', '.pkl')])

if list_save_name != '':
  with open(list_save_name, 'wb') as f:
      pickle.dump(GUV_Post_Analysis_df_list, f)

Pandas_list_plotting(GUV_Post_Analysis_df_list, 'Intensity')
Pandas_list_plotting(GUV_Post_Analysis_df_list,'Radius')