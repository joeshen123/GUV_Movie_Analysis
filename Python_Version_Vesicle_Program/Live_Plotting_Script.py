from matplotlib import gridspec
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import pickle
from skimage import io
import numpy as np
from matplotlib.pyplot import cm
from matplotlib.animation import ArtistAnimation
from matplotlib.colors import LinearSegmentedColormap

root = tk.Tk()
root.withdraw()

my_filetypes = [('all files', '.*'),('pkl files', '.pkl')]

c2_path = filedialog.askopenfilename(title='Please Select a File', filetypes = my_filetypes)

pickle_in = open(c2_path,"rb")
c2= pickle.load(pickle_in)
c2_df= c2[0]

lox5_path = filedialog.askopenfilename(title='Please Select a File', filetypes = my_filetypes)

pickle_in = open(lox5_path,"rb")
lox5 = pickle.load(pickle_in)
lox5_df= lox5[0]

pkc_path = filedialog.askopenfilename(title='Please Select a File', filetypes = my_filetypes)

pickle_in = open(pkc_path,"rb")
pkc = pickle.load(pickle_in)
pkc_df= pkc[0]

my_filetypes = [('all files', '.*'),('image files', '.tif')]
c2_path = filedialog.askopenfilename(title='Please Select a File', filetypes = my_filetypes)

c2 = io.imread(c2_path)

lox5_path = filedialog.askopenfilename(title='Please Select a File', filetypes = my_filetypes)

lox5 = io.imread(lox5_path)

pkc_path = filedialog.askopenfilename(title='Please Select a File', filetypes = my_filetypes)

pkc = io.imread(pkc_path)

#Make a colormap like Imagej Green Channel
cdict1 = {'red':  ((0.0, 0.0, 0.0),   # <- at 0.0, the red component is 0
                   (0.5, 0.0, 0.0),   # <- at 0.5, the red component is 1
                   (1.0, 0.0, 0.0)),  # <- at 1.0, the red component is 0

         'green': ((0.0, 0.0, 0.0),   # <- etc.
                   (0.5, 0.5, 0.5),
                   (1.0, 1.0, 1.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (1.0, 0.0, 0.0))
         }

Green = LinearSegmentedColormap('Green', cdict1)


# Define a function to live plot the protein intesnity data
def live_plotting_intensity(df1,df2,df3,im1,im2,im3):
       shortest_len = min(df1.shape[0],df2.shape[0],df3.shape[0])
       c2_Intensity_data = c2_df['GFP intensity'].tolist()[0:shortest_len]
       lox5_Intensity_data = lox5_df['GFP intensity'].tolist()[0:shortest_len]
       pkc_Intensity_data = pkc_df['GFP intensity'].tolist()[0:shortest_len]
       

       Time_point = np.linspace(0,60,shortest_len)

       c2_Intensity_with_Time = []
       c2_Intensity_with_Time.append(Time_point)
       c2_Intensity_with_Time.append(c2_Intensity_data)
       c2_Intensity_with_Time = np.array(c2_Intensity_with_Time)

       lox5_Intensity_with_Time = []
       lox5_Intensity_with_Time.append(Time_point)
       lox5_Intensity_with_Time.append(lox5_Intensity_data)
       lox5_Intensity_with_Time = np.array(lox5_Intensity_with_Time)

       pkc_Intensity_with_Time = []
       pkc_Intensity_with_Time.append(Time_point)
       pkc_Intensity_with_Time.append(pkc_Intensity_data)
       pkc_Intensity_with_Time = np.array(pkc_Intensity_with_Time)

       gs = gridspec.GridSpec(2,3)
       fig = plt.figure(figsize = (10,10))
       ax1 = fig.add_subplot(gs[0:1,0:1])
       ax2 = fig.add_subplot(gs[0:1,1:2])
       ax3 = fig.add_subplot(gs[0:1,2:3])
       ax4 = fig.add_subplot(gs[1:2,:])

       ax1.set_title('Cpla2 C2',fontsize=18)
       ax2.set_title('Lox-5 Plat',fontsize=18)
       ax3.set_title('PKC Gamma',fontsize=18)

       ax4.set_xlabel('Time Points (min)', fontsize = 16, fontweight = 'bold')
       ax4.set_ylabel('Protein Membrane Bindings',fontsize = 16, fontweight = 'bold')
    
       ims = []

       for time in range(len(c2_Intensity_data)):
           c2_im = ax1.imshow(im1[time],Green)
           ax1.axis('off')

           lox5_im = ax2.imshow(im2[time],Green)
           ax2.axis('off')

           kpc_im = ax3.imshow(im3[time],Green)
           ax3.axis('off')


           l_one, = ax4.plot(c2_Intensity_with_Time[0,:time],c2_Intensity_with_Time[1,:time],'r-')
           l_two, = ax4.plot(lox5_Intensity_with_Time[0,:time],lox5_Intensity_with_Time[1,:time],'b-')
           l_three, = ax4.plot(pkc_Intensity_with_Time[0,:time],pkc_Intensity_with_Time[1,:time],'k-')
           ax4.legend((l_one,l_two,l_three),('Cpla2 C2','Lox-5 Plat','PKC Gamma'),loc=0)


           ims.append([c2_im,lox5_im,kpc_im,l_one,l_two,l_three])

       line_ani = ArtistAnimation(fig,ims, interval=50,blit=False,repeat=True)
       plt.tight_layout()

       return line_ani


live_plot = live_plotting_intensity(c2_df, lox5_df,pkc_df,c2,lox5,pkc)

movie_save_name = filedialog.asksaveasfilename(parent=root,title="Please select a movie name for saving:",filetypes=[('Movie files', '.mp4')])

live_plot.save(movie_save_name,fps=4)