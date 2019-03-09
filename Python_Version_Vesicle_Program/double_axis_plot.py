import matplotlib.pyplot as plt
import pickle
import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

root = tk.Tk()
root.withdraw()

my_filetypes = [('all files', '.*'),('Image files', '.pkl')]

Original_test_path = filedialog.askopenfilename(title='Please Select a File', filetypes = my_filetypes)

pickle_in = open(Original_test_path,"rb")
df_list_one = pickle.load(pickle_in)

df = df_list_one[0]

Time = df['Time Point'].values

radius = df['radius_micron'].values

GFP_Intensity = df['GFP intensity'].values

fig, ax1 = plt.subplots()

ax1.set_xlabel('Time Points (min)', fontsize = 16, fontweight = 'bold')
ax1.set_ylabel('C2 Membrane Bindings', fontsize = 16,fontweight = 'bold',color ='r')

plt1 = ax1.plot(Time, GFP_Intensity, 'r-')
ax1.tick_params(axis = 'y', labelcolor = 'r')

ax2 = ax1.twinx()

ax2.set_xlabel('Time Points (min)', fontsize = 16, fontweight = 'bold')
ax2.set_ylabel('Radius Changes (um)', fontsize = 16,fontweight = 'bold',color='b')
plt2 = ax2.plot(Time, radius, 'b-')
ax2.tick_params(axis = 'y', labelcolor = 'b')

plt_total = plt1 + plt2
ax1.legend(plt_total,('C2 Membrane Bindings','Radius'),loc=0, fontsize='xx-large', title_fontsize='90')

fig.tight_layout()
plt.show()


