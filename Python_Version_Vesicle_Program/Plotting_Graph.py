import tkinter as tk
from tkinter import filedialog
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from functools import reduce
import pandas as pd
import glob,os

root = tk.Tk()
root.withdraw()

root.directory = filedialog.askdirectory()

os.chdir(root.directory)

df_list = []



df_filenames = glob.glob('*.pkl' )


for n in range(len(df_filenames)):
   df_name = df_filenames[n]
   pickle_in = open(df_name,"rb")
   
   df_list_one = pickle.load(pickle_in)
   df_list += df_list_one

df_final = pd.concat(df_list)


#print(df)
sns.set_style('ticks')
ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final)
#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_two)
#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_three)
plt.show()
#plt.savefig('C2_versus_mutant_rupture.tif', dpi=400)
