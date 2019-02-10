import tkinter as tk
from tkinter import filedialog
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from functools import reduce
import pandas as pd
import glob,os

# Make a function to combine all df together into one df. Input is the directory of all dfs
def curve_df_combine(directory):
   os.chdir(root.directory)

   df_list = []

   df_filenames = glob.glob('*.pkl' )


   for n in range(len(df_filenames)):
      df_name = df_filenames[n]
      pickle_in = open(df_name,"rb")
   
      df_list_one = pickle.load(pickle_in)
      df_list += df_list_one

   print(len(df_list))
   df_final = pd.concat(df_list)
   
   return df_final

root = tk.Tk()
root.withdraw()


root.directory = filedialog.askdirectory()

Name1 = root.directory
label_1 = Name1.split("/")[-1].split(" ")[0]
df_final_one = curve_df_combine(Name1)

root.directory = filedialog.askdirectory()
Name2 = root.directory
label_2 = Name2.split("/")[-1].split(" ")[0]
df_final_two = curve_df_combine(Name2)




#print(df)
sns.set_style('ticks')
sns.set(font_scale=1.3)
ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_one,label=label_1)
ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_two,label=label_2)
#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_three)
ax.set_title("%s and %s Binding Profile" %(label_1, label_2))
plt.show()
#plt.savefig('C2_versus_mutant_rupture.tif', dpi=400)
