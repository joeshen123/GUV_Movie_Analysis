import tkinter as tk
from tkinter import filedialog
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from functools import reduce
import pandas as pd
import glob,os
import warnings

#Ignore Warnings
warnings.simplefilter("ignore",UserWarning)
warnings.simplefilter("ignore",RuntimeWarning)

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

   df_len = len(df_list)
   df_final = pd.concat(df_list)
   
   return df_final, df_len

root = tk.Tk()
root.withdraw()


root.directory = filedialog.askdirectory()

Name1 = root.directory
label_1 = Name1.split("/")[-1].split(" ")[0]
df_final_one,df_final_one_len = curve_df_combine(Name1)


root.directory = filedialog.askdirectory()
Name2 = root.directory
label_2 = Name2.split("/")[-1].split(" ")[0]
df_final_two,df_final_two_len = curve_df_combine(Name2)

root.directory = filedialog.askdirectory()
Name3 = root.directory
label_3 = Name3.split("/")[-1].split(" ")[0]
df_final_three,df_final_three_len = curve_df_combine(Name3)

root.directory = filedialog.askdirectory()
Name4 = root.directory
label_4 = Name4.split("/")[-1].split(" ")[0]
df_final_four,df_final_four_len = curve_df_combine(Name4)

root.directory = filedialog.askdirectory()
Name5 = root.directory
label_5 = Name5.split("/")[-1].split(" ")[0]
df_final_five,df_final_five_len = curve_df_combine(Name5)

root.directory = filedialog.askdirectory()
Name6 = root.directory
label_6 = Name6.split("/")[-1].split(" ")[0]
df_final_six,df_final_six_len = curve_df_combine(Name6)

root.directory = filedialog.askdirectory()
Name7 = root.directory
label_7 = Name7.split("/")[-1].split(" ")[0]
df_final_seven,df_final_seven_len = curve_df_combine(Name7)



#print(df)
sns.set(style="ticks")
sns.set_context("paper", font_scale=1.5, rc={"font.size":14,"axes.labelsize":15})
ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_one,label='%s (n = %d)' %(label_1, df_final_one_len))
ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_two,label='%s (n = %d)' %(label_2, df_final_two_len))
#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_three,label='%s (n = %d)' %(label_3, df_final_three_len))
#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_four,label='%s (n = %d)' %(label_4, df_final_four_len))
#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_five,label='%s (n = %d)' %(label_5, df_final_five_len))
#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_six,label='%s (n = %d)' %(label_6, df_final_six_len))
#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_seven,label='%s (n = %d)' %(label_7, df_final_seven_len))
ax.set_xlabel('Time Points (min)', fontweight = 'bold') 
ax.set_ylabel('C2 Membrane Bindings', fontweight = 'bold')
ax.legend(fontsize='x-large', title_fontsize='40')
#ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_three)
#ax.set_title("%s and %s Binding Profile" %(label_1, label_2))

plt.show()
#plt.savefig('C2_versus_mutant_rupture.tif', dpi=400)
