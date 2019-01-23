import tkinter as tk
from tkinter import filedialog
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from functools import reduce
import pandas as pd
#root = tk.Tk()
#root.withdraw()


#my_filetypes = [('all files', '.*'),('Image files', '.pkl')]

#Original_test_path = filedialog.askopenfilename(title='Please Select a File', filetypes = my_filetypes)
#print(Original_test_path)
pickle_in = open('/Users/joeshen/Desktop/012319 Lab Meeting Analysis/C2_rupture.pkl',"rb")
df_list_one = pickle.load(pickle_in)

pickle_in = open('/Users/joeshen/Desktop/012319 Lab Meeting Analysis/L3233F_Rupture.pkl',"rb")
df_list_two = pickle.load(pickle_in)

pickle_in = open('/Users/joeshen/Desktop/012319 Lab Meeting Analysis/Y90F_Rupture.pkl',"rb")
df_list_three = pickle.load(pickle_in)

df_final_one = pd.concat(df_list_one)
df_final_two = pd.concat(df_list_two)
df_final_three = pd.concat(df_list_three)

#print(df)
sns.set_style('ticks')
ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_one)
ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_two)
ax= sns.lineplot(x='Time Point', y='GFP intensity', data = df_final_three)
plt.show()
#plt.savefig('C2_versus_mutant_rupture.tif', dpi=400)
