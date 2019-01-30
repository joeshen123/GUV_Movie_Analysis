import matplotlib.pyplot as plt
import pickle
import tkinter as tk
from tkinter import filedialog
import numpy as np

root = tk.Tk()
root.withdraw()


my_filetypes = [('all files', '.*'),('Image files', '.pkl')]

Original_test_path = filedialog.askopenfilename(title='Please Select a File', filetypes = my_filetypes)

pickle_in = open(Original_test_path,"rb")
df_list_one = pickle.load(pickle_in)

print(df_list_one[0])

df_list_one[0].plot(x = 'Time Point',y='GFP intensity',kind = 'scatter')

list_num = df_list_one[0]['GFP intensity']

print(np.diff(list_num))

plt.show()