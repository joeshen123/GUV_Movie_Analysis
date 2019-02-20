import matplotlib.pyplot as plt
import pickle
import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy.optimize import curve_fit

root = tk.Tk()
root.withdraw()


my_filetypes = [('all files', '.*'),('Image files', '.pkl')]

Original_test_path = filedialog.askopenfilename(title='Please Select a File', filetypes = my_filetypes)

pickle_in = open(Original_test_path,"rb")
df_list_one = pickle.load(pickle_in)

print(df_list_one[0])

df = df_list_one[0]
list_num = df_list_one[0]['GFP intensity']
df.plot(x = 'Time Point',y='GFP intensity',kind = 'scatter')

x_data = df_list_one[0].iloc[np.argmax(list_num):]['Time Point'].values
print(x_data)
y_data = df_list_one[0].iloc[np.argmax(list_num):]['GFP intensity'].values
print(y_data)


def func(x, a, b, c):
   return a * np.exp(-b*x) + c

popt, pcov = curve_fit(func, x_data,y_data,p0=(1,1e-9,1))

print(popt)
print(np.exp(-x_data))

plt.plot(x_data, func(x_data, *popt), 'r-',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

plt.show()

