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

def func(x, a,b,c):
   return a*np.exp(-b*x) + c

# define a function to fit first-order decay exponential fit to the data and output decay constant
def first_order_fit (data_frame, plot = False):
   list_num = data_frame['GFP intensity']

   x_data = data_frame.iloc[np.argmax(list_num):]['Time Point'].values

   y_data = data_frame.iloc[np.argmax(list_num):]['GFP intensity'].values

   data_frame.plot(x = 'Time Point',y='GFP intensity',kind = 'scatter')
   
   b_param_test = np.linspace(1e-10,1e-8,300)
   
   b_score = 0
   
   parameter = (1,1,1)
   for b in b_param_test:
      try:
       popt, pcov = curve_fit(func, x_data,y_data, p0=[1,b,1])
      except:
         continue
         
      r = r_square_compute(x_data,y_data,popt)

      if r >= b_score:
         b_score = r
         parameter = popt

   if plot == True:
     plt.plot(x_data, func(x_data, *parameter), 'r-', label = 'fit:a= {}, b={}, c= {}'.format(*parameter))
     plt.legend(loc='best')
     plt.show()
   
   return (parameter[1],b_score)
# define a function to compute r square
def r_square_compute (xdata, ydata, param):
   res = ydata - func(xdata, *param)
   ss_res = np.sum(res**2)
   ss_tot = np.sum((ydata-np.mean(ydata))**2)
   r_square = 1 - (ss_res/ss_tot)

   return r_square


# define a function to compute the decay constants for a list of data frames
def list_exponential_fit (df_list):
   decay_list = []
   for df in df_list:
      decay_const, r = first_order_fit(df)
      if r >= 0.8:
         decay_list.append(decay_const)
   
   return decay_list



print(list_exponential_fit(df_list_one))

#first_order_fit(df_list_one[4], plot=True)
