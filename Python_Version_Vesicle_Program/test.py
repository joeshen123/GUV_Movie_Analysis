import cv2
import skimage.io as io
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from skimage.filters import threshold_otsu
from skimage import morphology
from matplotlib import pyplot as plt
from scipy.misc import bytescale
from scipy.misc import bytescale
from skimage import img_as_ubyte
import warnings
from skimage.filters import sobel
from skimage import viewer
from skimage.viewer.plugins import lineprofile, Measure, CannyPlugin
from skimage import exposure
from GUV_Analysis_Module import *

#Ignore warnings issued by skimage through conversion to uint8
#warnings.simplefilter("ignore",UserWarning)
    
# Use tkinter to interactively select files to import
root = tk.Tk()
root.withdraw()

GUV_Post_Analysis_df_list = []

my_filetypes = [('all files', '.*'),('Image files', '.tif')]

Original_Image_path = filedialog.askopenfilename(title='Please Select a File', filetypes = my_filetypes)

img_stack = io.imread(Original_Image_path)

'''
def find_perfect_plane(img_stack):
   stack_len = img_stack.shape[0]
   max_num = 0
   best_n = 0

   for n in range(stack_len):
      img = img_as_ubyte(img_stack[n,:,:])
      img = cv2.normalize(img,None,alpha=0, beta=255,norm_type=cv2.NORM_MINMAX)
      img = cv2.equalizeHist(img)
      img = cv2.GaussianBlur(img,(11,11),0,0)
      
      
      circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,0.1,30,param1=200,param2=30,minRadius=0,maxRadius=60)
      
      if not (circles is None):
       circle = np.uint16(np.around(circles))
       circle_num = circle.shape[1]

       if circle_num >= max_num:
         max_num = circle_num
         best_n = n
    
   return best_n

a = find_perfect_plane (img_stack)

'''
im = img_stack[0,:,:]
im2 = img_stack[60,:,:]

_,_,img = enhance_blur_medfilter(im2, median_filter=True)

plt.imshow(img)
plt.show()


kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(100,100))
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

#_,opening,_ = enhance_blur_medfilter(opening,  median_filter=False)

print(img)
print(opening)
im_sub = cv2.subtract(img, opening)
print(np.median(img))
print(np.median(im_sub))
#np.set_printoptions(threshold=np.inf)
print(im_sub)
plt.imshow(im_sub)
plt.show()

new_viewer = viewer.ImageViewer(img) 
new_viewer += lineprofile.LineProfile() 
#new_viewer += Measure()
#new_viewer += CannyPlugin()
new_viewer.show() 


new_viewer = viewer.ImageViewer(im_sub) 
new_viewer += lineprofile.LineProfile() 
#new_viewer += Measure()
#new_viewer += CannyPlugin()
new_viewer.show() 

'''
img = img_as_ubyte(img_stack[16,:,:])
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.normalize(img,None,alpha=0, beta=255,norm_type=cv2.NORM_MINMAX)
img = cv2.equalizeHist(img)
img = cv2.GaussianBlur(img,(11,11),0,0)
plt.imshow(img)
plt.show()

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,0.1,30,
                                 param1=200,param2=30,minRadius=0,maxRadius=60)
      
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


_,_,img = enhance_blur_medfilter(img_stack[a,:,:],median_filter=False)
plt.imshow(img)
plt.show()
'''