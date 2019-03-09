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
from tkinter import simpledialog
from matplotlib import cm
from GUV_Analysis_Module import *
import cell_segmentation as cellseg

#from GUV_Analysis_Module import *

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
   circle_num_list = []
   for n in range(stack_len):
      img = img_as_ubyte(img_stack[n,:,:])
      img = cv2.normalize(img,None,alpha=0, beta=255,norm_type=cv2.NORM_MINMAX)
      img = cv2.equalizeHist(img)
      img = cv2.GaussianBlur(img,(11,11),0,0)
   
      circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,0.1,20,
                                 param1=200,param2=30,minRadius=0,maxRadius=80)
   
      if not (circles is None):
        circle = np.uint16(np.around(circles))
        circle_num = circle.shape[1]
        circle_num_list.append((n,circle_num))
      
      else:
         circle_num_list.append((n,0))
   

   circle_num_list = sorted(circle_num_list, key = lambda element: element[1], reverse = True)
   
   top_4 = circle_num_list[:4]

      
   slice_list = [x[0] for x in top_4]
      
    
   return slice_list

a = find_perfect_plane (img_stack)


#im = img_stack[0,:,:]
#im2 = img_stack[60,:,:]

#_,_,img = enhance_blur_medfilter(im, median_filter=True)

plt.imshow(img)
plt.show()


kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(100,100))
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

im_sub = cv2.subtract(img,opening)
#_,opening,_ = enhance_blur_medfilter(opening,  median_filter=False)

#print(img)
#print(opening)
#im_sub = cv2.subtract(img, opening)
#print(np.median(img))
#print(np.median(im_sub))
#np.set_printoptions(threshold=np.inf)
#print(im_sub)
plt.imshow(opening)
plt.show()

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
cl1, gaussian_blur_cl1, segmented_zlevel, centers = cellseg.enhance_blur_segment(img_stack[0,:,:],enhance = True, blur = True, kernel = 21, n_intensities = 2)

labeled = cellseg.watershedsegment(segmented_zlevel,smooth_distance = True,kernel = 11)

zlevel_image_color_regions  = cellseg.draw_contours(labeled,img_stack[0,:,:], with_labels = True, color = (255,0,0),width = 3 )


plt.imshow(cl1)
plt.show()

plt.imshow(segmented_zlevel)
plt.show()

plt.imshow(labeled)
plt.show()
'''
img = img_as_ubyte(img_stack[23,:,:])
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.normalize(img,None,alpha=0, beta=255,norm_type=cv2.NORM_MINMAX)
img = cv2.equalizeHist(img)
img = cv2.GaussianBlur(img,(11,11),0,0)
plt.imshow(img)
plt.show()

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,0.1,20,
                                 param1=200,param2=30,minRadius=0,maxRadius=80)
      
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


'''
_,_,img = enhance_blur_medfilter(img_stack[a,:,:],median_filter=False)
plt.imshow(img)
plt.show()



root = tk.Tk()
root.withdraw()
delete_answer = simpledialog.askstring("Input", "Which number you want to delete ?",
                                parent=root)

int_list = [int(x) for x in delete_answer.split()]
  
print(int_list)


#display_image_sequence(img_stack)
cmap_reversed = cm.get_cmap('Greens_r')
plt.imshow(img_stack[20,:,:], cmap=cmap_reversed)
plt.show()

# Uncomment the next two lines if you want to save the animation
#import matplotlib
#matplotlib.use("Agg")

import numpy
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.animation as animation



# Sent for figure
font = {'size'   : 9}
matplotlib.rc('font', **font)

# Setup figure and subplots
f0 = figure(num = 0, figsize = (12, 8))#, dpi = 100)
f0.suptitle("Oscillation decay", fontsize=12)
ax01 = subplot2grid((2, 2), (0, 0))
ax02 = subplot2grid((2, 2), (0, 1))
ax03 = subplot2grid((2, 2), (1, 0), colspan=2, rowspan=1)
ax04 = ax03.twinx()
#tight_layout()

# Set titles of subplots
ax01.set_title('Position vs Time')
ax02.set_title('Velocity vs Time')
ax03.set_title('Position and Velocity vs Time')

# set y-limits
ax01.set_ylim(0,2)
ax02.set_ylim(-6,6)
ax03.set_ylim(-0,5)
ax04.set_ylim(-10,10)

# sex x-limits
ax01.set_xlim(0,5.0)
ax02.set_xlim(0,5.0)
ax03.set_xlim(0,5.0)
ax04.set_xlim(0,5.0)

# Turn on grids
ax01.grid(True)
ax02.grid(True)
ax03.grid(True)

# set label names
ax01.set_xlabel("x")
ax01.set_ylabel("py")
ax02.set_xlabel("t")
ax02.set_ylabel("vy")
ax03.set_xlabel("t")
ax03.set_ylabel("py")
ax04.set_ylabel("vy")

# Data Placeholders
yp1=zeros(0)
yv1=zeros(0)
yp2=zeros(0)
yv2=zeros(0)
t=zeros(0)

# set plots
p011, = ax01.plot(t,yp1,'b-', label="yp1")
p012, = ax01.plot(t,yp2,'g-', label="yp2")

p021, = ax02.plot(t,yv1,'b-', label="yv1")
p022, = ax02.plot(t,yv2,'g-', label="yv2")

p031, = ax03.plot(t,yp1,'b-', label="yp1")
p032, = ax04.plot(t,yv1,'g-', label="yv1")

# set lagends
ax01.legend([p011,p012], [p011.get_label(),p012.get_label()])
ax02.legend([p021,p022], [p021.get_label(),p022.get_label()])
ax03.legend([p031,p032], [p031.get_label(),p032.get_label()])

# Data Update
xmin = 0.0
xmax = 5.0
x = 0.0

def updateData(self):
	global x
	global yp1
	global yv1
	global yp2
	global yv2
	global t

	tmpp1 = 1 + exp(-x) *sin(2 * pi * x)
	tmpv1 = - exp(-x) * sin(2 * pi * x) + exp(-x) * cos(2 * pi * x) * 2 * pi
	yp1=append(yp1,tmpp1)
	yv1=append(yv1,tmpv1)
	yp2=append(yp2,0.5*tmpp1)
	yv2=append(yv2,0.5*tmpv1)
	t=append(t,x)

	x += 0.05

	p011.set_data(t,yp1)
	p012.set_data(t,yp2)

	p021.set_data(t,yv1)
	p022.set_data(t,yv2)

	p031.set_data(t,yp1)
	p032.set_data(t,yv1)

	if x >= xmax-1.00:
		p011.axes.set_xlim(x-xmax+1.0,x+1.0)
		p021.axes.set_xlim(x-xmax+1.0,x+1.0)
		p031.axes.set_xlim(x-xmax+1.0,x+1.0)
		p032.axes.set_xlim(x-xmax+1.0,x+1.0)

	return p011, p012, p021, p022, p031, p032

# interval: draw new frame every 'interval' ms
# frames: number of frames to draw
simulation = animation.FuncAnimation(f0, updateData, blit=False, frames=200, interval=20, repeat=False)

# Uncomment the next line if you want to save the animation
#simulation.save(filename='sim.mp4',fps=30,dpi=300)

plt.show()
'''
