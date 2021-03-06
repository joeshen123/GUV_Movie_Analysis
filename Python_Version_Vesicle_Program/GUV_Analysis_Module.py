import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from skimage import io
from scipy import ndimage
from skimage.measure import ransac, CircleModel
from matplotlib.patches import Circle
import cv2
from skimage import filters
from skimage import color
from skimage import feature
from skimage import morphology
from scipy.misc import bytescale
import pandas as pd
import itertools
from matplotlib.animation import ArtistAnimation
from matplotlib import gridspec
from matplotlib.pyplot import cm
from skimage.filters import threshold_otsu, threshold_adaptive
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from skimage import measure
import seaborn as sns
from tqdm import tqdm
from colorama import Fore
from matplotlib.colors import LinearSegmentedColormap
from skimage import img_as_ubyte

#Make a colormap like Imagej Green Channel
cdict1 = {'red':  ((0.0, 0.0, 0.0),   # <- at 0.0, the red component is 0
                   (0.5, 0.0, 0.0),   # <- at 0.5, the red component is 1
                   (1.0, 0.0, 0.0)),  # <- at 1.0, the red component is 0

         'green': ((0.0, 0.0, 0.0),   # <- etc.
                   (0.5, 0.5, 0.5),
                   (1.0, 1.0, 1.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (1.0, 0.0, 0.0))
         }

Green = LinearSegmentedColormap('Green', cdict1)

#Make a colormap like Imagej Red Channel
cdict2 = {'red':  ((0.0, 0.0, 0.0),   # <- at 0.0, the red component is 0
                   (0.5, 0.5, 0.5),   # <- at 0.5, the red component is 1
                   (1.0, 1.0, 1.0)),  # <- at 1.0, the red component is 0

         'green': ((0.0, 0.0, 0.0),   # <- etc.
                   (0.5, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (1.0, 0.0, 0.0))
         }

Red = LinearSegmentedColormap('Red', cdict2)


# define a function to find the border of a circle (assume the edge is the brightest)
def circle_edge_detector(center, distance, image):
    theta = np.linspace(0., 2*np.pi)
    x0,y0 = center

    x_list = distance*np.cos(theta) + x0
    y_list = distance*np.sin(theta) + y0

    end_list = []

    for n in range(len(theta)):
        end = (x_list[n],y_list[n])
        end_list.append(end)

    x_edge_list = []
    y_edge_list = []

    for end in end_list:
        lineprofile, x,y = line_pixel_extractor(center, end, 1000, image)
        max_num = np.argmax(lineprofile)

        test_dist = np.linalg.norm(np.array((x[max_num], y[max_num])) - center)

        if test_dist >= 0.7*distance and test_dist <= 1.05*distance:
          x_edge_list.append(x[max_num])
          y_edge_list.append(y[max_num])


    edge_list = np.column_stack([np.array(y_edge_list),np.array(x_edge_list)])


    return edge_list


#Define a function to obtain all pixels in a line
def line_pixel_extractor(start,end,num,image):
    x0,y0=start
    x1,y1 =end
    x,y = np.linspace(x0,x1,num),np.linspace(y0,y1,num)

    zi = ndimage.map_coordinates(image,np.vstack((y,x)))

    return zi,x,y


# Define a function to plot both radius and protein intensity from a list of pandas df
def Pandas_list_plotting(pandas_list, keyword):
    fig= plt.figure(figsize = (10,6))
    gs = gridspec.GridSpec(1,1)

    ax = fig.add_subplot(gs[:,:])

    list_len = len(pandas_list)
    color = cm.tab20b(np.linspace(0,1,list_len))

    if keyword == 'Intensity':
      ax.set_title('Vesicles Protein Bindings Changes',fontsize=18)

      ax.set_xlabel('Time Points (min)', fontsize = 16, fontweight = 'bold')
      ax.set_ylabel('Protein Fluorescence Intensity', fontsize = 16,fontweight = 'bold')

      for n in range(list_len):
         df = pandas_list[n]
         Intensity_data = df['GFP intensity'].tolist()
         Time_point = df['Time Point'].tolist()
         ax = plt.plot(Time_point, Intensity_data, color = color[n], label = str(n))

    if keyword == 'Radius':
      ax.set_title('Vesicles Radius Changes (um)',fontsize=18)

      ax.set_xlabel('Time Points (min)', fontsize = 16, fontweight = 'bold')
      ax.set_ylabel('Vesicles Radius (um)', fontsize = 16,fontweight = 'bold')

      for n in range(list_len):
         df = pandas_list[n]
         radius_data = df['radius_micron'].tolist()
         Time_point = df['Time Point'].tolist()
         ax = plt.plot(Time_point, radius_data, color = color[n], label = str(n))


    plt.legend(loc='right')
    plt.tight_layout()
    plt.show()

def create_circular_mask(center, radius,h=512,w=512):

    Y,X = np.ogrid[:h,:w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask

def crop_image(Image,center,width,height,factor=3):
    low_point = (center[0] - 1.4 * width, center[1] - 1.4 * height)
    X,Y = low_point
    Crop_image = Image[int(round(Y)):int(round(Y+factor*height)),int(round(X)):int(round(X + factor * width))]
    return Crop_image

def creating_mask(Image,center,width,height,factor=2):
    Y,X = np.ogrid[:Image.shape[0],:Image.shape[1]]

    low_point = (center[0] - width, center[1] - height)
    X_low,Y_low = low_point

    X_low = int(X_low)
    Y_low = int(Y_low)

    X_high = int(round(X_low+factor*width))
    Y_high = int(round(Y_low+factor*height))


    mask_image = (Y >= Y_low) & (Y <= Y_high) & (X >= X_low) & (X <= X_high)

    return mask_image

def generate_df_from_list(pixel_attribute, center_list,r_list,intensity_list):

    time_point_list = np.linspace(0, 60, num = len(center_list))
    center_list = np.array(center_list)
    center_x = center_list [:,0]
    center_y = center_list[:,1]

    r_list = np.array(r_list) 
    r_list_micron = r_list * pixel_attribute
    intensity_list = np.array(intensity_list)

    stat_dict = {'Time Point': time_point_list,'center_x': center_x, 'center_y': center_y, 'radius': r_list, 'radius_micron':r_list_micron, 'GFP intensity': intensity_list}

    stat_df = pd.DataFrame(stat_dict)

    return stat_df

def draw_sample_points (image,Position,distance):
    mask = creating_mask(image,Position,width=distance,height=distance)
    local_thresh = threshold_otsu(image)
    binary_im = image > local_thresh

    closing_im = morphology.closing(binary_im)


    edges = feature.canny(closing_im,sigma=0.5)
    edges = edges * mask
    
    points = np.column_stack(np.nonzero(edges))


    return points


#This function to enhance and blur the images. Improve contrast
def enhance_blur_medfilter(img, enhance=True,blur=True,kernal=5,median_filter=True,size=8):

    if enhance:
        clahe = cv2.createCLAHE()
        cl1=clahe.apply(img)
    else:
        cl1=img.copy()

    if blur:
        gaussian_blur = cv2.GaussianBlur(cl1,(kernal,kernal),0,0)
    else:
        gaussian_blur=cl1.copy()

    if median_filter:
        medfilter = ndimage.median_filter(gaussian_blur,size)
    else:
        medfilter = gaussian_blur.copy()

    return (cl1, gaussian_blur,medfilter)


def fit_circle_contour(image,pt,width=20,height=20):
    sample_points = circle_edge_detector(pt,width, image)
    #print(sample_points)
    #print(len(sample_points))
    
    if len(sample_points) <= 25:
        sample_points = draw_sample_points(image,pt,width)
 
    model_robust, inliers = ransac(sample_points, CircleModel,min_samples=3,residual_threshold=2,max_trials=1000)
    y,x,r = model_robust.params
    center = (x,y)
    
        
    return (center,r)

def obtain_ring_pixel(center,radius,dif,image,choice='global'):
    inner_ring_mask = create_circular_mask(center,radius-dif)
    outer_ring_mask = create_circular_mask(center,radius+dif)

    ring_mask = np.logical_xor(outer_ring_mask,inner_ring_mask)

    where = np.where(ring_mask==True)
    
    if choice == 'local':
      raw_median_intensity_from_img = np.median(image[where[0],where[1]])

      background_outer_ring_mask = create_circular_mask(center,radius+dif+5)
      background_inner_ring_mask = create_circular_mask(center,radius+dif+4)

      background_ring_mask = np.logical_xor(background_outer_ring_mask,background_inner_ring_mask)

      background_where = np.where(background_ring_mask ==True)
      background_median_intensity = np.median(image[background_where[0],background_where[1]])

      median_intensity_from_img = raw_median_intensity_from_img - background_median_intensity

    elif choice == 'global':
      kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(50,50))
      opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
      sub_image = cv2.subtract(image,opening)
      
      median_intensity_from_img = np.median(sub_image[where[0],where[1]])


    return median_intensity_from_img

def draw_circle_fit (center,r ,image):
    Plot_center = (int(np.ceil(center[0])),int(np.ceil(center[1])))
    
    img = cv2.circle(image, Plot_center, int(np.ceil(r)), 0,2)
    
    return img

def display_image_sequence(image_stack,cmap_name):
   stack_len = len(image_stack)
   #print(image_stack.shape)
   img = None

   for i in range(stack_len):
     im = image_stack[i]
     if img is None:
        img = pylab.figimage(im, cmap = cmap_name)
     else:
        img.set_data(im)
     pylab.pause(.1)
     pylab.draw()
   plt.close()



# Define a class to store and process the time series Images
class Image_Stacks:

     def __init__ (self,vesicle_image,GFP_image):
         self.Image_stack = vesicle_image.copy()
         self.Intensity_stack = GFP_image.copy()
         self.Intensity_stack_med = np.zeros(GFP_image.shape)
         self.Image_stack_median = np.zeros(vesicle_image.shape)
         self.Rendering_Image_Stack = []
         self.Rendering_Intensity_Image = []
         self.Crop_Original_Stack = []
         self.Crop_Intensity_Stack = []
         self.Micron_Pixel = 0.33
         self._enhance = None
         self._blur = None
         self._kernal = None
         self.median_filter = None
         self.size = None
         self.point = None
         self.point_list = None
         self.end = None
         self.stats_df_list = []
         self.width = None
         self.height = None
         self.line_ani = None
         self.render_image_temp = None
         self.render_image_intensity_temp = None

        
     def set_parameter (self,enhance=True, blur = True, kernal = 5,median_filter = True,size=3):
        self._enhance = enhance
        self._blur = blur
        self._kernal = kernal
        self.median_filter = median_filter
        self.size = size

     def set_points (self):
         line_selection = line_drawing()
         line_selection.show_image(self.Image_stack_median)
         line_selection.draw_line()
         plt.show()

         self.point_list = line_selection.center
         self.end = line_selection.end

         self.width = line_selection.dist
         self.height = line_selection.dist
     

     def stack_enhance_blur(self):
         num_len = self.Image_stack.shape[0]
         
         pb = tqdm(range(num_len), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))

         for n in pb:
            pb.set_description ('Preprosessing Image Stack')
            img = self.Image_stack[n].copy()
            intensity_img = self.Intensity_stack[n].copy()
            _, _,medfilter = enhance_blur_medfilter(img, self._enhance,self._blur,self._kernal,self.median_filter,self.size)
            _, _,medfilter_intensity = enhance_blur_medfilter(intensity_img, False,self._blur,self._kernal,self.median_filter,self.size)

            self.Image_stack_median[n] = medfilter
            self.Intensity_stack_med[n] = medfilter_intensity
            

     def tracking_single_circle(self,num):
        num_len = self.Image_stack_median.shape[0]
        center_list = []
        r_list = []
        GFP_list = []
        self.point = self.point_list[num]

        for n in range(num_len):
            
            try:
             center,r = fit_circle_contour(self.Image_stack_median[n], self.point, self.width[num],self.height[num])
        
            
            except:
                pass
                print('Exception Raised!')
                center = self.point
                if n != 0:
                  r = r_list[-1]
                else:
                  print('None Exception')
                  r = self.width[num]

            
            self.point = center
            center_list.append(center)
            r_list.append(r)
        
        for n in range(num_len):
            Intensity = obtain_ring_pixel(center_list[n],r_list[n],1.5,self.Intensity_stack_med[n], choice='global')
            GFP_list.append(Intensity)

        stats_df = generate_df_from_list(self.Micron_Pixel, center_list,r_list,GFP_list)

        return stats_df
     
     # Design functions to track multiple circles based on single circle tracking function
     def tracking_multiple_circles(self):
        num = len(self.point_list)
        
        pb = tqdm(range(num), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET))
        for n in pb:
           pb.set_description("Fitting Circles on GUV and Measure Protein Fluorescence")
           df = self.tracking_single_circle(n)
           self.stats_df_list.append(df)



        


    # Generating Stacks to display as animation later
     def displaying_circle_movies(self):
        total_circle_list= []

        for df in self.stats_df_list:
          single_circles = df[['center_x','center_y','radius']].values.tolist()
          total_circle_list.append(single_circles)

        num_len = self.Image_stack_median.shape[0]
        
        pb = tqdm(range(num_len), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET))

        for n in pb:
          pb.set_description("Generating Image Stack Movies")
          self.render_image_intensity_temp =  self.Intensity_stack_med[n].copy()
          self.render_image_temp = self.Image_stack_median[n].copy()

          for single_circles in total_circle_list:
            center = (single_circles[n][0],single_circles[n][1])
            r = single_circles[n][2]

            self.render_image_temp = draw_circle_fit(center,r,self.render_image_temp)
            self.render_image_intensity_temp = draw_circle_fit(center,r, self.render_image_intensity_temp)

            #Crop_Original_Image = crop_image(original_image, center, r, r)
            #print(Crop_Original_Image.shape)
            #Crop_Intensity_Image = crop_image(intensity_based_image, center, r, r)

          self.Rendering_Image_Stack.append(self.render_image_temp)
          self.Rendering_Intensity_Image.append(self.render_image_intensity_temp)

            #self.Crop_Original_Stack.append(Crop_Original_Image)
            #self.Crop_Intensity_Stack.append(Crop_Intensity_Image)
    
        self.Rendering_Image_Stack = np.array(self.Rendering_Image_Stack)
        self.Rendering_Intensity_Image = np.array(self.Rendering_Intensity_Image)

        #self.Crop_Original_Stack = np.array(self.Crop_Original_Stack)
        #self.Crop_Intensity_Stack = np.array(self.Crop_Intensity_Stack)

'''
    # Define a function to live plot the protein intesnity data
     def live_plotting_intensity(self):
       Intensity_data = self.stats_df_list[0]['GFP intensity'].tolist()
       Radius_data = self.stats_df_list[0]['radius'].tolist()
       Time_point = np.arange(len(Intensity_data))
       Intensity_with_Time = []
       Intensity_with_Time.append(Time_point)
       Intensity_with_Time.append(Intensity_data)
       Intensity_with_Time = np.array(Intensity_with_Time)
       Radius_with_Time = []
       Radius_with_Time.append(Time_point)
       Radius_with_Time.append(Radius_data)
       Radius_with_Time = np.array(Radius_with_Time)
       gs = gridspec.GridSpec(2,2)
       fig = plt.figure(figsize = (10,10))
       ax1 = fig.add_subplot(gs[0:1,0:1])
       ax2 = fig.add_subplot(gs[0:1,1:2])
       ax3 = fig.add_subplot(gs[1:2,:])
       ax1.set_title('GUV Radius Changes',fontsize=18)
       ax2.set_title('Peripheral Protein Bindings Changes',fontsize=18)
       ax3.set_xlabel('Time Points (min)', fontsize = 16, fontweight = 'bold')
       ax3.set_ylabel('Protein Fluorescence Intensity', fontsize = 16,fontweight = 'bold')
       ax3.set_xlim(0,(len(Intensity_data)+1))
       ax3.set_ylim(0,(np.max(Intensity_data)+300))
       #l_one, =ax3.plot([], [], 'r-')
       ax4= ax3.twinx()
       #l_two, =ax4.plot([], [], 'b-')
       ax4.set_xlabel('Time Points (min)', fontsize = 16, fontweight = 'bold')
       ax4.set_ylabel('Radius Changes (um)',fontsize = 16, fontweight = 'bold')
       ax4.set_xlim(0,(len(Radius_data)+1))
       ax4.set_ylim((np.min(Radius_data)-1),(np.max(Radius_data)+1))
       #self.line_ani= FuncAnimation(fig, update_line, frames = len(Intensity_data), fargs=(Intensity_with_Time,Radius_with_Time,l_one,l_two),interval=100, blit=True, repeat=False)
       ims = []
       for time in range(len(Intensity_data)):
           GUV_im = ax1.imshow(self.Crop_Original_Stack[time],cmap = 'Reds')
           ax1.axis('off')
           Intensity_im = ax2.imshow(self.Crop_Intensity_Stack[time],cmap = 'Greens')
           ax2.axis('off')
           l_one, = ax3.plot(Intensity_with_Time[0,:time],Intensity_with_Time[1,:time],'r-')
           l_two, = ax4.plot(Radius_with_Time[0,:time],Radius_with_Time[1,:time],'b-')
           ax3.legend((l_one,l_two),('Protein Fluorescence Intensity','Radius'),loc=0)
           ims.append([GUV_im,Intensity_im,l_one,l_two])
       self.line_ani = ArtistAnimation(fig,ims, interval=50,blit=False,repeat=True)
       plt.tight_layout()
       plt.show()
    '''


# Define a class for drawing a line for tracking
class line_drawing():
    def __init__(self):
        self.fig, self.ax = None,None
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.center = None
        self.end = None
        self.dist = []
        self.lines = []

    def show_image(self,image):
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(image[0,:,:], cmap= Red)

    def draw_line(self):

        xy = self.fig.ginput(-1,timeout=0)
        x = [p[0] for p in xy]
        self.x0 = x[::2]
        self.x1 = x[1::2]

        y = [p[1] for p in xy]
        self.y0 = y[::2]
        self.y1 = y[1::2]

        self.center = list(zip(self.x0, self.y0))
        self.end = list(zip(self.x1,self.y1))
        
        for n in range(len(self.x0)):
           line = plt.plot((self.x0[n],self.x1[n]),(self.y0[n],self.y1[n]),'w-')
           self.lines.append(line)

        self.ax.figure.canvas.draw()
        
        for n in range(len(self.center)):
           dist = np.linalg.norm(np.array(self.center[n]) - np.array(self.end[n]))
           self.dist.append(dist)

        
'''
root = tk.Tk()
root.withdraw()
my_filetypes = [('all files', '.*'),('Image files', '.tif')]
Original_Image_path = filedialog.askopenfilename(title='Please Select a File', filetypes = my_filetypes)
 
im = io.imread(Original_Image_path)
line_selection = line_drawing()
line_selection.show_image(im)
line_selection.draw_line()
plt.show()
num = len(line_selection.center)
circle_list = []
fig,ax = plt.subplots()
ax.imshow(im[0,:,:])
for n in range(num):
   list = circle_edge_detector(line_selection.center[n],line_selection.dist[n],im[0,:,:])
   y,x= zip(*list)
   center,r = fit_circle_contour(im[0,:,:],line_selection.center[n],line_selection.dist[n],line_selection.dist[n])
   circle = Circle(center,radius=r,facecolor=None,fill=False,edgecolor='b',linewidth=2)
   circle_list.append(circle)
   #plt.scatter(x,y,c = 'r')
for c in circle_list:
  ax.add_patch(c)
plt.show()
'''