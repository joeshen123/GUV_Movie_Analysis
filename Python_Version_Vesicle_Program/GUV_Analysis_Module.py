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
import PIL
import pickle
from tqdm import tqdm
from scipy.misc import bytescale
import multiprocessing
import pandas as pd
import itertools
from matplotlib.patches import Rectangle
from matplotlib.animation import ArtistAnimation
from matplotlib import gridspec

def hough_circle_image(img,dp,mindist,param1,param2,minr,maxr):
    img = bytescale(img)

    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,dp,mindist,
                                param1=param1,param2=param2,minRadius=minr,maxRadius=maxr)

    return circles


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

def generate_df_from_list(center_list,r_list,intensity_list):

    center_list = np.array(center_list)
    center_x = center_list [:,0]
    center_y = center_list[:,1]

    r_list = np.array(r_list)

    intensity_list = np.array(intensity_list)

    stat_dict = {'center_x': center_x, 'center_y': center_y, 'radius': r_list, 'GFP intensity': intensity_list}

    stat_df = pd.DataFrame(stat_dict)

    return stat_df

def draw_sample_points (image,Position,width=20, height=20):
    #image = ndimage.median_filter(image,3)
    mask = creating_mask(image,Position,width,height)
    edges = feature.canny(image,sigma=5)
    edges = edges * mask

    #plt.imshow(edges)
    #plt.show()

    points = np.array(np.nonzero(edges)).T

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
    sample_points = draw_sample_points(image,pt,width,height)
    model_robust, inliers = ransac(sample_points, CircleModel,min_samples=3,residual_threshold=2,max_trials=1000)
    y,x,r = model_robust.params
    center = (x,y)
    return (center,r)

def obtain_ring_pixel(center,radius,dif,image,choice='local'):
    inner_ring_mask = create_circular_mask(center,radius-dif)
    outer_ring_mask = create_circular_mask(center,radius+dif)

    ring_mask = np.logical_xor(outer_ring_mask,inner_ring_mask)
    if choice == 'local':
      background_outer_ring_mask = create_circular_mask(center,radius+dif+2)
      background_inner_ring_mask = create_circular_mask(center,radius+dif+1)

      background_ring_mask = np.logical_xor(background_outer_ring_mask,background_inner_ring_mask)

      background_where = np.where(background_ring_mask ==True)
      background_mean_intensity = np.median(image[background_where[0],background_where[1]])

    else:
      background_mean_intensity = np.median(image)

    where = np.where(ring_mask==True)
    raw_mean_intensity_from_img = np.mean(image[where[0],where[1]])


    mean_intensity_from_img = raw_mean_intensity_from_img - background_mean_intensity

    return mean_intensity_from_img

def draw_circle_fit (center,r ,image):
    fig,ax = plt.subplots()
    ax.imshow(image)
    circle = Circle(center,radius=r,facecolor=None,fill=False,edgecolor='r',linewidth=2)
    ax.add_patch(circle)
    ax.axis('off')


    fig.canvas.draw()

    img = np.array(fig.canvas.renderer._renderer)

    plt.close()
    return img

def display_image_sequence(image_stack,string):
   stack_len = len(image_stack)
   #print(image_stack.shape)
   img = None
   image_length = tqdm(range(stack_len))
   for i in image_length:
     image_length.set_description(string)
     im = image_stack[i]
     if img is None:
        img = pylab.figimage(im)
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
         self.Image_stack_enhance = np.zeros(vesicle_image.shape)
         self.Image_stack_blur = np.zeros(vesicle_image.shape)
         self.Image_stack_median = np.zeros(vesicle_image.shape)
         self.Rendering_Image_Stack = []
         self.Rendering_Intensity_Image = []
         self.Crop_Original_Stack = []
         self.Crop_Intensity_Stack = []
         self._enhance = None
         self._blur = None
         self._kernal = None
         self.median_filter = None
         self.size = None
         self.point = None
         self.stats_df = None
         self.width = None
         self.height = None
         self.line_ani = None

     def set_parameter (self,enhance=True, blur = True, kernal = 5,median_filter = True,size=3):
        self._enhance = enhance
        self._blur = blur
        self._kernal = kernal
        self.median_filter = median_filter
        self.size = size

     def set_points (self):
         line_selection = line_drawing(self.Image_stack_median)
         line_selection.draw_line()
         plt.show()

         self.point = line_selection.center

         self.width = line_selection.dist
         self.height = line_selection.dist

     def stack_enhance_blur(self):
         num_len = self.Image_stack.shape[0]

         for n in range(num_len):
            img = self.Image_stack[n].copy()
            intensity_img = self.Intensity_stack[n].copy()
            cl1, gaussian_blur,medfilter = enhance_blur_medfilter(img, self._enhance,self._blur,self._kernal,self.median_filter,self.size)
            _, _,medfilter_intensity = enhance_blur_medfilter(intensity_img, self._enhance,self._blur,self._kernal,self.median_filter,self.size)

            self.Image_stack_blur[n] = cl1
            self.Image_stack_enhance[n] = gaussian_blur
            self.Image_stack_median[n] = medfilter
            self.Intensity_stack_med[n] = medfilter_intensity

     def tracking_single_circle(self):
        num_len = self.Image_stack_median.shape[0]
        center_list = []
        r_list = []
        GFP_list = []
        iter_len = tqdm(range(num_len))
        for n in iter_len:
            iter_len.set_description('Fitting circle to original image')
            center,r = fit_circle_contour(self.Image_stack_median[n], self.point, self.width,self.height)
            self.point = center
            center_list.append(center)
            r_list.append(r)

        for n in iter_len:
            iter_len.set_description('Measuring GFP Intensities')
            Intensity = obtain_ring_pixel(center_list[n],r_list[n],3,self.Intensity_stack_med[n],choice='global')
            GFP_list.append(Intensity)

        self.stats_df = generate_df_from_list(center_list,r_list,GFP_list)



     def displaying_circle_movies(self):

        single_circles = self.stats_df[['center_x','center_y','radius']].values.tolist()

        num_len = self.Image_stack_median.shape[0]

        list_len = tqdm(range(num_len))

        for n in list_len:
            list_len.set_description('Generating Rendering Images')
            original_image = self.Image_stack_median[n]
            intensity_based_image = self.Intensity_stack_med[n]
            center = (single_circles[n][0],single_circles[n][1])
            r = single_circles[n][2]
            Rendering_Image = draw_circle_fit(center,r,original_image)
            Rendering_Intensity_Image = draw_circle_fit(center,r,intensity_based_image)

            Crop_Original_Image = crop_image(original_image, center, r, r)
            #print(Crop_Original_Image.shape)
            Crop_Intensity_Image = crop_image(intensity_based_image, center, r, r)

            self.Rendering_Image_Stack.append(Rendering_Image)
            self.Rendering_Intensity_Image.append(Rendering_Intensity_Image)

            self.Crop_Original_Stack.append(Crop_Original_Image)
            self.Crop_Intensity_Stack.append(Crop_Intensity_Image)

        #self.Rendering_Image_Stack = np.array(self.Rendering_Image_Stack)
        #self.Rendering_Intensity_Image = np.array(self.Rendering_Intensity_Image)

        #self.Crop_Original_Stack = np.array(self.Crop_Original_Stack)
        #self.Crop_Intensity_Stack = np.array(self.Crop_Intensity_Stack)

    # Define a function to live plot the protein intesnity data
     def live_plotting_intensity(self):
       Intensity_data = self.stats_df['GFP intensity'].tolist()
       Radius_data = self.stats_df['radius'].tolist()
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

       ax1.set_title('Vesicles Radius Changes',fontsize=18)
       ax2.set_title('Vesicles Protein Bindings Changes',fontsize=18)


       ax3.set_xlabel('Time Points', fontsize = 16, fontweight = 'bold')
       ax3.set_ylabel('Protein Fluorescence Intensity', fontsize = 16,fontweight = 'bold')
       ax3.set_xlim(0,(len(Intensity_data)+1))
       ax3.set_ylim(-800,(np.max(Intensity_data)+300))
       #l_one, =ax3.plot([], [], 'r-')

       ax4= ax3.twinx()
       #l_two, =ax4.plot([], [], 'b-')

       ax4.set_xlabel('Time Points', fontsize = 16, fontweight = 'bold')
       ax4.set_ylabel('Radius Changes',fontsize = 16, fontweight = 'bold')
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

       self.line_ani = ArtistAnimation(fig,ims, interval=100,blit=False,repeat=True)
       plt.tight_layout()

       plt.show()



# Define a class for drawing a line for tracking
class line_drawing():
    def __init__(self,image):
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(image[0,:,:])
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.center = None
        self.end = None
        self.dist = None
        self.lines = []

    def draw_line(self):

        xy = plt.ginput(2)

        x = np.array([p[0] for p in xy])
        self.x0 = x[0]
        self.x1 = x[1]

        y = np.array([p[1] for p in xy])
        self.y0 = y[0]
        self.y1 = y[1]

        self.center = (self.x0,self.y0)

        self.end = (self.x1,self.y1)

        line = plt.plot(x,y,'r-')

        self.ax.figure.canvas.draw()

        self.dist = np.linalg.norm(np.array(self.center) - np.array(self.end))

        self.lines.append(line)
