import pickle
import Analysis as a
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pylab
from scipy import ndimage
from scipy.misc import bytescale
import cv2

file = open('Image_Array', 'rb')
Image_series = pickle.load(file)
file.close()

sample_images = Image_series[0,:,:,:]

test_image = bytescale(sample_images[35,:,:])

th2 = cv2.adaptiveThreshold(test_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY,3,5)
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(th2,kernel,iterations=1)
imgray=cv2.Canny(th2)
img,_,_ = a.enhance_blur_medfilter(imgray)

plt.imshow(imgray)
plt.show()

def hough_circle_image(img,dp,mindist,param1,param2,minr,maxr):
    img = bytescale(img)

    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,dp,mindist,
                                param1=param1,param2=param2,minRadius=minr,maxRadius=maxr)
    if circles is not None:
      circles = np.uint16(np.around(circles))

      for i in circles[0,:]:
         # draw the outer circle
         cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
         # draw the center of the circle
         cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)


    return (img,circles)
'''
img_series = []

for n in range(sample_images.shape[0]):
    img,_,_ = a.enhance_blur_medfilter(sample_images[n,:,:])
    img, _= hough_circle_image(img,5,10,200,30,10,40)
    img_series.append(img)

plt.imshow(img_series[20])
plt.show()
'''
