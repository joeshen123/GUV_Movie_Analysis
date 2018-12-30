import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import io
import cv2
from skimage import filters
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import numpy as np

# Import Image
Images = io.imread('C2.tif')

sample_image = Images[5,:,:]

def define_rect(image):
    """
    Define a rectangular window by click and drag your mouse.

    Parameters
    ----------
    image: Input image.
    """

    clone = image.copy()
    rect_pts = [] # Starting and ending points
    win_name = "image" # Window name

    def select_points(event, x, y, flags, param):

        nonlocal rect_pts
        if event == cv2.EVENT_LBUTTONDOWN:
            rect_pts = [(x, y)]

        if event == cv2.EVENT_LBUTTONUP:
            rect_pts.append((x, y))

            # draw a rectangle around the region of interest
            cv2.rectangle(clone, rect_pts[0], rect_pts[1], (0, 255, 0), 2)
            cv2.imshow(win_name, clone)

    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, select_points)

    while True:
        # display the image and wait for a keypress
        cv2.imshow(win_name, clone)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("r"): # Hit 'r' to replot the image
            clone = image.copy()

        elif key == ord("c"): # Hit 'c' to confirm the selection
            break

    # close the open windows
    cv2.destroyWindow(win_name)

    return rect_pts

#This function to enhance and blur the images. Improve contrast
def enhance_blur_segment(img, enhance=True,blur=True,kernal=9):

    if enhance:
        clahe = cv2.createCLAHE()
        cl1=clahe.apply(img)
    else:
        cl1=img.copy()

    if blur:
        gaussian_blur = cv2.GaussianBlur(cl1,(kernal,kernal),0,0)
    else:
        gaussian_blur=cl1.copy()

    centers = filters.threshold_otsu(gaussian_blur)
    segmented = gaussian_blur > centers

    return (cl1, gaussian_blur,segmented)


#Use Watershed algorithm to seperate touching object
def watershed_process (binary_im):
   distance = ndi.distance_transform_edt(binary_im)
   local_maxi=peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                            labels=binary_im)
   markers = ndi.label(local_maxi)[0]
   labels=watershed(-distance, markers,mask=binary_im)

   return (distance,labels)


(a,b,c) = enhance_blur_segment(sample_image)

_,ws = watershed_process(c)


plt.imshow(ws,cmap=plt.cm.nipy_spectral, interpolation='nearest')
plt.show()
