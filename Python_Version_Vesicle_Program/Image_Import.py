from nd2reader.reader import ND2Reader
import pickle as pk
import numpy as np
import time

start_time = time.time()

# Define a function to convert time series of ND2 images to a numpy list of Max Intensity Projection
# images.

def Z_Stack_Images_Extractor(address, fields_of_view, channel):
   Image_Sequence = ND2Reader(address)
   time_series = Image_Sequence.sizes['t']
   z_stack = Image_Sequence.sizes['z']

   Total_Slice = []
   for time in range(time_series):
     z_stack_images = []
     for z_slice in range(z_stack):
        slice = Image_Sequence.get_frame_2D(c=channel, t=time, z=z_slice, v=fields_of_view)
        z_stack_images.append(slice)
     z_stack_images = np.array(z_stack_images)
     Total_Slice.append(z_stack_images)

   Total_Slice = np.array(Total_Slice)

   return Total_Slice


#print(ND2Reader('/Users/joeshen/Desktop/niethammerlab/Niethammer Lab/Joe/Research Project/Vesicle Experiment/20181116 L3233F Vesicles binding/L3233F 1um binding_low_power.nd2').metadata)
Images = Z_Stack_Images_Extractor('/Users/joeshen/Desktop/niethammerlab/Niethammer Lab/Joe/Research Project/Vesicle Experiment/20181116 L3233F Vesicles binding/L3233F 1um binding_low_power.nd2',
                                   fields_of_view=0, channel=1)

ndarray = open('Image_Array','wb')

pk.dump(Images, ndarray)

ndarray.close()

#Print the time of this program
print("--- %s seconds ---" % (time.time() - start_time))
