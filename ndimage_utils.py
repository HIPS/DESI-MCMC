import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

def generate_peaks(data, threshold=-np.inf, neighborhood_size=5):
  """
  Returns a generator of x,y tuples.

  http://stackoverflow.com/questions/9111711/get-coordinates-of-local-maxima-in-2d-array-above-certain-value
  """
  data_max = filters.maximum_filter(data, neighborhood_size)
  maxima = (data == data_max)
  data_min = filters.minimum_filter(data, neighborhood_size)
  diff = ((data_max - data_min) > threshold)
  maxima[diff == 0] = 0 # sets values <= threshold as background

  labeled, num_objects = ndimage.label(maxima)
  slices = ndimage.find_objects(labeled)
  for dy,dx in slices:
    x_center = (dx.start + dx.stop - 1)/2
    y_center = (dy.start + dy.stop - 1)/2
    yield x_center, y_center
