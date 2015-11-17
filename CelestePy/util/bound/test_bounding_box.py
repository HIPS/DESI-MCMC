import numpy as np
from bounding_box import *

if __name__ == "__main__":
    boxes = np.array([[2, 6, -3, 5], [3, 9, -2, 7]])
    print get_bounding_boxes_idx(np.array([-10, -10]), boxes)
    print get_bounding_boxes_idx(np.array([3, 2]), boxes)
    print get_bounding_boxes_idx(np.array([8, 2]), boxes)
    print get_bounding_boxes_idx(np.array([4, -2.5]), boxes)
