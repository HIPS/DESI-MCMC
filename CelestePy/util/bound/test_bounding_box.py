import numpy as np
from bounding_box import *

if __name__ == "__main__":
    boxes = [BoundingBox(np.array([4, 3]), 1), BoundingBox(np.array([5, 2]), 6)]
    print get_bounding_boxes_idx(np.array([-10, -10]), boxes)
    print get_bounding_boxes_idx(np.array([3, 2]), boxes)
    print get_bounding_boxes_idx(np.array([11, 2]), boxes)
    print get_bounding_boxes_idx(np.array([-1, 7]), boxes)
