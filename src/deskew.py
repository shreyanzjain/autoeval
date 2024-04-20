import cv2
import numpy as np
from skimage.transform import rotate
from skimage.filters.edges import sobel
from skimage.util import invert

img_url = './public/image.jpg'

# find the horizontal projection of all the rows in the image
# rotate the image between angles -10 to 10 degrees 
# and find the angle which gives the highest median of horizontal projection

def horizontal_projections(sobel_image):
  sum_of_cols = []
  rows, _ = sobel_image.shape
  for row in range(rows - 1):
    sum_of_cols.append(np.sum(sobel_image[row,:]))
    
  return sum_of_cols

def deskew(img_url):
    img = cv2.imread(img_url, cv2.IMREAD_GRAYSCALE)
    sobel_img = invert(sobel(img))
    predicted_angle = 0
    highest_median_hp = 0
    for _, angle in enumerate(range(-10, 10)):
       hp = horizontal_projections(sobel_image=rotate(sobel_img, angle=angle, cval=1))
       median_hp = np.median(hp)
       if median_hp > highest_median_hp:
          predicted_angle = angle
          highest_median_hp = median_hp
    deskewed_img = rotate(img, predicted_angle, cval=1)
    # convert float 64bit to a 8 bit image
    result = cv2.normalize(deskewed_img, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(f"./public/saved/image.jpg", result)

deskew(img_url)