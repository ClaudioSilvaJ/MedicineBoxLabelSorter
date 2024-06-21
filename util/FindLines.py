import cv2
import numpy as np
from util.Helper import Helper

class FindLines(Helper):
  
  def finds_aux(self, img):
    bw = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if (cv2.boundingRect(cnt)[2] / cv2.boundingRect(cnt)[3])>=4.0]
    return filtered_contours

  def find_countours(self, img):
    kernel = np.ones((3,1), np.uint8)
    img = cv2.dilate(img, kernel)
    self.images.append(img)
    bw = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel_size = (25, 1) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    bw_closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    self.images.append(bw_closed)
    Helper.show_img(self, bw_closed)
    contours, _ = cv2.findContours(bw_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if (cv2.boundingRect(cnt)[2] / cv2.boundingRect(cnt)[3])>=3.0]
    min_width = 50
    sorted_contours = [contour for contour in filtered_contours if cv2.boundingRect(contour)[2] >= min_width]
    sorted_contours = sorted(sorted_contours, key=lambda contour: cv2.boundingRect(contour)[1])
    return sorted_contours

  def find_text(self, img):
    try:
      kernel = np.ones((15,70), np.uint8)
      erosion = cv2.erode(img, kernel)
      filtered_contours = self.finds_aux(erosion)
      sorted_contours = [contour for contour in filtered_contours if cv2.boundingRect(contour)[2] >= 40]
      sorted_contours = sorted(sorted_contours, key=lambda contour: cv2.boundingRect(contour)[1])
      x, y, w, h = cv2.boundingRect(sorted_contours[0])
      line_image = img[y:y + h, x:x+w]
      return line_image
    except:
      return img

  def draw_countours(self, img, contours):
    img_copy = img.copy()
    padding = 3
    for contour in contours:
      x, y, w, h = cv2.boundingRect(contour)
      x, y, w, h = (x-padding, y-padding, w+(padding*2), h+(padding*2)) 
      cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img_copy

  def texts_return(self, img, contours, debug=False):
    padding = 5
    imgs = []
    max_width = max_height = 0
    contours = sorted(contours, key=lambda ctr: (cv2.boundingRect(ctr)[1], -cv2.boundingRect(ctr)[0], cv2.boundingRect(ctr)[3]))
    imgs = []
    for contour in contours:
      x, y, w, h = cv2.boundingRect(contour)
      max_width = max(max_width, w)
      max_height = max(max_height, h)
    
    for contour in contours:
      x, y, w, h = cv2.boundingRect(contour)
      line_image = img[y:y + h, x:x+w]
      height, width = line_image.shape[:2]
      padding_x = max_width - width
      padding_y = max_height - height
      new_width = width + padding_x + 2 * padding
      new_height = height + padding_y + 2 * padding
      expanded_image = np.zeros((new_height, new_width), dtype=np.uint8) + 255
      expanded_image[padding:padding + height, padding:padding + width] = line_image
      imgs.append(expanded_image)
      self.images.append(expanded_image)
    return imgs

  def draw_lines(self, img, lines):
    img_copy = img.copy()
    for line in lines:
      x1, y1, x2, y2 = line[0]
      cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img_copy

  def erosionforLines(self, img):
    kernel = np.ones((1,30), np.uint8)
    erosion = cv2.erode(img, kernel)
    return erosion
