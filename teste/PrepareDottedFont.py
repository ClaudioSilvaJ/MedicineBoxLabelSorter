import cv2
import numpy as np

class PrepareDottedFont():

  def flatten_img(self, img):
    altura, largura = img.shape[:2]
    img_flat = img.reshape(altura * largura)
    return img_flat

  def segment(self, img):
    ret, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    return thresh

  def gaussian(self, img):
    img_flatten = self.flatten_img(img)
    img_blur = cv2.GaussianBlur(img_flatten, (5, 5), 0)
    adapt_gauss = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)
    return adapt_gauss.reshape(img.shape[0], img.shape[1])

  def erode(self, img):
    kernel = np.ones((3,3), np.uint8)
    erosao = cv2.erode(img, kernel)
    return erosao

  def dilate(self, img):
    kernel = np.ones((3,3), np.uint8)
    dilate = cv2.dilate(img, kernel)
    return dilate
