import cv2
import numpy as np
from v3.util.Helper import Helper


class ImageProcessor(Helper):

  def flatten_img(self, img):
    altura, largura = img.shape[:2]
    img_flat = img.reshape(altura * largura)
    return img_flat

  def segmenta2regioes(self, img):
    # for i in range(len(pixels)):
    #   if pixels[i] > 150:
    #     pixels[i] = 255
    #   else:
    #     pixels[i] = 0
    # seg_regiao = pixels.reshape(img.shape[0], img.shape[1])
    ret, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    return thresh


  def desfoqueGaussiano(self, img):
    

    img_flatten = self.flatten_img(img)
    img_blur = cv2.GaussianBlur(img_flatten, (5, 5), 0)
    adapt_gauss = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)
    
    self.images.append(adapt_gauss.reshape(img.shape[0], img.shape[1]))
    return adapt_gauss.reshape(img.shape[0], img.shape[1])

  def erode(self, img):
    kernel = np.ones((3,3), np.uint8)
    erosao = cv2.erode(img, kernel)
    return erosao

  def dilate(self, img):
    kernel = np.ones((3,3), np.uint8)
    dilate = cv2.dilate(img, kernel)
    return dilate
