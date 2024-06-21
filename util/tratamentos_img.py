import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def show_img(img):
  # plt.imshow(img, cmap='gray')
  # plt.show()
  pass

def flatten_img(img):
  altura, largura = img.shape[:2]
  img_flat = img.reshape(altura * largura)
  return img_flat

def segmenta2regioes(img):
  inicio = time.time()
  pixels = flatten_img(img).copy()
  # for i in range(len(pixels)):
  #   if pixels[i] > 150:
  #     pixels[i] = 255
  #   else:
  #     pixels[i] = 0
  # seg_regiao = pixels.reshape(img.shape[0], img.shape[1])
  ret, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
  fim = time.time()
  print(fim - inicio)
  return thresh

def segmenta2regioes2(img):
  pixels = flatten_img(img).copy()
  for i in range(len(pixels)):
    if pixels[i] > 150:
      pixels[i] = 255
    else:
      pixels[i] = 0
  seg_regiao = pixels.reshape(img.shape[0], img.shape[1])
  return seg_regiao

def desfoqueGaussiano(img):
  img_flatten = flatten_img(img)
  img_blur = cv2.GaussianBlur(img_flatten, (5, 5), 0)
  adapt_gauss = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)
  return adapt_gauss.reshape(img.shape[0], img.shape[1])

def erosao(img):
  kernel = np.ones((3,3), np.uint8)
  erosao = cv2.erode(img, kernel)
  return erosao

def dilate(img):
  kernel = np.ones((3,3), np.uint8)
  dilate = cv2.dilate(img, kernel)
  return dilate
