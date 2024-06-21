import cv2
import matplotlib.pyplot as plt
from filtros import filter_image
import numpy as np
from PIL import Image as img
from recorte import mark_text_regions


def predict_image(reader, image_path, debug=False):
  image = cv2.imread(image_path)
  #image = mark_text_regions(image_path, debug=debug)
  if debug:
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


  image = filter_image(image)

  if image is None:
      raise ValueError("Invalid image file or path.")

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (3, 3), 0) #Mexer aqui
  bw = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

  kernel_size = (20, 1) 
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
  bw_closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
  if debug:
      plt.imshow(cv2.cvtColor(bw_closed, cv2.COLOR_BGR2RGB))
      plt.show()
  
  contours, _ = cv2.findContours(bw_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  filtered_contours = [cnt for cnt in contours if (cv2.boundingRect(cnt)[2] / cv2.boundingRect(cnt)[3])>=3.0]
  min_width = 40


  sorted_contours = [contour for contour in filtered_contours if cv2.boundingRect(contour)[2] >= min_width]
  sorted_contours = sorted(sorted_contours, key=lambda contour: cv2.boundingRect(contour)[1])



  
  predict = []
  score = []
  padding = 3
  for contour in sorted_contours:
      if debug:
        print("Width: ", cv2.boundingRect(contour)[2])
        print("Height: ", cv2.boundingRect(contour)[3])
      x, y, w, h = cv2.boundingRect(contour)
      x, y, w, h = (x-padding, y-padding, w+(padding*2), h+(padding*2)) 
      line_image = bw[y:y + h, x:x+w]
      try:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,1))
        dilate_2 = cv2.dilate(line_image, kernel, iterations=1)
        blur_2 = cv2.GaussianBlur(dilate_2, (3, 3), 0)
      except:
        continue
      if debug:
        plt.imshow(cv2.cvtColor(blur_2, cv2.COLOR_BGR2RGB))
        plt.show()
      
      raw_predict = reader.readtext(blur_2)
      if(len(raw_predict)>0):
          cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
          predict.append(raw_predict[0][1])
          score.append(raw_predict[0][2])
          print("\nPredict: ", raw_predict[0][1])
          print("Score: ", raw_predict[0][2])

  if debug:
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
  return predict, score