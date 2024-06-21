import cv2
import easyocr
import cv2
import numpy as np


class ModelOCR:
    def __init__(self):
        self.model = easyocr.Reader(['en'])
        self.patterns = ["F1234567890/:;", "F1234567890/:", "V1234567890/", "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"]

    def predict(self, image):
        return self.model.readtext(image)

    def preprocess(self, image):
        return image

    def prediction(self, prediction):
        return prediction[0][1]
    
    def confiability(self, prediction):
        try:
            return prediction[0][2]
        except:
            return 0

    def run(self, image):
        image = self.preprocess(image)
        prediction = self.predict(image)
        return prediction
    
    def predictions(self, imgs_text):
      predicts = []
      count = 0
      for img_text in imgs_text:
          if(count == 2):
            kernel = np.ones((3,3), np.uint8)
            img_text = cv2.dilate(img_text, kernel)
            img_text = cv2.blur(img_text, (5,3))
            predicts.append(self.confiability(self.run(img_text)))
          else:
            if(count < len(self.patterns)):
              img_text = cv2.blur(img_text, (5,3))
              predicts.append(self.confiability(self.run(img_text)))
          count += 1
      return predicts
