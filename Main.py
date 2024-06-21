import sys
sys.path.append('c:/Users/Administrador/Documents/0 - Organização/Faculdade/3º Semestre/MedicineBoxLabelSorter/')
sys.path.append('/mnt/c/Users/Administrador/Documents/0 - Organização/Faculdade/3º Semestre/MedicineBoxLabelSorter/')
from util.FindLines import FindLines
from util.ImageProcessor import ImageProcessor
from util.ModelOCR import ModelOCR
from util.Helper import Helper
import numpy as np
import os
from PIL import Image
import time

class Main:
    def __init__(self, ImageProcessor, FindLines, debug=False):
        self.debug = debug
        self.ImageProcessor = ImageProcessor
        self.FindLines = FindLines
        self.Not_Readable = []
        self.Readable = []
        self.images = []

    def run(self):
      modelOCR = ModelOCR()
      filenames = self.loadingImagens()
      #filenames = [filenames[0]]
      
      start = time.time()
      for img_array, filename  in zip(self.images, filenames):
        predicts = self.useModel(img_array, modelOCR)
        if(self.debug): print(f"Predictions for {filename}: {predicts}")
        result = self.classify(predicts, filename)
        print(f"Imagem {filename} é {result}")
      end = time.time()
      print(f"Tempo de execução: {end-start} segundos")
      print(f"Tempo de execução para 300: {((end-start)/(len(self.Readable) + len(self.Not_Readable)))*300} segundos")

    def useModel(self, img, model):
        seg_regiao = self.ImageProcessor.segmenta2regioes(img)
        Helper.show_img(self, seg_regiao)
        text_area = self.FindLines.find_text(seg_regiao)
        contours = self.FindLines.find_countours(text_area)
        erosao_img = self.ImageProcessor.erode(text_area)
        imgs_text = self.FindLines.texts_return(erosao_img, contours)
        return model.predictions(imgs_text)
    
    def loadingImagens(self):
        directory = 'assets/images/'
        filenames = os.listdir(directory)
        for filename in os.listdir(directory):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                img_path = os.path.join(directory, filename)
                img = Image.open(img_path).convert('L')
                img_array = np.array(img)
                self.images.append(img_array)
        return filenames
    
    def classify(self, predicts, filename):
        if np.mean(predicts) > 0.45:
            self.Readable.append(filename)
            return "Legível"
        else:
            self.Not_Readable.append(filename)
            return "Ilegível"


if __name__ == "__main__":
    debugMode = False
    helper = Helper(debugMode)
    imageProcessor = ImageProcessor(debugMode)
    findLines = FindLines(debugMode)
    main_instance = Main(imageProcessor, findLines, debug=debugMode)
    main_instance.run()
    print(f"Legíveis: {main_instance.Readable}")
    print(f"Ilegíveis: {main_instance.Not_Readable}")
    print(f"Total lido: {len(main_instance.Readable) + len(main_instance.Not_Readable)} imagens")

