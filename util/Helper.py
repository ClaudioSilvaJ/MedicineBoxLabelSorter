import numpy as np
from PIL import Image

class Helper(object):
    def __init__(self, debug=False):
        self.debug = debug
        self.images = []

    def show_img(self, img):
        if not self.debug:
            return
        pil_img = Image.fromarray(img)
        pil_img.show()

    def display_accumulated_images(self):
        if not self.debug:
            print("Debug mode não está habilitado para acumular imagens.")
            return
        num_images = len(self.images)
        if num_images == 0:
            print("Nenhuma imagem acumulada para exibir.")
            return
        for i in range(num_images):
            pil_img = Image.fromarray(self.images[i])
            pil_img.show()
