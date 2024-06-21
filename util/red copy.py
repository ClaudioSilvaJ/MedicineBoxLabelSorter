import cv2
import os

def redimensionar_imagem(input_path):
    count = 0
    for nome_arquivo in os.listdir(input_path):
        if nome_arquivo.endswith('.jpg') or nome_arquivo.endswith('.png'):
            print(nome_arquivo)
            caminho_completo = os.path.join(input_path, nome_arquivo)
            image = cv2.imread(caminho_completo)

            altura_atual, largura_atual = image.shape[:2]
            largura_atual = 500
            altura_atual = 180
            image = cv2.resize(image, (int(largura_atual), int(altura_atual)))
          
            cv2.imwrite(input_path+"\\resize\\{}.png".format(count), image)
            count += 1

x = 36
tests = "v2\\files\\imgs_trat\\imgs\\Bad"

redimensionar_imagem(tests)
