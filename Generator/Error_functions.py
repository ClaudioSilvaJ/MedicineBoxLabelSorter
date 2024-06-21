import random
import numpy as np
import cv2 as cv
import math
import json
import os
import tqdm
#Lê todas as imagens da pasta good e aplicar distorções aleatórias

# Função para adicionar pontos aleatórios na imagem
def pontinhos(img):
        #define a quantidade de pontos
        for _ in range(random.randint(20, 100)):
            #define a cor do ponto , [187, 184, 175]
            cor_ponto = random.choice([[0, 0, 0], [187, 184, 175]])
            #define o posicionamento do ponto
            x, y= random.randint(230,1100), random.randint(230,550)
            #posiciona o ponto na imagem
            if cor_ponto == [0, 0, 0]:
                cv.circle(img, (x, y), 3, (cor_ponto), -1)
            else:
                cv.circle(img, (x, y), 10, (cor_ponto), -1)
        return img
#listar os diretórios Good
lista = os.listdir('MedicineBoxLabelSorter/Good')
# Loop para ler todas as imagens da pasta Good
for i in lista:

    #Ler imagem por imagem no diretório Good sem a necessidade de saber a quantidade de imagens
    img = cv.imread(f'MedicineBoxLabelSorter/Good/{i}')
    img = pontinhos(img)

    # #criando um array kernel 5x5 numeros inteiros aleatórios
    # kernel =  random.randint(-1 , 8, (5, 5))
    # Lê o arquivo kernel.json e seleciona uma matriz para usar como kernel
    with open('MedicineBoxLabelSorter/AlfaDB/kernel.json', 'r') as file:
        data = file.read()
        qntd_kernels = len(json.loads(data)) -1
        kernel = np.array(json.loads(data)[random.randint(0, qntd_kernels)])

    #escolhe uma distorção aleatória  **"rotacao"
    distortion_func = random.choice([cv.GaussianBlur, cv.medianBlur,  "ondulacao"])

    # Aplica a GaussianBlur usando um kernel diferente para cada distorção
    if distortion_func == cv.GaussianBlur:
        distorted_img = distortion_func(img, (5, 5), 30)
        # print(f'A matriz de kernel da imagem {i} é: ',kernel)
    
    #Aplica medianBlur usando um kernel diferente para cada distorção
    elif distortion_func == cv.medianBlur:
        distorted_img = distortion_func(img, 5)
        # print(f'A matriz de kernel da imagem {i} é: ',kernel)

    # Aplica a distorção de ondulação bidirecional (Horizontal + Vertical) **Wrapping distortion
    elif distortion_func == "ondulacao":
        #define a imagem de saída
        rows, cols, _ = img.shape
        #cria uma imagem preta
        img_output = np.zeros(img.shape, dtype=img.dtype)
        #aplica a distorção 
        for n in range(rows):
            for j in range(cols):
                offset_x = 1
                offset_y = int(7 * math.cos(2 * math.pi * j / 300))
                if n+offset_y < rows and j+offset_x < cols:
                    img_output[n,j] = img[(n+offset_y)%rows,(j+offset_x)%cols]
                else:
                    img_output[n,j] = 0
        distorted_img = img_output
    
        # #Rotaciona a imagem em 30 graus
        # else:
        #     #define a imagem de saída
        #     rows, cols, _ = img.shape
        #     #cria uma matriz de rotação
        #     M = cv.getRotationMatrix2D((cols / 2, rows / 2), (random.randint(10,30)), 1)
        #     #aplica a matriz de rotação
        #     distorted_img = cv.warpAffine(img, M, (cols, rows))
    #Mostrando as imagens distorcidas
    # cv.imshow(i, distorted_img)
    # cv.waitKey(0)
    #salva a imagem com o nome da distorção aplicada
    cv.imwrite(f"MedicineBoxLabelSorter/Bad/{i}.png", distorted_img)
