import random
import numpy as np
import string
import cv2 as cv
import os
import json
import math
from PIL import Image, ImageDraw, ImageFont
import tqdm
import pandas as pd

#apagar as imagens da pasta god e bad
def remove_files(directory):
    #Remove todos os arquivos em um diretório.
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Removendo arquivos da pasta "Good"
remove_files("MedicineBoxLabelSorter/Generator/Good")
# Removendo arquivos da pasta "Bad"
remove_files("MedicineBoxLabelSorter/Generator/Bad")
    
# Função para gerar texto aleatório com números aleatórios
def generate_random_text():
    #template do texto a ser gerado
    datas = 'F05/27      67:11\nV18/30\nL23K248'
    #randomiza os caracteres do texto
    random_lote = ''.join(random.choice(string.ascii_uppercase) if char.isalpha() and char not in ["F", "V"] else char for char in datas)
    #randomiza os números do texto (inclusive datas, mesmo que invalidas tem o intuíto de testar o maior número de combinações)
    random_text = ''.join(random.choice(string.digits) if char.isdigit() else char for char in random_lote)
    return random_text

# Função para adicionar pontos aleatórios na imagem
def print_white_points(img):
        #define a quantidade de pontos
        for _ in range(random.randint(20, 100)):
            #define a cor do ponto , [187, 184, 175]
            cor_ponto = random.choice([[0, 0, 0], [186, 184, 185]])
            #define o posicionamento do ponto
            x, y= random.randint(230,1800), random.randint(230,700)
            #posiciona o ponto na imagem
            if cor_ponto == [0, 0, 0]:
                cv.circle(img, (x, y), 6, (cor_ponto), -1)
            else:
                cv.circle(img, (x, y), 15, (cor_ponto), -1)
        return img

# Quantidade de imagens a serem geradas
num_img = 10
# Cria um db para os armazenar os ground truth
df_labels = pd.DataFrame(columns=['ID', 'Ground_Truth'])

with tqdm.tqdm(total=num_img*2) as pbar:
    # Loop para gerar imagens boas
    for i in range(num_img):

        # Abre a imagem fonte
        img = Image.open("MedicineBoxLabelSorter/Generator/Imagem_base.jpg") 
        medice_box = ImageDraw.Draw(img)
        #cria um numero rândomico para o tamanho da fonte
        font_size = random.randint(72,90)
        #define a fonte usando o path do arquivo true type e tamanho da fonte
        myFont = ImageFont.truetype('MedicineBoxLabelSorter/Generator/5by7.ttf', font_size)
        #chama a função para gerar o texto
        text = generate_random_text()
        #escreve o texto na imagem
        # medice_box.multiline_text((800, 350), text, fill=(3, 11, 12), font=myFont)
        medice_box.multiline_text((random.randint(230,800), random.randint(230,350)), text, fill=(3, 11, 12), font=myFont)
        #salva a imagem
        img.save(f"MedicineBoxLabelSorter/Generator/Good/{i}.png")
        # img.save(f"MedicineBoxLabelSorter\\Good\\1.png")
        #armazena no df_labels o ground truth
        text_value = text.replace("      ", "").replace("\n","")
        df_labels = pd.concat([pd.DataFrame([[i,text_value]], columns=df_labels.columns), df_labels], ignore_index=True)
        pbar.update(1)
    #listar os diretórios Good
    lista = os.listdir('MedicineBoxLabelSorter/Generator/Good')

    # Loop para ler todas as imagens da pasta Good
    for i in lista:

        #Ler imagem por imagem no diretório Good
        img = cv.imread(f'MedicineBoxLabelSorter/Generator/Good/{i}')
        img = print_white_points(img)

        # Lê o arquivo kernel.json e seleciona uma matriz para usar como kernel
        with open('MedicineBoxLabelSorter/Generator/kernel.json', 'r') as file:
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

        #Aplica a distorção de ondulação bidirecional (Horizontal + Vertical) **Wrapping distortion
        elif distortion_func == "ondulacao":
            #define a imagem de saída
            rows, cols, _ = img.shape
            #cria uma imagem de destino
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
        pbar.update(1)
        #salva a imagem com o nome da distorção aplicada
        cv.imwrite(f"MedicineBoxLabelSorter/Generator/Bad/{i}.png", distorted_img)
df_labels.to_csv("MedicineBoxLabelSorter/Generator/Labels.csv", index=False)