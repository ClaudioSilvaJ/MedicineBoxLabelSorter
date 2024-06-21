import cv2

from colorama import init as colorama_init
from colorama import Fore
from colorama import Style


import numpy as np
from PIL import Image
import easyocr
import joblib
import pygame
from pygame.locals import *

from util.tratamentos_img import segmenta2regioes, erosao, dilate
from util.find_lines import find_countours, draw_countours, find_text, texts_return

colorama_init()
modelo_carregado_svm = joblib.load('modelo_svm_v2.pkl')
reader = easyocr.Reader(['en'])

patterns = ["F1234567890/:", "F1234567890/:", "V1234567890/", "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"]

video_path = 'video.mp4'
video_capture = cv2.VideoCapture(video_path)

pygame.init()
screen_width, screen_height = 480,360
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()

x, y, width, height = 700, 525, 530, 180

frame_number = 0

while True:
    ret, frame = video_capture.read()
    if(frame_number % 10 == 0):
        if not ret:
            break
        
        roi_frame = frame[y:y+height, x:x+width]
    
        
        frame_resized = cv2.resize(roi_frame, (500, 180))
        img_test = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        img_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        seg_regiao = segmenta2regioes(img_gray)
        text_area = find_text(seg_regiao)
        contours = find_countours(text_area)
        img_contours = draw_countours(text_area, contours)
        erosao_img = erosao(text_area)
        imgs_text = texts_return(erosao_img, contours)
        
        count = 0
        accu = []
        #print(len(imgs_text))
        if(len(imgs_text) >= 3):
            for img in imgs_text:
                try:
                    if count == 2:
                        img = dilate(img)
                        img = cv2.blur(img, (5, 3))
                    
                    predict = reader.readtext(img, allowlist=patterns[count])
                    #print(predict)
                    count += 1
                    try:
                        accu.append(predict[0][2])
                    except:
                        accu.append(0)

                except Exception as e:
                    continue
                    #print("Erro ao processar a imagem:", e)

        try:
            confidence_mean = np.mean(accu)
            if(confidence_mean > 0.4):
                print(f"{Fore.GREEN}Imagem válida{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Imagem inválida{Style.RESET_ALL}")
        except:
            print("Imagem inválida")

        frame_resized = cv2.flip(frame_resized, 1) 
        frame_pygame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_pygame = np.rot90(frame_pygame)
        
        pygame_frame = pygame.surfarray.make_surface(frame_pygame)
        pygame_frame = pygame.transform.scale(pygame_frame, (screen_width, screen_height))
        
        screen.blit(pygame_frame, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == QUIT:
                video_capture.release()
                pygame.quit()
                cv2.destroyAllWindows()
                quit()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    frame_number += 1

video_capture.release()
cv2.destroyAllWindows()
