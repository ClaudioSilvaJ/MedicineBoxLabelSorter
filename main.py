from util.tratamentos_img import *
from util.find_lines import find_countours, draw_countours, draw_lines, erosionforLines, find_text, texts_return
import cv2
import easyocr
import time
import cProfile


def show_imgs(imgs):
  for img in imgs:
    show_img(img)



test = "files/new_usable_images/1.png"

means = []
reader = easyocr.Reader(['en'])


def main():
  
  img = cv2.imread(test, 0)
  seg_regiao = segmenta2regioes(img)
  #show_img(seg_regiao)
  text_area = find_text(seg_regiao)
  contours = find_countours(text_area)
  erosao_img = erosao(text_area)
  imgs_text = texts_return(erosao_img, contours)

  #show_imgs([img, seg_regiao, text_area, erosao_test, img_contours])
  for(img) in imgs_text:
    #print("Texto encontrado:")
    show_img(img)
    img = cv2.blur(img, (5,3))
    predict = reader.readtext(img)
    print(predict)
    show_img(img)

inicio = time.time()
main()
fim = time.time()
tempo = fim - inicio
print(tempo)
print((tempo)*300)
print(means)
results = [0,0]
for mean in means:
  print(mean)
  if mean > 0.38:
    results[0] += 1
    print("A imagem é um documento")
  else:
    results[1] += 1
    print("A imagem não é um documento") 

print(results)



#show_imgs([img, seg_regiao, text_area, erosao_img, dilate_img, erosao_test, img_contours])

