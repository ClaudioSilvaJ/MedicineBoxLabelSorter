from util.tratamentos_img import *
from util.find_lines import find_countours, draw_countours, draw_lines, erosionforLines, find_text, texts_return
import cv2
import easyocr
import time
import cProfile


def show_imagems(imagems):
  for imagem in imagems:
    show_img(imagem)



test = "files/new_usable_images/1.png"

means = []
reader = easyocr.Reader(['en'])


def main():
  
  imagem = cv2.imread(test, 0)
  seg_regiao = segmenta2regioes(imagem)
  #show_imagem(seg_regiao)
  text_area = find_text(seg_regiao)
  contours = find_countours(text_area)
  erosao_imagem = erosao(text_area)
  imagems_text = texts_return(erosao_imagem, contours)

  imagems_blur = []
  for img in imagems_text:
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    imagems_blur.append(blurred_img)

  predictions = reader.readtext_batched(imagems_blur)

  for imagem, predict in zip(imagems_text, predictions):
    show_img(imagem)
    print(predict)

cProfile.run('main()')
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



#show_imagems([imagem, seg_regiao, text_area, erosao_imagem, dilate_imagem, erosao_test, imagem_contours])

