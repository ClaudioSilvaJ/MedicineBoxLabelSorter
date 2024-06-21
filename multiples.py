from util.tratamentos_img import *
from util.find_lines import find_countours, draw_countours, draw_lines, erosionforLines, find_text, texts_return
import cv2
import easyocr
import time

def show_imgs(imgs):
  for img in imgs:
    show_img(img)

x = 47

tests = ["files/new_usable_images/" + str(indice+1) + ".png" for indice in range(x)]


# tests = [
#     "images_to_test/img1.png",
#     "images_to_test/img2.png",
#     "images_to_test/img3.png",
#     "images_to_test/img4.png",
#     "images_to_test/img5.png",
#     "images_to_test/img6.png",
#     "images_to_test/img7.png",
#     "images_to_test/img8.png",
#     "images_to_test/img9.png",
#     "images_to_test/img10.png",
#     "images_to_test/img11.png",
# ]

means = []
reader = easyocr.Reader(['en'])
inicio = time.time()
for test in tests:
  img = cv2.imread(test, 0)
  seg_regiao = segmenta2regioes(img)
  #show_img(seg_regiao)
  text_area = find_text(seg_regiao)
  contours = find_countours(text_area)
  img_contours = draw_countours(text_area, contours)
  #show_img(img_contours)
  erosao_img = erosao(text_area)
  dilate_img = dilate(erosao_img)
  imgs_text = texts_return(erosao_img, contours)

  #show_imgs([img, seg_regiao, text_area, erosao_test, img_contours])
  accu = []
  for(img) in imgs_text:
    #print("Texto encontrado:")
    show_img(img)
    img = cv2.blur(img, (5,3))
    predict = reader.readtext(img)
    print(predict)
    show_img(img)
    try:
      accu.append(predict[0][2])
    except:
      accu.append(0)
  means.append(sum(accu)/len(accu))
fim = time.time()
tempo = fim - inicio
print("Tempo total: ")
print(tempo)
print("Tempo médio para 300 imagens: ")
print((tempo/len(tests))*300)
results = [0,0]
not_document = []
index = 0
for mean in means:
  if mean > 0.38:
    results[0] += 1
    #print("A imagem é valida")
  else:
    results[1] += 1
    not_document.append(index+1)
    #print("A imagem não é valida") 

  index += 1

print("Imagens que não são validas:")
print(not_document)
print(results)



#show_imgs([img, seg_regiao, text_area, erosao_img, dilate_img, erosao_test, img_contours])

