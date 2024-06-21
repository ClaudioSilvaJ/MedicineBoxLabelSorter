from util.tratamentos_img import *
from util.find_lines import find_countours, draw_countours, draw_lines, erosionforLines, find_text, texts_return
import cv2
import easyocr
import time
from tensorflow.keras.models import load_model

modelo_carregado = load_model('modelo_1_90.h5')

def show_imgs(imgs):
  for img in imgs:
    show_img(img)

x = 255

patterns = ["F1234567890/:", "F1234567890/:", "V1234567890/", "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"]

# tests = ["images_ger/result_" + str(indice) + ".png" for indice in range(x)]

tests = [
    "files/images_to_test/img1.png",
    "files/images_to_test/img2.png",
    "files/images_to_test/img3.png",
    "files/images_to_test/img4.png",
    "files/images_to_test/img5.png",
    "files/images_to_test/img6.png",
    "files/images_to_test/img7.png",
    "files/images_to_test/img8.png",
    "files/images_to_test/img9.png",
    "files/images_to_test/img10.png",
    "files/images_to_test/img11.png",
]

means = []
reader = easyocr.Reader(['en'])
inicio = time.time()
for test in tests:
  img = cv2.imread(test, 0)
  seg_regiao = segmenta2regioes(img)
  text_area = find_text(seg_regiao)
  contours = find_countours(text_area)
  img_contours = draw_countours(text_area, contours)
  erosao_img = erosao(text_area)
  imgs_text = texts_return(erosao_img, contours)

  show_imgs([img, seg_regiao, text_area, img_contours, erosao_img])
  accu = []
  count = 0
  new_img = cv2.resize(erosao_img, (200,400))
  backto = cv2.cvtColor(new_img, cv2.COLOR_GRAY2BGR)
  backto = np.expand_dims(backto, axis=0)
  resultado = modelo_carregado.predict(backto)
  print(resultado)
  for(img) in imgs_text:
    #print("Texto encontrado:")
    show_img(img)
    if(count == 2):
      img = dilate(img)
      img = cv2.blur(img, (5,3))
      predict = reader.readtext(img, allowlist=patterns[count])
    else:
      img = cv2.blur(img, (5,3))
      predict = reader.readtext(img, allowlist=patterns[count])
    print(predict)
    count += 1
  

    show_img(img)
    try:
      accu.append(predict[0][2])
    except:
      accu.append(0)
  #print(accu)
  means.append(sum(accu)/len(accu))
fim = time.time()
tempo = fim - inicio
print("Tempo total de execução:")
print(tempo)
print("Tempo estimado de execução de 300:")
print((tempo/len(tests))*300)
results = [0,0]
for mean in means:
  print(mean)
  if mean > 0.5:
    results[0] += 1
    print("A imagem é legível")
  else:
    results[1] += 1
    print("A imagem não é legível") 

print(results)



#show_imgs([img, seg_regiao, text_area, erosao_img, dilate_img, erosao_test, img_contours])


