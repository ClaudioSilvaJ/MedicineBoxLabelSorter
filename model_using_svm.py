from util.tratamentos_img import *
from util.find_lines import find_countours, draw_countours, draw_lines, erosionforLines, find_text, texts_return
import cv2
import easyocr
import time
import joblib
from PIL import Image

modelo_carregado_svm = joblib.load('modelo_svm_v2.pkl')

def show_imgs(imgs):
  for img in imgs:
    show_img(img)

numbers_imgs = 47

patterns = ["F1234567890/:", "F1234567890/:", "V1234567890/", "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"]

# tests = ["images_ger/result_" + str(indice) + ".png" for indice in range(x)]

tests = ["files/new_usable_images/" + str(indice+1) + ".png" for indice in range(numbers_imgs)]

results_list_svm = []
means = []
reader = easyocr.Reader(['en'])
inicio = time.time()


for test in tests:
  print(test)
  img_test = Image.open(test)
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
  
  nova_imagem_redimensionada = cv2.resize(img, (500, 180))

  img_test = img_test.resize((500, 180))
  imagem_array = np.array(img_test)
  imagem_achatada = imagem_array.flatten()
  imagem_reshaped = imagem_achatada.reshape(1, -1)
  try:
    resultado = modelo_carregado_svm.predict(imagem_reshaped)
  except:
    print("Erro ao processar a imagem")
    continue


  print(resultado)
  results_list_svm.append(resultado)
  for(img) in imgs_text:
    #print("Texto encontrado:")
    show_img(img)
    if(count == 2):
      img = dilate(img)
      img = cv2.blur(img, (5,3))
      predict = reader.readtext(img, allowlist=patterns[count])
    else:
      if(count < len(patterns)):
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
results_easyocr = [0,0]
certas_easyocr = []
erros_easyocr = []
for mean in means:
  print(mean)
  if mean > 0.5:
    results_easyocr[0] += 1
    certas_easyocr.append(sum(results_easyocr))
    #print("A imagem é legível")
  else:
    results_easyocr[1] += 1
    erros_easyocr.append(sum(results_easyocr))
    #print("A imagem não é legível") 



certas_svm = []
erros_svm = []

results_svm = [0,0]
for resultSVM in results_list_svm:
  if(resultSVM[0] == 1):
    #print("A imagem é legível")
    results_svm[0] += 1
    certas_svm.append(sum(results_svm))
  else:
    #print("A imagem não é legível")
    results_svm[1] += 1
    erros_svm.append(sum(results_svm))

print("Modelo svm:")
print(certas_svm)
print(erros_svm)
print(results_svm)

print("Modelo easyocr:")
print(certas_easyocr)
print(erros_easyocr)
print(results_easyocr)

print("Tempo total de execução:")
print(tempo)
print("Tempo estimado de execução de 300:")
print((tempo/len(tests))*300)