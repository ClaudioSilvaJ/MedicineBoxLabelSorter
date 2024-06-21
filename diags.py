import cv2
import easyocr
import time
import cProfile
from util.tratamentos_img import *
from util.find_lines import find_countours, draw_countours, draw_lines, erosionforLines, find_text, texts_return

def show_imgs(imgs):
    for img in imgs:
        show_img(img)

test = "files/new_usable_images/1.png"
means = []
reader = easyocr.Reader(['en'])

def main():
    fims = []
    inicio_total = time.time()

    inicio = time.time()
    img = cv2.imread(test, 0)
    fims.append(time.time() - inicio)

    inicio = time.time()
    seg_regiao = segmenta2regioes(img)
    fims.append(time.time() - inicio)

    inicio = time.time()
    text_area = find_text(seg_regiao)
    fims.append(time.time() - inicio)

    inicio = time.time()
    contours = find_countours(text_area)
    fims.append(time.time() - inicio)

    inicio = time.time()
    erosao_img = erosao(text_area)
    fims.append(time.time() - inicio)

    inicio = time.time()
    imgs_text = texts_return(erosao_img, contours)
    fims.append(time.time() - inicio)

    inicio = time.time()
    
    for img in imgs_text:
        show_img(img)
        predict = reader.readtext(img)
    
    fims.append(time.time() - inicio)
    print("Fim do OCR ", len(fims))

    fim_total = time.time()

    total_time = fim_total - inicio_total
    print("Tempo total:", total_time)

    total_percentage = 0
    for i, fim in enumerate(fims):
        percentage = (fim / total_time) * 100
        total_percentage += percentage
        print(f"Porcentagem de tempo gasto na função {i}: {percentage}%")
    
    print("Porcentagem total:", total_percentage)

cProfile.run('main()')

main()
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

