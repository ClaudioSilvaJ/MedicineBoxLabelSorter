import cv2

def redimensionar_imagem(input_path):
    image = cv2.imread("images_to_test/"+input_path)

    altura_atual, largura_atual = image.shape[:2]
    largura_atual = 500
    altura_atual = 180
    image = cv2.resize(image, (int(largura_atual), int(altura_atual)))
  
    cv2.imwrite("images_to_test/"+input_path, image)

x = 255
# tests = ["img" + str(indice) + ".png" for indice in range(x)]
tests = [
    "result_2.jpg",
    "result_3.jpg",
]

for test in tests:
  redimensionar_imagem(test)
