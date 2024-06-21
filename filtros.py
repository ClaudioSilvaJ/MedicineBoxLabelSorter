import cv2
import imgaug.augmenters as iaa

def sharpen(img):
  sharpen = iaa.Sharpen(alpha=1.0, lightness = 1.0)
  sharpen_img = sharpen.augment_image(img)
  return sharpen_img

def filter_image(imagem):
  sharpened_image = sharpen(imagem)
  _, binary_image = cv2.threshold(sharpened_image, 1, 255, cv2.THRESH_BINARY)

  thresh = cv2.threshold(binary_image, 115, 255, cv2.THRESH_BINARY_INV)[1]
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,2))
  dilate_2 = cv2.dilate(thresh, kernel, iterations=1)
  final = cv2.threshold(dilate_2, 115, 255, cv2.THRESH_BINARY_INV)[1]
  return final

# plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()

# cv2.imwrite('img_tratada.png', final)