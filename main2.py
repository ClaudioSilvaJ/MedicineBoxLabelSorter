
import easyocr
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imgaug.augmenters as iaa
import mlflow
import mlflow.pyfunc
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os
from PIL import Image as img

expected_format = ["NN:NN", "LNN/NN", "LNN/NN", "LNNLNNN"]
expected_size = 4

letter_mapping = {
        'S': '5',
        'o': '0',
        'O': '0',
        'l': '1',
        'U': 'V',
        'z': '2',
        'I': '1',
    }

def measure_similarity(predicted):
    if len(predicted) != expected_size or len(predicted) == 0:
        return 0
    identify_F_or_hour(predicted)
    total_score = 0
    total_expected_chars = 0
    total_predicted_chars = 0

    for pred, exp in zip(predicted, expected_format):

        for p, e in zip(pred, exp):
            if e == 'N' and p.isdigit():
                total_score += 1
            elif e == 'L' and p.isalpha():
                total_score += 1
            elif e == p:
                total_score += 1
            else:
                total_score -= 1

            if e in ('N', 'L'):
                total_expected_chars += 1
            if p.isdigit() or p.isalpha():
                total_predicted_chars += 1
            if(len(pred) != len(exp)):
                print("Predict: ", pred)
                print("Expected: ", exp)
                print("Tamanhos diferentes")
                total_score -= 0.15

    similarity = total_score / max(total_expected_chars, total_predicted_chars)
    return similarity


def identify_F_or_hour(predicted):
    print("Predicted: ", predicted[0])
    print("Predicted: ", predicted[0][1])
    if(predicted[0][0] == "F"):
        aux = predicted[0]
        predicted[0] = predicted[1]
        predicted[1] = aux
    return predicted


def replace_chars(predict):
    for i, pred in enumerate(predict):
        predict[i] = ''.join(letter_mapping.get(char, char) for char in pred)
    return predict


def mark_text_regions(image_path, debug=False):
    image = cv2.imread(image_path)
    original_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobelx, sobely)
    _, binary_image = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    closed_image = cv2.convertScaleAbs(closed_image)
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    #largest_contours = contours[:2]
    cv2.drawContours(original_image, contours, -1, (0, 255, 0), 2)
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h
    padding = 8
    x_min -= padding
    y_min -= padding
    x_max += padding
    y_max += padding

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.shape[1], x_max)
    y_max = min(image.shape[0], y_max)

    cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    text_region = image[y_min:y_max, x_min:x_max]

    if debug:
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    return text_region


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


def predict_image(reader, image_path, debug=False):
  image = cv2.imread(image_path)
  if debug:
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
  image = filter_image(image)

  if image is None:
      raise ValueError("Invalid image file or path.")

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (3, 3), 0) #Mexer aqui
  bw = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

  kernel_size = (20, 1) 
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
  bw_closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
  if debug:
      plt.imshow(cv2.cvtColor(bw_closed, cv2.COLOR_BGR2RGB))
      plt.show()
  
  contours, _ = cv2.findContours(bw_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  filtered_contours = [cnt for cnt in contours if (cv2.boundingRect(cnt)[2] / cv2.boundingRect(cnt)[3])>=3.0]
  min_width = 40
  sorted_contours = [contour for contour in filtered_contours if cv2.boundingRect(contour)[2] >= min_width]
  sorted_contours = sorted(sorted_contours, key=lambda contour: cv2.boundingRect(contour)[1])
  predict = []
  score = []
  padding = 3
  for contour in sorted_contours:
      if debug:
        print("Width: ", cv2.boundingRect(contour)[2])
        print("Height: ", cv2.boundingRect(contour)[3])
      x, y, w, h = cv2.boundingRect(contour)
      x, y, w, h = (x-padding, y-padding, w+(padding*2), h+(padding*2)) 
      line_image = bw[y:y + h, x:x+w]
      try:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,1))
        dilate_2 = cv2.dilate(line_image, kernel, iterations=1)
        blur_2 = cv2.GaussianBlur(dilate_2, (3, 3), 0)
      except:
        continue
      if debug:
        plt.imshow(cv2.cvtColor(blur_2, cv2.COLOR_BGR2RGB))
        plt.show()
      
      raw_predict = reader.readtext(blur_2)
      if(len(raw_predict)>0):
          cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
          predict.append(raw_predict[0][1])
          score.append(raw_predict[0][2])
          print("\nPredict: ", raw_predict[0][1])
          print("Score: ", raw_predict[0][2])

  if debug:
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
  return predict, score


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('process_file', filename=filename))
    return render_template('upload.html')

@app.route('/process/<filename>')
def process_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    reader = easyocr.Reader(['en'])

    mlflow.set_tracking_uri("http://mlflow:5001")
    mlflow.set_experiment("OCR Experiment")

    with mlflow.start_run():
        predict, score = predict_image(reader, filepath)
        try:
            score_med = sum(score) / len(score)
        except:
            score_med = 0
        predict = replace_chars(predict)
        similarity_format = measure_similarity(predict)
        
        if (score_med > 0.5) or (similarity_format > 0.5):
            result = "Legível"
        else:
            result = "Ilegível"

        mlflow.log_param("filename", filename)
        mlflow.log_metric("average_score", score_med)
        mlflow.log_metric("similarity_format", similarity_format)
        mlflow.log_param("result", result)

        with open("results.txt", "w") as f:
            f.write(f"Filename: {filename}\n")
            f.write(f"Result: {result}\n")
            f.write(f"Average Score: {score_med}\n")
            f.write(f"Similarity Format: {similarity_format}\n")
        mlflow.log_artifact("results.txt")

    return f"Processed {filename} with result: {result}"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)






