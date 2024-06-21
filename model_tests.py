from predict import predict_image
from utils import measure_similarity
from Levenshtein import distance
import easyocr

tests = {
    "images_to_test/img1.png": ["10:22", "F10/23", "V10/25", "L23K427"],
    "images_to_test/img2.png": ["05:41", "F10/23", "V10/25", "L23K428"],
    "images_to_test/img3.png": ["10:55", "F09/23", "V09/25", "L23J439"],
    "images_to_test/img4.png": ["12:00", "F10/23", "V10/25", "L23K427"],
    "images_to_test/img5.png": ["12:00", "F10/23", "V10/25", "L23K427"],
    "images_to_test/img6.png": ["12:00", "F10/23", "V10/25", "L23K427"],
}
expect_results = ["Legível", "Legível", "Legível", "Ilegível", "Ilegível", "Ilegível"]

letter_mapping = {
    'S': '5',
    'o': '0',
    'O': '0',
    'l': '1',
    'U': 'V'
}

reader = easyocr.Reader(['en'])
results = []

for image_path in tests.keys():
    predict, score = predict_image(reader, image_path)
    expected_values = tests.get(image_path)
    score_med = sum(score) / len(score)

    for i, pred in enumerate(predict):
        predict[i] = ''.join(letter_mapping.get(char, char) for char in pred)

    print("Previsão:", predict)
    print("Original:", expected_values)
    print("Media de Score:", score_med)

    if expected_values:
        similarities = []
        for i in range(len(predict)):
            sim_score = 1 - (distance(predict[i], expected_values[i]) / max(len(predict[i]), len(expected_values[i])))
            similarities.append(sim_score)

        similarity_format = measure_similarity(predict)
        average_similarity = sum(similarities) / len(similarities)
        print("Média de Similaridade:", average_similarity)
        print("Média de Simularidade com formato:", similarity_format)
    else:
        print("Não há valores esperados para a imagem fornecida.")

    med_total = (average_similarity + score_med) / 2
    print("Media Geral Total:", med_total)


    if(score_med < 0.5 and similarity_format < 0.5):
        results.append("Ilegível")
        print("Ilegível")
    elif((score_med > 0.5) or (similarity_format > 0.6)):
        print("Legível")
        results.append("Legível")

print("Resultados:", results)

matchs = sum(1 for x, y in zip(expect_results, results) if x == y)
print(matchs, "/", len(expect_results))
matchs_perc = (matchs / len(expect_results)) * 100

print("Porcentagem de acertos:", matchs_perc, "%")
