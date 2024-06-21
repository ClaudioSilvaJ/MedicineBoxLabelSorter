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