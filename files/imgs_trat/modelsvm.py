from PIL import Image
import os
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
import joblib
from sklearn.metrics import accuracy_score, classification_report

def carregar_imagens(tamanho_alvo=(500, 180)):
    imagens = []
    labels = []
    diret_bad = "files/imgs_trat/imgs/Bad/"
    diret_good = "files/imgs_trat/imgs/Good/"

    for nome_arquivo in os.listdir(diret_bad):
        if nome_arquivo.endswith('.jpg') or nome_arquivo.endswith('.png'):
            caminho_completo = os.path.join(diret_bad, nome_arquivo)
            try:
                imagem = Image.open(caminho_completo)
                imagem = imagem.resize(tamanho_alvo)
                imagem_array = np.array(imagem)
                imagens.append(imagem_array.flatten())
                labels.append(0)
            except Exception as e:
                print(f"Erro ao processar {caminho_completo}: {e}")

    for nome_arquivo in os.listdir(diret_good):
      if nome_arquivo.endswith('.jpg') or nome_arquivo.endswith('.png'):
          caminho_completo = os.path.join(diret_good, nome_arquivo)
          try:
              imagem = Image.open(caminho_completo)
              imagem = imagem.resize(tamanho_alvo)
              imagem_array = np.array(imagem)
              imagens.append(imagem_array.flatten())
              labels.append(1)
          except Exception as e:
              print(f"Erro ao processar {caminho_completo}: {e}")

    return np.array(imagens), np.array(labels)

imagens, labels = carregar_imagens()
print(labels)


X_train, X_test, y_train, y_test = train_test_split(imagens, labels, test_size=0.2, random_state=42)

n_samples, n_features = X_train.shape
print("Número de amostras (n_samples):", n_samples)
print("Número de características (n_features):", n_features)

pipeline = Pipeline([
    ('feature_selection', SelectKBest()), 
    ('svm', SVC()),
])

parameters = {
    'feature_selection__k': [10, 20, 50],
    'svm__C': [0.1, 1, 2, 3, 4, 5, 10, 50, 100],
    'svm__gamma': [1, 0.1, 0.01, 0.001],
    'svm__kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy', cv=5)
grid_search.fit(X_train.reshape(len(X_train), -1), y_train) 

print("Melhores parâmetros encontrados:\n", grid_search.best_params_)

predictions = grid_search.predict(X_test)
print(classification_report(y_test, predictions))
#classifier = svm.SVC(kernel='linear')
# classifier = svm.LinearSVC(dual="auto")
# classifier.fit(X_train.reshape(len(X_train), -1), y_train) 
# y_pred = classifier.predict(X_test.reshape(len(X_test), -1))
# accuracy = accuracy_score(y_test, y_pred)
# print("Acurácia do modelo SVM: {:.2f}%".format(accuracy * 100))
# joblib.dump(classifier, 'modelo_svm.pkl')