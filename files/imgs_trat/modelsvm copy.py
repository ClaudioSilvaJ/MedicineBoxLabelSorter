from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report
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

svm_model = SVC(C=0.1, gamma=1, kernel='linear')


feature_selector = SelectKBest(f_classif)


final_model = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', feature_selector),
    ('svm', svm_model)
])
final_model.fit(X_train, y_train)

predictions = final_model.predict(X_test)



print("Relatório de Classificação:")
print(classification_report(y_test, predictions))
joblib.dump(final_model, 'modelo_svm_v3.pkl')