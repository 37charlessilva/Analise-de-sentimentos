from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from dados import Dados
import pandas as pd

# Carregar os dados a partir de um arquivo CSV
df = Dados('Bases/b2w.csv')

# Transformar os textos em vetores numéricos

def alterardados(n):
    if n <= 2:
        return 1
    elif n == 3:
        return 2
    else:
        return 3

X = df.transform_data('review_text_processed')

# Separar as características (X) e os rótulos (y)
y = df.get_coluna('rating').map(alterardados)


# Dividir os dados em conjuntos de treinamento e teste
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print('Conjuntos de treinamento e teste separados!')
print('Tamanho do conjunto de treinamento:', train_x.shape)
print('Tamanho do conjunto de teste:', test_x.shape)

# Treinar o modelo Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(train_x, train_y)

# Fazer previsões no conjunto de teste
test_y_pred = nb_model.predict(test_x)

# Avaliar o desempenho do modelo
accuracy = accuracy_score(test_y, test_y_pred)
report = classification_report(test_y, test_y_pred)
conf_matrix = confusion_matrix(test_y, test_y_pred)

print('Acurácia do modelo:', accuracy)
print('Relatório de classificação:')
print(report)
print('Matriz de confusão:')
print(conf_matrix)