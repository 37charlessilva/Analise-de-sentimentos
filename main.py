from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from naive_bayes import Naive_bayes
from dados import Dados
from graficos import Graficos
#from sklearn.metrics import plot_confusion_matrix

# nome das bases de dados
bases = ['Bases/b2w.csv', 'Bases/buscape.csv', 'Bases/olist.csv', 'Bases/utlc_apps.csv']

# inicializa a classe Naive_bayes
nv = Naive_bayes()

# inicializa a classe Graficos
graficos = Graficos

while True:
    # Menu inicial
    print("Escolha sua base de dados\n"
        "1: B2w\n"
        "2: Buscape\n"
        "3: Olist\n" 
        "4: Utlc_apps")
    i = int(input("Resposta: "))

    # Carregar os dados a partir de um arquivo CSV
    df = Dados(bases[i - 1])

    # Separar as características (X) e os rótulos (y)
    X = df.transform_data('review_text_processed')
    y = df.get_rating()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar modelo Naive Bayes
    nv.start(X_train, y_train)

    print(df.get_head())
    print('\nConjuntos de treinamento e teste separados!')
    print()
    print('Tamanho do conjunto de treinamento:', X_train.shape)
    print('Tamanho do conjunto de teste:', X_test.shape)

    # Avaliar o desempenho do modelo
    """accuracy = nv.accuracy_score(X_test, y_test)
    report = nv.classification_report(X_test, y_test)
    conf_matrix = nv.confusion_matrix(X_test, y_test)"""
    accuracy = nv.accuracy_score()
    report = nv.classification_report()
    conf_matrix = nv.confusion_matrix()

    print('\nAcurácia do modelo:', accuracy)
    print('\nRelatório de classificação:')
    print(report)
    print('\nMatriz de confusão:')
    print(conf_matrix)
    print("\n")
    
    nv_classes_ = df.get_classes()
    graficos.plot_confusion_matrix(conf_matrix, nv_classes_)
