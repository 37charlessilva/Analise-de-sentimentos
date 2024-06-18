from sklearn.model_selection import train_test_split
from naive_bayes import Naive_bayes
from svm import SVM_model
from dados import Dados
import graficos
from time import sleep

# nome das bases de dados
bases = ['Bases/b2w.csv', 'Bases/buscape.csv', 'Bases/olist.csv', 'Bases/utlc_apps.csv']

# inicializa a classe Naive_bayes
nv = Naive_bayes()

# inicializa a classe SVM_model
sv =  SVM_model()

while True:
    # Menu inicial
    print("Escolha sua base de dados\n"
        "1: B2w\n"
        "2: Buscape\n"
        "3: Olist\n" 
        "4: Utlc_apps")
    i = int(input("Resposta: "))

    # carrega os dados a partir de um arquivo CSV
    df = Dados(bases[i - 1])
    sleep(0.5)

    # Seperando o conjunto de treino e teste
    df.trainig_data()

    print('\nConjuntos de treinamento e teste separados!')
    print('Tamanho do conjunto de treinamento:', df.get_train_x().shape)
    print('Tamanho do conjunto de teste:', df.get_test_x().shape)

    nv.start(df.get_train_x(), df.get_test_x(), df.get_train_y(), df.get_test_y())

    # avalia o desempenho do modelo Naive_bayes
    accuracy = nv.accuracy_score()
    report = nv.classification_report()
    conf_matrix = nv.confusion_matrix()

    print('\nAcurácia do modelo nv:', accuracy)
    print('\nRelatório de classificação:')
    print(report)
    print('\nMatriz de confusão:')
    print(conf_matrix)
    print("\n")

    sv.start(df.get_train_x(), df.get_test_x(), df.get_train_y(), df.get_test_y())

    # avalia o desempenho do modelo SVM_model
    accuracy1 = sv.accuracy_score()
    report1 = sv.classification_report()
    conf_matrix1 = sv.confusion_matrix()

    print('\nAcurácia do modelo sv:', accuracy1)
    print('\nRelatório de classificação:')
    print(report1)
    print('\nMatriz de confusão:')
    print(conf_matrix1)
    print("\n")

    # implementa grafico da matriz de confusão
    nv_classes_ = df.get_classes()
    graficos.plot_confusion_matrix(conf_matrix, nv_classes_)
