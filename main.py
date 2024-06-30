from joblib import load, dump
from sklearn.ensemble import RandomForestClassifier
from naive_bayes import Naive_bayes
from svm import SVM_model
from dados import Dados
import graficos

# nome das bases de dados e modelos
bases = ['Bases/b2w', 'Bases/buscape', 'Bases/olist', 'Bases/utlc_apps']
s = ['svm_model', 'random_forest_model']

# inicializa a classe Naive_bayes
nv = Naive_bayes()

# inicializa a classe SVM_model
sv = SVM_model()

# inicializa o modelo random forest
rd = RandomForestClassifier()

modelos = [nv, sv, rd]
while True:
    # Menu inicial
    print("Escolha sua base de dados\n"
          "1: B2w\n"
          "2: Buscape\n"
          "3: Olist\n"
          "4: Utlc_apps")
    i = int(input("Resposta: "))

    # carrega os dados a partir de um arquivo CSV
    df = Dados(bases[i - 1] + ".csv")

    # Seperando o conjunto de treino e teste
    df.trainig_data()

    print('\nConjuntos de treinamento e teste separados!')
    print('Tamanho do conjunto de treinamento:', df.get_train_x().shape)
    print('Tamanho do conjunto de teste:', df.get_test_x().shape)

    # Treinamento do Naive bayes
    nv.start(df.get_train_x(), df.get_test_x(), df.get_train_y(), df.get_test_y())
    print("\nModelo Naive bayes treinado\n")

    # Treinamento do SVM e random forest
    for c in range(1, len(modelos)):
        if df.verify(f"{bases[i - 1]}_{s[c - 1]}_.pk1") == False:
            print(f"{s[c - 1].upper()} em treinamento, pode levar algum tempo\n")
            modelos[c].start(df.get_train_x(), df.get_test_x(), df.get_train_y(), df.get_test_y())
            dump(modelos[c], f"{bases[i - 1]}_{s[c - 1]}_.pk1")
            print(f"{s[c - 1].upper()} treinado")
        else:
            print(f"Modelo {s[c - 1]} recuperado\n")
            modelos[c] = load(f"{bases[i - 1]}_{s[c - 1]}_.pk1")

    # Treinamento (supondo que 'start' seja um método válido para esses modelos)
    sv.start(df.get_train_x(), df.get_test_x(), df.get_train_y(), df.get_test_y())
    rd.start(df.get_train_x(), df.get_test_x(), df.get_train_y(), df.get_test_y())

    # Salvando os modelos treinados
    dump(sv, f"{bases[i - 1]}_{s[0]}_.pk1")
    dump(rd, f"{bases[i - 1]}_{s[1]}_.pk1")

    # Carregando os modelos treinados
    sv = load(f"{bases[i - 1]}_{s[0]}_.pk1")
    rd = load(f"{bases[i - 1]}_{s[1]}_.pk1")

    sv = modelos[1]
    rd = modelos[2]

    # avalia o desempenho do modelo Naive_bayes
    accuracy = nv.accuracy_score()
    report = nv.classification_report()
    conf_matrix = nv.confusion_matrix()

    # avalia o desempenho do modelo SVM_model
    accuracy1 = sv.accuracy_score()
    report1 = sv.classification_report()
    conf_matrix1 = sv.confusion_matrix()

    accuracy2 = rd.accuracy_score()
    report2 = rd.classification_report()
    conf_matrix2 = rd.confusion_matrix()

    while i != 0:
        print("\n0: Para voltar\n"
              "1: Comparacao de acuracia\n"
              "2: Comparacao dos relatorios de classificacao\n"
              "3: Matriz de confusao")
        i = int(input("Resposta: "))
        if i == 1:
            graficos.plot_model_comparison(df.get_train_x().shape, df.get_test_x().shape, accuracy, accuracy1, accuracy2)
        elif i == 2:
            classes = ["1", "2", "3"]
            graficos.plot_classification_reports(report, report1, report2, classes, "nv", "sv", "rd")
        elif i == 3:
            graficos.plot_confusion_matrix(conf_matrix, df.get_classes())
            graficos.plot_confusion_matrix(conf_matrix1, df.get_classes())
            graficos.plot_confusion_matrix(conf_matrix2, df.get_classes())
        print()


