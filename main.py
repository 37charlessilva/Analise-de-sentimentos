from joblib import load, dump
from naive_bayes import Naive_bayes
from svm import SVM_model
from dados import Dados
import graficos

# nome das bases de dados
bases = ['Bases/b2w', 'Bases/buscape', 'Bases/olist', 'Bases/utlc_apps']
s = ['svm_model']
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
    df = Dados(bases[i - 1] + ".csv")

    # Seperando o conjunto de treino e teste
    df.trainig_data()

    print('\nConjuntos de treinamento e teste separados!')
    print('Tamanho do conjunto de treinamento:', df.get_train_x().shape) 
    print('Tamanho do conjunto de teste:', df.get_test_x().shape)
    
    #Treinamento do Naive bayes
    nv.start(df.get_train_x(), df.get_test_x(), df.get_train_y(), df.get_test_y()) 
    print("\nModelo Naive bayes treinado\n")

    # Treinamento do SVM
    if(df.verify(f"{bases[i - 1]}_{s[0]}_.pk1") == False):
        print("Modelo SVM sendo treinado, pode levar algum tempo")
        sv.start(df.get_train_x(), df.get_test_x(), df.get_train_y(), df.get_test_y())
        dump(sv, f"{bases[i - 1]}_{s[0]}_.pk1")
        print("Modelo SVM treinado")
    else:
        print("\nModelo SVM recuperado\n")
        sv = load(f"{bases[i - 1]}_{s[0]}_.pk1")
        
    # avalia o desempenho do modelo Naive_bayes
    accuracy = nv.accuracy_score()
    report = nv.classification_report()
    conf_matrix = nv.confusion_matrix()

    # avalia o desempenho do modelo SVM_model
    accuracy1 = sv.accuracy_score()
    report1 = sv.classification_report()
    conf_matrix1 = sv.confusion_matrix()

    while(i != 0):
        print("\n0: Para voltar\n"
            "1: Comparacao de acuracia\n"
            "2: Comparacao dos relatorios de classificacao\n"
            "3: Matriz de confusao")
        i = int(input("Resposta: "))
        if(i == 1):
            graficos.plot_model_comparison(df.get_train_x().shape, df.get_test_x().shape, accuracy, accuracy1)
        elif(i == 2):
            classes = ["1", "2", "3"]
            graficos.plot_classification_reports(report, report1, classes, "nv", "svm")
        elif(i == 3):
            graficos.plot_confusion_matrix(conf_matrix, df.get_classes())
            graficos.plot_confusion_matrix(conf_matrix1, df.get_classes())
        print()
