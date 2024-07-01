from joblib import load, dump
from naive_bayes import Naive_bayes
from svm import SVM_model
from random_forest import RandomForest 
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
rd = RandomForest()

modelos = [nv, sv, rd]

while True:
    # Menu inicial
    print("Escolha sua base de dados\n"
          "0: Para sair\n"
          "1: B2w\n"
          "2: Buscape\n"
          "3: Olist\n" 
          "4: Utlc_apps")
    i = int(input("Resposta: "))

    # carrega os dados a partir de um arquivo CSV
    df = Dados(bases[i - 1] + ".csv")

    # Separando o conjunto de treino e teste
    df.trainig_data()

    print('\nConjuntos de treinamento e teste separados!')
    print('Tamanho do conjunto de treinamento:', df.get_train_x().shape) 
    print('Tamanho do conjunto de teste:', df.get_test_x().shape)
    
    # Treinamento do Naive bayes
    nv.start(df.get_train_x(), df.get_test_x(), df.get_train_y(), df.get_test_y()) 
    print("\nModelo Naive bayes treinado\n")

    # Treinamento do SVM e Random Forest
    for c in range(1, len(modelos)):
        if df.verify(f"{bases[i - 1]}_{s[c - 1]}_.pk1") == False:
            print(f"{s[c - 1].upper()} em treinamento, pode levar algum tempo\n")
            modelos[c].start(df.get_train_x(), df.get_test_x(), df.get_train_y(), df.get_test_y())
            dump(modelos[c], f"{bases[i - 1]}_{s[c - 1]}_.pk1")
            print(f"{s[c - 1].upper()} treinado")
        else:
            print(f"Modelo {s[c - 1]} recuperado\n")
            modelos[c] = load(f"{bases[i - 1]}_{s[c - 1]}_.pk1")

    
    sv = modelos[1]
    rd = modelos[2]

    # Avaliação do desempenho do modelo Naive_bayes
    accuracy = nv.accuracy_score()
    report = nv.classification_report()
    conf_matrix = nv.confusion_matrix()

    # Avaliação do desempenho do modelo SVM_model
    accuracy1 = sv.accuracy_score()
    report1 = sv.classification_report()
    conf_matrix1 = sv.confusion_matrix()

    # Avaliação do desempenho do modelo Random_foresnt
    accuracy2 = rd.accuracy_score()
    report2 = rd.classification_report()
    conf_matrix2 = rd.confusion_matrix()
    

    while i != 0:
        print("\n0: Para voltar\n"
              "1: Comparação de acurácia\n"
              "2: Comparação dos relatórios de classificação\n"
              "3: Matriz de confusão\n"
              "4: Gráfico Pizza\n"
              "5: Distribuição dos dados")
        i = int(input("Resposta: "))

        if i == 1:
            graficos.plot_model_comparison(df.get_train_x().shape, df.get_test_x().shape, accuracy, accuracy1, accuracy2)

        elif i == 2:
            classes = ['1', '2', '3']
            graficos.plot_classification_reports(report, report1, report2, classes, "nv", "sv", "rd")

        elif i == 3:
            c = -1
            while(c != 0):
                print("\nQual modelo:\n"
                    "0: para sair\n"
                    "1: Naive Bayes\n"
                    "2: Support Vector machine\n"
                    "3: random Forest\n")
                c = int(input("Resposta: "))

                if(c == 1):
                    graficos.plot_confusion_matrix(conf_matrix, df.get_classes(), text = 'nv')

                elif(c == 2):
                    graficos.plot_confusion_matrix(conf_matrix1, df.get_classes(), text = 'sv')

                elif(c == 3):
                    graficos.plot_confusion_matrix(conf_matrix2, df.get_classes(), text = 'rd')

                else:
                    print("\nResposta invalida")

        elif i == 4:
            c = -1
            while(c != 0):
                print("\nQual modelo:\n"
                    "0: para sair\n"
                    "1: Naive Bayes\n"
                    "2: Support Vector machine\n"
                    "3: random Forest\n")
                c = int(input("Resposta: "))

                if(c == 1):           
                    graficos.plot_dual_pie_charts(df.get_rating_percentage(nv.test_y_pred), df.get_rating_percentage(df.get_rating()))

                elif(c == 2):
                    graficos.plot_dual_pie_charts(df.get_rating_percentage(sv.test_y_pred), df.get_rating_percentage(df.get_rating()))
                
                elif(c == 3):
                    graficos.plot_dual_pie_charts(df.get_rating_percentage(rd.test_y_pred), df.get_rating_percentage(df.get_rating()))
                
                else:
                    print("\nResposta invalida")

        elif i == 5:
            graficos.plot_pie_chart(df.get_rating_percentage(df.get_rating()))
        print()
