from joblib import load, dump
from sklearn.ensemble import RandomForestClassifier
from naive_bayes import Naive_bayes
from svm import SVM_model
from random_forest import RandomForest 
from sklearn.metrics import accuracy_score, confusion_matrix
from dados import Dados
import graficos

"""
# Listas de bases e modelos
bases = ['Bases/b2w', 'Bases/buscape', 'Bases/olist', 'Bases/utlc_apps']
s = ['svm_model', 'random_forest_model']

# Inicializa os modelos
nv = Naive_bayes()
sv = SVM_model()
rd = RandomForestClassifier()

modelos = [nv, sv, rd]

while True:
    # Menu inicial para seleção da base de dados
    print("Escolha sua base de dados\n"
          "1: B2w\n"
          "2: Buscape\n"
          "3: Olist\n" 
          "4: Utlc_apps")
    i = int(input("Resposta: "))

    # Carrega os dados a partir de um arquivo CSV
    df = Dados(bases[i - 1] + ".csv")

    # Separação do conjunto de treino e teste
    df.trainig_data()

    print('\nConjuntos de treinamento e teste separados!')
    print('Tamanho do conjunto de treinamento:', df.get_train_x().shape) 
    print('Tamanho do conjunto de teste:', df.get_test_x().shape)
    
    # Treinamento do Naive Bayes
    nv.start(df.get_train_x(), df.get_test_x(), df.get_train_y(), df.get_test_y()) 
    print("\nModelo Naive Bayes treinado\n")

    # Treinamento do SVM e Random Forest
    for c in range(1, len(modelos)):
        if df.verify(f"{bases[i - 1]}_{s[c - 1]}_.pk1") == False:
            if isinstance(modelos[c], RandomForestClassifier):
                print(f"{s[c - 1].upper()} do sklearn, treinamento padrão\n")
                modelos[c].fit(df.get_train_x(), df.get_train_y())
                dump(modelos[c], f"{bases[i - 1]}_{s[c - 1]}_.pk1")
                print(f"{s[c - 1].upper()} treinado")
            else:
                print(f"{s[c - 1].upper()} em treinamento, pode levar algum tempo\n")
                modelos[c].start(df.get_train_x(), df.get_test_x(), df.get_train_y(), df.get_test_y())
                dump(modelos[c], f"{bases[i - 1]}_{s[c - 1]}_.pk1")
                print(f"{s[c - 1].upper()} treinado")
        else:
            print(f"Modelo {s[c - 1]} recuperado\n")
            modelos[c] = load(f"{bases[i - 1]}_{s[c - 1]}_.pk1")
    
    sv = modelos[1]
    rd = modelos[2]

    # Avaliação do desempenho do modelo Naive Bayes
    accuracy = nv.accuracy_score()
    report = nv.classification_report()
    conf_matrix = nv.confusion_matrix()

    # Avaliação do desempenho do modelo SVM
    if df.verify(f"{bases[i - 1]}_{s[0]}_.pk1"):
        sv = load(f"{bases[i - 1]}_{s[0]}_.pk1")
    else:
        print(f"Modelo {s[0]} não encontrado. Realize o treinamento primeiro.")
        continue

    accuracy1 = sv.accuracy_score()
    report1 = sv.classification_report()
    conf_matrix1 = sv.confusion_matrix()

    # Avaliação do desempenho do modelo Random Forest
    predictions = rd.predict(df.get_test_x())
    accuracy2 = accuracy_score(df.get_test_y(), predictions)
    report2 = classification_report(df.get_test_y(), predictions)
    conf_matrix2 = confusion_matrix(df.get_test_y(), predictions)

    # Menu de opções para visualização
    while True:
        print("\n0: Para voltar\n"
              "1: Comparação de acurácia\n"
              "2: Comparação dos relatórios de classificação\n"
              "3: Matriz de confusão\n"
              "4: Gráfico de Pizza")
        choice = int(input("Resposta: "))
        
        if choice == 0:
            break
        elif choice == 1:
            graficos.plot_model_comparison(df.get_train_x().shape, df.get_test_x().shape, accuracy, accuracy1, accuracy2)
        elif choice == 2:
            classes = df.get_classes()
            graficos.plot_classification_reports(report, report1, report2, classes, "Naive Bayes", "SVM", "Random Forest")
        elif choice == 3:
            graficos.plot_confusion_matrix(conf_matrix, df.get_classes())
            graficos.plot_confusion_matrix(conf_matrix1, df.get_classes())
            graficos.plot_confusion_matrix(conf_matrix2, df.get_classes())
        elif choice == 4:
            predict_counts = [list(report.values()).count(classe) for classe in classes]
            actual_counts = [list(df.get_test_y()).count(classe) for classe in classes]
            graficos.plot_dual_pie_charts(predict_counts, actual_counts)
        else:
            print("Opção inválida. Tente novamente.")

    print()
"""

# nome das bases de dados e modelos
bases = ['Bases/b2w', 'Bases/buscape', 'Bases/olist', 'Bases/utlc_apps']
s = ['svm_model', 'random_forest_model']

# inicializa a classe Naive_bayes
nv = Naive_bayes()

# inicializa a classe SVM_model
sv = SVM_model()

# inicializa o modelo random forest
rd = RandomForestClassifier()

# inicializa a classe classes
classes = ['positivo', 'negativo', 'neutro']

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
            if isinstance(modelos[c], RandomForestClassifier):
                print(f"{s[c - 1].upper()} do sklearn, treinamento padrão\n")
                modelos[c].fit(df.get_train_x(), df.get_train_y())
                dump(modelos[c], f"{bases[i - 1]}_{s[c - 1]}_.pk1")
                print(f"{s[c - 1].upper()} treinado")
            else:
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
    
    # 1. Calcular a acurácia
    predictions = rd.predict(df.get_test_x())
    accuracy2 = accuracy_score(df.get_test_y(), predictions)
    print(f'Acurácia do RandomForestClassifier: {accuracy2}')
    
    # 2. Gerar o relatório de classificação (opcional)
    # Se você quiser um relatório detalhado, pode usar sklearn.metrics.classification_report
    from sklearn.metrics import classification_report
    report2 = classification_report(df.get_test_y(), predictions)
    print(f'Relatório de classificação:\n{report2}')
    
    # 3. Gerar a matriz de confusão
    conf_matrix2 = confusion_matrix(df.get_test_y(), predictions)
    print(f'Matriz de Confusão:\n{conf_matrix2}')
    
    while i != 0:
        print("\n0: Para voltar\n"
              "1: Comparação de acurácia\n"
              "2: Comparação dos relatórios de classificação\n"
              "3: Matriz de confusão\n"
              "4: Gráfico Pizza")
        i = int(input("Resposta: "))
        if i == 1:
            graficos.plot_model_comparison(df.get_train_x().shape, df.get_test_x().shape, accuracy, accuracy1, accuracy2)
        elif i == 2:
            classes = ["1", "2", "3"]
            graficos.plot_classification_reports(report, report1, report2, classes, "nv", "sv", "rd")
        elif i == 3:
            graficos.plot_confusion_matrix(conf_matrix, df.get_classes())
            graficos.plot_confusion_matrix(conf_matrix1, df.get_classes())
        elif i == 4:
            predict_counts = [list(report.values()).count(classe) for classe in classes]
            actual_counts = [list(df.get_test_y()).count(classe) for classe in classes]
            graficos.plot_dual_pie_charts(predict_counts, actual_counts)
        print()


