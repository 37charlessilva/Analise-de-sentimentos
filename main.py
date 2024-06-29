from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from joblib import load, dump
from dados import Dados
from graficos import plot_model_comparison, plot_classification_reports, plot_confusion_matrix

# Definindo os modelos e seus nomes
modelos = {
    'svm_model': LinearSVC(),
    'random_forest_model': RandomForestClassifier()
}

while True:
    # Menu inicial
    print("Escolha sua base de dados\n"
          "1: B2w\n"
          "2: Buscape\n"
          "3: Olist\n"
          "4: Utlc_apps")
    resposta = int(input("Resposta: "))
    base_de_dados = bases[resposta - 1]

    # Carrega os dados a partir de um arquivo CSV
    df = Dados(base_de_dados + ".csv")

    # Separa o conjunto de treino e teste
    df.trainig_data()
    print('\nConjuntos de treinamento e teste separados!')
    print('Tamanho do conjunto de treinamento:', df.get_train_x().shape)
    print('Tamanho do conjunto de teste:', df.get_test_x().shape)
    
    # Treinamento e avaliação dos modelos
    for nome, modelo in modelos.items():
        arquivo_modelo = f"{base_de_dados}_{nome}.pk1"
        
        if not df.verify(arquivo_modelo):
            print(f"{nome.upper()} em treinamento, pode levar algum tempo...\n")
            
            # Criação de um pipeline com pré-processamento (opcional) e SVM
            pipeline = Pipeline([
                ('scaler', StandardScaler()),  # Normalização dos dados
                ('pca', PCA(n_components=0.95)),  # Redução de dimensionalidade com PCA
                ('svm', modelo)  # SVM (pode ser LinearSVC ou SVC com kernel específico)
            ])
            
            # Definição dos parâmetros para Grid Search
            params = {
                'svm__C': [0.1, 1, 10],  # Valores para C
                'svm__class_weight': [None, 'balanced']  # Opções de balanceamento de classe
            }
            
            # Realiza Grid Search para encontrar os melhores parâmetros
            grid_search = GridSearchCV(pipeline, params, cv=5, n_jobs=-1)
            grid_search.fit(df.get_train_x(), df.get_train_y())
            
            # Melhor modelo encontrado
            melhor_modelo = grid_search.best_estimator_
            
            # Salva o modelo treinado
            dump(melhor_modelo, arquivo_modelo)
            
            print(f"{nome.upper()} treinado.\n")
        
        else:
            print(f"Modelo {nome} recuperado.\n")
            melhor_modelo = load(arquivo_modelo)
        
        # Avaliação do modelo
        accuracy = melhor_modelo.score(df.get_test_x(), df.get_test_y())
        report = classification_report(df.get_test_y(), melhor_modelo.predict(df.get_test_x()))
        conf_matrix = confusion_matrix(df.get_test_y(), melhor_modelo.predict(df.get_test_x()))
        
        # Exibe métricas de desempenho
        print(f"Acurácia do modelo {nome}: {accuracy:.2f}\n")
        print(f"Relatório de classificação do modelo {nome}:\n{report}\n")
        print(f"Matriz de confusão do modelo {nome}:\n{conf_matrix}\n")
    
    # Menu de comparação e visualização de métricas
    opcao = int(input("Escolha uma opção:\n"
                      "0: Voltar\n"
                      "1: Comparação de acurácia\n"
                      "2: Comparação dos relatórios de classificação\n"
                      "3: Matriz de confusão\n"
                      "Resposta: "))
    
    if opcao == 1:
        plot_model_comparison(df.get_train_x().shape, df.get_test_x().shape, accuracy, accuracy1, accuracy2)
    elif opcao == 2:
        classes = ["1", "2", "3"]
        plot_classification_reports(report, report1, report2, classes, "nv", "sv", "rd")
    elif opcao == 3:
        plot_confusion_matrix(conf_matrix, df.get_classes())
        plot_confusion_matrix(conf_matrix1, df.get_classes())
        plot_confusion_matrix(conf_matrix2, df.get_classes())

