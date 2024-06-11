from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from dados import Dados
from naive_bayes import Naive_bayes

# nome das bases de dados
bases = ['Bases/b2w.csv', 'Bases/buscape.csv', 'Bases/olist.csv', 'Bases/utlc_apps.csv']

# inicializa a classe Naive_bayes
nv = Naive_bayes()

while(True):
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
    nv.start(df.transform_data('review_text_processed'), 
            y = df.get_rating())

    print('\nConjuntos de treinamento e teste separados!')
    print('Tamanho do conjunto de treinamento:', nv.get_train_x().shape)
    print('Tamanho do conjunto de teste:', nv.get_test_x().shape)

    # Avaliar o desempenho do modelo
    accuracy = nv.accuracy_score()
    report = nv.classification_report()
    conf_matrix = nv.confusion_matrix()

    print('\nAcurácia do modelo:', accuracy)
    print('\nRelatório de classificação:')
    print(report)
    print('\nMatriz de confusão:')
    print(conf_matrix)
    print("\n")
    
    # Plot confusion matrix for the current classifier
    plot_confusion_matrix(conf_matrix, nv_classes_)
