from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Naive_bayes:
    def __init__(self) -> None:
        self.train_x = None
        self.test_x = None
        self.train_y = None
        self.test_y = None
        self.test_y_pred = None
        self.nb_model = MultinomialNB()
        
    def start(self, train_x, test_x, train_y, test_y):
        # Temos os conjuntos de teste e treinamento divididos
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y

        # Treinar o modelo Naive Bayes
        self.nb_model.fit(train_x, train_y)

        # Fazer previsões no conjunto de teste
        self.test_y_pred = self.nb_model.predict(test_x)

    def get_test_y_pred(self):
        return self.test_y_pred
    
    # Avaliar o desempenho do modelo
    def accuracy_score(self):
        # Retorna a acuracia do modelo
        return accuracy_score(self.test_y, self.test_y_pred)

    def classification_report(self):
        # Retorna um relatório detalhado das métricas de classificação
        return classification_report(self.test_y, self.test_y_pred, output_dict=True)
    
    def confusion_matrix(self):
        # Retorna a matriz de confusão
        return confusion_matrix(self.test_y, self.test_y_pred)
