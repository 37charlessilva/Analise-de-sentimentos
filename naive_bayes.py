from modelos import Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Naive_bayes(Models):
    def __init__(self) -> None:
        super().__init__()
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
    