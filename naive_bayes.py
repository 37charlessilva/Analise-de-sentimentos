from sklearn.model_selection import train_test_split
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
        
    def start(self, X, y):
        """X vai ser os dados contindos em review_text_processed e y 
        as avaliacoes dos produtos"""
        # Temos os conjuntos de teste e treinamento divididos
        # 80 para treinamento e 20 para teste
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
        
        # Treinar o modelo Naive Bayes
        self.nb_model.fit(self.train_x, self.train_y)

        # Fazer previsões no conjunto de teste
        self.test_y_pred = self.nb_model.predict(self.test_x)

    def get_train_x(self):
        # Retorna o conjunto de treino x
        return self.train_x
    
    def get_train_y(self):
        # Retorna o conjunto de treino y
        return self.train_y
    
    def get_test_x(self):
        # Retorna o conjunto de teste x
        return self.test_x
    
    def get_test_y(self):
        # Retorna o conjunto de teste y
        return self.test_y
    
    # Avaliar o desempenho do modelo
    def accuracy_score(self):
        # Retorna a acuracia do modelo
        return accuracy_score(self.test_y, self.test_y_pred)

    def classification_report(self):
        # 
        return classification_report(self.test_y, self.test_y_pred)
    
    def confusion_matrix(self):
        # Retorna a matriz de confusão
        return confusion_matrix(self.test_y, self.test_y_pred)
