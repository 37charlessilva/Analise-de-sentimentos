from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

class RandomForest:
    def __init__(self) -> None:
        self.train_x = None
        self.test_x = None
        self.train_y = None
        self.test_y = None
        self.test_y_pred = None
        self.training_time = None
        self.rf_model = RandomForestClassifier(n_estimators=75, random_state=42)
        
    def start(self, train_x, test_x, train_y, test_y):
        # Temos os conjuntos de teste e treinamento divididos
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y

        # Treinar o modelo Random Forest
        start_time = time.time()
        self.rf_model.fit(train_x, train_y)
        end_time = time.time()

        self.training_time = (end_time - start_time) / 60
        print(f"Tempo de treinamento do random forest: {self.training_time} minutos\n")

        # Fazer previsões no conjunto de teste
        self.test_y_pred = self.rf_model.predict(test_x)

    def get_test_y_pred(self):
        return self.test_y_pred
    
    def get_training_time(self):
        return self.training_time
    
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
