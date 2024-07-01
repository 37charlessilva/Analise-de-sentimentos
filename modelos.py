from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Models:
    def __init__(self) -> None:
        self.train_x = None
        self.test_x = None
        self.train_y = None
        self.test_y = None
        self.test_y_pred = None

    def start(self, train_x, test_x, train_y, test_y):
        pass

    def get_test_y_pred(self):
        return self.test_x

    def accuracy_score(self):
        # Retorna a acuracia do modelo
        return accuracy_score(self.test_y, self.test_y_pred)
    
    def classification_report(self):
        # Retorna um relatório detalhado das métricas de classificação
        return classification_report(self.test_y, self.test_y_pred, output_dict=True)
    
    def confusion_matrix(self):
        # Retorna a matriz de confusão
        return confusion_matrix(self.test_y, self.test_y_pred)
