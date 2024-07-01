from modelos import Models
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

class SVM_model(Models):
    def __init__(self, kernel='linear', C=1.0) -> None:
        super().__init__()
        self.training_time = None
        self.svm_model = SVC(kernel=kernel, C=C)
        
    def start(self, train_x, test_x, train_y, test_y):
        # Temos os conjuntos de teste e treinamento divididos
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y
        
        # Treinar o modelo SVM
        start_time = time.time()
        self.svm_model.fit(self.train_x, self.train_y)
        end_time = time.time()
        
        self.training_time = (end_time - start_time) / 60
        print(f"Tempo de treinamento do SVM: {self.training_time} minutos\n")

        # Fazer previsões no conjunto de teste
        self.test_y_pred = self.svm_model.predict(self.test_x)
    
    def get_training_time(self):
        return self.training_time
