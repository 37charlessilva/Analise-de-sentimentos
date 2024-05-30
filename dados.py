import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


class Dados:
    def __init__(self, file_path) -> None:
        # Carregando dados apartir de um arquivo cvs
        self.df = pd.read_csv(file_path)
        self.vetorizador = TfidfVectorizer()
        
    def get_head(self, tamanho = 10):
        # Returna uma quantidade de linhas do arquivo
        return self.df.head(tamanho)
    
    def get_coluna(self, coluna):
        return self.df[coluna]
    
    def transform_data(self, coluna):
        # Vetoriza os dados da culuna
        x = self.vetorizador.fit_transform(self.df[coluna])
        return x
