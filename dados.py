import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class Dados:
    def __init__(self, file_path) -> None:
        # Carregando dados apartir de um arquivo cvs
        self.df = pd.read_csv(file_path)
        self.vetorizador = TfidfVectorizer()
        
    def get_head(self, tamanho = 10):
        # Returna uma quantidade de linhas do arquivo
        return self.df.head(tamanho)
    
    def get_column(self, column):
        # Retorna um coluna
        return self.df[column]
    
    def get_rating(self):
        # Retorna a coluna rating com os dados separadas em 
        # 1: positivo, 2: Neltro, 3: negativo 
        def alterardados(n):
            if n <= 2:
                return 1
            elif n == 3:
                return 2
            else:
                return 3
        return self.df['rating'].map(alterardados)
    
    def get_rating_percentage(self, rating):
        # Pega a porcetagem de valores contidos na coluna ratting e retorna a porcetagem
        neg = nel = pos = 0
        total = len(rating)
        
        for r in rating:
            if r == 1:
                neg += 1
            elif r == 2:
                nel += 1
            else:
                pos += 1
        return [round((neg / total) * 100), round((nel / total) * 100), round((pos / total) * 100)]
    
    def transform_data(self, coluna):
        # Vetoriza os dados da culuna
        self.df[coluna] = self.df[coluna].fillna("")
        x = self.vetorizador.fit_transform(self.df[coluna])
        return x
        
    def get_classes(self):        
        return ['Negativo', 'Neutro', 'Positivo']
