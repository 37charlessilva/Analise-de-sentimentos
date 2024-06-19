import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
from tqdm import tqdm
import os

class Dados:
    def __init__(self, file_path) -> None:
        # Carregando dados apartir de um arquivo cvs
        self.df = self.load_data_with_progress(file_path)
        self.vetorizador = TfidfVectorizer()

        # Dados para treino
        self.train_x = None
        self.test_x = None
        self.train_y = None
        self.test_y = None
    
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
    
    def load_data_with_progress(self, file_path):
        print("\nCarregando a base de dados, por favor aguarde...")
        chunks = pd.read_csv(file_path, chunksize=10000, iterator=True)
        df = pd.DataFrame()
        for chunk in tqdm(chunks, desc="Carregando", unit="chunk"):
            df = pd.concat([df, chunk], ignore_index=True)
        return df

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
    
    def get_classes(self):        
        return ['Negativo', 'Neutro', 'Positivo']
    
    def transform_data(self, coluna):
        # Vetoriza os dados da culuna
        self.df[coluna] = self.df[coluna].fillna("")
        x = self.vetorizador.fit_transform(self.df[coluna])
        return x

    def trainig_data(self):
        """X vai ser os dados contindos em review_text_processed e y 
        as avaliacoes dos produtos"""
        # Temos os conjuntos de teste e treinamento divididos
        # 80 para treinamento e 20 para teste
        X = self.transform_data('review_text_processed')
        y = self.get_rating()
        
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    def verify(self, pasta_modelos, nome_modelo):
        # Verificar a existência dos modelos
        caminho_modelo = os.path.join(pasta_modelos, nome_modelo) 
        if os.path.exists(caminho_modelo + ".pk1"):
            return True
        else:
            return False
