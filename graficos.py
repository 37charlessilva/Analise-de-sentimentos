import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_confusion_matrix(conf_matrix, nv_classes_):
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', 
                xticklabels=nv_classes_, yticklabels=nv_classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_model_comparison(train_size, test_size, accuracy_nv, accuracy_sv):
    """
    Plota a comparação de acurácia entre dois modelos.

    Parâmetros:
    - train_size: Tamanho do conjunto de treinamento (número de amostras).
    - test_size: Tamanho do conjunto de teste (número de amostras).
    - accuracy_nv: Acurácia do modelo nv.
    - accuracy_sv: Acurácia do modelo sv.
    """
    models = ['Modelo nv', 'Modelo sv']
    accuracies = [accuracy_nv, accuracy_sv]

    plt.figure(figsize=(10, 6))
    plt.bar(models, accuracies, color=['blue', 'green'])

    plt.title(f'Comparação de Acurácia entre Modelos, Tamanho total: {train_size[0] + test_size[0]}\n'
              f'Tamanho do Conjunto de Treinamento: {train_size[0]}, Tamanho do Conjunto de Teste: {test_size[0]}')
    plt.xlabel('Modelos')
    plt.ylabel('Acurácia')
    plt.ylim(0, 1)  # Define o limite do eixo y de 0 a 1 para representar a acurácia em porcentagem
    plt.grid(True, axis='y')

    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.2f}', ha='center', va='bottom')

    plt.show()

def plot_classification_reports(report1, report2, classes, title1='Modelo 1', title2='Modelo 2'):
    metrics = ['precision', 'recall', 'f1-score']
    x = np.arange(len(classes))
    width = 0.3  # Diminuir a largura das barras

    fig, ax = plt.subplots(3, 1, figsize=(10, 14), sharex=True)
    
    for i, metric in enumerate(metrics):
        values1 = [report1[c][metric] for c in classes]
        values2 = [report2[c][metric] for c in classes]
        
        bars1 = ax[i].bar(x - width/2, values1, width, label=title1)
        bars2 = ax[i].bar(x + width/2, values2, width, label=title2)
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(classes)
        ax[i].set_ylabel(metric.capitalize())
        ax[i].legend()
        ax[i].set_ylim(0, 1)
        
        # Adicionar os valores em cima das barras
        for bar in bars1:
            height = bar.get_height()
            ax[i].text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            ax[i].text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')

    plt.suptitle('Comparação de Relatórios de Classificação')
    plt.xlabel('Classes')
    plt.xticks(rotation=0)  # Ajustar a rotação dos rótulos das classes
    plt.show()
