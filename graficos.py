import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_dual_pie_charts(predict_counts, actual_counts):
    # Verificar se predict_counts e actual_counts têm o mesmo número de elementos
    if len(predict_counts) != len(actual_counts):
        raise ValueError("predict_counts e actual_counts devem ter o mesmo número de elementos.")

    # Definir rótulos de acordo com o número de fatias
    labels = ['Negativo', 'Neutro', 'Positivo'] 
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  
    
    # Verificar se o número de rótulos corresponde ao número de fatias
    if len(labels) != len(predict_counts):
        raise ValueError("Número incorreto de rótulos fornecidos para o número de fatias.")

    def autopct_generator(pct):
        return f'{pct:.1f}%' if pct > 5 else ''

    def check_100_percent(counts):
        total = sum(counts)
        for count, label in zip(counts, labels):
            if count == total:
                return label
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Verificar se a distribuição predita é 100% uma única categoria
    predict_100_percent_label = check_100_percent(predict_counts)
    if predict_100_percent_label:
        ax1.pie([1], labels=[predict_100_percent_label], colors=[colors[labels.index(predict_100_percent_label)]], startangle=90)
    else:
        wedges1, texts1, autotexts1 = ax1.pie(predict_counts, labels=labels, autopct=autopct_generator, colors=colors, startangle=90)
        for text in autotexts1:
            text.set_fontsize(10)
            text.set_color('white')

    ax1.set_title('Predicted Class Distribution')

    # Verificar se a distribuição real é 100% uma única categoria
    actual_100_percent_label = check_100_percent(actual_counts)
    if actual_100_percent_label:
        ax2.pie([1], labels=[actual_100_percent_label], colors=[colors[labels.index(actual_100_percent_label)]], startangle=90)
    else:
        wedges2, texts2, autotexts2 = ax2.pie(actual_counts, labels=labels, autopct=autopct_generator, colors=colors, startangle=90)
        for text in autotexts2:
            text.set_fontsize(10)
            text.set_color('white')

    ax2.set_title('Actual Class Distribution')

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(conf_matrix, nv_classes_, text):
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', 
                xticklabels=nv_classes_, yticklabels=nv_classes_)
    plt.title(text + ' Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_model_comparison(train_size, test_size, accuracy_nv, accuracy_sv, accuracy_rd):
    """
    Plota a comparação de acurácia entre três modelos.
    """
    models = ['Modelo nv', 'Modelo sv', 'Modelo rd']
    accuracies = [accuracy_nv, accuracy_sv, accuracy_rd]

    plt.figure(figsize=(10, 6))
    plt.bar(models, accuracies, color=['blue', 'green', 'red'])

    plt.title(f'Comparação de Acurácia entre Modelos, Tamanho total: {train_size[0] + test_size[0]}\n'
              f'Tamanho do Conjunto de Treinamento: {train_size[0]}, Tamanho do Conjunto de Teste: {test_size[0]}')
    plt.xlabel('Modelos')
    plt.ylabel('Acurácia')
    plt.ylim(0, 1)  # Define o limite do eixo y de 0 a 1 para representar a acurácia em porcentagem
    plt.grid(True, axis='y')

    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.2f}', ha='center', va='bottom')

    plt.show()

def plot_classification_reports(report1, report2, report3, classes, title1='Modelo 1', title2='Modelo 2', title3='Modelo 3'):
    metrics = ['precision', 'recall', 'f1-score']
    x = np.arange(len(classes))
    width = 0.2  # Ajuste da largura das barras para acomodar três conjuntos de barras

    fig, ax = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    
    for i, metric in enumerate(metrics):
        values1 = [report1[c][metric] for c in classes]
        values2 = [report2[c][metric] for c in classes]
        values3 = [report3[c][metric] for c in classes]
        
        bars1 = ax[i].bar(x - width, values1, width, label=title1)
        bars2 = ax[i].bar(x, values2, width, label=title2)
        bars3 = ax[i].bar(x + width, values3, width, label=title3)
        
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(classes)
        ax[i].set_ylabel(metric.capitalize())
        ax[i].legend()
        ax[i].set_ylim(0, 1)
        
        # Adicionar os valores em cima das barras
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax[i].text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')

    plt.suptitle('Comparação de Relatórios de Classificação')
    plt.xlabel('Classes')
    plt.xticks(rotation=0)  # Ajustar a rotação dos rótulos das classes
    plt.show()

def plot_pie_chart(percentages):
    """
    Plota um gráfico de pizza com base nas porcentagens fornecidas.
    """

    # Verificando se a lista tem exatamente três elementos
    if len(percentages) != 3:
        raise ValueError("A lista deve conter exatamente três porcentagens.")
    
    # Rótulos para cada fatia da pizza
    labels = ['Negativo', 'Neltro', 'Positivo']

    # Cores para cada fatia
    colors = ['#ff9999', '#66b3ff', '#99ff99']

    # Criando o gráfico de pizza
    plt.figure(figsize=(8, 8))
    plt.pie(percentages, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)

    # Título do gráfico
    plt.title('Distribuição dos Dados na Base de Dados')

    # Exibir o gráfico
    plt.show()
