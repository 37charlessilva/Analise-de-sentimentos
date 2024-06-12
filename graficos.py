#import matplotlib.pyplot as plt
import sklearn.metrics import plot_confusion_matrix
import pandas as pd
#import seaborn as sns
import numpy as np

class Graficos:
    def plot_confusion_matrix(conf_matrix, nv_classes_):
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', 
                    xticklabels=nv_classes_, yticklabels=nv_classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()



