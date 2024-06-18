import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(conf_matrix, nv_classes_):
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', 
                xticklabels=nv_classes_, yticklabels=nv_classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
