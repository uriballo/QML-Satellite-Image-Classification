import matplotlib.pyplot as plt
import numpy as np
import seaborn


def confusion_matrix_plot(confusion_matrix, labels, title='Confusion Matrix', save_path='', filename='confusion_matrix.png'):
    plt.figure(figsize=(8, 6))
    
    # Plotting the confusion matrix with heatmap
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='crest', 
                 xticklabels=labels, yticklabels=labels, cbar=True)

    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path + filename, bbox_inches='tight')
    plt.show()
