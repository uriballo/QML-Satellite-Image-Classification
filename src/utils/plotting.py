import matplotlib.pyplot as plt
import seaborn as sns


def confusion_matrix_plot(confusion_matrix, labels, title='Confusion Matrix', save_path='', filename='confusion_matrix.png'):

    for i in range(len(labels)):
        if labels[i] == "AnnualCrop":
            labels[i] = "Annual\nCrop"
        elif labels[i] == "HerbaceousVegetation":
            labels[i] = "Herbaceous\nVegetation"
        elif labels[i] == "PermanentCrop":
            labels[i] = "Permanent\nCrop"

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
