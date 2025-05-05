import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def data_anal(Y):
    class_counts = np.sum(Y, axis=0) if Y.ndim > 1 else np.bincount(Y)
    plt.bar(range(len(class_counts)), class_counts)
    plt.title("Class distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.show()

def var_scores(y_true, y_pred):
    # binary/multiclass
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

def confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()