from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix#, plot_confusion_matrix



# y_true: List of true labels.
# y_pred: List of predicted labels.


def get_accuracy(y_true, y_pred):
    correct = 0.
    for (index, val) in enumerate(y_true):
        if (y_true[index] == y_pred[index]):
            correct += 1
    accuracy = correct/len(y_true)
  
    return {'accuracy': accuracy}


def get_precision(y_true, y_pred): 
    fp = 0.
    tp = 0.
    for (index, val) in enumerate(y_true):
        for (i, v) in enumerate(val):
            if (y_true[index][i] == 1 and y_pred[index][i] == 1):
                tp += 1
            elif (y_true[index][i] == 0 and y_pred[index][i] == 1):
                fp += 1
    precission = tp/(tp + fp)
    
    return {'precission': precission}



def get_recall(y_true, y_pred):
    fn = 0.
    tp = 0.
    for (index, val) in enumerate(y_true):
        for (i, v) in enumerate(val):
            if (y_true[index][i] == 1 and y_pred[index][i] == 1):
                tp += 1
            elif (y_true[index][i] == 1 and y_pred[index][i] == 0):
                fn += 1
    recall = tp/(tp + fn)    
    
    return {'recall: ': recall}


def get_f1(y_true, y_pred):
    precision = get_precision(y_true, y_pred)
    recall = get_recall(y_true, y_pred)
    f1 = 2 * ((precision * recall)/(precision + recall))
    
    return {'f1 score: ': f1}


def get_classification_report(y_true, y_pred):
    accuracy = get_accuracy(y_true, y_pred)
    precision = get_precision(y_true, y_pred)
    recall = get_recall(y_true, y_pred)
    f1 = get_f1(y_true, y_pred)

    report = {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precission"],
        "recall": recall["recall: "],
        "f1_score": f1["f1 score: "],
    }
    return {'classification_repor: ', report}



    