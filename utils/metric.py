from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix#, plot_confusion_matrix


# def metric():

#     X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.2, 0.8], random_state=0)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#     clf = LogisticRegression(random_state=0).fit(X_train, y_train)
#     y_pred = clf.predict(X_test)

#     return classification_report(y_test, y_pred)




def get_accuracy(labels, predictions):
#   predictions = preds.predictions.argmax(axis=-1)
#   labels = preds.label_ids
  accuracy = accuracy_score(labels, predictions)
  return {'accuracy': accuracy}


def get_precision(labels, predictions):
   precission =  precision_score(labels, predictions)
   return {'precission': precission}


def get_recall(labels, predictions):
    recall = recall_score(labels, predictions)
    return {'recall: ', recall}


def get_f1(labels, predictions):
    f1 = f1_score(labels, predictions)
    return {'f1 score: ', f1}


def get_classification_report(labels, predictions):
    cr = classification_report(labels, predictions)
    return {'classification_repor: ', cr}
