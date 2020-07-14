import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from time import time

val_features = pd.read_csv('val_features.csv')
val_labels = pd.read_csv('val_labels.csv', header=None)

test_features = pd.read_csv('test_features.csv')
test_labels = pd.read_csv('test_labels.csv', header=None)

models = {}
for model in ['LR', 'SVM', 'MLP', 'RF', 'GB']:
    models[model] = joblib.load('{}_model.pkl'.format(model))


def evaluate_model(name, mdl, features, labels):
    start = time()
    pred = mdl.predict(features)
    end = time()
    accuracy = round(accuracy_score(labels, pred), 3)
    precision = round(precision_score(labels, pred), 3)
    recall = round(recall_score(labels, pred), 3)
    print('{} -- Accuracy: {}, Precision: {}, Recall:{}, Latency: {}ms'.format(name, accuracy, precision, recall,
                                                                               end - start))


for name, mdl in models.items():
    evaluate_model(name, mdl, val_features, val_labels)

# We can see that the RF_model is the best fit
print("------------------------------ Best Model is RF - Random Forest -------------------------------")
evaluate_model('Random Forest', models['RF'], test_features,test_labels)

# Output:
# LR -- Accuracy: 0.77, Precision: 0.707, Recall:0.631, Latency: 0.0ms
# SVM -- Accuracy: 0.747, Precision: 0.672, Recall:0.6, Latency: 0.0ms
# MLP -- Accuracy: 0.747, Precision: 0.667, Recall:0.615, Latency: 0.0ms
# RF -- Accuracy: 0.82, Precision: 0.824, Recall:0.646, Latency: 0.0ms
# GB -- Accuracy: 0.798, Precision: 0.784, Recall:0.615, Latency: 0.007988929748535156ms
# ------------------------------ Best Model is RF - Random Forest -------------------------------
# Random Forest -- Accuracy: 0.81, Precision: 0.875, Recall:0.645, Latency: 0.0ms
