import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

tr_features = pd.read_csv('train_features.csv')
tr_labels = pd.read_csv('train_labels.csv')


def print_results(results):
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))
    print('\nBest Parameters: {}\n'.format(results.best_params_))


gbc = GradientBoostingClassifier()
parameters = {
    'n_estimators': [5, 50, 250, 500],
    'max_depth': [2, 4, 6, 8, 10, None],
    'learning_rate': [0.01, 0.1, 1, 10, 100]
}

cv = GridSearchCV(gbc, parameters, cv=5)
cv.fit(tr_features, tr_labels.values.ravel())

print_results(cv)

joblib.dump(cv.best_estimator_, 'GB_model.pkl')
