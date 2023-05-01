from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.joblib import Memory
from timeit import default_timer as timer

mem = Memory(cachedir='cache')

@mem.cache
def make_data():
    return datasets.make_classification(n_samples=700, n_features=200, random_state=0)

X, y = make_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
parameters = {
    'n_estimators': [10, 50, 100, 500],
    'max_depth':    [5, 10, None],
    'max_features': [.1, .5, .8, 1.]
    }

clf = GridSearchCV(RandomForestClassifier(n_jobs=-1), parameters, cv=5, n_jobs=-1,
                       scoring='precision_macro' % score)
t0 = timer()
clf.fit(X_train, y_train)
t1 = timer()
print(t1-t0)
