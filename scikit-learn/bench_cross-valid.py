from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.joblib import Memory
from timeit import default_timer as timer

mem = Memory(cachedir='cache')

@mem.cache
def make_data():
    return datasets.make_classification(n_samples=70000, n_features=200, random_state=0)

X, y = make_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)

t0 = timer()
cross_val_score(clf, X_train, y_train, n_jobs=5, cv=5)
t1 = timer()
print(t1-t0)

