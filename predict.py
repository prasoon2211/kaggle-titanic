import pandas as pd
import numpy as np

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
#from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

def add_title(df):
    df["Title"] = df["Name"].apply(lambda x: x.split(',')[1].split('.')[0].strip())

def extrapolate_age(df):
    is_female = df["Sex"] == "female"
    is_male = df["Sex"] == "male"
    mean_age_men = np.mean(df[is_male]["Age"])
    mean_age_women = np.mean(df[is_female]["Age"])

    for i, age in enumerate(df["Age"]):
       if np.isnan(age):
           if df["Sex"][i] == "female":
               df["Age"][i] = int(mean_age_women)
           else:
               title = df["Title"][i]
               if title == "Master":
                   df["Age"][i] = 8
               else:
                   df["Age"][i] = int(mean_age_men)

def factor_to_dummy(df, col_name):
    dummy = pd.get_dummies(df[col_name], prefix=col_name)
    return pd.concat([df, dummy], axis=1)

def fill_fare(train, test):
    temp = pd.concat([train["Fare"], test["Fare"]], axis=0)
    test['Fare'] = test['Fare'].fillna(np.nanmean(temp))

def family_col(df):
    df["Family"] = df["SibSp"] + df["Parch"]

def check_accuracy(X, y):
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X, y, test_size=0.40, random_state=0)
    clf = RandomForestClassifier(n_estimators=100, max_depth=None)
    clf.fit(X_train, y_train)
    print "Score on test set1: ", clf.score(X_test, y_test)
    clf = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0)
    clf.fit(X_train, y_train)
    print "Score on test set2: ", clf.score(X_test, y_test)

    # clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    # clf.fit(X_train, y_train)
    # print "Score on test set: ", clf.score(X_test, y_test)

    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    print "Score on test set3 : ", clf.score(X_test, y_test)

    clf1 = ExtraTreesClassifier(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=0)
    clf3 = AdaBoostClassifier(n_estimators=50)
    clf4 = GaussianNB()

    clf = VotingClassifier(estimators=[('et', clf1), ('rf', clf2), ('abc', clf3), ('gn', clf4)], voting='hard')
    clf.fit(X_train, y_train)
    print "Score on test set 4: ", clf.score(X_test, y_test)


def get_relevant(df):
    cols = [col for col in df.columns
            if col not in ["PassengerId", "Survived", "Name", "Sex",
                           "Ticket", "Cabin", "Embarked", "Title", "Age", "SibSp", "Parch"]]
    X = df[cols]
    print cols
    y = df["Survived"]
    return X, y

def run_all():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    test['Survived'] = 0
    fill_fare(train, test)

    # family_col(train)
    # family_col(test)

    add_title(train)
    add_title(test)

    extrapolate_age(train)
    extrapolate_age(test)

    # Check no age left NaN
    print train["Age"].isnull().values.any()
    print test["Age"].isnull().values.any()

    # Convert from categorical to dummy
    train = factor_to_dummy(train, 'Sex')
    test = factor_to_dummy(test, 'Sex')

    check_accuracy(*get_relevant(train))

    # clf = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0)
    # clf.fit(*get_relevant(train))
    clf1 = ExtraTreesClassifier(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=0)
    clf3 = AdaBoostClassifier(n_estimators=50)
    clf4 = GaussianNB()

    clf = VotingClassifier(estimators=[('et', clf1), ('rf', clf2), ('abc', clf3), ('gn', clf4)], voting='hard')
    clf.fit(*get_relevant(train))

    X, y = get_relevant(test)
    prediction = clf.predict(X)

    test['Survived'] = prediction

    result = test[["PassengerId", "Survived"]]
    result.to_csv('prediction.csv', index=False)

run_all()
