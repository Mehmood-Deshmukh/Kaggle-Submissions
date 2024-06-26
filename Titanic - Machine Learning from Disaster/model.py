import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('train.csv')


dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
dataset['IsAlone'] = (dataset['FamilySize'] == 1).astype(int)
dataset['Age*Class'] = dataset['Age'] * dataset['Pclass']
dataset['FarePerPerson'] = dataset['Fare'] / dataset['FamilySize']
dataset['Title'] = dataset['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())


X = dataset.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'])
y = dataset['Survived']


numerical_features = ['Age', 'Fare', 'FamilySize', 'Age*Class', 'FarePerPerson']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'SibSp', 'Parch', 'IsAlone', 'Title']


imputer = SimpleImputer(strategy='mean')
X[numerical_features] = imputer.fit_transform(X[numerical_features])


scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])


ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), categorical_features)], remainder='passthrough')
X = ct.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)


cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


from sklearn.model_selection import RandomizedSearchCV

parameters = {'n_estimators': [50, 100, 150, 200, 250],
                'criterion': ['gini', 'entropy'],
                'max_depth': [3, 4, 5, 6, 7, 8],
                'min_samples_split': [2, 5, 10]}

random_search = RandomizedSearchCV(estimator = classifier,
                                param_distributions = parameters,
                                scoring = 'accuracy',
                                cv = 10,
                                n_jobs = -1)

random_search.fit(X_train, y_train)
best_accuracy = random_search.best_score_
best_parameters = random_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)


test_data = pd.read_csv('test.csv')
test_ids = test_data['PassengerId']


test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
test_data['IsAlone'] = (test_data['FamilySize'] == 1).astype(int)
test_data['Age*Class'] = test_data['Age'] * test_data['Pclass']
test_data['FarePerPerson'] = test_data['Fare'] / test_data['FamilySize']
test_data['Title'] = test_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

X_test = test_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
X_test[numerical_features] = imputer.transform(X_test[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])
X_test = ct.transform(X_test).toarray()


y_pred = random_search.best_estimator_.predict(X_test)


output = pd.DataFrame({'PassengerId': test_ids, 'Survived': y_pred})
output.to_csv('submission.csv', index=False)
