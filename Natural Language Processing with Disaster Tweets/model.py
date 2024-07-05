import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.remove('not')
ps = PorterStemmer()

dataset = pd.read_csv('train.csv')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    url = re.compile(r'https?://\S+|www\.\S+')
    text = url.sub(r'',text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = ' '.join(ps.stem(word) for word in text.split() if word not in stop_words)
    return text

dataset['cleaned_text'] = dataset['text'].apply(clean_text)

dataset['keyword'].fillna('None', inplace=True)
dataset['location'].fillna('None', inplace=True)
dataset['Keyword'] = dataset['keyword'].apply(lambda x: x.replace('%20', ' '))

dataset['combined_text'] = dataset['cleaned_text'] + ' ' + dataset['keyword'] + ' ' + dataset['location']

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(dataset['combined_text'])
y = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
print("accuracy: ", accuracy_score(y_test, y_pred))

test_data = pd.read_csv('test.csv')
test_data['cleaned_text'] = test_data['text'].apply(clean_text)
test_data['keyword'].fillna('', inplace=True)
test_data['location'].fillna('', inplace=True)
test_data['combined_text'] = test_data['cleaned_text'] + ' ' + test_data['keyword'] + ' ' + test_data['location']
X_test = vectorizer.transform(test_data['combined_text'])
y_pred = classifier.predict(X_test)

submission = pd.DataFrame({'id': test_data['id'], 'target': y_pred})
submission.to_csv('submission.csv', index=False)