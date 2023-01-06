import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("survey lung cancer.csv")
df.head()

df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.shape

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['LUNG_CANCER']=encoder.fit_transform(df['LUNG_CANCER'])
df['GENDER']=encoder.fit_transform(df['GENDER'])
df.head()

plt.figure(figsize=(15,15))
import seaborn as sns
sns.heatmap(df.corr(),annot=True,linewidth=0.5,fmt='0.2f')

##bağımlı ve bağımsız unsurlaırı ayırma
X=df.drop(['LUNG_CANCER'],axis=1)
y=df['LUNG_CANCER']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)
print(f'Train shape : {X_train.shape}\nTest shape: {X_test.shape}')

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

#Create KNN 
knn = KNeighborsClassifier(n_neighbors=3)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred_knn = knn.predict(X_test)

# Model Accuracy,
print("KNN Accuracy:",metrics.accuracy_score(y_test, y_pred_knn))

from sklearn.metrics import classification_report
target_names = ['without cancer (0)', 'with cancer (1)']
print(classification_report(y_test, y_pred_knn, target_names=target_names))

cnf_matrix = metrics.confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(8,8))
sns.heatmap(cnf_matrix,annot=True)
plt.title("KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# svm model

#Create a svm 
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model 
clf.fit(X_train, y_train)

#Predict - test dataset
y_pred_svm = clf.predict(X_test)


# Model Accuracy
print("SVM Accuracy:",metrics.accuracy_score(y_test, y_pred_svm))
from sklearn.metrics import classification_report
target_names = ['without cancer (0)', 'with cancer (1)']
print(classification_report(y_test, y_pred_svm, target_names=target_names))

cnf_matrix = metrics.confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(8,8))
sns.heatmap(cnf_matrix,annot=True)
plt.title("SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")



logreg = LogisticRegression(random_state=42)

# fit the model with data
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,8))
sns.heatmap(cnf_matrix,annot=True)
plt.title("Logistic R.")
plt.xlabel("Predicted")
plt.ylabel("Actual")

from sklearn.metrics import classification_report
target_names = ['without cancer (0)', 'with cancer (1)']
print(classification_report(y_test, y_pred, target_names=target_names))

# Model Accuracy: how often is the classifier correct?
print("Logistic R. Accuracy:",metrics.accuracy_score(y_test, y_pred))

