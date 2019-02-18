from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score
bankdata=datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(bankdata.data, bankdata.target, test_size = 0.20)

svclassifier = svm.SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(bankdata.data)
recall_str=classification_report(bankdata.target,y_pred)
print(recall_str)