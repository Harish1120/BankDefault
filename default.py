import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt  
from sklearn.tree import export_graphviz

df = pd.read_csv(r'C:\Users\Harish Sundaralingam\Desktop\Stats and ML\Machine Learning\bank.csv')

print(df.describe())
print(df.info())
print(df.isnull().sum()) # no null data 
print(df.nunique())

sns.pairplot(df)
plt.show()

def preprocessor(df):
    dt_df = df.copy()
    
    le = preprocessing.LabelEncoder()

    dt_df['job'] = le.fit_transform(dt_df['job'])
    dt_df['marital'] = le.fit_transform(dt_df['marital'])
    dt_df['education'] = le.fit_transform(dt_df['education'])
    dt_df['default'] = le.fit_transform(dt_df['default'])
    dt_df['housing'] = le.fit_transform(dt_df['housing'])
    dt_df['loan'] = le.fit_transform(dt_df['loan'])
    dt_df['month'] = le.fit_transform(dt_df['month'])
    dt_df['poutcome'] = le.fit_transform(dt_df['poutcome'])
    dt_df['deposit'] = le.fit_transform(dt_df['deposit'])
    
    return dt_df

encoded_df = preprocessor(df)
encoded_df.drop('contact',axis=1,inplace=True)
x = encoded_df.drop(['deposit'],axis =1).values
y = encoded_df['deposit'].values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


model_dt = DecisionTreeClassifier(random_state=111)

from sklearn.model_selection import GridSearchCV

np.random.seed(42)
param_dist = {'max_depth': [8,9,10],
             'max_features':['auto','sqrt','log2',None],
             'criterion':['gini','entropy']
             }

cv_rf = GridSearchCV(model_dt, cv=10,
                    param_grid = param_dist,
                    n_jobs =3)

cv_rf.fit(x_train,y_train)
print('Best parameters using Grid Search: ', cv_rf.best_params_)

model_dt = DecisionTreeClassifier(criterion='gini', max_depth=8, max_features=None)
model_dt.fit(x_train,y_train)
y_pred_dt = model_dt.predict_proba(x_test)[:,1]

fpr_dt,tpr_dt,_ = roc_curve(y_test,y_pred_dt)
roc_auc_dt = auc(fpr_dt,tpr_dt)

predictions = model_dt.predict(x_test)

y_actual_result = 0
for i in range(len(predictions)):
    if predictions[i] == 1:
        y_actual_result = np.vstack((y_actual_result,y_test[i]))


y_actual_result = y_actual_result.flatten()
count = 0
for result in y_actual_result:
    if result == 1:
        count += 1 
print('true yes|predicted yes:')
print(count/(len(y_actual_result)))

plt.figure(1)
lw = 2
plt.plot(fpr_dt, tpr_dt, color='green',
         lw=lw, label='Decision Tree(AUC = %0.2f)' % roc_auc_dt)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Area Under Curve')
plt.legend(loc="lower right")
plt.show()

accuracy_score(y_test, predictions)

import itertools
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(model, normalize=False): # This function prints and plots the confusion matrix.
    cm = confusion_matrix(y_test, model, labels=[0, 1])
    classes=["True", "False"]
    cmap = plt.cm.Blues
    title = "Confusion Matrix"
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=3)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

plt.figure(figsize=(6,6))
plot_confusion_matrix(predictions, normalize=False)
plt.show()





