import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from scipy.special import expit
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
import pickle

path = '/content/drive/MyDrive/Colab Notebooks/Wine Quality Project/winequality-white.csv'
global dataframe2
dataframe2 = pd.read_csv(path)
dataframe2.isnull().sum()

x = dataframe2.drop(['residual sugar', 'quality'], axis=1)
y = dataframe2['quality']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=30)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier(n_neighbors=5, weights='uniform'))
])

_model_3 = GridSearchCV(estimator=pipe, param_grid={
    'model__n_neighbors': [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
    'model__weights': ['uniform', 'distance']
}, cv=10)

_model_3.fit(x_train, y_train)

pickle.dump(_model_3, open('model.pkl','wb'))

pred = _model_3.predict(x_test)

print('%.2f'%_model_3.score(x_test, y_test), 'Model score')
accuracy = accuracy_score(y_test, pred)
print('%.2f'%accuracy, 'Accuracy')

rmse = np.sqrt(mean_squared_error(y_test, pred))
print('%.3f'%rmse, 'RMSE')

precision = precision_score(y_test, pred, average='micro')
print('%.4f'%precision, 'Precision')

recall = recall_score(y_test, pred, average='micro')
print('%.2f'%recall, 'Recall')

f1 = f1_score(y_test, pred, average='micro')
print('%.2f'%f1, 'F1-score')
