#!/usr/bin/env python
# coding: utf-8

# **Voy a realizar un modelo de clasificación para saber qué tipo de usuario accede al sistema.**
# 
# I am going to carry out a classification model to know what type of user accesses the system.
# 

# In[30]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn import metrics
from sklearn import svm
from tabulate import tabulate
from sklearn.model_selection import RepeatedStratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


# In[31]:


# importo el archivo
# importing the file.
df = pd.read_csv('C:/remaining_behavior_ext.csv')
df.head()


# In[32]:


df.info()


# **voy a eliminar las variables menos importantes**
# 
# I'm going to remove the less important variables

# In[33]:


df.drop(['_id','behavior'], axis=1)


# **genero las variables explicativas y la variable objetivo**
# 
# generating feature and label

# In[34]:


feature_cols = ['inter_api_access_duration(sec)',
               'api_access_uniqueness',
               'sequence_length(count)',
               'vsession_duration(min)',
               'ip_type',
               'num_sessions',
               'num_users',
               'num_unique_apis',
               'source' 
               ]
label_col = ['behavior_type']
X = df[feature_cols]
Y = df[label_col]


# In[35]:


X.describe()


# **creo las columnas dummies para las variables categóricas**
# 
# Generating dummies columns for categorical variables

# In[36]:


df_X = pd.get_dummies(data=X, drop_first = True)
columns_dummies = df_X.columns.values
df_X.head()


# **defino train/test**
# 
# generating train/test

# In[37]:


# definimos los criterios de entrenamiento y test
# defining the training / test criteria
X_train, X_test, y_train, y_test = train_test_split(df_X,Y, test_size = 0.3, random_state = 101)
# para evitar mensaje de error conforme hay infinito o nan en X
# to avoid error message as there infinity or nan in X
X_train= X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
X_test= X_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# **genero el modelo**
# 
# creating the model

# In[38]:


# modelo de árbol de decisión
# decission tree model
dtc = DecisionTreeClassifier()
# entrenamos el modelo
# training the model
dtc = dtc.fit(X_train, y_train)
# realizamos las predicciones
# generating predictions
y_pred_dtc = dtc.predict(X_test)


# **genero la matriz de confusión**
# 
# generating confusion matrix

# In[39]:


# matriz de confusión
# confusion matrix
disp = plot_confusion_matrix(dtc, X_test, y_test, cmap = plt.cm.Blues)


# In[40]:


print(classification_report(y_test, y_pred_dtc))


# **validación cruzada**
# 
# cross validation

# In[41]:


# realizo la validación cruzada
# generating cross validation
kf = KFold(n_splits=5)
score = cross_val_score(dtc, X_train, y_train.values.ravel(), cv = kf)
score_mean = score.mean().round(2)
print(score_mean)


# **Optimizo los parámetros**
# 
# Optimizing the parameters

# In[42]:


# buscamos los mejores parámetros
# searching the best parameters
dtc = DecisionTreeClassifier()
# definimos los parámetros
# defining the parameters
criterion = ['gini','entropy']
splitter = ['best','random']
max_depth = [2,3,5,10,20]
min_samples_split = [5,10,20,50,100]
min_samples_leaf = [5,10,20,50,100]
# definimos la red de búsqueda
# defining grid search
grid = dict(criterion = criterion,
           splitter = splitter,
           max_depth = max_depth,
           min_samples_split = min_samples_split,
           min_samples_leaf = min_samples_leaf)
cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 1)
grid_search = GridSearchCV(estimator = dtc,
                          param_grid = grid,
                          n_jobs = -1,
                          scoring = 'accuracy',
                          error_score = 0)
grid_result = grid_search.fit(X_train, y_train.values.ravel())
# resumimos resultados
# summarizing results
print("los mejores parámetros: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# In[43]:


# utilizamos los mejores parámetros
# implementing the best parameters
dtc_2 = DecisionTreeClassifier(criterion = "entropy",
                              max_depth = 10,
                              min_samples_leaf = 10,
                              min_samples_split = 20,
                              splitter = "best")
# entrenamos el modelo
# fitting the model
dtc_2.fit(X_train, y_train.values.ravel())
# realizamos las predicciones
# realizing predictions
y_pred_dtc_2 = dtc_2.predict(X_test)
# obtenemos los datos
# summarizing results.
print(classification_report(y_test, y_pred_dtc_2))


# **como podemos observar ha mejorado la precisión de clasificación de bot y attack, manteniendo attack, normal y outlier y ha empeorado la de bot.**
# 
# 
# As we can see, the bot and attack classification accuracy has improved, keeping attack, normal and outlier, and the bot has worsened.
# 
# **en cuanto a F-1 el modelo ha mejorado en attack y bot, el modelo es especialmente bueno en detectar normal y outlier, y es peor detectando ataques**
# 
# 
# As for F-1, the model has improved in attack and bot, the model is especially good at detecting normal and outlier, and is worse at detecting attacks
# 
# 

# In[44]:


# imprimimos la matriz de confusión
# confusion matrix
disp = plot_confusion_matrix(dtc_2, X_test, y_test, cmap = plt.cm.Blues)


# **Como podemos observar ha mejorado la clasificación de outlier y attack, se ha mantenido la del usuario normal, y ha empeorado la de bot.**
# 
# Como podemos observar ha mejorado la clasificación de outlier y attack, se ha mantenido la del usuario normal, y ha empeorado la de bot.

# In[45]:


# realizo la validación cruzada
# generating cross validation
kf = KFold(n_splits=5)
score = cross_val_score(dtc_2, X_train, y_train.values.ravel(), cv = kf)
score_mean = score.mean().round(2)
print(score_mean)


# **la validación cruzada mantiene el modelo en un 99%**
# 
# cross validation maintains the model at 99%
