import pandas as pd
import numpy as np
import xlrd
import xlwt
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn.tree as sklt
import sklearn.metrics as sklm
import sklearn.model_selection as sklms
import sklearn.linear_model as skllm
import sklearn.svm as sklsvm
import sklearn.cluster as sklc
import sklearn.preprocessing as sklp
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.models as tkm
import tensorflow.keras.layers as tkl
import tensorflow.keras.wrappers.scikit_learn as tkwcl
import os


def build_model(activation='relu',learning_rate=0.01):
    model = tkm.Sequential()
    model.add(tkl.Dense(16,input_dim=12,activation=activation))
    model.add(tkl.Dense(8,activation=activation))
    model.add(tkl.Dense(1,activation='relu'))   
    
    model.compile(loss='mae',
                  optimizer=tfk.optimizers.Adam(learning_rate),
                  metrics=['mae','mse'])
    return model

def plot_learning_curve(history,metric='neg_mean_absolute_error'):
    plt.figure(figsize=(8,4))
    plt.title('Training Loss vs Validation Loss')
    plt.plot(history.epoch,history.history['loss'],label='train')
    plt.plot(history.epoch,history.history['val_loss'],label='val')
    plt.ylabel('Training '+metric)
    plt.xlabel('Epochs '+metric)
    plt.legend()


def valueToASD(asd):
    if asd == 0:
      return 'None'
    elif asd == 1:
      return 'Low'
    elif asd == 2:
      return 'Medium'
    elif asd == 3:
      return 'High'
    elif asd == 4:
      return 'Very_High'
    else:
      return None


def scaleASD(asd):
    if asd == 0:
      return 0.0
    elif asd == 1:
      return 0.25
    elif asd == 2:
      return 0.5
    elif asd == 3:
      return 0.75
    elif asd == 4:
      return 1
    else:
      return None


def unscaleASD(arr):
    ret = np.copy(arr)
    for i in range(arr.shape[0]):
        if arr[i] <= 0.20:
            ret[i] = 0
        elif arr[i] <= 0.4:
            ret[i] = 1
        elif arr[i] <= 0.6:
            ret[i] = 2
        elif arr[i] <= 0.8:
            ret[i] = 3
        elif arr[i] <= 1:
            ret[i] = 4
        else:
            ret[i] = None
    return ret



def choose_model(sub,swi,x,y,df_test):
    x_train, x_test, y_train, y_test = sklms.train_test_split(x,y,test_size=0.25,random_state=2021)
    
    print('\n\n\nxtrain:')
    print(x_train)
    print('\n\n\nxtest:')
    print(x_test)
    print('\n\n\nytrain:')
    print(y_train)
    print('\n\n\nytest:')
    print(y_test)
  
    predictions = {}
    if(swi == 1):
        if(sub):
            x_test = df_test
        clf = sklt.DecisionTreeClassifier()
        clf.fit(x_train,y_train)
      
        predictions = clf.predict(x_test)
      
        
        if(not sub):
            print('\n\nModelo Arvore de Decisão:')      
            sklm.confusion_matrix(y_test, predictions)
            print(f'Accuracy: {sklm.accuracy_score(y_test,predictions)}')
            print(f'Precision: {sklm.precision_score(y_test,predictions,average="micro")}')
            print(f'Recall: {sklm.recall_score(y_test,predictions,average="micro")}')
  
    elif(swi == 2):
        if(sub):
            x_test = df_test
        lm = skllm.LinearRegression()
        lm.fit(x_train,y_train)

        predictions = lm.predict(x_test)
        
        if(not sub):
            print('\n\nModelo Regressão Linear:')
            print(f'Mean Absolute Error: {sklm.mean_absolute_error(y_test,predictions)}')
            print(f'Mean Squared Error: {sklm.mean_squared_error(y_test,predictions)}')
            print(f'Root Mean Squared Error: {np.sqrt(sklm.mean_squared_error(y_test,predictions))}')
  
    elif(swi == 3):
        if(sub):
            x_test = df_test
        lm = skllm.LogisticRegression(max_iter=10000)
        lm.fit(x_train,np.ravel(y_train))

        predictions = lm.predict(x_test)

        if(not sub):
            print('\n\nModelo Regressão Logistica:')
            print(sklm.classification_report(y_test, predictions))
    
    elif(swi == 4):
        if(sub):
            x_test = df_test
        model = sklsvm.SVC(random_state=2021)
        model.fit(x_train,np.ravel(y_train))
      
        predictions = model.predict(x_test)
        
        if(not sub):
            print('\n\nSupport Vector Machine Model:')
            print(f'Accuracy: {sklm.accuracy_score(y_test,predictions)}')
  
    elif(swi == 5):
        if(sub):
            x_test = df_test
            
        param_grid = {'C' : [16], 'gamma' : [0.000345], 'kernel' : ['rbf']}
        model = sklms.GridSearchCV(sklsvm.SVC(random_state=2021), param_grid, refit=True,verbose=3)
    
        model.fit(x_train,np.ravel(y_train))

        print(model.best_params_)

        print(model.best_estimator_)
  

        model = sklsvm.SVC(C=model.best_params_['C'], gamma=model.best_params_['gamma'], random_state=2021)

        model.fit(x_train,np.ravel(y_train))

        predictions = model.predict(x_test)

        if(not sub):
            print('\n\nSupport Vector Machine Model w/ Hyperparameters:')
            print(f'Accuracy: {sklm.accuracy_score(y_test,predictions)}')
            print(sklm.classification_report(y_test, predictions))
    elif(swi == 6):
        scaler_x = sklp.MinMaxScaler(feature_range = (0,1)).fit(x)
        x_scaled = pd.DataFrame(scaler_x.transform(x[x.columns]),columns=x.columns)
        #y_scaled = pd.DataFrame(scaler_y.transform(y_train[y.columns]),columns=y.columns)
        
        print(x)
        print(x.isna().sum())
        print(x_scaled)
        print("\n\n\n\n\n\n\nyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
        print(y)
        y_scaled = y.apply(np.vectorize(scaleASD))
        
        print(y_scaled)
        x_train, x_test, y_train, y_test = sklms.train_test_split(x_scaled,y_scaled,test_size=0.25,random_state=2021)
        
        if(sub):
            scaler_x_test = sklp.MinMaxScaler(feature_range = (0,1)).fit(df_test)
            x_test_scaled = pd.DataFrame(scaler_x_test.transform(df_test[df_test.columns]),columns=df_test.columns)
            x_test = x_test_scaled
            x_train = x
            y_train = y
        
        print(x_test)
    
        TUNING_DICT = {'activation': ['relu'],
                       'learning_rate': [0.005]}
    
        kf = sklms.KFold(n_splits=2,shuffle=True,random_state=2021)    
        model = tkwcl.KerasRegressor(build_fn=build_model, 
                                     epochs=1,
                                     batch_size=1)
        grid_search = sklms.GridSearchCV(estimator=model, 
                                         param_grid=TUNING_DICT,
                                         cv = kf,
                                         scoring='neg_mean_absolute_error',
                                         refit='True',
                                         verbose=1)
        grid_search.fit(x_train,y_train,validation_split=0.2,verbose=1)
        
        
        print(f'Best: {grid_search.best_score_} using: {grid_search.best_params_}')
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        params = grid_search.cv_results_['params']
        
        for mean,stdev,param in zip(means,stds,params):
            print(f'{mean}({stdev}) with: {param}')
            
            
        best_mlp_model = grid_search.best_estimator_
        plot_learning_curve(best_mlp_model.model.history,metric='neg_mean_absolute_error')
    
        predictions_scaled = best_mlp_model.predict(x_test)

        print(predictions_scaled)
        
        predictions = unscaleASD(predictions_scaled)
    
        print(predictions)
        if(not sub):
            print('\n\nMultilayer Perceptrons Model:')
            print(sklm.classification_report(y_test, predictions))
        
    return predictions


df = pd.read_csv('datasets/training_data_clean.csv', encoding ='utf-8')
df_test = pd.read_csv('datasets/test_data_clean.csv', encoding ='utf-8')
df_submit = pd.read_csv('datasets/example_submission.csv')

df.fillna(df.mean(), inplace=True)
df_test.fillna(df_test.mean(), inplace=True)

x = df.drop(['AVERAGE_SPEED_DIFF'],axis=1)
y = df['AVERAGE_SPEED_DIFF'].to_frame()

print('\n\nOutput to file?[y/n]:')
sw = input()


print('\n\nChoose the model:\n1 - Decision Tree\n2 - Linear Regression\n3 - Logistic Regression\n4 - Support Vector Machine\n5 - Support Vector Machine w/ Hyperparameters\n')
swi = int(input())


sub = False
if(sw == 'y' or sw == 'Y'):
    sub = True
    
predictions = choose_model(sub,swi,x,y,df_test)

if(sub):
  for ind,row in df_submit.iterrows():
    df_submit.iat[ind,1] = valueToASD(predictions[ind])

  print(df_submit)
  df_submit.to_csv("datasets/G45Submission.csv",index=False)