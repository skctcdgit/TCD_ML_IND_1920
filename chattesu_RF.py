# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 21:02:56 2019

@author: SuhridKrishna
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import category_encoders as CatEnc


def RemoveFeatures(dataset):
    
    #Removing the features 'Hair Color', 'Instance' and 'Wears Glasses'
    dataset = dataset.drop('Hair Color', axis=1)
    dataset = dataset.drop('Instance', axis=1)
    dataset = dataset.drop('Wears Glasses', axis=1)
    return dataset
    
def PreprocessData(dataset, dataset_test):
    #Storing the predictors in variable X and denoting that it is to be used for training the model using the column 'training_or_not'
    X = pd.DataFrame(dataset.iloc[:,:-1])
    X['training_or_not'] = 1
    
    #Extracting the target variable 'Income in EUR' into Y
    Y = pd.DataFrame(dataset['Income in EUR'])
    
    #Storing the testing data [TCD ML Data] into X_test and denoting that it is to be used for prediction
    X_test = pd.DataFrame(dataset_test.iloc[:,:-1])
    X_test['training_or_not'] = 0
    
    #Concatinating the training and testing data for preprocessing in both the datasets
    dataset_total = pd.concat([X, X_test])
    
    #Deleting the existing X and X_test variables so that the preprocessed data can be stored in the same variables
    del X_test, X
    
    #---------------------------------------PREPROCESSING : DATA IMPUTATION-------------------------------------------------
    
    #Filling NA and 0 values in 'Gender' with 'unknown'
    dataset_total['Gender'] = dataset_total['Gender'].fillna('unknown')
    dataset_total['Gender'] = dataset_total['Gender'].replace('0', 'unknown')
    
    #Filling NA and 0 values in 'University Degree' with 'No'
    dataset_total['University Degree'] = dataset_total['University Degree'].fillna('No')
    dataset_total['University Degree'] = dataset_total['University Degree'].replace('0', 'No')
    
    #Fill missing values in 'Profession', 'Age', 'Year of Record', 'Body Height cm' and 'Size of City' by the mean values of the respective columns
    dataset_total['Profession'].fillna(dataset_total['Profession'].mode()[0], inplace=True)
    
    dataset_total['Age'].fillna(dataset_total['Age'].mean(), inplace=True)
    
    dataset_total['Year of Record'].fillna(dataset_total['Year of Record'].mean(), inplace=True)
    
    dataset_total['Body Height cm'].fillna(dataset_total['Body Height cm'].mean(), inplace=True)
    
    dataset_total['Size of City'].fillna(dataset_total['Size of City'].mean(), inplace=True)
    
    #Filling missing values 'Country' with the mode of the feature
    dataset_total['Country'].fillna(dataset_total['Country'].mode()[0], inplace=True)
    
    #Creating TargetEncoder() and MinMaxScaler() instances
    TargEnc = CatEnc.TargetEncoder()
    MMScaler = MinMaxScaler()
    
    #Separating the training and testing datasets into X and X_test respectively,
    #encoding them both using TargetEncoder and 
    #finally applying MinMaxScaler over them to normalize the data
    X = dataset_total[dataset_total['training_or_not'] == 1]
    X = TargEnc.fit_transform(X, Y, verbose = 1)
    X = pd.DataFrame(MMScaler.fit_transform(X))
    
    X_test = dataset_total[dataset_total['training_or_not'] == 0]    
    X_test = TargEnc.transform(X_test)    
    X_test = pd.DataFrame(MMScaler.transform(X_test))

    return X, Y, X_test

#import the training and testing datasets
dataset = pd.read_csv(r'H:\income_pred\unedited_data\with_labels_unedited.csv')
dataset_test = pd.read_csv(r'H:\income_pred\unedited_data\without_labels_unedited.csv')

#Perform preprocessing on the datasets
dataset = RemoveFeatures(dataset)
dataset_test = RemoveFeatures(dataset_test)
X, Y, X_test = PreprocessData(dataset, dataset_test)

#Normalizing Y values
MMScaler_Y = MinMaxScaler()
MMScaler_Y = MMScaler_Y.fit(Y)
Y = MMScaler_Y.transform(Y)

#Using RandomForestRegressor to fit the training data and predicting the Incomes into Y_pred
RFregressor = RandomForestRegressor(n_estimators=1000, verbose=1, n_jobs=-1)
RFregressor.fit(X, Y)
Y_pred = RFregressor.predict(X_test)

#Undo scaling of the predictions to get the actual values
Y_pred = MMScaler_Y.inverse_transform(Y_pred.reshape(-1,1))

#Finally, store the predictions into a .csv file
Y_pred = np.array(Y_pred)
with open('chattesu_pred_tcd.csv', 'w') as file:
    for i in np.array(Y_pred):
        file.write(str(i[0]) + "\n")

#print("RMSE: %.4f" % sqrt(mean_squared_error(Y_test,np.exp(Y_pred))))

