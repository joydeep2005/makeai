# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder 
# Loading the dataset.
# iris_df = pd.read_csv("iris-species.csv")
st.title('upload the file')
df  = st.file_uploader(' ')
if df is not None:
  iris_df = pd.read_csv(df)
  # if st.button('clk'):
  ap = st.multiselect(
      label='Select the features you want to consider',
      options=iris_df.columns[:-1].tolist()  # <-- Use a list of column names
  )
  # else:
  #   ap = iris_df.columns.tolist() 


  le = LabelEncoder()  # <-- Create a LabelEncoder object
  iris_df['Label'] = le.fit_transform(iris_df.iloc[:,-1])
  # Creating a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.

  # Creating features and target DataFrames.

  X = iris_df[ap]
  y = iris_df['Label']

  # Splitting the data into training and testing sets.
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

  # Creating the SVC model and storing the accuracy score in a variable 'score'.
  
  a = RandomForestClassifier(n_jobs=-1,n_estimators = 100)

  b = LogisticRegression()
  a.fit(X_train,y_train)
  b.fit(X_train,y_train)
  svc_model = SVC(kernel = 'linear')
  svc_model.fit(X_train, y_train)
  score = svc_model.score(X_train, y_train)
  s  = a.score(X_train,y_train)
  c = b.score(X_train,y_train)


  st.title('iris flower prediction')
  st.sidebar.subheader('select values')
  for i in range(len(ap)):
      globals()[ap[i]] = st.sidebar.slider(ap[i],float(iris_df[ap[i]].min()),float(iris_df[ap[i]].max()))
  o = ['LogisticRegression','RandomForestClassifier','SVC']
  z = st.sidebar.radio('select classisfier',o)
  l1=[]
  k = ['x','y','z']
  for i in ap:
    l1.append(globals()[i])

  def predict(z):
    y_pred = z.predict([l1])
    if y_pred[0] ==0:
    	return 'Iris Setosa'
    if y_pred[0] ==1:
    	return 'Iris-virginica'
    else:
    	return 'Iris-versicolor'
  btn = st.button('click')
  if not btn:
    if st.sidebar.button('predict'):

      if z == 'LogisticRegression':
        pred 	=		predict(b)
        st.write('flower: ' ,pred)
        st.write('score:',c,font=50,bold=True)
      if z == 'RandomForestClassifier':

        pred = predict(a)
        st.write('flower: ',pred)
        st.write('score: ',s,font=50,bold=True)
      if z == 'SVC':
        pred = predict(svc_model)
        st.write('flower: ',pred)
        st.write('score: ',score,font=50,bold=True)
  else:
    if st.sidebar.button('predict'):

      if z == 'LogisticRegression':
        pred 	=		predict(b)
        st.write('flower: ' ,pred)
        st.write('score:',c,font=50,bold=True)
      if z == 'RandomForestClassifier':

        pred = predict(a)
        st.write('flower: ',pred)
        st.write('score: ',s,font=50,bold=True)
      if z == 'SVC':
        pred = predict(svc_model)
        st.write('flower: ',pred)
        st.write('score: ',score,font=50,bold=True)   