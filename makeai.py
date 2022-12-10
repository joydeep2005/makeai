

# Importing the necessary libraries.
#from PIL import Image
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
#icn = Image.open('./Untitled2.png')
#st.set_page_config(page_title='MakeAi',
 #                 page_icon=icn)

# Loading the dataset.
# df = pd.read_csv("iris-species.csv")
st.header('Upload the Data')
df1  = st.file_uploader(' ')
if df1 is not None:
  df = pd.read_csv(df1)
  # if st.button('clk'):
  ap = st.multiselect(
      label='Select the features you want to consider',
      options=df.columns[:-1].tolist()  # <-- Use a list of column names
  )
  # else:
  #   ap = df.columns.tolist() 

  if len(ap)>0:
    le = LabelEncoder()  # <-- Create a LabelEncoder object
    df['Label'] = le.fit_transform(df.iloc[:,-1])
    # Creating a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.
    # Creating features and target DataFrames.
    X = df[ap]
    y = df['Label']

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
        globals()[ap[i]] = st.sidebar.slider(ap[i],float(df[ap[i]].min()),float(df[ap[i]].max()))
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
    st.write('select at least 1 feature')
