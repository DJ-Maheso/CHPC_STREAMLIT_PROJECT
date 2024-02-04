import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

header = st.container()
dataset = st.container()
features = st.container()
modeltraining = st.container()

with header:
    st.title('IM trying streamlit')
    st.text('I have gamma-ray sources and i want the app to illustre their parameters')

    
with dataset:
    st.header('IM trying streamlit')
    
    mydata = pd.read_csv('data_2.csv')
    st.write(mydata.head())
    
    st.subheader('')
    distribution = pd.DataFrame(mydata['t90'].value_counts())
    st.write(distribution)
    st.bar_chart(mydata, x="source", y=['Epeak', 't90','redshift'])

with features:
    st.header('IM trying streamlit')
    
with modeltraining:
    st.header('IM trying streamlit')
    st.text('change redshift and see how the wole thing changes')
    sel_col, disp_col = st.columns(2)
    slide = sel_col.slider('what kind of source is this', min_value=0, max_value=10, value=1)
    n_estimators = sel_col.selectbox('im not sure', options=[1,3,5,10],index=0)
    input_features = sel_col.text_input('somethingg','redshift')
    regr = RandomForestRegressor(max_depth = slide, n_estimators=n_estimators)
    x_1= mydata[[input_features]]
    y_1= mydata[['redshift']]

    regr.fit(x_1,y_1)
    prediction = regr.predict(y_1)

    disp_col.subheader('mean absolute error of the model is:')
    disp_col.write(mean_absolute_error(y_1,prediction))

    disp_col.subheader('mean absolute error of the model is:')
    disp_col.write(mean_squared_error(y_1,prediction))

    disp_col.subheader('mean absolute error of the model is:')
    disp_col.write(r2_score(y_1,prediction))
