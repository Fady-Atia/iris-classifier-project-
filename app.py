import streamlit as st  
import pandas as pd 
from sklearn.datasets import load_iris 
from model_final.model import predict 

st.title("Hi my is :blue[Fady] :sunglasses:")

st.divider()

st.subheader("Explore Data")
df = pd.DataFrame(load_iris().data, columns=load_iris().feature_names) 
st.dataframe(df, width=1000, height=300)

st.divider() 

col1,col2=st.columns(2) 
with col1:
    st.metric(label="number of rows", value=df.shape[0]) 
with col2:
        st.metric(label="number of columns", value=df.shape[1])  

tab1, tab2 = st.tabs(["line chart", "scatter chart"])  
with tab1:
    st.line_chart(df, x="sepal length (cm)", y="sepal width (cm)", use_container_width=True)
with tab2:
    st.scatter_chart(df, x="sepal length (cm)", y="sepal width (cm)", use_container_width=True) 
 

x1 = st.slider("PETAL WIDTH", 0.0, 10.0, 1.0, step=0.1) 
x2 = st.slider("PETAL LENGTH", 0.0, 10.0, 1.0, step=0.1)
x3 = st.slider("SEPAL WIDTH", 0.0, 10.0, 1.0, step=0.1)
x4 = st.slider("SEPAL LENGTH", 0.0, 10.0, 1.0, step=0.1) 

if st.button ("Predict", type="primary"):
   result=predict(x1,x2,x3,x4)
   st.success(f"The predicted iris species is: {result}")
 

