from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier 
import joblib 

iris=load_iris() 
x=iris.data
y=iris.target

load_pipeline=joblib.load('iris_pipeline_model.pkl')

def predict(petal_length,petal_width,sepal_length,sepal_width):
   if load_pipeline.predict([[petal_length,petal_width,sepal_length,sepal_width]])==0:
      return "setosa"
   elif load_pipeline.predict([[petal_length,petal_width,sepal_length,sepal_width]])==1:
      return "versicolor"
   else:
      return "virginica"


