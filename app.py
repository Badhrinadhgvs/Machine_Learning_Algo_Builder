import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.svm import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from xgboost import XGBRegressor,XGBClassifier
import pickle

st.html("<h1>Machine Learning Model Builder</h1>")
st.error("Note: Make Sure the Dataset is preprocessed and numerical")
problem_type = st.selectbox(label="Select type of problem:",options=['None','Regression','classification'])


if problem_type == 'Regression':
  algo = st.selectbox(label="Select any one algorithm",options=['None','Multiple Linear Regression','Ridge','Lasso','Elastic Net','Support Vector Machine(SVR)','Decision Tree Regressor','Random Forest Regressor','XGB Regressor','KNN Regressor'])
  if algo != 'None':
    test_size=st.number_input(label="Enter test size",min_value=0.20,max_value=1.0)
    df=st.file_uploader(label="Provide CSV File",type='CSV')
    if df:
      df=pd.read_csv(df)
      st.html(f'<h4><p style="color:red">Columns in CSV:</p> {list(df.columns)}</h4>')
      dep=st.selectbox(label="Enter Dependent Variable:",options=df.columns)
      ss=st.checkbox(label="Perform Standard Scaling ?",value=True)
      X=df.drop([dep],axis=1)
      y=df[dep]
      X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size)
      if ss:
        scale=StandardScaler()
        X_train=scale.fit_transform(X_train)
        X_test=scale.transform(X_test)
      if algo == 'Multiple Linear Regression':
        model = LinearRegression().fit(X_train,y_train)
        y_pred=model.predict(X_test)
        st.warning(f"Accuracy Score: {r2_score(y_test,y_pred)*100}")
      elif algo == 'Ridge':
        model = Ridge().fit(X_train,y_train)
        y_pred=model.predict(X_test)
        st.warning(f"Accuracy Score: {r2_score(y_test,y_pred)*100}")
      elif algo == 'Lasso':
        model = Lasso().fit(X_train,y_train)
        y_pred=model.predict(X_test)
        st.warning(f"Accuracy Score: {r2_score(y_test,y_pred)*100}")
      elif algo== 'Elastic Net':
        model = ElasticNet().fit(X_train,y_train)
        y_pred=model.predict(X_test)
        st.warning(f"Accuracy Score: {r2_score(y_test,y_pred)*100}")
      elif algo=='Support Vector Machine(SVR)':
        model = SVR().fit(X_train,y_train)
        y_pred=model.predict(X_test)
        st.warning(f"Accuracy Score: {r2_score(y_test,y_pred)*100}")
      elif algo == 'Decision Tree Regressor':
        model = DecisionTreeRegressor().fit(X_train,y_train)
        y_pred=model.predict(X_test)
        st.warning(f"Accuracy Score: {r2_score(y_test,y_pred)*100}")
      elif algo == 'XGB Regressor':
        model=XGBRegressor().fit(X_train,y_train)
        y_pred=model.predict(X_test)
        st.warning(f"Accuracy Score: {r2_score(y_test,y_pred)*100}")
      elif algo=='KNN Regressor':
        model=KNeighborsRegressor().fit(X_train,y_train)
        y_pred=model.predict(X_test)
        st.warning(f"Accuracy Score: {r2_score(y_test,y_pred)*100}")
      else:
        model = RandomForestRegressor().fit(X_train,y_train)
        y_pred=model.predict(X_test)
        st.warning(f"Accuracy Score: {r2_score(y_test,y_pred)*100}")
      model=pickle.dumps(model)
      st.download_button(label="Download Model",data=model,file_name="Model.pkl",mime='application/octet-stream')
      st.write("Note : Check 👈 to know how to use model")  
  else:
    st.html("<h1>Please Select type of Algorithm</h1>")
  
      
elif problem_type == 'None':
  st.html("<h1>Please Select type of problem</h1>")
else:
  algo = st.selectbox(label="Select any one algorithm",options=['None','Logistic Regression','Support Vector Machine(SVC)','Decision Tree Classifier','Random Forest Classifier','XGB Classifier','KNN Classifier','Naive Bayes'])
  if algo != 'None':
    test_size=st.number_input(label="Enter test size",min_value=0.20,max_value=1.0)
    df=st.file_uploader(label="Provide CSV File",type='CSV')
    if df:
      df=pd.read_csv(df)
      st.html(f'<h4><p style="color:red">Columns in CSV:</p> {list(df.columns)}</h4>')
      dep=st.selectbox(label="Enter Dependent Variable:",options=df.columns)
      ss=st.checkbox(label="Perform Standard Scaling ?",value=True)
      X=df.drop([dep],axis=1)
      y=df[dep]
      X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size)
      if ss:
        scale=StandardScaler()
        X_train=scale.fit_transform(X_train)
        X_test=scale.transform(X_test)
      if algo == 'Logistic Regression':
        model = LogisticRegression().fit(X_train,y_train)
        y_pred=model.predict(X_test)
        st.warning(f"Accuracy Score: {r2_score(y_test,y_pred)*100}")
      elif algo=='Support Vector Machine(SVC)':
        model = SVC().fit(X_train,y_train)
        y_pred=model.predict(X_test)
        st.warning(f"Accuracy Score: {r2_score(y_test,y_pred)*100}")
      elif algo == 'Decision Tree Classifier':
        model = DecisionTreeClassifier().fit(X_train,y_train)
        y_pred=model.predict(X_test)
        st.warning(f"Accuracy Score: {r2_score(y_test,y_pred)*100}")
      elif algo == 'XGB Classifier':
        model=XGBClassifier().fit(X_train,y_train)
        y_pred=model.predict(X_test)
        st.warning(f"Accuracy Score: {r2_score(y_test,y_pred)*100}")
      elif algo=='KNN Classifier':
        model=KNeighborsClassifier().fit(X_train,y_train)
        y_pred=model.predict(X_test)
        st.warning(f"Accuracy Score: {r2_score(y_test,y_pred)*100}")
      else:
        model = RandomForestClassifier().fit(X_train,y_train)
        y_pred=model.predict(X_test)
        st.warning(f"Accuracy Score: {r2_score(y_test,y_pred)*100}")
      model=pickle.dumps(model)
      st.download_button(label="Download Model",data=model,file_name="Model.pkl",mime='application/octet-stream')
      st.write("Note : Check 👈 to know how to use model")
  else:
    st.html("<h1>Please Select type of Algorithm</h1>")
  
  
  
  
footer_html = """
<style>
  .footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #1f2937; /* Dark background color */
    color: white;
    text-align: center;
    padding: 10px 0;
  }
  .footer a {
    color: #9ca3af; /* Light gray color for links */
    margin: 0 10px;
    text-decoration: none;
  }
  .footer a:hover {
    color: white; /* White color on hover */
  }
</style>
<footer class="footer">
    <div>
        <p>&copy; 2024 Created by <b>Venkata Sai Badhrinadh.</p>
        <div>
            <a href="https://www.linkedin.com/in/badhrinadhgvs/">Linkedin</a>
            <a href="https://github.com/Badhrinadhgvs">Github</a>
        </div>
    </div>
</footer>
"""

# Display the footer

st.markdown(footer_html, unsafe_allow_html=True)