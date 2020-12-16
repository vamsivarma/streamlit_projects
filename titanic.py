import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# For Machine Learning
from sklearn.model_selection import train_test_split

# Preprocessing
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()

# Metrics for analysing the performance

# Classification metrics - Start
# Accuracy score
from sklearn.metrics import accuracy_score
# Example usage: accuracy_score(y_test, y_pred)

# Classification report
from sklearn.metrics import classification_report
# Example usage: print(classification_report(y_test, y_pred))

# Confusion matrix
from sklearn.metrics import confusion_matrix
# Example usage: print(confusion_matrix(y_test, y_pred))

# Precision and Recall
from sklearn.metrics import precision_score, recall_score
# Example usage: print("Precision:", precision_score(Y_train, Y_pred))
# Example usage: print("Recall:",recall_score(Y_train, Y_pred))

# F1-score
from sklearn.metrics import f1_score
# Example usage: f1_score(Y_train, Y_pred)

# ROC curve
from sklearn.metrics import roc_auc_score
# Example usage: roc_auc_score(Y_train, Y_pred)

# Loading Machine Learning Algorithms
# @TODO: Need to load the optimal hyparameters list
# to each of these algorithms
from sklearn import linear_model

# GB
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=100,max_depth=5)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)

# Logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

# SVM
#from sklearn.svm import SVC, LinearSVC
from sklearn.svm import SVC
svc = SVC(kernel='linear')

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

from sklearn.linear_model import Perceptron

# SGD
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()

# DT
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

# Wrapper for EDA operations
import eda_wrapper as edaw
# Wrapper which contains code for doing Exploratory data analysis
eda_obj = edaw.EDA_Wrapper()

@st.cache
def get_data():
    return pd.read_csv("https://raw.githubusercontent.com/vamsivarma/datasets/master/data_science/pandas/titanic.csv")

df = get_data()
st.title("ML workflow using titanic")

st.header("Machine Learning workflow")
st.markdown("The first five records of the data we downloaded.")
st.dataframe(df.head())

st.markdown("Filter dataset by choosing only the columns that interest you")
cols = st.multiselect("Columns", df.columns.tolist(), default=df.columns.tolist())
st.dataframe(df[cols])

st.markdown("<hr>", unsafe_allow_html=True)

st.header("Dataset Analysis")
ds_summary = eda_obj.get_ds_summary(df)

st.subheader("Dataset Structure")
st.markdown(ds_summary['properties'], unsafe_allow_html=True)

st.subheader("Summmary of Numerical Features")
st.markdown(ds_summary['numeric'], unsafe_allow_html=True)

st.subheader("Summary of Categorical Features")
st.markdown(ds_summary['categorical'], unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

st.header("Exploratory Data Analysis")

numerics = ['float16', 'float32', 'float64'] # 'int16', 'int32', 'int64', 
new_df = df.select_dtypes(include=numerics)

# Scatter plot
st.subheader('Comparing numerical features using Scatter plot')
s_col1 = new_df.columns[0]
s_col2 = new_df.columns[1]

s_col1 = st.selectbox('Which feature on x?', new_df.columns)
s_col2 = st.selectbox('Which feature on y?', new_df.columns)

# create figure using plotly express
fig = px.scatter(new_df, x =s_col1,y=s_col2)
# Plot!
st.plotly_chart(fig)

# Histogram
st.subheader('Histogram for understanding distribution of a feature')
hist_col = st.selectbox('Select a feature', new_df.columns)

# create figure using plotly express
fig2 = px.histogram(new_df, x=hist_col, marginal="rug")

# Plot!
st.plotly_chart(fig2)

cats = ['object', 'bool']
cat_df = df.select_dtypes(include=cats)

# Pie chart
st.subheader('Understanding Categorical features with Pie chart')
cat_col = st.selectbox('Select a feature', cat_df.columns)

# Get the distribution of label column
label_vc = cat_df[cat_col].value_counts()

# Convert it in to dictionary
label_vc_dict = pd.DataFrame(label_vc).to_dict('dict')

# Get the label field categories
vc_dict_keys = list(label_vc_dict[cat_col].keys())

# values for that categories
vc_dict_values = []

for k in vc_dict_keys:
    vc_dict_values.append(label_vc_dict[cat_col][k])

labels = vc_dict_keys
values = vc_dict_values

# create figure using plotly express
cat_title = "<b>Distribution of " + cat_col + "</b>"

#fig3 = px.pie(cat_grp_df, values=values, names=labels)
#fig3.update_layout(title=cat_title)

data = [go.Pie(labels=labels, values=values)]

layout = go.Layout(
    title = cat_title,
    hovermode ='closest'
)

fig3 = go.Figure(data=data, layout=layout)
st.plotly_chart(fig3)

st.subheader('Data Agregation')
groupby_col = st.selectbox('Select a feature for group by', cat_df.columns)
agg_col = st.selectbox('Select a feature for aggregation', new_df.columns)
tabh_name = "Average " + agg_col

st.table(df.groupby(groupby_col)[agg_col].mean().reset_index()\
.round(2).sort_values(agg_col, ascending=False)) #\
#.assign( agg_col = lambda x: x.pop(agg_col).apply(lambda y: "%.2f" % y)))

st.markdown("<hr>", unsafe_allow_html=True)

st.header('Applying Machine Learning')

cat_cols = ['object', 'bool']
cats_df = df.select_dtypes(include=cat_cols)
ml_df = df.copy()

for col in cats_df.columns:
    ml_df[col] = enc.fit_transform(ml_df[col].astype(str))

ml_df.fillna(0, inplace=True)

output_col = st.selectbox('Select a output feature', cats_df.columns)

alg_list = ["Logistic regression", "Random Forest", "SGD", "Decision tree", "K Nearest Neighbour", 
            "Gaussian Naive Bayes", "Gradient Boosting"]

alg_map = {
    "Logistic regression": lr,
    "Random Forest": rfc,
    "SGD": sgd,
    "Decision tree": dt,
    "K Nearest Neighbour": knn,
    "Gaussian Naive Bayes": gnb,
    "Gradient Boosting": gbc
}

t_cols_list = [
    'Algoritm', 'Train Accuracy', 'Test Accuracy', 
    'Train Precision', 'Test Precision', 
    'Train Recall', 'Test Recall',
    'Train F1 Score', 'Test F1 Score' #,
    #'Train ROC', 'Test ROC'
]

if output_col != '':

    features = ml_df.drop(output_col, axis=1).astype(float).values
    labels = ml_df[output_col].values

    X_train,X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)
    
    classifier_list = st.multiselect("Select ML algorithms: ", alg_list)

    #initialize model list and dicts
    models = []

    metric_values = []

    index = 0

    

    if classifier_list: 

        model_obj_list = []
        for c in classifier_list:
            if c in alg_map:
                model_obj_list.append(alg_map[c])
        


        models.extend(model_obj_list)
        
    # Find accuracy for every model
    for model in models:

        model.fit(X_train, y_train)
        # Prediction
        y_train_pred = model.predict(X_train)

        model.fit(X_test, y_test)
        # Prediction
        y_test_pred = model.predict(X_test)

        cm = confusion_matrix(y_test,y_test_pred)
        st.write('Confusion matrix for ' + classifier_list[index]  + ' :', cm)


        cur_metric_values = [classifier_list[index]]

        # accuracy
        acc_model_train = accuracy_score(y_train,y_train_pred)
        acc_model_test = accuracy_score(y_test,y_test_pred)
        cur_metric_values.append(acc_model_train)
        cur_metric_values.append(acc_model_test)

        # precision
        p_score_train = precision_score(y_train, y_train_pred, average='weighted')
        p_score_test = precision_score(y_test, y_test_pred, average='weighted')
        cur_metric_values.append(p_score_train)
        cur_metric_values.append(p_score_test)

        # recall
        r_score_train = recall_score(y_train, y_train_pred, average='weighted')
        r_score_test = recall_score(y_test, y_test_pred, average='weighted')
        cur_metric_values.append(r_score_train)
        cur_metric_values.append(r_score_test)

        # F1 score
        f1m_score_train = f1_score(y_train, y_train_pred, average='weighted')
        f1m_score_test = f1_score(y_test, y_test_pred, average='weighted')
        cur_metric_values.append(f1m_score_train)
        cur_metric_values.append(f1m_score_test)

        # ROC score - since ROC only supports binary classification
        #roc_score_train = roc_auc_score(y_train, y_train_pred)
        #roc_score_test = roc_auc_score(y_test, y_test_pred)
        #cur_metric_values.append(roc_score_train)
        #cur_metric_values.append(roc_score_test)

        metric_values.append(cur_metric_values)

        index += 1

    # Convert the metric results in to dataframe sorted by accuracy value
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    metric_df = pd.DataFrame(metric_values, columns = t_cols_list)
    metric_df = metric_df.sort_values("Train Accuracy",ascending=False)

    st.markdown(metric_df.to_html(index=False), unsafe_allow_html=True)
