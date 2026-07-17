import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🌸 Iris Flower Species Classifier")

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    return pd.read_csv(url)

df = load_data()

st.header("Dataset Preview")
st.dataframe(df.head())

# EDA
st.header("Exploratory Data Analysis")
fig = sns.pairplot(df, hue="species")
st.pyplot(fig)

# Train model
@st.cache_resource
def train_model():
    X = df.drop('species', axis=1)
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    joblib.dump(model, 'iris_model.pkl')
    return model, acc

model, accuracy = train_model()
st.success(f"Model trained! Accuracy: {accuracy:.2f}")

# Prediction
st.header("Make a Prediction")
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

if st.button("Predict Species"):
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Species: **{prediction}**")