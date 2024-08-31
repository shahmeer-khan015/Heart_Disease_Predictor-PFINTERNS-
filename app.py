import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
from sklearn.impute import SimpleImputer
import base64

# Function to add background image from a local file
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        image_bytes = image.read()
        base64_image = base64.b64encode(image_bytes).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{base64_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Path to your background image
image_file = r"C:\Users\siddiqui taj 2024\PycharmProjects\pythonProject\Heart_Disease_Prediction\blue-red-human-heart.jpeg"

# Call the function to set the background
add_bg_from_local(image_file)



# Step 1: Load the data
heart_df = pd.read_csv("heart_disease_uci.csv")

# Step 2: Check for missing values
print(heart_df.isna().sum())

# Step 3: Handle missing values
# Numerical columns imputed with mean
num_imputer = SimpleImputer(strategy='mean')
numerical_columns = ['trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
for col in numerical_columns:
    heart_df[col] = num_imputer.fit_transform(heart_df[[col]])

# Categorical columns imputed with the most frequent value
cat_imputer = SimpleImputer(strategy='most_frequent')
categorical_columns = ['fbs', 'restecg', 'exang', 'slope', 'thal']
for col in categorical_columns:
    heart_df[col] = cat_imputer.fit_transform(heart_df[[col]]).ravel()

# Step 4: Map categorical features to numerical equivalents
heart_df['sex'] = heart_df['sex'].map({'Male': 0, 'Female': 1})

cp_mapping = {
    'typical angina': 0,
    'atypical angina': 1,
    'non-anginal': 2,
    'asymptomatic': 3
}
heart_df['cp'] = heart_df['cp'].map(cp_mapping)

restecg_mapping = {
    'normal': 0,
    'lv hypertrophy': 2,
    'having ST-T wave abnormality': 1
}
heart_df['restecg'] = heart_df['restecg'].map(restecg_mapping)

slope_mapping = {
    'upsloping': 0,
    'flat': 1,
    'downsloping': 2
}
heart_df['slope'] = heart_df['slope'].map(slope_mapping)

thal_mapping = {
    'normal': 3,
    'fixed defect': 6,
    'reversable defect': 7
}
heart_df['thal'] = heart_df['thal'].map(thal_mapping)

# Step 5: Handle the target variable
# Check the unique values in 'num'
print(heart_df['num'].unique())

# Map the target column to binary (0: no heart disease, 1: heart disease)
heart_df['num'] = heart_df['num'].apply(lambda x: 1 if x > 0 else 0)

# Step 6: Drop unnecessary columns
if 'dataset' in heart_df.columns:
    heart_df = heart_df.drop(columns=['dataset'])

# Step 7: Split the dataset into training and testing sets
X = heart_df.drop('num', axis=1)
y = heart_df['num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=2)

# Step 8: Instantiate and fit the model
model = HistGradientBoostingClassifier()
model.fit(X_train, y_train)

# Step 9: Predict on the training data and calculate accuracy
hist_train_pred = model.predict(X_train)
hist_accuracy = accuracy_score(y_train, hist_train_pred)
print(f"HistGradientBoostingClassifier Training Accuracy: {hist_accuracy}")

# Step 10: Streamlit web app for prediction
st.title('Heart Disease Prediction Model')

# Input for the features
input_text = st.text_input('Provide comma separated features to predict heart disease')

# Predict button
if st.button('Predict'):
    sprted_input = input_text.split(',')
    try:
        np_df = np.asarray(sprted_input, dtype=float)
        reshaped_df = np_df.reshape(1, -1)
        prediction = model.predict(reshaped_df)
        if prediction[0] == 0:
            st.write("This person does not have heart disease.")
        else:
            st.write("This person has heart disease.")
    except ValueError:
        st.write('Please provide comma separated values')

# Display data and model performance in Streamlit
st.subheader("About Data")
st.write(heart_df)
st.subheader("Model Performance on Training Data")
st.write(hist_accuracy)
