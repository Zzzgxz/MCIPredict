#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# Load data
data_path = 'charsl2015Smote0723.csv'  # Replace with your data path
data = pd.read_csv(data_path)

# Select features and target
features = ['Education', 'GS', 'Height', 'Weight', 'CRE', 'PLT','MCV']
X = data[features]
y = data['MCI']

# Define the model
model = RandomForestClassifier(n_estimators=100, random_state=0)

# Setup cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


# Perform cross-validation
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the model
    model.fit(X_train, y_train)

# 创建一个标题
st.title('MCI Screening Model')


# In[7]:


# 创建输入字段，将Education改为下拉选择框
# 创建带描述的下拉选择框
education_options = {
    '1 (文盲)': 1,
    '2 (小学及以下（小学）)': 2,
    '3 (中学及以上（初中、高中或中专、大专及以上）)': 3
}
education = st.selectbox('Education Level', list(education_options.keys()))

# 将描述映射回数字
education_value = education_options[education]

gs = st.number_input('Grip Strength (KG)', min_value=0.0, max_value=100.0, value=30.0)
height = st.number_input('Height (cm)', min_value=100.0, max_value=250.0, value=170.0)
weight = st.number_input('Weight (kg)', min_value=30.0, max_value=200.0, value=70.0)
cre = st.number_input('Creatinine Level (mg/dl)', min_value=0.0, max_value=15.0, value=1.0)
mcv = st.number_input('Mean Corpuscular Volume (fL)', min_value=50.0, max_value=150.0, value=90.0)
plt = st.number_input('Platelet Count (10^9/L)', min_value=1, max_value=500000, step=10000, value=250000)


# In[8]:


# 按钮进行预测
if st.button('Predict'):
    features = np.array([[education_value, gs, height, weight, cre, mcv, plt]])
    prediction = model.predict(features)
    if prediction[0] == 0:
        st.success('患MCI风险小')
    else:
        st.error('患MCI风险大')





