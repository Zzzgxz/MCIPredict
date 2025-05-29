#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, roc_auc_score

# Function to calculate sensitivity and specificity
def sensitivity_specificity(conf_matrix):
    tn, fp, fn, tp = conf_matrix.ravel()
    sensitivity = tp / (tp + fn)  # Same as recall
    specificity = tn / (tn + fp)
    return sensitivity, specificity

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


# In[14]:


import shap
import streamlit as st
# SHAP解释器初始化
explainer = shap.TreeExplainer(model)

# Streamlit Web 页面
st.title('MCI Prediction and Explanation Dashboard')


# In[5]:


import warnings
warnings.filterwarnings('ignore')


# In[6]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib


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

gs = st.number_input('Grip Strength (KG)', 0.0, 100.0, 30.0)
height = st.number_input('Height (cm)', 100.0, 250.0, 170.0)
weight = st.number_input('Weight (kg)', 30.0, 200.0, 70.0)
cre = st.number_input('Creatinine (mg/dl)', 0.0, 15.0, 1.0)
plt_count = st.number_input('Platelet Count (10^9/L)', 1, 500000, 250000, step=10000)
mcv = st.number_input('Mean Corpuscular Volume (fL)', 50.0, 150.0, 90.0)


# In[13]:


if st.button('Predict'):
    input_data = np.array([[education_value, gs, height, weight, cre, plt_count, mcv]])
    input_df = pd.DataFrame(input_data, columns=features)

    prediction = model.predict(input_data)
    if prediction[0] == 0:
        st.success('✅ 患 MCI 的风险较低')
    else:
        st.error('⚠️ 患 MCI 的风险较高')

    st.subheader("🔍 SHAP Force Plot: Explanation for This Prediction")
    
    try:
        shap_values = explainer.shap_values(input_df)
        sv = shap_values[0, :, 1]
        expected_value = explainer.expected_value[1]
        feature_row = input_df.iloc[0]

        from streamlit.components.v1 import html
        force_plot_obj = shap.plots.force(
            expected_value,
            sv,
            feature_row,
            matplotlib=False
        ).html()
        html(shap.getjs(), height=0)
        html(force_plot_html, height=300)  # ✅ 只嵌入纯 HTML 字符串
    except Exception as e:
        st.error("❌ SHAP force plot 生成失败:")
        st.write(str(e))


# In[ ]:




