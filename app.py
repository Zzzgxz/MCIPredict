#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, confusion_matrix, recall_score, f1_score

# 读取数据
data = pd.read_csv('/Users/wuxuan/jupyter/202406MCI-ML/imputedData/charls2015MICE/charls2015_imputed_1.csv')

# 选取特征和目标变量
selected_features = ['Education', 'GS', 'Height', 'Weight', 'CRE', 'MCV', 'PLT']
X = data[selected_features]
y = data['MCI']

# 使用SMOTE进行过采样
smote = SMOTE(random_state=42)
X_upsampled, y_upsampled = smote.fit_resample(X, y)

# 初始化随机森林模型
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 计算特异性函数
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# 使用交叉验证来评估模型
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
sensitivity_scores = cross_val_score(rf_classifier, X_upsampled, y_upsampled, cv=cv, scoring=make_scorer(recall_score))
specificity_scores = cross_val_score(rf_classifier, X_upsampled, y_upsampled, cv=cv, scoring=make_scorer(specificity_score))
auc_scores = cross_val_score(rf_classifier, X_upsampled, y_upsampled, cv=cv, scoring='roc_auc')
f1_scores = cross_val_score(rf_classifier,X_upsampled, y_upsampled, cv=cv, scoring='f1')

# 输出平均敏感性、特异性和AUC值
avg_sensitivity = np.mean(sensitivity_scores)
avg_specificity = np.mean(specificity_scores)
avg_auc = np.mean(auc_scores)
avg_f1 = np.mean(f1_scores)

print("Average Sensitivity:", avg_sensitivity)
print("Average Specificity:", avg_specificity)
print("Average AUC:", avg_auc)
print("Average F1 Score:", avg_f1)

# 训练随机森林模型使用全部上采样后的数据
rf_classifier.fit(X_upsampled, y_upsampled)


# In[8]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


# In[18]:


# 创建一个标题
st.title('MCI Prediction Model')


# In[16]:


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


# In[20]:


# 按钮进行预测
if st.button('Predict'):
    features = np.array([[education_value, gs, height, weight, cre, mcv, plt]])
    prediction = model.predict(features)
    if prediction[0] == 0:
        st.success('患MCI风险小')
    else:
        st.error('患MCI风险大')


# In[ ]:





# In[ ]:




