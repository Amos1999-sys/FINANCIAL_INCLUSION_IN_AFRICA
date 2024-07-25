#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
#machine learning
#from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
plt.rcParams["axes.labelsize"] = 18
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

train = pd.read_csv('C:\\Users\Methuselah\\Desktop\\financial-inclusion-in-africa\\Train.csv')
test = pd.read_csv('C:\\Users\Methuselah\\Desktop\\financial-inclusion-in-africa\\Test.csv')
ss = pd.read_csv('C:\\Users\Methuselah\\Desktop\\financial-inclusion-in-africa\\SampleSubmission.csv')
variables = pd.read_csv('C:\\Users\Methuselah\\Desktop\\financial-inclusion-in-africa\\VariableDefinitions.csv')


# In[2]:


# Let's observe the shape of our datasets
print('train data shape :',train.shape)
print('test data shape :', test.shape)


# In[3]:


#show list of columns in train data
list(train.columns)


# In[4]:


list(test.columns)


# In[5]:


#inspect train data
train.head()


# In[6]:


#check for missing values
print('missing values:', train.isnull().sum().sum())


# In[7]:


#Explore Target bistribution
sb.catplot(x="bank_account", kind="count", data=train, palette="Set1")


# In[8]:


# view the submission file
ss.head()


# In[9]:


#show some information about the dataset
print(train.info())


# In[10]:


# Let's view the variables
variables.T


# In[11]:


train['bank_account'].value_counts()


# In[12]:


sb.catplot(x="country", kind="count", data=train, palette="colorblind")


# In[13]:


sb.catplot(x="location_type", kind="count", data=train, palette="colorblind")


# In[14]:


sb.catplot(x="year", kind="count", data=train, palette="colorblind")


# In[15]:


sb.catplot(x="cellphone_access", kind="count", data=train, palette="colorblind")


# In[16]:


sb.catplot(x="gender_of_respondent", kind="count", data=train, palette="colorblind")


# In[17]:


sb.catplot(x="relationship_with_head", kind="count", data=train, palette="colorblind")

plt.xticks(
rotation=45,
horizontalalignment='right',
fontweight='light',
fontsize='x-large')


# In[18]:


sb.catplot(x="marital_status", kind="count", data=train, palette="colorblind")

plt.xticks(
rotation=45,
horizontalalignment='right',
fontweight='light',
fontsize='x-large')


# In[19]:


sb.catplot(x="education_level", kind="count", data=train, palette="colorblind")

plt.xticks(
rotation=45,
horizontalalignment='right',
fontweight='light',
fontsize='x-large')


# In[20]:


sb.catplot(x="job_type", kind="count", data=train, palette="colorblind")

plt.xticks(
rotation=45,
horizontalalignment='right',
fontweight='light',
fontsize='x-large')


# In[21]:


plt.figure(figsize=(16,6))
train.household_size.hist()
plt.xlabel('household  size')


# In[22]:


plt.figure(figsize=(16,6))
sb.catplot(x='location_type', hue='bank_account', kind='count', data=train)
plt.xticks(
fontweight='light',
fontsize='x-large')


# In[23]:


plt.figure(figsize=(16,6))
sb.catplot(x='cellphone_access', hue='bank_account', kind='count', data=train)
plt.xticks(
fontweight='light',
fontsize='x-large')


# In[24]:


#import preprocessing module
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
 
#convert target label to numerical Data
le = LabelEncoder()
train['bank_account'] = le.fit_transform(train['bank_account'])

#separate training features from target
x_train = train.drop(['bank_account'], axis=1)
y_train = train['bank_account']

print(y_train)


# In[25]:


#funtion of preprocess our data from train models
def preprocessing_data(data):
    
    #convert the following numerical labels from interger to float
  float_array = data[["household_size", "age_of_respondent", "year"]].values.astype(float)

    #categorical features to be onverted to one Hot Encoding
  categ = ["relationship_with_head", "marital_status", "education_level", "job_type", "country"]

   #one Hot Encoding conversion
  data = pd.get_dummies(data, prefix_sep="_", columns=categ)

    #Label Encoder conversion
  data["location_type"] = le.fit_transform(data["location_type"])
  data["cellphone_access"] = le.fit_transform(data["cellphone_access"])
  data["gender_of_respondent"] = le.fit_transform(data["gender_of_respondent"])

    #drop uniqueid column
  data = data.drop(["uniqueid"], axis=1)

   #scale our data into range of 0 and 1
  scaler = MinMaxScaler(feature_range=(0, 1))
  data = scaler.fit_transform(data)

  return data


# In[26]:


#prerocess the train data
preprocessed_train=preprocessing_data(x_train)
preprocessed_test=preprocessing_data(test)


# In[27]:


#view the first row of the processed_train dataset after preprocessing
#Inclusive of Start, Exclusive of End
print(preprocessed_train[:2])


# In[28]:


import sklearn.model_selection


# In[29]:


#split train_data
from sklearn.model_selection import train_test_split
    
X_Train, x_Val, y_Train, y_val = train_test_split(preprocessed_train, y_train, stratify = y_train, test_size = 28, random_state=42)
    


# In[37]:


#import classifier algorithm here
from xgboost import XGBClassifier

#create models
xg_model = XGBClassifier()

#fitting the models
xg_model.fit(x_Train, y_Train)


# In[55]:


# import evaluation metrics
from sklearn.metrics import confusion_matrix, accuracy_score

# evaluate the model
xg_y_model = xg_model.predict(x_Val)
accuracy_score = accuracy_score(y_val,xg_y_model)
print('Accuracy is = ', accuracy_score)

# Get error rate
print("Error rate of XGB classifier: ", 1 - accuracy_score)


# In[56]:


#print the classification report
from sklearn.metrics import classification_report
    
report = classification_report(y_val, xg_y_model)
print(report)


# In[45]:


# calculate the accuracy and prediction of the model
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay 
    
    
xgboost_model_predicted = xg_model.predict(x_Val)
score = accuracy_score(y_val, xgboost_model_predicted)
print("Error rate for XGBClassifier model is:", 1-score)
#Calculate confusion matrix
cm = confusion_matrix(y_val, xgboost_model_predicted, normalize='true')
print(".conda\confusion Matrix:")
print(cm)
#Plot confusion matrix as aheatmap
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_val))
disp.plot(cmap='viridis', values_format='.2f')
plt.title("Confusion Matrics")
plt.show


# In[59]:


#Get the predicted result for the test Data
test.bank_account = xg_model.predict(preprocessed_test)


# In[60]:


#Create submission DataFrame
submission = pd.DataFrame({"uniqueid": test["uniqueid"] + " x " + test["country"], "bank_account": test.bank_account})


# In[61]:


#show the five sample
submission.sample(15)


# In[62]:


#create a Submission file in Jupyter notebook and downloadit
from IPython.display import FileLink
submission.to_csv('submission1.csv', index=False)


# In[63]:


#Display a download link
FileLink('submission1.csv')


# In[64]:


from IPython.display import FileLink
submission.to_excel('submission2.xlsx', index=False)


# In[65]:


FileLink('submission2.xlsx')

