#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[47]:


plt.rcParams["figure.figsize"] = (20,12)


# In[48]:


import warnings 
warnings.filterwarnings('ignore')


# In[49]:


#Load datanya!

df = pd.DataFrame(pd.read_csv('survey 2.csv'))


# In[50]:


#Lihat dataset nya!

df.head(5)


# In[51]:


#Struktur dataset!

df.info()


# In[52]:


df.shape


# In[53]:


#Persentase missing value

total_missing = df.isnull().sum().sort_values(ascending = False)
total_non_missing = df.count().sort_values(ascending = False)
percent_missing = total_missing/(total_non_missing+total_missing)*100
df_missing = pd.concat([total_missing, total_non_missing, percent_missing], axis = 1, keys = ['Total Missing Values',
                                                                                                                        'Total Non Missing Values',
                                                                                                                        'Persentase Missing Values'])
df_missing.head(10)


# In[54]:


#Hapus kolom yang memiliki persentase missing values >= 70%

df.drop('comments', axis=1, inplace=True)
df.columns


# In[55]:


#Pisah data ke kelompok tipe data numerik dan object!

tipe_object = ['Timestamp', 'Gender', 'Country', 'state', 'self_employed',
       'family_history', 'treatment', 'work_interfere', 'no_employees',
       'remote_work', 'tech_company', 'benefits', 'care_options',
       'wellness_program', 'seek_help', 'anonymity', 'leave',
       'mental_health_consequence', 'phys_health_consequence', 'coworkers',
       'supervisor', 'mental_health_interview', 'phys_health_interview',
       'mental_vs_physical', 'obs_consequence']

tipe_numerik = ['Age']


# In[56]:


#Statistik sederhana untuk kelompok numerik!

df[tipe_numerik].describe().transpose()


# In[57]:


#Statistik sederhana untuk kelompok object!

df[tipe_object].describe().transpose()


# ## **Imputasi Missing Values**

# In[58]:


#Visualisasi Kolom 'state'

sns.countplot(df['state'])
plt.title("Count Plot fitur state!", fontsize=15)

plt.show()


# In[59]:


# Imputasi Missing Values kolom 'state'

df['state'].fillna(df['state'].mode()[0], inplace = True)


# In[60]:


#Visualisasi Kolom 'work_interfere'

plt.subplot(1, 2, 1)
sns.countplot(df['work_interfere'])
plt.title("Count Plot fitur state!", fontsize=20)

plt.subplot(1, 2, 2)
pie_chart = df['work_interfere'].value_counts()
pie_chart.plot.pie(autopct = '%1.1f%%', figsize = (15,10))

plt.title('Persentase pada fitur state', fontsize = 20)

plt.show()


# In[61]:


# Imputasi Missing Values kolom 'work_interfere'

df['work_interfere'].fillna(df['work_interfere'].mode()[0], inplace = True)


# In[62]:


#Visualisasi Kolom 'self_employed'

plt.subplot(1, 2, 1)
sns.countplot(df['self_employed'])
plt.title("Count Plot fitur self_employed!", fontsize=20)

plt.subplot(1, 2, 2)
pie_chart = df['self_employed'].value_counts()
pie_chart.plot.pie(autopct = '%1.1f%%', figsize = (15,10))

plt.title('Persentase pada fitur self_employed', fontsize = 20)

plt.show()


# In[63]:


# Imputasi Missing Values kolom 'work_interfere'

df['self_employed'].fillna(df['self_employed'].mode()[0], inplace = True)


# In[64]:


# Cek missing values pada data

df.isnull().sum()


# ## **Data Preprocessing**

# In[65]:


#Perhatikan bahwa nilai unik pada fitur Gender berjumlah 49 (Harusnya ada 3, laki-laki, perempuan, atau kategori lainnya)

df['Gender'].value_counts()


# In[66]:


df['Gender'].unique()


# In[67]:


#Jadikan 3 Gender saja

df['Gender'].replace(['M', 'Male', 'male', 'm', 'Male-ish', 'maile',
       'something kinda male?', 'Cis Male', 'Mal', 'Male (CIS)', 'Make','Guy (-ish) ^_^', 
       'Male ', 'Man', 'msle', 'Mail', 'cis male', 'Malr', 'Cis Man'], 'Male', inplace = True)

df['Gender'].replace(['Female', 'female', 'Cis Female', 'F', 
       'Woman', 'f', 'queer/she/they','non-binary', 'Femake', 'woman', 'Genderqueer', 'Female ', 
       'cis-female/femme', 'Female (cis)', 'femail'], 'Female', inplace = True)

df['Gender'].replace(['Trans-female', 'non-binary','Nah', 'All', 'Enby', 'fluid', 'Genderqueer',  'Androgyne', 'Agender',
       'male leaning androgynous', 'Trans woman', 'Neuter', 'Female (trans)','queer','A little about you','p',  'ostensibly male, unsure what that really means'],
       'other_categories', inplace = True)


# In[68]:


df['Gender'].value_counts()


# In[69]:


# Perhatikan bahwa nilai fitur Age atau umur ada yang negatif dan ada yang lebih besar dari 100 (bahkan 1000 an)
# Ganti nilai2 aneh tsb dengan median

df['Age'] = df[(df['Age'] >= 0) & (df['Age'] <= 100)]['Age']
df['Age'].fillna(df['Age'].median(), inplace = True)

df['Age'].describe()


# In[ ]:





# In[70]:


#Hapus kolom Timestamp yang sekiranya kurang berguna

df.drop('Timestamp', axis = 1, inplace = True)


# ## **Modelling**

# In[71]:


df['treatment'].value_counts()


# In[72]:


#Ubah variabel kategorik ke numerik!

df['treatment'].replace('Yes', 1, inplace = True)
df['treatment'].replace('No', 0, inplace = True)

df = pd.get_dummies(df)
df.shape


# In[73]:


#Visualisasi variabel target 'treatment'

plt.subplot(1, 2, 1)
sns.countplot(df['treatment'])
plt.title("Count Plot fitur treatment!", fontsize=20)

plt.subplot(1, 2, 2)
pie_chart = df['treatment'].value_counts()
pie_chart.plot.pie(autopct = '%1.1f%%', figsize = (15,10))

plt.title('Persentase pada fitur treatment', fontsize = 20)

plt.show()


# In[74]:


# Pisah dataset menjadi variabel2 prediktor dan variabel target

X = df.drop('treatment', axis = 1)
y = df['treatment']


# In[75]:


X.shape


# In[76]:


y.shape


# In[77]:


# Train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2, random_state = 42)


# In[78]:


from sklearn.metrics import accuracy_score


# In[79]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV


# In[80]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter=1000, solver="sag")
logreg.fit(X_train, y_train)

y_pred_logreg = logreg.predict(X_test)
print(classification_report(y_test, y_pred_logreg))


# In[81]:


#Skor ROC-AUC Model Gradient Boosting (Setelah Hyperparameter Tuning)

print(roc_auc_score(y_test, y_pred))


# In[82]:


#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dec_tree = DecisionTreeClassifier(criterion="gini", max_depth=8)
dec_tree.fit(X_train, y_train)

y_pred_dectree = dec_tree.predict(X_test)
print(classification_report(y_test, y_pred_dectree))


# In[83]:


#Skor ROC-AUC Model Gradient Boosting (Setelah Hyperparameter Tuning)

print(roc_auc_score(y_test, y_pred))


# In[84]:


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
r_forest=RandomForestClassifier(max_depth=22, warm_start=True)
r_forest.fit(X_train,y_train)

y_pred_forest = r_forest.predict(X_test)
print(classification_report(y_test, y_pred_forest))


# In[85]:


#Skor ROC-AUC Model Gradient Boosting (Setelah Hyperparameter Tuning)

print(roc_auc_score(y_test, y_pred))


# In[86]:


#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gboost=GradientBoostingClassifier(learning_rate=0.2)
gboost.fit(X_train, y_train)

y_pred_gboost = gboost.predict(X_test)
print(classification_report(y_test, y_pred_gboost))


# In[87]:


#Skor ROC-AUC Model Gradient Boosting (Setelah Hyperparameter Tuning)

print(roc_auc_score(y_test, y_pred))


# In[88]:


#Logistic Regression dengan Hyperparameter Tuning

hyperparameters = dict(penalty=['l1', 'l2', 'elasticnet'], C=np.logspace(-4,4,20))
clf = GridSearchCV(logreg, hyperparameters, cv=5)

#Fitting Model
logreg_best_model = clf.fit(X_train,y_train)

#Print hyperparameters terbaik
print('Best Penalty:', logreg_best_model.best_estimator_.get_params()['penalty'])
print('Best C:', logreg_best_model.best_estimator_.get_params()['C'])

#Prediksi dengan model hasil parameter tuning
y_pred = logreg_best_model.predict(X_test)

print(classification_report(y_test, y_pred))


# In[89]:


#Skor ROC-AUC Model Gradient Boosting (Setelah Hyperparameter Tuning)

print(roc_auc_score(y_test, y_pred))


# In[90]:


#Decision Tree dengan Hyperparameter Tuning

dectree_params = {
    'criterion' : ['gini', 'entropy'],
    'max_depth' : range(1,10),
    'min_samples_split' : range(1,5),
    'min_samples_leaf' : range(1,5)
}

clf = GridSearchCV(dec_tree, dectree_params, cv=5)

#Fitting Model
dectree_best_model = clf.fit(X_train,y_train)

#Print hyperparameters terbaik
print('Best Hyperparameters:', dectree_best_model.best_estimator_.get_params())

#Prediksi dengan model hasil parameter tuning
y_pred = dectree_best_model.predict(X_test)

print(classification_report(y_test, y_pred))


# In[91]:


#Skor ROC-AUC Model Gradient Boosting (Setelah Hyperparameter Tuning)

print(roc_auc_score(y_test, y_pred))


# In[92]:


#Gradient Boosting Classifier dengan Hyperparameter Tuning

gb_params = {
              'learning_rate': [0.05, 0.1, 0.2, 0.3],
              'max_depth': [4, 6, 8],
              'min_samples_leaf': [20, 50,100,150],
              'max_features': ['auto', 'sqrt', 'log2', 'None'] 
              }

clf = GridSearchCV(gboost, gb_params, cv=5)

#Fitting Model
gboost_best_model = clf.fit(X_train,y_train)

#Print hyperparameters terbaik
print('Best Hyperparameters:', gboost_best_model.best_estimator_.get_params())

#Prediksi dengan model hasil parameter tuning
y_pred = gboost_best_model.predict(X_test)

print(classification_report(y_test, y_pred))


# In[93]:


#Skor ROC-AUC Model Gradient Boosting (Setelah Hyperparameter Tuning)

print(roc_auc_score(y_test, y_pred))


# In[94]:


import simplejson as json


# In[96]:


class MyLogReg(LogisticRegression):

    # Override the class constructor
    def __init__(self, C=0.08858667904100823, solver='liblinear', max_iter=100, X_train=None, Y_train=None):
        LogisticRegression.__init__(self, C=C, solver=solver, max_iter=max_iter)
        self.X_train = X_train
        self.Y_train = y_train

    # A method for saving object data to JSON file
    def save_json(self, filepath):
        dict_ = {}
        dict_['C'] = self.C
        dict_['max_iter'] = self.max_iter
        dict_['solver'] = self.solver
        dict_['X_train'] = self.X_train.values.tolist() if self.X_train is not None else 'None'
        dict_['Y_train'] = self.Y_train.values.tolist() if self.Y_train is not None else 'None'

        # Creat json and save to file
        json_txt = json.dumps(dict_, indent=4)
        with open(filepath, 'w') as file:
            file.write(json_txt)

    # A method for loading data from JSON file
    def load_json(self, filepath):
        with open(filepath, 'r') as file:
            dict_ = json.load(file)

        self.C = dict_['C']
        self.max_iter = dict_['max_iter']
        self.solver = dict_['solver']
        self.X_train = np.asarray(dict_['X_train']) if dict_['X_train'] != 'None' else None
        self.Y_train = np.asarray(dict_['Y_train']) if dict_['Y_train'] != 'None' else None


# In[97]:


filepath = "LOGREG_BANGKIT.json"

# Create a model and train it
mylogreg = MyLogReg(X_train=X_train, Y_train=y_train)  
mylogreg.save_json(filepath)

# Create a new object and load its data from JSON file
json_mylogreg = MyLogReg()  
json_mylogreg.load_json(filepath)  
json_mylogreg


# In[ ]:




