

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from sklearn.preprocessing import LabelEncoder#for encoding
from sklearn.model_selection import train_test_split#for train test splitting
from sklearn.tree import DecisionTreeClassifier#for decision tree object
from sklearn.metrics import classification_report, confusion_matrix#for checking testing results
from sklearn.tree import plot_tree#for visualizing tree

#reading the data
df = pd.read_csv('C:/Users/Masum/Downloads/Company_Data.csv')
df.head()

#getting information of dataset
df.info()

df.shape

df.isnull().any()

# let's plot pair plot to visualise the attributes all at once
sns.pairplot(data=df, hue = 'ShelveLoc')

#Creating dummy vairables dropping first dummy variable
df=pd.get_dummies(df,columns=['Urban','US'], drop_first=True)

df

df.info()

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

df['ShelveLoc']=df['ShelveLoc'].map({'Good':1,'Medium':2,'Bad':3})

df.head()

x=df.iloc[:,0:6]
y=df['ShelveLoc']

x

y

df['ShelveLoc'].unique()

df.ShelveLoc.value_counts()

colnames = list(df.columns)
colnames

# Splitting data into training and testing data set
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)

"""# Building Decision Tree Classifier using Entropy Criteria"""

model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)

from sklearn import tree

#PLot the decision tree
tree.plot_tree(model);

fn=['Sales','CompPrice','Income','Advertising','Population','Price']
cn=['1', '2', '3']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn,
               class_names=cn,
               filled = True);

#Predicting on test data
preds = model.predict(x_test) # predicting on test data set
pd.Series(preds).value_counts() # getting the count of each category

preds

pd.crosstab(y_test,preds) # getting the 2 way table to understand the correct and wrong predictions

# Accuracy
np.mean(preds==y_test)

"""# Building Decision Tree Classifier (CART) using Gini Criteria"""

from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)

model_gini.fit(x_train, y_train)

#Prediction and computing the accuracy
pred=model.predict(x_test)
np.mean(preds==y_test)

"""# Decision Tree Regression Example"""

# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor

array = df.values
X = array[:,0:3]
y = array[:,3]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

#Find the accuracy
model.score(X_test,y_test)

