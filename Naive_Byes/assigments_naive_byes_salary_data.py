


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for statistical data visualization
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')

data = r'C:\Users\Masum\Downloads\SalaryData_Test.csv'
df = pd.read_csv(data)
df.shape

df.head()

col_names = ['age', 'workclass', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
df.columns = col_names
df.columns

# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :\n\n', categorical)

# view the categorical variables

df[categorical].head()

# check missing values in categorical variables

df[categorical].isnull().sum()

# view frequency counts of values in categorical variables

for var in categorical:

    print(df[var].value_counts())

# view frequency distribution of categorical variables

for var in categorical:

    print(df[var].value_counts()/np.float(len(df)))

# check labels in workclass variable

df.workclass.unique()

# check frequency distribution of values in workclass variable

df.workclass.value_counts()

# replace '?' values in workclass variable with `NaN`
df['workclass'].replace('?', np.NaN, inplace=True)

# again check the frequency distribution of values in workclass variable
df.workclass.value_counts()

# check labels in occupation variable
df.occupation.unique()

# check frequency distribution of values in occupation variable
df.occupation.value_counts()

# replace '?' values in occupation variable with `NaN`
df['occupation'].replace('?', np.NaN, inplace=True)

# again check the frequency distribution of values in occupation variable
df.occupation.value_counts()

df.native_country.unique()

# check frequency distribution of values in native_country variable

df.native_country.value_counts()

# replace '?' values in native_country variable with `NaN`

df['native_country'].replace('?', np.NaN, inplace=True)

# again check the frequency distribution of values in native_country variable

df.native_country.value_counts()

df[categorical].isnull().sum()

# check for cardinality in categorical variables

for var in categorical:

    print(var, ' contains ', len(df[var].unique()), ' labels')

# find numerical variables

numerical = [var for var in df.columns if df[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)

# view the numerical variables

df[numerical].head()

# check missing values in numerical variables

df[numerical].isnull().sum()

X = df.drop(['income'], axis=1)

y = df['income']

# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# check the shape of X_train and X_test

X_train.shape, X_test.shape

# display categorical variables

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical

# display numerical variables

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical

# print percentage of missing values in the categorical variables in training set

X_train[categorical].isnull().mean()

# print categorical variables with missing data

for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, (X_train[col].isnull().mean()))

# impute missing categorical variables with most frequent value

for df2 in [X_train, X_test]:
    df2['workclass'].fillna(X_train['workclass'].mode()[0], inplace=True)
    df2['occupation'].fillna(X_train['occupation'].mode()[0], inplace=True)
    df2['native_country'].fillna(X_train['native_country'].mode()[0], inplace=True)

# check missing values in categorical variables in X_train

X_train[categorical].isnull().sum()

# check missing values in categorical variables in X_test

X_test[categorical].isnull().sum()

# check missing values in X_train

X_train.isnull().sum()

# check missing values in X_test

X_test.isnull().sum()

# print categorical variables

categorical

X_train[categorical].head()

# import category encoders
import category_encoders as ce

# encode remaining variables with one-hot encoding

encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'marital_status', 'occupation', 'relationship',
                                 'race', 'sex', 'native_country'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

X_train.head()

X_train.shape

X_test.head()

X_test.shape

cols = X_train.columns

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])

X_test = pd.DataFrame(X_test, columns=[cols])

X_train.head()

# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB


# instantiate the model
gnb = GaussianNB()


# fit the model
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
y_pred

from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

y_pred_train = gnb.predict(X_train)

y_pred_train

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

# print the scores on training and test set

print('Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))