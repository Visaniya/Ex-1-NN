<H3>ENTER YOUR NAME: S.VISANIYA</H3>
<H3>ENTER YOUR REGISTER NO: 212225040492 </H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

~~~
import pandas as pd
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
~~~
~~~
df=pd.read_csv("Churn_Modelling.csv")

### Checking Data

df.head()
df.tail()
df.columns
~~~
~~~
df.isnull().sum()


### Check for Duplicates

df.duplicated()
~~~
~~~
y = df.iloc[:, -1].values
print(y)
~~~
~~~
df.describe()
~~~
~~~
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
~~~
~~~
data.head()
~~~
~~~
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
~~~
~~~
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(X)
print(y)
~~~
~~~
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
~~~


## OUTPUT:
## Data checking
<img width="873" height="153" alt="image" src="https://github.com/user-attachments/assets/1a5af1b9-2e1f-40ac-b5b4-e9ca73dff849" />


## Duplicates identification
<img width="891" height="368" alt="image" src="https://github.com/user-attachments/assets/375428a2-5db0-448e-98e6-b376737662ca" />


## Vakues of 'Y'
<img width="877" height="62" alt="image" src="https://github.com/user-attachments/assets/28ac50fd-146d-4475-ba56-b1d85c7808d2" />


## Outliers
<img width="877" height="385" alt="image" src="https://github.com/user-attachments/assets/9cc0300e-bfe0-409b-a6e1-86a2f11e313f" />


## Checking datasets after dropping string values data from dataset
<img width="905" height="278" alt="image" src="https://github.com/user-attachments/assets/90a5def3-889c-45f9-b680-3473324cd909" />


## Normalize the dataset
<img width="893" height="793" alt="image" src="https://github.com/user-attachments/assets/34e91d62-2ad4-43b6-8c8c-414a658d9a24" />


## Split the dataset
<img width="878" height="252" alt="image" src="https://github.com/user-attachments/assets/9b315b8e-0a97-4f9c-a499-1626e6d7ca05" />


## Training and testing model
<img width="781" height="655" alt="image" src="https://github.com/user-attachments/assets/75019c40-301f-4275-9786-85a957d0c7b4" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


