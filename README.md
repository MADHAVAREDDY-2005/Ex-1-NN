<H3>NAME: K MADHAVA REDDY</H3>
<H3>REGISTER NO: 212223240064</H3> 
<H3>EX. NO.1</H3>
<H3>DATE: 13-09-2025</H3>
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
```py
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
data
data.head()

X=data.iloc[:,:-1].values
X

y=data.iloc[:,-1].values
y

data.isnull().sum()

data.duplicated()

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train

X_test

print("Lenght of X_test ",len(X_test))

```

## OUTPUT:
### Dataset:
<img width="1269" height="244" alt="image" src="https://github.com/user-attachments/assets/4271923a-0ceb-447b-b978-775218d2a503" />

### X Values:
<img width="836" height="232" alt="image" src="https://github.com/user-attachments/assets/ffff8c6a-21e6-4c4c-b0fc-eb1b21a694df" />

### Y Values:
<img width="577" height="100" alt="image" src="https://github.com/user-attachments/assets/44b72aec-8ee5-44ab-a56a-0906efbd9da0" />
### Null Values:
<img width="353" height="379" alt="image" src="https://github.com/user-attachments/assets/ab61b004-2f0b-40f4-9750-3583c0172534" />
### Duplicated Values:
<img width="400" height="330" alt="image" src="https://github.com/user-attachments/assets/0687c719-8353-46cd-8135-f5b6a2ca6b08" />
### Description:
<img width="1248" height="348" alt="image" src="https://github.com/user-attachments/assets/e9a8e158-c8a3-4588-9470-869f57e09dfc" />
### Normalized Dataset:
<img width="764" height="592" alt="image" src="https://github.com/user-attachments/assets/c95580bc-3703-4b5c-922c-07874c33901e" />
### Training Data:
<img width="729" height="171" alt="image" src="https://github.com/user-attachments/assets/a5fd31c7-09c5-4fba-acda-0f6d512e70b4" />
### Testing Data:
<img width="729" height="171" alt="image" src="https://github.com/user-attachments/assets/730207bf-b14d-41af-9a02-5f62c160489b" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


