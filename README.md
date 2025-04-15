# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
   import pandas as pd
   
   import numpy as np

   df=pd.read_csv("C:\\Users\\sherwin\\OneDrive\\Documents\\bmi (1).csv")
   
   df

   ![Screenshot 2025-04-15 105719](https://github.com/user-attachments/assets/18a4f2c3-0237-4994-b9dc-7e59aac331c4)

   df.head()

   ![Screenshot 2025-04-15 105810](https://github.com/user-attachments/assets/1d365fa4-bfff-4d4d-9c03-d74df06473a0)

  df.dropna()

  ![Screenshot 2025-04-15 105930](https://github.com/user-attachments/assets/ba02d8cd-9cb8-47c0-ac79-9b19dc73704d)

   max_vals=np.max(np.abs(df[['Height','Weight']]))
   
   max_vals

   ![Screenshot 2025-04-15 110014](https://github.com/user-attachments/assets/522b0f94-5941-46c6-a3f6-1d9893ec1a50)

   from sklearn.preprocessing import MinMaxScaler
   
   scaler=MinMaxScaler()
   
   df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])fr
   
   df.head(10)

![Screenshot 2025-04-15 110102](https://github.com/user-attachments/assets/25ecaf74-b06c-4e7f-b3bb-6577c300bace)

   df1=pd.read_csv("C:\\Users\\sherwin\\OneDrive\\Documents\\bmi (1).csv")
   
   df2=pd.read_csv("C:\\Users\\sherwin\\OneDrive\\Documents\\bmi (1).csv")
   
   df3=pd.read_csv("C:\\Users\\sherwin\\OneDrive\\Documents\\bmi (1).csv")
   
   df4=pd.read_csv("C:\\Users\\sherwin\\OneDrive\\Documents\\bmi (1).csv")
   
   df5=pd.read_csv("C:\\Users\\sherwin\\OneDrive\\Documents\\bmi (1).csv")
   
   df1

![image](https://github.com/user-attachments/assets/18c485cc-9c41-4051-9ce9-1d891f13b527)

   from sklearn.preprocessing import StandardScaler
   
   sc=StandardScaler()
   
   df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
   
   df1.head(10)

![Screenshot 2025-04-15 110258](https://github.com/user-attachments/assets/603c43a1-78c0-4d68-8162-12b02d0b4baa)

   from sklearn.preprocessing import Normalizer
   
   scaler=Normalizer()
   
   df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
   
   df2

   ![Screenshot 2025-04-15 110416](https://github.com/user-attachments/assets/51a78e22-8e2a-41c9-9167-b4907ee501dc)

   from sklearn.preprocessing import MaxAbsScaler
   
   scaler=MaxAbsScaler()
   
   df3[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
   
   df3 

![Screenshot 2025-04-15 110457](https://github.com/user-attachments/assets/6f458d07-dfda-47ab-a2b0-07cbaa858ecd)

   from sklearn.preprocessing import RobustScaler
   
   scaler=RobustScaler()
   
   df4[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
   
   df4

   ![Screenshot 2025-04-15 110534](https://github.com/user-attachments/assets/17521830-ed92-4536-85cc-c4d262b70e5f)

   import seaborn as sns
   
   feature selection 
   
   import pandas as pd

   import numpy as np 
   
   import seaborn as sns
   
   import seaborn as sns
   
   import pandas as pd
   
   from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_classif
   
   from sklearn.feature_selection import chi2
   
   data=pd.read_csv("C:\\Users\\priya\\Downloads\\titanic_dataset (1).csv")
   
   data

   ![Screenshot 2025-04-15 110623](https://github.com/user-attachments/assets/d2c27ff3-7415-41ba-a534-e420bfd09768)

   data=data.dropna()
   
   x=data.drop(['Survived','Name','Ticket'],axis=1)
   
   y=data['Survived']
   
   data["Sex"]=data["Sex"].astype("category")
   
   data["Cabin"]=data["Cabin"].astype("category")
   
   data["Embarked"]=data["Embarked"].astype("category")
   
   data["Sex"]=data["Sex"].cat.codes
   
   data["Cabin"]=data["Cabin"].cat.codes
   
   data["Embarked"]=data["Embarked"].cat.codes
   
   data

   ![Screenshot 2025-04-15 110713](https://github.com/user-attachments/assets/32d5075f-e9bb-450d-845d-0587edb0aa81)

   k=5 selector=SelectKBest(score_func=chi2, k=k) x=pd.get_dummies(x) x_new=selector.fit_transform(x,y)

   x_encoded =pd.get_dummies(x) selector=SelectKBest(score_func=chi2, k=5) x_new = selector.fit_transform(x_encoded,y)

   selected_feature_indices=selector.get_support(indices=True)
   
   selected_features=x.columns[selected_feature_indices]
   
   print("Selected_Feature:")
   
   print(selected_features)

   ![Screenshot 2025-04-15 110759](https://github.com/user-attachments/assets/461480c8-8405-458d-a869-faeb76c7c7cb)

   selector=SelectKBest(score_func=mutual_info_classif, k=5)
   
   x_new = selector.fit_transform(x,y)
   
   selected_feature_indices=selector.get_support(indices=True)
   
   selected_features=x.columns[selected_feature_indices]
   
   print("Selected Features:")
   
   print(selected_features)

   ![Screenshot 2025-04-15 110859](https://github.com/user-attachments/assets/f1bbca65-fd59-478f-97c3-a66890de22ac)

   selector=SelectKBest(score_func=mutual_info_classif, k=5)
   
   x_new = selector.fit_transform(x,y)
   
   selected_feature_indices=selector.get_support(indices=True)
   
   selected_features=x.columns[selected_feature_indices]
   
   print("Selected Features:")
   
   print(selected_features)

   ![Screenshot 2025-04-15 110953](https://github.com/user-attachments/assets/38350f01-c710-422f-bd25-069ae165ef29)

   from sklearn.feature_selection import SelectFromModel
   
   from sklearn.ensemble import RandomForestClassifier
   
   
   model=RandomForestClassifier()
   
   sfm=SelectFromModel(model,threshold='mean')
   
   x=pd.get_dummies(x)
   
   sfm.fit(x,y)
   
   selected_features=x.columns[sfm.get_support()]
   
   print("Selected Features:")
   
   print(selected_features)

   ![Screenshot 2025-04-15 111057](https://github.com/user-attachments/assets/3f09ec29-1415-4700-ad17-1653ae7698ec)

   from sklearn.ensemble import RandomForestClassifier
   
   model=RandomForestClassifier(n_estimators=100,random_state=42)
   
   model.fit(x,y)
   
   feature_importances=model.feature_importances_
   
   threshold=0.1
   
   selected_features = x.columns[feature_importances>threshold]
   
   print("Selected Features:")
   
   print(selected_features)

   ![Screenshot 2025-04-15 111136](https://github.com/user-attachments/assets/d9f53cd3-04e7-4fa5-9c13-d8b91248cebe)

   from sklearn.ensemble import RandomForestClassifier
   
   model=RandomForestClassifier(n_estimators=100,random_state=42)
   
   model.fit(x,y)
   
   feature_importances=model.feature_importances_
   
   threshold=0.15
   
   selected_features = x.columns[feature_importances>threshold]
   
   print("Selected Features:")
   
   print(selected_features)

   ![Screenshot 2025-04-15 111229](https://github.com/user-attachments/assets/cb6a50b0-db76-40f5-a024-1fb916b5178c)

# RESULT:
Thus,Feature selection and Feature scaling has been used on the given dataset.
