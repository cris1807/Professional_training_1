import numpy as np
import pandas as pd
import pickle

# Loading the dataset
data=pd.read_csv(r'C:\Users\samya\Desktop\pride_ml\diabetes.csv')
df=data.copy()

df['Gender'] = df ['Gender'].replace({'Female':0,'Male':1 })
for column in df.columns.drop(['Age','Gender','class']):
     df[column]= df[column].replace({'No':0 , 'Yes': 1})
     
df['class'] = df ['class'].replace({'Positive':0,'Negative':1 })

y=df["class"]
X=df.drop("class", axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,shuffle=True,random_state=123) 

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=pd.DataFrame(scaler.fit_transform(X_train),index=X_train.index , columns=X_train.columns)


from sklearn.tree import DecisionTreeClassifier
model_2=DecisionTreeClassifier()
model_2.fit(X_train,y_train)



# Renaming DiabetesPedigreeFunction as DPF
# df = df.rename(columns={'DiabetesPedigreeFunction':'DPF'})

# # Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
# df_copy = df.copy(deep=True)
# df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

# # Replacing NaN value by mean, median depending upon distribution
# df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
# df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
# df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
# df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
# df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)

# # Model Building
# from sklearn.model_selection import train_test_split
# X = df.drop(columns='Outcome')
# y = df['Outcome']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# # Creating Random Forest Model
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import metrics
# clf_dc = DecisionTreeClassifier()
# clf_dc.fit(X_train, y_train)
# pred_dc = clf_dc.predict(X_test)
# acc_dc = metrics.accuracy_score(pred_dc, y_test)
# metrics.confusion_matrix(pred_dc, y_test)

# Training the model
# clf_dc.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'diabetes.pkl'
pickle.dump(model_2, open(filename, 'wb'))
