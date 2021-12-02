import pandas as pd
import numpy
import xlrd
import xlwt
import matplotlib
import seaborn as sns
import sklearn
from sklearn import preprocessing

#Read csv file

df = pd.read_csv('datasets/stroke.csv',encoding='latin-1')

#DataExploration

print(df.columns)
print('\n')
print(df.head())
print('\n')
print(df.info())
print('\n')

print(df['age'].describe())
print('\n')

#Missing values

print(df.isna().any)
print(df.isna().sum())
print('\n')

#Unique values

print(df.Residence_type.unique().size)
print(df.Residence_type.unique())
print('\n')

#NewColumns

def bmiclass(bmi):
    if bmi > 0 and bmi < 18.5:
        return "underweight "
    elif bmi >= 18.5 and bmi < 24.9:
        return "healthy_weight"
    elif bmi >= 24.9 and bmi < 30:
        return "overweight"
    elif bmi >= 30 :
        return "obese"
    
df['bmi_classification'] = df['bmi'].apply(bmiclass)


def glucose(value):
    if value > 0 and value < 100:
        return "good"
    elif value >= 100 and value < 140:
        return "not_bad"
    elif value >= 140 :
        return "dangerous"
    
df['avg_glucose_classification'] = df['avg_glucose_level'].apply(glucose)

def age_category(age):
    if age > 0 and age < 60:
        return "adult_or_less"
    elif age >= 60 and age < 80:
        return "elderly"
    elif age >= 80 :
        return "very_elderly"
    

df['age_Category'] = df['age'].apply(age_category)

def issmoker(ss):
    if ss=="smokes" or ss=="formerly smoked":
        return "true"
    else:    
        return "false"

df['isSmoker'] = df['smoking_status'].apply(issmoker)

def diseases(hy,he):
    if hy==1 and he==1:
        return "2"
    elif hy==1 or he==1:    
        return "1"
    else :
        return "0"

df['diseases'] = df.apply(lambda x: diseases(x.hypertension, x.heart_disease), axis=1)

#Grouping

df_bmiglucperc=df.groupby(by=['bmi_classification','avg_glucose_classification']).mean()  
df_stroke=df.groupby(by=['stroke']).mean()  
df_strokesmoke=df.groupby(by=['stroke','smoking_status']).mean()  
df_male=df.loc[lambda df: df['gender'] == "Male"]
df_female=df.loc[lambda df: df['gender'] == "Female"]

#Normalization
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
df[['age_normalization']]=min_max_scaler.fit_transform(df[['age']])

#Binning
age=preprocessing.KBinsDiscretizer(n_bins=4,encode='ordinal',strategy='quantile')
df['age_binned']=age.fit_transform(df[['age']])
df_agebinned=df.groupby(by=['age_binned']).count()

#Filter missing values

df_missingvalues=df.dropna(axis=0,how="any")

#Plots#

#sns.histplot(df_missingvalues['bmi_classification'])
#sns.histplot(df['avg_glucose_classification'])
#gluc = sns.boxplot(x=df['avg_glucose_level'])
#glucbmi = sns.boxplot(x=df['avg_glucose_level'],y=df['bmi'])
#agegender = df.plot.scatter(x='age',y='gender')


#Scatter Plots
#cols=['age','bmi','avg_glucose_level','hypertension','heart_disease']
#_=sns.pairplot(df[cols],height= 2.5)

#Correlation Matrixes
corr_matrix=df.corr()
sns.heatmap(corr_matrix,vmin=-1,vmax=1,square=True,annot=True)