import pandas as pd
import numpy
import xlrd
import xlwt
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import os



#Definição de algumas funções para transformar dados
#categóricos em numerais, mapeando cada entrada de texto 
#para uma valor inteiro

def arToValue(ar):
 if ar == 'chuvisco fraco':
   return 0
 elif ar == 'chuva fraca':
   return 1
 elif ar == 'chuvisco e chuva fraca':
   return 2
 elif ar == 'chuva leve':
   return 3
 elif ar == 'aguaceiros fracos':
   return 4
 elif ar == 'aguaceiros':
   return 5
 elif ar == 'chuva moderada':
   return 6
 elif ar == 'chuva':
   return 6
 elif ar == 'chuva forte':
   return 7
 elif ar == 'chuva de intensidade pesado':
   return 8
 elif ar == 'chuva de intensidade pesada':
   return 8
 elif ar == 'trovoada com chuva leve':
   return 9
 elif ar == 'trovoada com chuva':
   return 10
 else:
   return None
   

def acToValue(ac):
 if ac == 'céu limpo':
   return 0
 elif ac == 'céu claro':
   return 1
 elif ac == 'algumas nuvens':
   return 2
 elif ac == 'nuvens dispersas':
   return 2
 elif ac == 'céu pouco nublado':
   return 3
 elif ac == 'nuvens quebrados':
   return 3
 elif ac == 'nuvens quebradas':
   return 3
 elif ac == 'tempo nublado':
   return 4
 elif ac == 'nublado':
   return 4
 else:
   return None

def lToValue(l):
  if l == 'DARK':
    return 0
  elif l == 'LOW_LIGHT':
    return 1
  elif l == 'LIGHT':
    return 2
  else:
    return None


def asdToValue(asd):
  if asd == 'None':
    return 0
  elif asd == 'Low':
    return 1
  elif asd == 'Medium':
    return 2
  elif asd == 'High':
    return 3
  elif asd == 'Very_High':
    return 4
  else:
    return None

def dayPart(hour):
  if hour > 0 and hour <= 8:
    return 'Madrugada'
  elif hour > 8 and hour <=16:
    return 'Horário_trabalho'
  else: 
    return 'Noite'

def dpToValue(asd):
  if asd == 'Madrugada':
    return 0
  elif asd == 'Horário_trabalho':
    return 1
  elif asd == 'Noite':
    return 2
  else:
    return None

def wdToValue(asd):
  if asd == 'Monday':
    return 0
  elif asd == 'Tuesday':
    return 1
  elif asd == 'Wednesday':
    return 2
  elif asd == 'Thursday':
    return 3
  elif asd == 'Friday':
    return 4
  elif asd == 'Saturday':
    return 5
  elif asd == 'Sunday':
    return 6
  else:
    return None



#Criação dos novos ficheiros com o encoding utf-8

sourcetr = open('datasets/training_data.csv', encoding ='latin-1')
targettr = open('datasets/training_data_utf8.csv', 'w+', encoding ='utf-8')

sourcete = open('datasets/test_data.csv', encoding ='latin-1')
targette = open('datasets/test_data_utf8.csv', 'w+', encoding ='utf-8')

filetr = sourcetr.read()
targettr.write(filetr)

filete = sourcete.read()
targette.write(filete)




#Leitura dos ficheiros com o novo encoding e consulta 
#de algumas informações básicas do dataset de treino

df = pd.read_csv('datasets/training_data_utf8.csv', encoding ='utf-8')
df_test = pd.read_csv('datasets/test_data_utf8.csv', encoding ='utf-8')
print('Colunas do dataset de treino')
print(f'{df.columns.size}\n{df.columns}')
print('Colunas do dataset de teste')
print(f'{df_test.columns.size}\n{df.columns}')
print('\nDescribe dataset de treino')
print(df.describe())
print(df.isna().sum())
print('\nDescribe dataset de teste')
print(df_test.describe())
print(df_test.isna().sum())


for k in df:
    print(k)
    print(df[k].unique().size,end='\n\n')



#Reparou-se que todos os valores da coluna AVERAGE_PRECIPITATION
#são 0.0 e a cidade é sempre o Porto, por isso removem-se
#essas colunas
#De seguida desdobra-se a coluna record_date em diversas 
#contendo informação pertinente


df = df.drop('city_name',axis=1)
df = df.drop('AVERAGE_PRECIPITATION',axis=1)
df.record_date = pd.to_datetime(df.record_date)
df['MONTH'] = df.record_date.dt.month
df['DAY'] = df.record_date.dt.day
df['WEEK_DAY'] = df.record_date.dt.day_name()
df['HOUR'] = df.record_date.dt.hour
df['DAY_PART'] = df['HOUR'].apply(dayPart)
df.head()


df_test = df_test.drop('city_name',axis=1)
df_test = df_test.drop('AVERAGE_PRECIPITATION',axis=1)
df_test.record_date = pd.to_datetime(df_test.record_date)
df_test['MONTH'] = df_test.record_date.dt.month
df_test['DAY'] = df_test.record_date.dt.day
df_test['WEEK_DAY'] = df_test.record_date.dt.day_name()
df_test['HOUR'] = df_test.record_date.dt.hour
df_test['DAY_PART'] = df_test['HOUR'].apply(dayPart)
df_test.head()


#Aplicação das funções de transformação às colunas

df['AVERAGE_RAIN'] = df.apply(lambda row: arToValue(row.AVERAGE_RAIN), axis=1)
df['AVERAGE_CLOUDINESS'] = df.apply(lambda row: acToValue(row.AVERAGE_CLOUDINESS), axis=1)
df['LUMINOSITY'] = df.apply(lambda row: lToValue(row.LUMINOSITY), axis=1)
df['AVERAGE_SPEED_DIFF'] = df.apply(lambda row: asdToValue(row.AVERAGE_SPEED_DIFF), axis=1)
df['DAY_PART'] = df.apply(lambda row: dpToValue(row.DAY_PART), axis=1)
df['WEEK_DAY'] = df.apply(lambda row: wdToValue(row.WEEK_DAY), axis=1)
df['IS_WEEKEND'] = df.WEEK_DAY.apply(lambda x : 1 if x in [5,6] else 0)
df.head()

df_test['AVERAGE_RAIN'] = df_test.apply(lambda row: arToValue(row.AVERAGE_RAIN), axis=1)
df_test['AVERAGE_CLOUDINESS'] = df_test.apply(lambda row: acToValue(row.AVERAGE_CLOUDINESS), axis=1)
df_test['LUMINOSITY'] = df_test.apply(lambda row: lToValue(row.LUMINOSITY), axis=1)
df_test['DAY_PART'] = df_test.apply(lambda row: dpToValue(row.DAY_PART), axis=1)
df_test['WEEK_DAY'] = df_test.apply(lambda row: wdToValue(row.WEEK_DAY), axis=1)
df_test['IS_WEEKEND'] = df_test.WEEK_DAY.apply(lambda x : 1 if x in [5,6] else 0)
df_test.head()


#Criação de tabelas para melhor entendimento do dataset
plt.figure()
sns.histplot(df['AVERAGE_RAIN'], kde=True, discrete=True)

plt.figure()
sns.histplot(df['AVERAGE_CLOUDINESS'], kde=True, discrete=True)

plt.figure()
sns.histplot(df['LUMINOSITY'], kde=True, discrete=True)

plt.figure()
sns.histplot(df['AVERAGE_SPEED_DIFF'], kde=True, discrete=True)


#Matrix de correlação dos diversos atributos relevantes do dataset

dims = (12, 12)
fig, ax = plt.subplots(figsize=dims)

newdf = df.drop(['record_date','HOUR'],axis=1)

corr_matrix = newdf.corr()

plt.figure()
sns.heatmap(corr_matrix,vmin = -1,vmax = 1,square = True,annot = True,ax=ax)


#Mais alguns gráficos de correlação

plt.figure()
sns.lineplot(x="WEEK_DAY", y="AVERAGE_TIME_DIFF", data=df)

plt.figure()
sns.lineplot(x="WEEK_DAY", y="AVERAGE_SPEED_DIFF",data=df)

plt.figure()
sns.lineplot(x="AVERAGE_RAIN", y="AVERAGE_TIME_DIFF",data=df)

plt.figure()
sns.lineplot(x="AVERAGE_RAIN", y="AVERAGE_SPEED_DIFF",data=df)


#Remover colunas redundantes

df = df.drop('record_date',axis=1)
df = df.drop('DAY_PART',axis=1)
df = df.drop('DAY',axis=1)
df = df.drop('IS_WEEKEND',axis=1)
df = df.drop('MONTH',axis=1)


df_test = df_test.drop('record_date',axis=1)
df_test = df_test.drop('DAY_PART',axis=1)
df_test = df_test.drop('DAY',axis=1)
df_test = df_test.drop('IS_WEEKEND',axis=1)
df_test = df_test.drop('MONTH',axis=1)


#Limpeza e escrita
targettr.close()
targette.close()

os.remove("datasets/training_data_utf8.csv")
os.remove("datasets/test_data_utf8.csv")

df.to_csv("datasets/training_data_clean.csv",index=False)
df_test.to_csv("datasets/test_data_clean.csv",index=False)
