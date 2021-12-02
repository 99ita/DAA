import pandas as pd
import numpy
import xlrd
import xlwt
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import os





df = pd.read_csv('datasets/training_data_clean.csv', encoding ='utf-8')
df_test = pd.read_csv('datasets/test_data_clean.csv', encoding ='utf-8')
