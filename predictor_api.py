import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL

import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph. I like it most for plot
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn
def impute_nulls_data(dataframe):
    print("****impute_nulls_data****'")
    imp = Imputer(missing_values="NaN",strategy="mean",axis =0) #check the strategy options strategy="most_frequent"
    dfencodeimpute = imp.fit_transform(dataframe)
    dfencodeimpute = pd.DataFrame(dfencodeimpute)
    dfencodeimpute.columns = dataframe.columns
    datarame = dfencodeimpute
    #from sklearn.impute import SimpleImputer(univariate), IterativeImputer(multivariate)
    #Explore Univariate vs. Multivariate Imputation
    return dataframe

def encode_data(dataframe): #sklean Label encoder, onehot encoding, pandas dummy
    print("****encode_data****'")
    try:
        # Note : lable encoding 
        print(dataframe.head())
        print(dataframe.columns)
        #data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
        columnlist = list(dataframe.columns)
        enc = LabelEncoder()
        for column in columnlist:
            if (dataframe[column].nunique() <= 2 or dataframe[column].nunique() > 50 ):
                #binary encoding
                enc.fit(dataframe[column])
                print(column)
                columnenc = column + "_encoded"
                dataframe[columnenc] = enc.transform(dataframe[column])
                dataframe.drop([column], axis = 1,inplace = True)
            else:
                #onehot, dummy encoding
                # Get dummies
                dataframe = pd.get_dummies(dataframe[column], prefix_sep='_', drop_first=True)
                dataframe.head()
                

        print(dataframe.head())
        #dfencoded = pd.concat([dfnumeric,dfcategorical],axis = 1)
        #print(dfencoded.head())
    except:
         print("An exception occurred in encode")

    return dataframe

def data_wangling(dforig):
    # 3. Data Wrangling/ preprocessing  ####
    # Note : lable encoding categorical data 
    print("****drop complete null value columns")
    dforig.dropna(axis = 1,how='all', inplace = True)
    print("****delete complete null rows")
    dforig.dropna(axis=0, how='all', inplace=True)
    print("****Remove duplicate rows")
    dupes = dforig.duplicated()
    print("remove duplicate number of rows:\n",sum(dupes))
    #Removing Duplicates
    dforig = dforig.drop_duplicates()
    # Remove rows where label column is null
    print("remove label column value is null\n")
    #???dforig = dforig[dforig[labelcolumn].notnull()]
    print(dforig.head())
    
    #convert Date to Date and time
    # Converts date string column to python datetime type
    # `infer_datetime_format=True` says method to guess date format from string
    #df['datetime'] = pd.to_datetime(df['date_string'], infer_datetime_format=True)
    # Converts date string column to python datetime type
    # `format` argument specifies the format of date to parse, fails on errors
    #df['datetime'] = pd.to_datetime(df['date_string'], format='%Y.%m.%d')
    
    # Coverts column in python datetime type to timestamp
    #df['timestamp'] = df['datetime'].values.astype(np.int64) // 10 ** 9
    
    
    #Remove white spaces
    #Make sure integers are not float or character
    # Convert data to numerical and categorical dataframes
    dfcategorical, dfnumeric = convert_data_numerical_categorical(dforig)
    dfwolables = dfnumeric
    print("Numeric data:\n",dfwolables.head())
    print("Categorical data:\n",dfcategorical.head())
    
    #impute_nulls_data numerical data
    dfnumericimpute = impute_nulls_data(dfnumeric)
    # we can also use KNN to propogate null values in the columns
    
    
    # encode categorical data
    dfcategoricalencode = encode_data(dfcategorical)  
    # If cetegorical feature is  null will be encoded to 0 etc
    
    # join encoded and numeric data
    dfimputeencoded = pd.concat([dfnumericimpute,dfcategoricalencode],axis = 1)
    dfimputeencoded.head()
    
    #scale data normalize data (reduce the impact of outliers)
    dfv = scale_data_StandardScalar(dfimputeencoded)
    dfv.head()
    
    return

def make_classification(chat_in):
    file = chat_in
    dataframe = pd.read_csv(file, skipinitialspace = True, thousands=',')
    data_wangling(dataframe)
    return
    

if __name__ == '__main__':
    from pprint import pprint
    print("Checking to see what empty string predicts")
    print('input string is ')
    #chat_in = 'bob'
    chat_in = '../Data/breast-cancer-data.csv'
    pprint(chat_in)
    X_input = make_classification(chat_in)

