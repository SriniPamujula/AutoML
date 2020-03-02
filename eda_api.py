import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL

import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph. I like it most for plot
from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import Imputer
from sklearn.Imputer import SimpleImputer as Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split # to split the data into two parts

#Classification Models
from sklearn.linear_model import LogisticRegression # to apply the Logistic regression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import GaussianNB

#Classification metrics
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn

def read_analyze_data(file):
    print("****read_analyze_data****")
    dirlog = '../Log/'
    dataframe = pd.read_csv(file, skipinitialspace = True, thousands=',')
    print("***Exploratory Data Analysis***")
    print("****Top 5 rows***")
    #dfhtml = dataframe.head().style.set_properties(**{'background-color': 'black','color': 'lawngreen','border-color': 'white'}).hide_index().render()
    dfhtml = dataframe.head()
    
    html = dfhtml.to_html()
    
    print(dataframe.head())
    print("****Last 5 rows***")
    print(dataframe.tail())
    print("****dataframe shape***")
    print(dataframe.shape)
    print("****dataframe info***")
    print(dataframe.info())
    print("****data frame describe statistics***")
    print(dataframe.describe())
    print("****data frame column sums ***")
    print(dataframe.sum())
    # check any null values in the DF
    print("****dataframe check nulls***")
    print(pd.isnull(dataframe).any())
    print("****dataframe check column nulls count***")
    print(dataframe.isna().sum())
    # check non unique values, helps in lable encoding, onehot encoding or pandas dummy columns
    print("****dataframe check non unique***")
    print(dataframe.nunique())
    print("****dataframe columns***")
    print(dataframe.columns)
    print("****delete complete null columns")
    

    
    # find lable column
    dataframe.select_dtypes(include=[object])
    return dataframe,html

def scale_data_minmax(dataframe):
    #explore StandardScalara, minmaxscaler etc
    print("****scale_data min max 0 and 1 ****'")
    scaler = MinMaxScaler(feature_range=[0, 1])
    scaler = scaler.fit_transform(dataframe)
    #dataframe = pd.DataFrame(dataframe,columns=columnnames)
    scaler = pd.DataFrame(scaler)
    scaler.columns = dataframe.columns
    dataframe = scaler
    return dataframe

def scale_data_StandardScalar(dataframe):
    #explore StandardScalara, minmaxscaler etc
    print("****scale_data standard ****'")
    scaler = StandardScaler()
    scaler = scaler.fit_transform(dataframe)
    scaler = pd.DataFrame(scaler)
    scaler.columns = dataframe.columns
    dataframe = scaler
    return dataframe
    

def half_masked_corr_heatmap(dataframe, title=None, file=None):
    print("plot heatmap")
    image_directory = "./static/images/"
    # Required parameter: dataframe ... the reference pandas dataframe
    # Optional parameters: title ... (string) chart title
    #                      file  ... (string) path+filename if you want to save image
    plt.figure(figsize=(14,14))
    sns.set(font_scale=1)

    mask = np.zeros_like(dataframe.corr())
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style("white"):
        sns.heatmap(dataframe.corr(), mask=mask, annot=False, cmap='coolwarm')

    if title: plt.title(f'\n{title}\n', fontsize=18)
    plt.xlabel('')    # optional in case you want an x-axis label
    plt.ylabel('')    # optional in case you want a  y-axis label
    if file: plt.savefig(image_directory+file, bbox_inches='tight')
    #plt.show();
    plt.clf()
    
    return

def plot_boxplot(dataframe, title=None, file=None): 
    print("plot box plot")
    image_directory = "./static/images/"
    # Required parameter: dataframe ... the reference pandas dataframe
    # Optional parameters: title ... (string) chart title
    #                      file  ... (string) path+filename if you want to save image
    plt.figure(figsize=(14,6))
    dfbox = scale_data_minmax(dataframe)
    ax = sns.boxplot(data=dfbox, orient="v", palette="Set2")
    plt.setp(ax.get_xticklabels(), rotation=90)
    if title: plt.title(f'\n{title}\n', fontsize=18)
   
    #if file: plt.savefig(file, bbox_inches='tight')
    if file: 
        plt.savefig(image_directory+file, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None, metadata=None)
       
    #plt.show()
    plt.clf()    
        
    
    #plt.figure(figsize=(14, 6))
    #ax = sns.boxplot(data=dfbox, orient="v", palette="Set2")
    #plt.setp(ax.get_xticklabels(), rotation=90)
    #plt.show()
    
    return 

def plot_countplot(dataframe,title=None,file=None): 
    print("plot count plot")
    image_directory = "./static/images/"
    for column in dataframe.columns:
        print("in column loop",column)
        sns.countplot(x = column , data = dataframe, label="Count")
        if file: 
            plt.savefig(image_directory+file+column, dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches='tight', pad_inches=0.1,
            frameon=None, metadata=None)
            plt.clf()
        
        
        #plt.show()
        #for column in dfcategorical.columns:
        #print("in column loop",column)
        #sns.countplot(x = column , data = dfcategorical, label="Count")
        #plt.show()
               
       
    return 

def plot_analysis_data(dfwolables, dfcategorical):
    print("****plot_analysis_data****'")
    labl = list(dfwolables.columns)
    print("lables",labl)

    # scale(lognormal) data to keep every number between 0 and 1
    # box plot requires all features in same scale to plot together
    print(dfwolables.columns.values)
    dfbox = scale_data_minmax(dfwolables)
    
    
    #Boxplot to analyze the statistical parameters,outliers and distribution
    #plt.figure(figsize=(14, 6))
    #ax = sns.boxplot(data=dfbox, orient="v", palette="Set2")
    #plt.setp(ax.get_xticklabels(), rotation=90)
    #plt.show()
    
    plot_boxplot(dfbox,'Features Boxplot','box')
   
    # plot the data 
    ## Heatmap to check the correlation between features
    #corr = dfwolables.corr() # .corr is used for find corelation
    #plt.figure(figsize=(14,14))
    #sns.heatmap(corr,cbar = True,square = True,cmap= 'coolwarm')
    #plt.show()
    half_masked_corr_heatmap(dfwolables,'Features Correlations','heatmap')


    ## Pairpolot to see the correlation using scatter plots
    #corr = dfwolables.corr() # .corr is used for find corelation bw features
    #sns.pairplot(corr)
    #plt.show()
    
    plot_countplot(dfcategorical,'Lable Count Plot','countplot')   
    #for column in dfcategorical.columns:
    #    print("in column loop",column)
    #    sns.countplot(x = column , data = dfcategorical, label="Count")
    #    plt.show()
        
    return 

def convert_data_numerical_categorical(dataframe):
    print("****convert data to numerical and Categorical dataframes ****'")
    #Check the categorical data(nominal or ordinal)
    dfcategorical = dataframe.select_dtypes(include=[object])
    
    #Check the numerical data(discrete or continuous)
    #dataframe = dataframe.select_dtypes(exclude=['object'])
    dfnumeric = dataframe.select_dtypes(exclude=['object'])
    
    #dataframelabel = dataframe.select_dtypes(include=['object'])
    return dfcategorical, dfnumeric

    #def check_data_lables(dataframe):
    #Drop rows with nulls
    #Drop column with complete nulls
    #df = df.dropna(axis = 0, how = 'any')
    #Find duplicates data
    #repeat_patients = df.groupby(by = 'patient_id').size().sort_values(ascending =False)
    #Filter rows more than 2 times
    #filtered_patients = repeat_patients[repeat_patients > 2].to_frame().reset_index()
    #filtered_df = df[~df.patient_id.isin(filtered_patients.patient_id)]

def impute_nulls_data(dataframe):
    print("****impute_nulls_data****'")
    imp = Imputer(missing_values="NaN",strategy="mean",axis =0) #check the strategy options strategy="most_frequent"
    dfencodeimpute = imp.fit_transform(dataframe)
    dfencodeimpute = pd.DataFrame(dfencodeimpute)
    dfencodeimpute.columns = dataframe.columns
    dataframe = dfencodeimpute
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


def exploratorydataanalysis(chat_in):
    print("here:",chat_in)
    dir = './Data/'
    dirlog = './Log/'
    #file = 'breast-cancer-data.csv'
    file = chat_in
    predictmodel = 'classification'
    # Find the input file
    with os.scandir(dir) as entries:
        for entry in entries:
           #print(entry.name)
            if(entry.name == file):
                print(entry.name)
                print("Yes")

    # 1. Exploratory Data Analysis EDA
    print("here2:",dir+chat_in)
    dforig, html = read_analyze_data(dir+file)
    print(html)
    html = html.replace('dataframe','table table-responsive',1)
    
    dfcategorical, dfwolables = convert_data_numerical_categorical(dforig)
    
    plot_analysis_data(dfwolables, dfcategorical)
    
    #text_file = open("index.html", "w")
    #text_file.write(html)
    #text_file.close()

    return html
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
    
    #scale data normalize data (reduce the impact of outliers)
    dfv = scale_data_StandardScalar(dfnumericimpute)
    dfv.head()
    # join encoded and numeric data
    dfv = pd.concat([dfv,dfcategoricalencode],axis = 1)
    
       
    return dfv





def make_classification(chat_in):
    file = chat_in
    dataframe = pd.read_csv(file, skipinitialspace = True, thousands=',')
    dfv = data_wangling(dataframe)
    
   
    return


if __name__ == '__main__':
    from pprint import pprint
    print("Checking to see what empty string predicts")
    print('input string is ')
    #chat_in = 'bob'
    chat_in = '../Data/breast-cancer-data.csv'
    pprint(chat_in)
    X_input = exploratorydataanalysis(chat_in)
    make_classification(chat_in)

    #print(f'Input values: {x_input}')
    #print('Output probabilities')
    #pprint(probs)
    
