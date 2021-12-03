#!/usr/bin/env python
# coding: utf-8

# ## Functions to Wrangle the Zillow Data

# ### Import

# In[23]:


import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from env import host, user, password

import warnings
warnings.filterwarnings("ignore")

import sklearn.preprocessing


# ### Acquire Data

# In[2]:


# Connect to SQL database
def get_db_url(db_name):
    '''
    This function contacts Codeup's SQL database and uses the info from 
    my env to create a connection URL.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db_name}'


# In[3]:


# Get Zillow data

def get_zillow_data():
    '''
    This function reads the zillow data from Codeup database to a dataframe,
    and returns the dataframe.
    '''
    # SQL query
    sql_query =  '''
            
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
    FROM properties_2017
    LEFT JOIN propertylandusetype USING(propertylandusetypeid)
    WHERE propertylandusedesc IN ("Single Family Residential",                       
                                  "Inferred Single Family Residential")'''
    
    # Read in DataFrame
    df = pd.read_sql(sql_query, get_db_url('zillow'))
    
    df = df.rename(columns = {'bedroomcnt': 'num_beds',
                                      'bathroomcnt': 'num_baths',
                                      'calculatedfinishedsquarefeet': 'square_footage',
                                      'taxvaluedollarcnt': 'tax_value',
                                      'yearbuilt': 'year_built', 'taxamount': 'tax_amount'})
    return df


# In[4]:


df = get_zillow_data()
df.head()


# ### Remove Outliers

# In[5]:


# Function to remove outliers
def remove_outliers(df, k, col_list):
    ''' 
        This function removes outliers from a list of columns in a df
        then reurns that df. 
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe sans outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df


# In[6]:


split_df = df[['num_beds', 'num_baths', 'square_footage', 'tax_value']]


# In[7]:


target = df.tax_value


# ### Visually Explore Individual Variables

# In[8]:


def get_hist(df):
    ''' Gets histographs of acquired continuous variables'''
    
    plt.figure(figsize=(16, 3))

    # List of columns
    cols = [col for col in df.columns if col not in ['fips', 'year_built']]

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        df[col].hist(bins=5)

        # Hide gridlines.
        plt.grid(False)

        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)

        plt.tight_layout()

    plt.show()


# In[9]:


def get_box(df):
    ''' Gets boxplots of acquired continuous variables'''
    
    # List of columns
    cols = ['num_beds', 'num_baths', 'square_footage', 'tax_value', 'tax_amount']

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):

        # i starts at 0, but plot should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[[col]])

        # Hide gridlines.
        plt.grid(False)

        # sets proper spacing between plots
        plt.tight_layout()

    plt.show()


# ### Prepare Data

# In[10]:


def prepare_zillow(df):
    ''' Prepare zillow data for exploration'''

    # removing outliers
    df = remove_outliers(df, 1.5, ['num_beds', 'num_baths', 'square_footage', 'tax_value', 'year_built', 'tax_amount'])
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # converting column datatypes
    #df.num_beds = df.num_beds.astype(object)
    #df.num_baths = df.num_baths.astype(object)
    #df.square_footage = df.square_footage.astype(object)
    df.year_built = df.year_built.astype(object)
    df.fips = df.fips.astype(object)
    
    # get distributions of numeric data
    get_hist(df)
    get_box(df)

    
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    

    return train, validate, test 


# ### Acquire, Prep, and Split

# In[11]:


def wrangle_zillow():
    '''
    This function will both aquire and prepare the zillow data.
    It displays the histogram and boxplots before splitting the df.
    After running this function the data is ready for exploritory analysis.
    '''
    train, validate, test = prepare_zillow(get_zillow_data())
    
    return train, validate, test


# In[12]:


train, validate, test = wrangle_zillow()


# In[13]:


train.head()


# In[14]:


train.shape


# In[15]:


validate.shape


# In[16]:


test.shape


# ### Split into X and y variables

# In[17]:


def split_tvt_into_variables(train, validate, test, target):

# split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target, 'year_built', 'fips'])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target, 'year_built', 'fips'])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target, 'year_built', 'fips'])
    y_test = test[target]
    
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test


# In[18]:


train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test = split_tvt_into_variables(train, validate, test, target='tax_value')


# In[19]:


X_train.shape


# ### Scaling Data

# In[24]:


def Min_Max_Scaler(X_train, X_validate, X_test):
    """
    Takes in X_train, X_validate and X_test dfs with numeric values only
    Returns scaler, X_train_scaled, X_validate_scaled, X_test_scaled dfs 
    """
    #Fit the thing
    scaler = sklearn.preprocessing.MinMaxScaler().fit(X_train)
    
    #transform the thing
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate), index = X_validate.index, columns = X_validate.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index = X_test.index, columns = X_test.columns)
    
    return scaler, X_train_scaled, X_validate_scaled, X_test_scaled


# In[25]:


scaler, X_train_scaled, X_validate_scaled, X_test_scaled = Min_Max_Scaler(X_train, X_validate, X_test)


# In[26]:


X_train_scaled.shape


# In[30]:


def Standard_Scaler(X_train, X_validate, X_test):
    """
    Takes in X_train, X_validate and X_test dfs with numeric values only
    Returns scaler, X_train_scaled, X_validate_scaled, X_test_scaled dfs
    """

    scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate), index = X_validate.index, columns = X_validate.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index = X_test.index, columns = X_test.columns)
    
    return scaler, X_train_scaled, X_validate_scaled, X_test_scaled


# In[31]:


scaler, X_train_scaled, X_validate_scaled, X_test_scaled = Standard_Scaler(X_train, X_validate, X_test)


# In[32]:


X_train_scaled.shape


# ### Visualize Data

# In[27]:


#--------VISUALIZE THE SCALED DATA
def visualize_scaled_date(scaler, scaler_name, feature):
    scaled = scaler.fit_transform(train[[feature]])
    fig = plt.figure(figsize = (12,6))

    gs = plt.GridSpec(2,2)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])

    ax1.scatter(train[[feature]], scaled)
    ax1.set(xlabel = feature, ylabel = 'Scaled' + feature, title = scaler_name)

    ax2.hist(train[[feature]])
    ax2.set(title = 'Original')

    ax3.hist(scaled)
    ax3.set(title = 'Scaled')
    plt.tight_layout();


# In[29]:


# visualize_scaled_date(sklearn.preprocessing.MinMaxScaler(), 'Min Max Scaler', 'num_beds')


# In[ ]:




