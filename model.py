"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    df = feature_vector_df.copy()
    def clean_column_names(df): 

        """ Function removes special characters from the dataframe
        column headings. 
        
        Takes in a dataframe as argument.

        Removes:' ','_','_','','-',''
        """
    df.columns=[i.replace(' ','_') for i in df.columns]
    df.columns=[i.replace('_','') for i in df.columns]
    df.columns=[i.replace('-','') for i in df.columns]
    
    return df
    def total_missing(df): 
        """ Code counts number of nulls and returns a dictionary
        of percentage null values and column name.

        Takes in a dataframe as argument.
        """
        #code counts number of nulls and returns a list
        #of percentage null values and column name
        values = {}
        rows = len(df)
        for column in df:
            x = ((df[column].isnull().sum())/(rows))*100
            values[column] = round(x,3)
    return values
    def drop_columns(input_df, threshold, unique_value_threshold): 
        """ Code will drop columns that are over a specified threshold
        for null values and unique values

        Takes in a dataframe as argument, null threshold and 
        unique value threshold.
        """
        func1_df = input_df.copy()

        rows = len(func1_df)
        for column in func1_df:
            x = ((func1_df[column].isnull().sum())/(rows))*100
            y = ((func1_df[column].nunique(dropna=True))/(rows)*100)
        if x > threshold or y < unique_value_threshold:
           func1_df = func1_df.drop(column, 1)
    return func1_df
    def impute(input_df, column, choice='median'): 
        """ Code will impute mean or median as specified for a 
        specified column

        Takes in a dataframe as argument, column name and
        choice between mean or median.
        """
    
        mean_df = input_df.copy()
        median_df = input_df.copy()
    
        if choice in ['median', 'mean']:
            if choice == 'mean':
              mean_df[column].fillna(round(median_df[column].mean(), 1), inplace=True)
              return mean_df
        
            else:
              median_df[column].fillna(round(median_df[column].median(), 1), inplace=True)
              return median_df

        else:
            raise ValueError ("choose median or mean as a choice parameter")
    def column_replacer(df, replaced, new): 
        """ Code will replace a column name with another as specified and
        add the new identical column on at the end.

        Takes in a dataframe as argument, column name and
        a new name to replace it with.
        """    
    
        df[new] = df[replaced]
        return df
    def drop_specific_columns(df, cols): 
        """ Code will drop columns specified in the "cols"
        list.

        Takes in a dataframe as argument and a list of column names
        to drop.
        """
    
        df2 = df.copy()
        for column in df2:
            if column in cols:
               df2 = df2.drop(column, 1)
        return df2
    
def categorical_time_changer(df): 
    """ Code will return categorical values for the timestamp columns.
        Unique code for the Zindi challenge.

        Takes in a dataframe as argument.
    """
    train2_df = df.copy()
    time=[]
    for i in range(len(train2_df['PickupTime'])):

      idx=train2_df['PickupTime'][i].index(':')

      if train2_df['PickupTime'][i][-2:]=='AM' and int(train2_df['PickupTime'][i][:idx]) in [3,4,5,6,7,8,9]:
        time.append('EarlyMorning')
      elif train2_df['PickupTime'][i][-2:]=='AM' and int(train2_df['PickupTime'][i][:idx]) in [10,11]:
        time.append('LateMorning')
      elif train2_df['PickupTime'][i][-2:]=='PM' and int(train2_df['PickupTime'][i][:idx]) in [12,1,2,3]:
        time.append('EarlyAfternoon')
      elif train2_df['PickupTime'][i][-2:]=='PM' and int(train2_df['PickupTime'][i][:idx]) in [4,5,6,7]:
        time.append('Late')
      elif train2_df['PickupTime'][i][-2:]=='AM':
        time.append('EarlyMorning')
      else:
        time.append('EarlyAfternoon')

    train2_df['PickupTime']=time
    return train2_df
    def shift_to_last(df, y_variable):
        """ Function will swap the specified column in a dataframe
        to the end of the dataframe

        Takes in a dataframe as argument and a string column name.
        """
        spare = df.pop(y_variable)
        df[y_variable] = spare

        return df
    df_to_train = clean_column_names(train_df)
    df_to_train = drop_columns(df_to_train, 95, 0)
    df_to_train = impute(df_to_train, column='Temperature', choice='mean')
    df_to_train = column_replacer(df_to_train, replaced = 'PlacementDayofMonth', new='DayofMonth')
    df_to_train = column_replacer(df_to_train, replaced = 'ConfirmationWeekday(Mo=1)', new='Weekday')
    cols = ['VehicleType', 'PlacementDayofMonth','PlacementWeekday(Mo=1)','ConfirmationDayofMonth','ConfirmationWeekday(Mo=1)', 'ArrivalatPickupDayofMonth','ArrivalatPickupWeekday(Mo=1)',
        'PickupDayofMonth','PickupWeekday(Mo=1)', 'ArrivalatDestinationDayofMonth','ArrivalatDestinationWeekday(Mo=1)','OrderNo','UserId' ]
    df_to_train = drop_specific_columns(df_to_train, cols)
    df_to_train = categorical_time_changer(df_to_train)
    cols = ['PlacementTime','ConfirmationTime','ArrivalatPickupTime','ArrivalatDestinationTime']
    df_to_train = drop_specific_columns(df_to_train, cols)
    df_to_train = pd.get_dummies(df_to_train,columns=['PickupTime', 'PersonalorBusiness','PlatformType'], drop_first=1 )
    df_to_train = shift_to_last(df_to_train, y_variable='TimefromPickuptoArrival')

    riders_df= clean_column_names(riders_df)
    #Innner join Train_df and Rider_df on Rider_Id
    #you are changing the df by using merge
    #use a left join to ensure train data stays

    df_to_train=pd.merge(df_to_train,riders_df,how='left',on='RiderId')
    df_to_train = drop_specific_columns(df_to_train, cols=['RiderId'])
    df_to_train = shift_to_last(df_to_train, y_variable='TimefromPickuptoArrival')

    # split data into predictors and response
    # THIS X AND Y VALUE ARE BASED ON MERGED DATAFRAME AND THEREFORE IS NOT THE SAME AS THE UNMERGE ONE. X COLS WILL DIFFER

    X = df_to_train.drop('TimefromPickuptoArrival', axis=1)
    y = df_to_train['TimefromPickuptoArrival']

    # import scaler method from sklearn
    from sklearn.preprocessing import StandardScaler
    # create scaler object
    scaler = StandardScaler()
    # create scaled version of the predictors (there is no need to scale the response)
    X_scaled = scaler.fit_transform(X)
    # convert the scaled predictor values into a dataframe
    X_standardise = pd.DataFrame(X_scaled,columns=X.columns)




    # ------------------------------------------------------------------------

    return X_standardise

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
