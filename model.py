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
    
    feature_vector_df = clean_column_names(feature_vector_df)
    feature_vector_df = drop_columns(feature_vector_df, threshold = 95, unique_value_threshold = 0)
    feature_vector_df = impute(feature_vector_df, column='Temperature', choice='mean')
    feature_vector_df = column_replacer(feature_vector_df, replaced = 'PlacementDayofMonth', new='DayofMonth')
    feature_vector_df = column_replacer(feature_vector_df, replaced = 'ConfirmationWeekday(Mo=1)', new='Weekday')
    #we need to keep the order number
    cols = ['VehicleType', 'PlacementDayofMonth','PlacementWeekday(Mo=1)','ConfirmationDayofMonth','ConfirmationWeekday(Mo=1)', 'ArrivalatPickupDayofMonth','ArrivalatPickupWeekday(Mo=1)',
            'PickupDayofMonth','PickupWeekday(Mo=1)', 'ArrivalatDestinationDayofMonth','ArrivalatDestinationWeekday(Mo=1)','UserId', 'RiderId' ]
    feature_vector_df = drop_specific_columns(feature_vector_df, cols)
    feature_vector_df = categorical_time_changer(feature_vector_df)
    cols = ['PlacementTime','ConfirmationTime','ArrivalatPickupTime','ArrivalatDestinationTime']
    feature_vector_df = drop_specific_columns(feature_vector_df, cols)
    #feature_vector_df = pd.get_dummies(feature_vector_df,columns=['PickupTime', 'PersonalorBusiness','PlatformType'], drop_first=1 )
    feature_vector_df.rename(columns={"Distance(KM)": "Distance"}, inplace=True)
    feature_vector_df.drop('OrderNo',axis=1,inplace=True)
    # ------------------------------------------------------------------------

    return feature_vector_df 

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
