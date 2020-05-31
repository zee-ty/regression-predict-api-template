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
    df_to_train = feature_vector_df.copy()

    df_to_train.columns=[i.replace(' ','_') for i in df_to_train.columns]
    df_to_train.columns=[i.replace('_','') for i in df_to_train.columns]
    df_to_train.columns=[i.replace('-','') for i in df_to_train.columns]

    time=[]

    for i in range(len(df_to_train['PickupTime'])):

        idx=df_to_train['PickupTime'][i].index(':')

        if df_to_train['PickupTime'][i][-2:]=='AM' and int(df_to_train['PickupTime'][i][:idx]) in [3,4,5,6,7,8,9]:
           time.append('Early Morning')
        elif df_to_train['PickupTime'][i][-2:]=='AM' and int(df_to_train['PickupTime'][i][:idx]) in [10,11]:
            time.append('Late Morning')
        elif df_to_train['PickupTime'][i][-2:]=='PM' and int(df_to_train['PickupTime'][i][:idx]) in [12,1,2,3]:
             time.append('Early Afternoon')
        elif df_to_train['PickupTime'][i][-2:]=='PM' and int(df_to_train['PickupTime'][i][:idx]) in [4,5,6,7]:
             time.append('Late')
        elif df_to_train['PickupTime'][i][-2:]=='AM':
             time.append('Early Morning')
        else:
            time.append('Early Afternoon')


    df_to_train ['PickupTime']= time
    df_to_train.drop('OrderNo',axis=1,inplace=True)
    df_to_train.drop('UserId',axis=1,inplace=True)
    df_to_train.drop('VehicleType',axis=1,inplace=True)

    df_to_train.drop('ConfirmationDayofMonth',axis=1,inplace=True)
    df_to_train.drop('ConfirmationWeekday(Mo=1)',axis=1,inplace=True)
    df_to_train.drop('ArrivalatPickupDayofMonth',axis=1,inplace=True)
    df_to_train.drop('ArrivalatPickupWeekday(Mo=1)',axis=1,inplace=True)
    df_to_train.drop('PickupDayofMonth',axis=1,inplace=True)
    df_to_train.drop('PickupWeekday(Mo=1)',axis=1,inplace=True)

    df_to_train.drop('ArrivalatDestinationDayofMonth',axis=1,inplace=True)
    df_to_train.drop('ArrivalatDestinationWeekday(Mo=1)',axis=1,inplace=True)
    df_to_train.drop('ArrivalatDestinationTime',axis=1,inplace=True)
    df_to_train.drop('Precipitationinmillimeters',axis=1,inplace=True)

    df_to_train.drop('PlacementTime',axis=1,inplace=True)
    df_to_train.drop('ConfirmationTime',axis=1,inplace=True)
    df_to_train.drop('ArrivalatPickupTime',axis=1,inplace=True)
    df_to_train.drop('RiderId',axis=1,inplace=True)

    df_to_train['Temperature'].fillna((df_to_train['Temperature'].median()), inplace=True)

    df_to_train.columns=[i.replace('PlacementDayofMonth','DayofMonth') for i in df_to_train.columns]
    df_to_train.columns=[i.replace('PlacementWeekday(Mo=1)','Weekday') for i in df_to_train.columns]

    df_to_train = pd.get_dummies(df_to_train,columns=['PickupTime', 'PersonalorBusiness','PlatformType'], drop_first=1 )

    df_to_train['Temperature'] = df_to_train['Temperature'].astype(float)
    df_to_train['PickupLat'] = df_to_train['PickupLat'].astype(float)
    df_to_train['PickuoLong'] = df_to_train['PickupLong'].astype(float)
    df_to_train['AverageRating'] = df_to_train['AverageRating'].astype(float)
    df_to_train['DestinationLong'] = df_to_train['DestinationLong'].astype(float)
    df_to_train['DestinationLat'] = df_to_train['DestinationLat'].astype(float)

    # split data into predictors and response
    # THIS X AND Y VALUE ARE BASED ON MERGED DATAFRAME AND THEREFORE IS NOT THE SAME AS THE UNMERGE ONE. X COLS WILL DIFFER

    X = df_to_train.drop('TimefromPickuptoArrival', axis=1)
    #y = df_to_train['TimefromPickuptoArrival']

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
