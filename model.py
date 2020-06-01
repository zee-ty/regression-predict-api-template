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
from sklearn.preprocessing import StandardScaler

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
    zindi2_df = feature_vector_df.copy()
    # ----------- Replace this code with your own preprocessing steps --------
    zindi2_df.columns=[i.replace(' ','_') for i in zindi2_df.columns]
    zindi2_df.columns=[i.replace('_','') for i in zindi2_df.columns]
    zindi2_df.columns=[i.replace('-','') for i in zindi2_df.columns]
    zindi2_df.drop('Precipitationinmillimeters',axis=1,inplace=True)
    zindi2_df["Temperature"].fillna(zindi2_df['Temperature'].mean(), inplace=True)

    zindi2_df.drop('VehicleType',axis=1,inplace=True)
    zindi2_df['DayofMonth'] = zindi2_df['PlacementDayofMonth']
    zindi2_df['Weekday'] = zindi2_df['ConfirmationWeekday(Mo=1)']

    zindi2_df.drop('PlacementDayofMonth',axis=1,inplace=True)
    zindi2_df.drop('PlacementWeekday(Mo=1)',axis=1,inplace=True)
    zindi2_df.drop('ConfirmationDayofMonth',axis=1,inplace=True)
    zindi2_df.drop('ConfirmationWeekday(Mo=1)',axis=1,inplace=True)
    zindi2_df.drop('ArrivalatPickupDayofMonth',axis=1,inplace=True)
    zindi2_df.drop('ArrivalatPickupWeekday(Mo=1)',axis=1,inplace=True)
    zindi2_df.drop('PickupDayofMonth',axis=1,inplace=True)
    zindi2_df.drop('PickupWeekday(Mo=1)',axis=1,inplace=True)
    #zindi2_df.drop('ArrivalatDestinationDayofMonth',axis=1,inplace=True)
    #zindi2_df.drop('ArrivalatDestinationWeekday(Mo=1)',axis=1,inplace=True)

    #Drop order no and User ID since they do not impact logistics
    #zindi2_df.drop('OrderNo',axis=1,inplace=True)
    zindi2_df.drop('UserId',axis=1,inplace=True)
    zindi2_df.drop('RiderId',axis=1,inplace=True)

    time=[]

    for i in range(len(zindi2_df['PickupTime'])):

        idx=zindi2_df['PickupTime'][i].index(':')

        if zindi2_df['PickupTime'][i][-2:]=='AM' and int(zindi2_df['PickupTime'][i][:idx]) in [3,4,5,6,7,8,9]:
            time.append('EarlyMorning')
        elif zindi2_df['PickupTime'][i][-2:]=='AM' and int(zindi2_df['PickupTime'][i][:idx]) in [10,11]:
            time.append('LateMorning')
        elif zindi2_df['PickupTime'][i][-2:]=='PM' and int(zindi2_df['PickupTime'][i][:idx]) in [12,1,2,3]:
            time.append('EarlyAfternoon')
        elif zindi2_df['PickupTime'][i][-2:]=='PM' and int(zindi2_df['PickupTime'][i][:idx]) in [4,5,6,7]:
            time.append('Late')
        elif zindi2_df['PickupTime'][i][-2:]=='AM':
            time.append('EarlyMorning')
        else:
            time.append('EarlyAfternoon')
    zindi2_df['PickupTime']=time

    zindi2_df.drop('PlacementTime',axis=1,inplace=True)
    zindi2_df.drop('ConfirmationTime',axis=1,inplace=True)
    zindi2_df.drop('ArrivalatPickupTime',axis=1,inplace=True)
    #zindi2_df.drop('ArrivalatDestinationTime',axis=1,inplace=True)
    zindi2_df.drop('OrderNo',axis=1,inplace=True)
    zindi2_df.rename(columns={"Distance(KM)": "Distance"}, inplace=True)

    zindi4_df = zindi2_df.copy()

    

    #####################################
    # get dummies manually
    if zindi4_df['PlatformType'].values[0] == 1:
        zindi4_df['PlatformType_2'] = 0
        zindi4_df['PlatformType_3'] = 0
        zindi4_df['PlatformType_4'] = 0

    elif zindi4_df['PlatformType'].values[0] == 2:
        zindi4_df['PlatformType_2'] = 1
        zindi4_df['PlatformType_3'] = 0
        zindi4_df['PlatformType_4'] = 0

    elif zindi4_df['PlatformType'].values[0] == 3:
        zindi4_df['PlatformType_2'] = 0
        zindi4_df['PlatformType_3'] = 1
        zindi4_df['PlatformType_4'] = 0

    elif zindi4_df['PlatformType'].values[0] == 4:
        zindi4_df['PlatformType_2'] = 0
        zindi4_df['PlatformType_3'] = 0
        zindi4_df['PlatformType_4'] = 1

    if zindi4_df['PersonalorBusiness'].values[0] == 'Personal':
        zindi4_df['PersonalorBusiness_Personal'] = 1
    else:
        zindi4_df['PersonalorBusiness_Personal'] = 0
    
    if zindi4_df['PickupTime'].values[0] == 'EarlyAfternoon':
        zindi4_df['PickupTime_EarlyMorning'] = 0
        zindi4_df['PickupTime_Late'] = 0
        zindi4_df['PickupTime_LateMorning'] = 0

    elif zindi4_df['PickupTime'].values[0] == 'EarlyMorning':
        zindi4_df['PickupTime_EarlyMorning'] = 1
        zindi4_df['PickupTime_Late'] = 0
        zindi4_df['PickupTime_LateMorning'] = 0

    elif zindi4_df['PickupTime'].values[0] == 'Late':
        zindi4_df['PickupTime_EarlyMorning'] = 0
        zindi4_df['PickupTime_Late'] = 1
        zindi4_df['PickupTime_LateMorning'] = 0

    elif zindi4_df['PickupTime'].values[0] == 'LateMorning':
        zindi4_df['PickupTime_EarlyMorning'] = 0
        zindi4_df['PickupTime_Late'] = 0
        zindi4_df['PickupTime_LateMorning'] = 1
    

    zindi4_df.drop('PickupTime',axis=1,inplace=True)
    zindi4_df.drop('PersonalorBusiness',axis=1,inplace=True)
    zindi4_df.drop('PlatformType',axis=1,inplace=True)
    #####################################
    #scaler here
    scale_model = load_model(
    path_to_model='assets/trained-models/FinalSetupScaler.pkl')
    
    X = zindi4_df
    zindi4_df = scale_model.transform(zindi4_df)
    X_standardise = pd.DataFrame(zindi4_df,columns=X.columns)
    predict_vector = X_standardise.copy()
        #['Distance', 'Temperature', 'PickupLat', 'PickupLong', 'DestinationLat',
       #'DestinationLong', 'NoOfOrders', 'Age', 'AverageRating', 'NoofRatings',
       #'DayofMonth', 'Weekday', 'PickupTime_EarlyMorning', 'PickupTime_Late',
       #'PickupTime_LateMorning', 'PersonalorBusiness_Personal',
       #'PlatformType_2', 'PlatformType_3', 'PlatformType_4']]
    
    # ------------------------------------------------------------------------

    return predict_vector

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
