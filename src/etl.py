import utils
import pandas as pd
import numpy as np
from datetime import timedelta

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    
    '''
    TODO: This function needs to be completed.
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv')
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'feature_map.csv')

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 a

    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''
    #change mortality data column from  str to datetime 
    mortality['timestamp'] = pd.to_datetime(mortality['timestamp'],format = '%Y-%m-%d',errors = 'ignore')
    dead_id = mortality['patient_id']
    
    #caculate index date in deceased people
    mort_index = mortality[['patient_id','timestamp']]
    mort_index['timestamp'] = mort_index['timestamp'] - timedelta(days=30)
    
    #get the live events
    events_alive = events.loc[~events['patient_id'].isin(dead_id)]
    events_alive['timestamp'] = pd.to_datetime(events_alive['timestamp'],format = '%Y-%m-%d',errors = 'ignore')
    events_alive_record = events_alive.groupby(['patient_id'])['timestamp'].unique()
    alive_index = events_alive_record.apply(lambda x:max(x))
    alive_index = alive_index.to_frame().reset_index()
    
    indx_date = mort_index.append(alive_index,ignore_index=True)
    indx_date.columns = ['patient_id','indx_date']
    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', index=False)
    
    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 b

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''
    #change timestamp from str to datetime object
    events['timestamp'] = pd.to_datetime(events['timestamp'],format = '%Y-%m-%d',errors = 'ignore')
    #merge ind_date to events
    events_ind = pd.merge(events,indx_date,how='outer',on = 'patient_id')
    
    #filter events
    filtered = events_ind.loc[events_ind['timestamp'] > (events_ind['indx_date']- timedelta(days=2000))]
    filtered = filtered.loc[filtered['timestamp']<filtered['indx_date']]
    filtered_events = filtered[['patient_id', 'event_id', 'value']]
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', index=False)
    
    return filtered_events


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 c

    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and count to calculate feature value
    4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return filtered_events
    '''
    #remove rows with null value
    filtered_events = filtered_events_df.dropna(how = 'any',axis=0)
    
    #change all lab value to 1
    filtered_events.loc[:,'value']=1
    
    #groupby on 'patient_id' and 'events' then count
    aggregated_events = filter_events.groupby(by = ['patient_id','event_id'],as_index=False).count()
    
    #change feature map df to a dictionary
    feature_map_dic = dict(zip(feature_map_df['event_id'],feature_map_df['idx']))
    
    #replace the event_id with the feature_id in feature_map_dic
    aggregated_events['event_id'] = aggregate_events['event_id'].map(feature_map_dic)
    
    aggregate_events.columns =  ['patient_id','feature_id','value']

    aggregate_events.to_csv(deliverables_path + 'etl_aggregated_events.csv',index = False)
    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''
    patient_features = {}
    mortality = {}

    return patient_features, mortality

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    TODO: This function needs to be completed

    Refer to instructions in Q3 d

    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')
    
    deliverable1.write(bytes((""),'UTF-8')); #Use 'UTF-8'
    deliverable2.write(bytes((""),'UTF-8'));

def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()