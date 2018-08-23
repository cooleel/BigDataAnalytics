import time
import pandas as pd
import numpy as np
from datetime import datetime

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    '''
    TODO : This function needs to be completed.
    Read the events.csv and mortality_events.csv files. 
    Variables returned from this function are passed as input to the metric functions.
    '''
    events = pd.read_csv(filepath + 'events.csv')
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    return events, mortality

def event_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the event count metrics.
    Event count is defined as the number of events recorded for a given patient.
    '''
    dead_Id = mortality['patient_id']
    #get the dead events
    events_dead = events.loc[events['patient_id'].isin(dead_Id)]
    #get the live events
    events_alive = events.loc[~events['patient_id'].isin(dead_Id)]
    
    #get the event counts in dead patients
    events_dead_counts = events_dead.groupby(['patient_id'])['event_id'].count()
    
    #get the event counts in alive patients
    events_alive_counts = events_alive.groupby(['patient_id'])['event_id'].count()
    
    avg_dead_event_count = events_dead.shape[0]/len(dead_Id)
    max_dead_event_count = events_dead_counts.sort_values().values[-1]
    min_dead_event_count = events_dead_counts.sort_values().values[0]
    avg_alive_event_count = events_alive.shape[0]/events_alive['patient_id'].nunique()
    max_alive_event_count = events_alive_counts.sort_values().values[-1]
    min_alive_event_count = events_alive_counts.sort_values().values[0]

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the encounter count metrics.
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
    '''
    dead_Id = mortality['patient_id']
    #get the dead events
    events_dead = events.loc[events['patient_id'].isin(dead_Id)]
    #get the live events
    events_alive = events.loc[~events['patient_id'].isin(dead_Id)]
    
    
    #get the unique encounter counts in deceased people as an array
    unique_dead_encounter = events_dead.groupby(['patient_id'])['timestamp'].nunique().sort_values().values
    #get the unique encounter counts in alive people as an array
    unique_alive_encounter = events_alive.groupby(['patient_id'])['timestamp'].nunique().sort_values().values
    
    avg_dead_encounter_count = sum(unique_dead_encounter)/len(dead_Id)
    max_dead_encounter_count = unique_dead_encounter[-1]
    min_dead_encounter_count = unique_dead_encounter[0] 
    avg_alive_encounter_count =sum(unique_alive_encounter)/events_alive['patient_id'].nunique()
    max_alive_encounter_count = unique_alive_encounter[-1]
    min_alive_encounter_count = unique_alive_encounter[0]

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    TODO: Implement this function to return the record length metrics.
    Record length is the duration between the first event and the last event for a given patient. 
    '''
    #change the timestamp to datetime object
    events['timestamp'] = pd.to_datetime(events['timestamp'],format = '%Y-%m-%d',errors = 'ignore')
    dead_Id = mortality['patient_id']
    #get the dead events
    events_dead = events.loc[events['patient_id'].isin(dead_Id)]
    #get the live events
    events_alive = events.loc[~events['patient_id'].isin(dead_Id)]
    
    def record_len(arr):
        return (max(arr)-min(arr))/np.timedelta64(1,'D').astype(int)
    events_dead_record = events_dead.groupby(['patient_id'])['timestamp'].unique()
    events_alive_record = events_alive.groupby(['patient_id'])['timestamp'].unique()
    
    dead_record_len = events_dead_record.apply(record_len)
    alive_record_len = events_alive_record.apply(record_len)
    
    avg_dead_rec_len = dead_record_len.sum().days/len(dead_Id)
    max_dead_rec_len = dead_record_len.max().days
    min_dead_rec_len = dead_record_len.min().days
    avg_alive_rec_len = alive_record_len.sum().days/events_alive['patient_id'].nunique()
    max_alive_rec_len = alive_record_len.max().days
    min_alive_rec_len = alive_record_len.min().days

    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    '''
    DO NOT MODIFY THIS FUNCTION.
    '''
    # You may change the following path variable in coding but switch it back when submission.
    train_path = '../data/train/'

    # DO NOT CHANGE ANYTHING BELOW THIS ----------------------------
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute event count metrics: " + str(end_time - start_time) + "s"))
    print(event_count)

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute encounter count metrics: " + str(end_time - start_time) + "s"))
    print(encounter_count)

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute record length metrics: " + str(end_time - start_time) + "s"))
    print(record_length)
    
if __name__ == "__main__":
    main()
