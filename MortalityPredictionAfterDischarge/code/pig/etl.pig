-- ***************************************************************************
-- TASK
-- Aggregate events into features of patient and generate training, testing data for mortality prediction.
-- Steps have been provided to guide you.
-- You can include as many intermediate steps as required to complete the calculations.
-- ***************************************************************************

-- ***************************************************************************
-- TESTS
-- To test, please change the LOAD path for events and mortality to ../../test/events.csv and ../../test/mortality.csv
-- 6 tests have been provided to test all the subparts in this exercise.
-- Manually compare the output of each test against the csv's in test/expected folder.
-- ***************************************************************************

-- register a python UDF for converting data into SVMLight format
REGISTER utils.py USING jython AS utils;

-- load events file
events = LOAD '../data/events.csv' USING PigStorage(',') AS (patientid:int, eventid:chararray, eventdesc:chararray, timestamp:chararray, value:float);

-- select required columns from events
events = FOREACH events GENERATE patientid, eventid, ToDate(timestamp, 'yyyy-MM-dd') AS etimestamp, value;

-- load mortality file
mortality = LOAD '../data/mortality.csv' USING PigStorage(',') as (patientid:int, timestamp:chararray, label:int);

mortality = FOREACH mortality GENERATE patientid, ToDate(timestamp, 'yyyy-MM-dd') AS mtimestamp, label;

--To display the relation, use the dump command e.g. DUMP mortality;

-- ***************************************************************************
-- Compute the index dates for dead and alive patients
-- ***************************************************************************
eventswithmort = JOIN events BY patientid LEFT, mortality BY patientid;
-- perform join of events and mortality by patientid;

filtered_dead = FILTER eventswithmort BY mortality::label ==1;
deadevents = FOREACH filtered_dead GENERATE events::patientid as patientid, events::eventid as eventid, events::value as value, mortality::label as label, DaysBetween(mortality::mtimestamp, events::etimestamp)-30 as time_difference;
-- detect the events of dead patients and create it of the form (patientid, eventid, value, label, time_difference) where time_difference is the days between index date and each event timestamp

filtered_alive = FILTER eventswithmort BY (mortality::label is null);
aliveevents_group = GROUP filtered_alive BY events::patientid;
aliveevents_idx = FOREACH aliveevents_group GENERATE group, MAX(filtered_alive.events::etimestamp) as index_date;
aliveevents_join = JOIN filtered_alive BY events::patientid LEFT, aliveevents_idx BY group;
aliveevents = FOREACH aliveevents_join GENERATE filtered_alive::events::patientid as patientid, filtered_alive::events::eventid as eventid, filtered_alive::events::value as value, 0 as label, DaysBetween(aliveevents_idx::index_date, filtered_alive::events::etimestamp) as time_difference;
-- detect the events of alive patients and create it of the form (patientid, eventid, value, label, time_difference) where time_difference is the days between index date and each event timestamp

--TEST-1
deadevents = ORDER deadevents BY patientid, eventid;
aliveevents = ORDER aliveevents BY patientid, eventid;
STORE aliveevents INTO 'aliveevents' USING PigStorage(',');
STORE deadevents INTO 'deadevents' USING PigStorage(',');

-- ***************************************************************************
-- Filter events within the observation window and remove events with missing values
-- ***************************************************************************
filtered_dead = FILTER deadevents BY (time_difference >=0L) and (time_difference <= 2000L) and (value is not null);
filtered_alive = FILTER aliveevents BY (time_difference >=0L) and (time_difference <= 2000L) and (value is not null);
filtered = UNION filtered_dead, filtered_alive;

-- contains only events for all patients within the observation window of 2000 days and is of the form (patientid, eventid, value, label, time_difference)

--TEST-2
filteredgrpd = GROUP filtered BY 1;
filtered = FOREACH filteredgrpd GENERATE FLATTEN(filtered);
filtered = ORDER filtered BY patientid, eventid,time_difference;
STORE filtered INTO 'filtered' USING PigStorage(',');

-- ***************************************************************************
-- Aggregate events to create features
-- ***************************************************************************
group_features = GROUP filtered BY (patientid, eventid);
featureswithid = FOREACH group_features GENERATE FLATTEN(group) as (patientid, eventid), COUNT(filtered) as featurevalue;


-- for group of (patientid, eventid), count the number of  events occurred for the patient and create relation of the form (patientid, eventid, featurevalue)

--TEST-3
featureswithid = ORDER featureswithid BY patientid, eventid;
STORE featureswithid INTO 'features_aggregate' USING PigStorage(',');

-- ***************************************************************************
-- Generate feature mapping
-- ***************************************************************************
events_name = FOREACH featureswithid GENERATE eventid;
events_name = DISTINCT events_name;
events_name_ordered = ORDER events_name BY eventid;
events_rank = RANK events_name_ordered;
all_features = FOREACH events_rank GENERATE ($0-1) as idx, $1 as eventid;


-- compute the set of distinct eventids obtained from previous step, sort them by eventid and then rank these features by eventid to create (idx, eventid). Rank should start from 0.

-- store the features as an output file
STORE all_features INTO 'features' using PigStorage(' ');

features_all = JOIN featureswithid BY eventid, all_features BY eventid;
features = FOREACH features_all GENERATE featureswithid::patientid as patientid, all_features::idx as idx, featureswithid::featurevalue as featurevalue;

-- perform join of featureswithid and all_features by eventid and replace eventid with idx. It is of the form (patientid, idx, featurevalue)

--TEST-4
features = ORDER features BY patientid, idx;
STORE features INTO 'features_map' USING PigStorage(',');

-- ***************************************************************************
-- Normalize the values using min-max normalization
-- Use DOUBLE precision
-- ***************************************************************************
maxvalues = GROUP features BY idx;
maxvalues = FOREACH maxvalues GENERATE group as idx, MAX(features.featurevalue) as maxvalue;


-- group events by idx and compute the maximum feature value in each group. I t is of the form (idx, maxvalue)

normalized = JOIN features BY idx, maxvalues BY idx;
-- join features and maxvalues by idx

features = FOREACH normalized GENERATE features::patientid as patientid, features::idx as idx, features::featurevalue*1.0/maxvalues::maxvalue as normalizedfeaturevalue;

-- compute the final set of normalized features of the form (patientid, idx, normalizedfeaturevalue)

--TEST-5
features = ORDER features BY patientid, idx;
STORE features INTO 'features_normalized' USING PigStorage(',');

-- ***************************************************************************
-- Generate features in svmlight format
-- features is of the form (patientid, idx, normalizedfeaturevalue) and is the output of the previous step
-- e.g.  1,1,1.0
--  	 1,3,0.8
--	     2,1,0.5
--       3,3,1.0
-- ***************************************************************************

grpd = GROUP features BY patientid;
grpd_order = ORDER grpd BY $0;
features = FOREACH grpd_order
{
    sorted = ORDER features BY idx;
    generate group as patientid, utils.bag_to_svmlight(sorted) as sparsefeature;
}

-- ***************************************************************************
-- Split into train and test set
-- labels is of the form (patientid, label) and contains all patientids followed by label of 1 for dead and 0 for alive
-- e.g. 1,1
--	2,0
--      3,1
-- ***************************************************************************

labels = FOREACH filtered GENERATE patientid, label;
labels = DISTINCT labels;

-- create it of the form (patientid, label) for dead and alive patients

--Generate sparsefeature vector relation
samples = JOIN features BY patientid, labels BY patientid;
samples = DISTINCT samples PARALLEL 1;
samples = ORDER samples BY $0;
samples = FOREACH samples GENERATE $3 AS label, $1 AS sparsefeature;

--TEST-6
STORE samples INTO 'samples' USING PigStorage(' ');

-- randomly split data for training and testing
DEFINE rand_gen RANDOM('6505');
samples = FOREACH samples GENERATE rand_gen() as assignmentkey, *;
SPLIT samples INTO testing IF assignmentkey <= 0.20, training OTHERWISE;
training = FOREACH training GENERATE $1..;
testing = FOREACH testing GENERATE $1..;

-- save training and tesing data
STORE testing INTO 'testing' USING PigStorage(' ');
STORE training INTO 'training' USING PigStorage(' ');
