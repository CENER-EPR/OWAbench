# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:49:33 2019

@author: Pawel Gancarski
"""

import pandas as pd
import datetime


# file paths
input_filepath = 'anh00a.csv'
output_filepath = 'anh00a.csv'


# read raw data
data = pd.read_csv(input_filepath, ',',
                   header = None,
                   skiprows = 1
                   ) 
# fix headers
turbines = pd.read_csv("../inputs/Anholt_layout.csv")
var_names = turbines['VDC ID'].tolist()
var_names.insert(0, 'time')
data.rename(columns=lambda x: var_names[x], inplace=True)

# fix time format
def fix_time(time_str):
	###                012345678901234567890

    ### inputs format: 02/01/2013 01:00
#    year    = int(time_str[6:10])
#    month   = int(time_str[3:5])
#    day     = int(time_str[0:2])
#    hours   = int(time_str[11:13])
#    minutes = int(time_str[14:16]) 

    ### inputs format: 2013-01-01 00:00:00
    year    = int(time_str[0:4])
    month   = int(time_str[5:7])
    day     = int(time_str[8:10])
    hours   = int(time_str[11:13])
    minutes = int(time_str[14:16]) 

    dt = datetime.datetime(year, month, day, hours, minutes)

    seconds_since = (dt - datetime.datetime(1970,1,1)).total_seconds()
    return seconds_since

data['time'] = data['time'].apply(fix_time)

# fix power format to MW 
out_data = data
out_data['time'] = data['time']


# write outputs
out_data.to_csv(output_filepath, index=False)

