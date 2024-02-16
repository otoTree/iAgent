from datetime import datetime
import time
'''
{time:{
memmory:{
session:[],
vision:[],
do:[]}
}
}
{'2024-02-16 18:39:30': {'memmory': session:[]},}
'''
memmory={'2024-02-16 18:39:30': {'memmory': [10]},
         '2024-02-16 18:39:31': {'memmory': [11]},
         '2024-02-16 18:39:32': {'memmory': [12]},
         '2024-02-16 18:39:34': {'memmory': [13]},
         '2024-02-16 18:39:35': {'memmory': [14]}}

def get_time_now():
    now = datetime.now()

    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time

def storage(session=[],vision=[],do=[]):
    memmory[get_time_now()]={'memmory':{'session':session,'vision':vision,'do':do}}


storage()

print(memmory)