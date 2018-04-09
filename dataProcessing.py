#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 18:29:12 2018

@author: moyuli
"""                
        
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Get files from directory:
import os
path = '/Users/moyuli/Documents/Capstone/database'
      


#for loop to go over all the databases
for filename in os.listdir(path):
    #if filename=='clip_1.sqlite':
    connection = sqlite3.connect(os.path.join(path,filename))
    
    #Defining The number of objects we have
    sql_command = "SELECT COUNT(*) FROM objects"

    max_objects_sql = connection.execute(sql_command)
    max_objects = max_objects_sql.fetchone()
    max_objects = max_objects[0]

    ## Enter the SQL command
    
    # Getting the positions of every object
    sql_query_positions = "SELECT a.object_id, b.frame_number, x_coordinate, y_coordinate\
                           FROM objects_features AS a \
                           JOIN positions AS b ON a.trajectory_id = b.trajectory_id"
    
    df_positions = pd.read_sql_query(sql_query_positions, connection)

    # Getting the velocities of every object
    sql_query_velocities = "SELECT a.object_id, b.frame_number, b.x_coordinate, b.y_coordinate\
                            FROM objects_features AS a \
                            JOIN velocities AS b ON a.trajectory_id = b.trajectory_id"
    df_velocities = pd.read_sql_query(sql_query_velocities, connection)

    # Merging the velocities and positions
    df = pd.DataFrame()

    df = df_positions.copy()
    df['v_x'] = df_velocities['x_coordinate']
    df['v_y'] = df_velocities['y_coordinate']
    df['type'] = 0
    df.columns = ['object', 'frame', 'x', 'y', 'v_x', 'v_y', 'type']
    
    # group by averaging    
    df = df.groupby(['object','frame']).mean().reset_index()
    
    for i in range(max_objects):
        
        x_diff = df.loc[df['object']==i]['x'].iloc[-1] - df.loc[df['object']==i]['x'].iloc[0]
        y_diff = df.loc[df['object']==i]['y'].iloc[-1] - df.loc[df['object']==i]['y'].iloc[0]
        
        # check if is vehicle
        if abs(x_diff) > 10:            
            df.loc[df['object'] == i,'type'] = 1
            
        # check if is pedestrian
        elif abs(y_diff) > 2:
            df.loc[df['object'] == i,'type'] = 2    
                       
                      
        
        X0 = pd.Series.tolist(df.loc[df['object'] == i]['x'])
        Y0 = pd.Series.tolist(df.loc[df['object'] == i]['y'])
        plt.scatter(X0,Y0)
        plt.xlabel("X positions")
        plt.ylabel("Y positions")
        plt.title("Trajectories of moving objects of " + filename)
        
    plt.show()

print(df[df['type']==1].object.unique())
print(df[df['type']==2].object.unique())

# extract the ped and veh
vehicle = df.loc[df['type'] == 1]
pedestrian = df.loc[df['type'] == 2]

## Left as the how because we take the pedestrian as a base
interaction = pd.merge(vehicle, pedestrian, how ='inner', on = ['frame'])
interaction.columns = ['object1', 'frame', 'x1', 'y1', 'v_x1', 'v_y1', 'type1', 'object2', 'x2', 'y2', 'v_x2', 'v_y2', 'type2']
#print interaction

# plot the interaction 
X = pd.Series.tolist((interaction)['x1'])
Y = pd.Series.tolist((interaction)['y1'])
X2 = pd.Series.tolist((interaction)['x2'])
Y2 = pd.Series.tolist((interaction)['y2'])
plt.scatter(X,Y)
plt.scatter(X2,Y2)
plt.show()


# calculate relative 
MLInput = pd.DataFrame()
MLInput['frame'] = interaction['frame']
MLInput['relative_X'] = interaction['x1'] - interaction['x2']
MLInput['relative_Y'] = interaction['y1'] - interaction['y2']
MLInput['relative_Vx'] = interaction['v_x1'] - interaction['v_x2']
MLInput['relative_Vy'] = interaction['v_y1'] - interaction['v_y2']
MLInput['brake'] = 0
#MLInput.columns = ['frame','relative_X','relative_Y','relative_Vx','relative_Vy','brake']


for frame in interaction['frame']:
    if len(interaction.loc[interaction['frame']==frame-1]['v_x1'].values)>0: 
        v_prev = interaction.loc[interaction['frame']==frame-1]['v_x1'].values[0]
        v = interaction.loc[interaction['frame']==frame]['v_x1'].values[0]
        if v - v_prev < 0:
            MLInput.loc[MLInput['frame']==frame,'brake'] = 1
        

MLInput.to_csv("Input1")