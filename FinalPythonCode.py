#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 14:39:03 2018

@author: maxg
"""

#Import database modules
import pandas as pd
import os
import glob
import numpy as np
from scipy.signal import savgol_filter
#Import linear algebra functions
from numpy.linalg import eig, inv
#Import module for graphing
import pylab
#Import smooth data algorithm 
from statsmodels.nonparametric.smoothers_lowess import lowess

#Starting text file that outputs statstical data
myfile = open('FilesStatistics.txt', 'w')


def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])


def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))


def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])


def ellipse_angle_of_rotation2( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else: 
        if a > c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2
        
print(__file__)

#Identifies Directory of Files
dir_path = os.path.dirname(os.path.realpath(__file__))
#Calls all files in the directory with .csv
filenames = sorted(glob.glob('*.csv')) 



alldata = []
allfit = []
speeds = []
perimeters = []
closures = []

AvgSpeed = np.zeros(400, dtype=float)
AvgClosure = np.zeros(400, dtype=float)
AvgPerimeter = np.zeros(400, dtype=float)

speed = np.zeros(400, dtype=float)
cclose = np.zeros(400, dtype=float)
perimeter = np.zeros(400, dtype=float)

counter = 0

for f in filenames:
    counter += 1
    #Pandas reads through each csv file in directory
    df = pd.read_csv(f, header=None)
    #Initiating columns X,Y
    resultsX = df[0]
    resultsY = df[1]
    #Find length of X/Y column
    totalX = len(resultsX)-1
    totalY = len(resultsY)-1
    #Initiating dictionary
    mydataX = {}
    #Initiating index to start at for each data frame
    spacer = 0
    #Counting each data frame
    m = 0
    #For loop for conditional if cell is NaN for column X
    for i in range(0,totalX):
        if np.isnan(resultsX[i]):
            mydataX[m] = resultsX[spacer:i]
            m = m + 1
            spacer = i + 1
    #Initiate new dictionary for Y values
    mydataY = {}
    #Recall m and spacer values
    m = 0
    spacer = 0
    for i in range(0,totalY):
        if np.isnan(resultsY[i]):
            mydataY[m] = resultsY[spacer:i]
            m = m + 1
            spacer = i + 1
    #Combining the X and Y dictionaries
    mydata = [mydataX, mydataY]
    #New array of length 3 to hold xc, yc, and R_1 for each dictionary
    #Fit0[0]=xm, Fit0[1]=ym, Fit0[2]=R_1
    fit0 = np.zeros((m,3), dtype=float)
    
    
    #Append each file circle fit data in directory path
    allfit.append(fit0)
    #Append all XY scatter points from directory path into an array 
    alldata.append(mydata)
    
    #Start EllipseFit-------------------------------------------------------------
    #Arbitrary values for arc/R
    arc = 2.0
    R = np.arange(0,arc*np.pi, 0.01)
    
    #Finding Major axis a/b
    Perimeter = np.zeros(len(mydataX)-1, dtype=float)
    Radius = np.zeros(len(mydataX)-1, dtype=float)
    Closure = np.zeros(len(mydataX)-1, dtype=float)
    RawSpeed = np.zeros(len(mydataX)-1, dtype=float)
    Time = np.arange(len(mydataX)-1)
    C = np.zeros(len(mydataX)-1, dtype=float)


    #Calculates Ellipse fit
    counter = 0  
    for i in range(len(mydataX)-1):
        a = fitEllipse(mydata[0][i],mydata[1][i])
        center = ellipse_center(a)
        #phi = ellipse_angle_of_rotation(a)
        phi = ellipse_angle_of_rotation2(a)
        axes = ellipse_axis_length(a)
        a, b = axes
        xx = center[0] + a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
        yy = center[1] + a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)
        h = (a - b)**2 / (a + b)**2
        #Elimin rest of perimeter
        #C is minimum between a and b
        C[i] = np.mean([a,b])
        
        Perimeter[i] = np.pi * (a + b) * (1 + ((3 * h) / (10 + np.sqrt(4 - 3*h))))

        
    Perimeter[0] = Perimeter[2]
    Perimeter[1] = Perimeter[2]
        
    for i in range(len(mydataX)-1):
                
        if counter < 1 and C[i] > .01:
            counter = counter
        
        else:
            counter += 1
            Perimeter[i] = np.nan
        

    #Plotting smoothed data for Perimeter
    #Perimeter Lowess of .25 tested to be best fit
    NewPerimeter = lowess(Perimeter, Time, frac=.25)
    perimeter = NewPerimeter[:,1].copy()
    myperimeter = np.empty(400)
    myperimeter[:] = np.nan
    counter = 0 
    for i in range(len(mydataX)-1):
        if counter < 1 and C[i] > .01:
            Radius[i] = perimeter[i] / (2 * np.pi)
            Closure[i] = 100. - (100. * (perimeter[i] / perimeter[0]))
        
        else:
            counter += 1
            Closure[i] = np.nan
            Radius[i] = np.nan
            
    for i in range(len(perimeter)):
        myperimeter[i] = perimeter[i]    
    cPerimeter = []
    for n in range(len(myperimeter)):
        cPerimeter.append(myperimeter[n])
    perimeters.append(cPerimeter)
    
    for i in range(len(mydataX)-2):
        RawSpeed[i] = np.absolute((Radius[i+1] - Radius[i]) * 1000)
    
    for i in range(len(mydataX)-2):
        if RawSpeed[i] < np.float(0) and Closure[i] > np.float(95):
            Closure[i+1] = 100
   
    #Figure0
    figure0 = pylab.plot(NewPerimeter[:,0], NewPerimeter[:,1], 'c')
    pylab.xlabel('Time vs. Perimeter w/ frac=.25')
    pylab.plot(Time, Perimeter)
    pylab.savefig("Perimeter" + str(f) + ".png")
    pylab.show()
    
    #Plotting smoothed data of Closure to RawSpeed
    NewSpeed = lowess(RawSpeed, Closure, frac=.10)
    speed = NewSpeed[:,1].copy()
    myspeed = np.empty(400)
    myspeed[:] = np.nan
    for i in range(len(speed)):
        myspeed[i] = speed[i]
        
    cSpeed = []
    for n in range(len(myspeed)):
        cSpeed.append(myspeed[n])
    speeds.append(cSpeed)
    #Figure1
    figure1 = pylab.plot(NewSpeed[:,0], NewSpeed[:,1], 'r')
    pylab.xlabel('Closure vs. Raw Speed w/ frac=.10')
    pylab.plot(Closure, RawSpeed)
    pylab.savefig("Closure" + str(f) + ".png")
    pylab.show()
    
    MaxClosure = max(Closure)
    #Statistical max value for speed of percent closure
    cclose = Closure.copy()
    myclose = np.empty(400)
    myclose[:] = np.nan
    for i in range(len(cclose)):
        myclose[i] = cclose[i]
        
    cClosure = []
    for n in range(len(myclose)):
        cClosure.append(myclose[n])
    closures.append(cClosure)
    
    FixedClosure, FixedRawSpeed = np.hsplit(NewSpeed, [1])
    
    MaxValue = max(FixedRawSpeed)
    
    #Saving textfile for statistics
    myfile.write("" + str(f) + ":," + str(MaxValue) + ",")
    for i in range(len(FixedRawSpeed)):
        if FixedRawSpeed[i] == MaxValue:
            Location = Closure[i]
            myfile.write("" + str(Location) + ",")
    
    myfile.write("" + str(MaxClosure) + "\n")

    
#End EllipseFit---------------------------------------------------------------

#Calculate statistical measurements of speed, perimeter, closure percentage
speeds = np.array(speeds)
perimeters = np.array(perimeters)
closures = np.array(closures)
speedStdev= np.zeros(400, dtype=float)
closeStdev= np.zeros(400, dtype=float)
perimeterStdev= np.zeros(400, dtype=float)
for i in range(len(speedStdev)):
    speedStdev[i] = np.nanstd(speeds[:,i])
    closeStdev[i] = np.nanstd(closures[:,i])
    perimeterStdev[i] = np.nanstd(perimeters[:,i])
    AvgSpeed[i] = np.nanmean(speeds[:,i])
    AvgClosure[i] = np.nanmean(closures[:,i])
    AvgPerimeter[i] = np.nanmean(perimeters[:,i])
    
    
    
AvgPerimeterNaN = AvgPerimeter[np.logical_not(np.isnan(AvgPerimeter))]

GolayAvgPerimeter = savgol_filter(AvgPerimeterNaN,7,4)


#Closing text file
myfile.close()
    
    
    
    