#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:21:08 2019

@author: tatumhennig
"""
import numpy as np
import pandas as pd

## ROT rotates and flips a quadrant appropriately.
#  Parameters:
#    Input, integer N, the length of a side of the square.  
#    N must be a power of 2.
#    Input/output, integer X, Y, the coordinates of a point.
#    Input, integer RX, RY, ???
def rot( n, x, y, rx, ry ):
  if ( ry == 0 ):
#  Reflect.
    if ( rx == 1 ):
      x = n - 1 - x
      y = n - 1 - y
#  Flip.
    t = x
    x = y
    y = t

  return x, y


## XY2D converts a 2D Cartesian coordinate to a 1D Hilbert coordinate.
#  Discussion:
#    It is assumed that a square has been divided into an NxN array of cells,
#    where N is a power of 2.
#    Cell (0,0) is in the lower left corner, and (N-1,N-1) in the upper 
#    right corner.
#  Parameters:
#    integer M, the index of the Hilbert curve.
#    The number of cells is N=2^M.
#    0 < M.
#    Input, integer X, Y, the Cartesian coordinates of a cell.
#    0 <= X, Y < N.
#    Output, integer D, the Hilbert coordinate of the cell.
#    0 <= D < N * N.
def xy2d(x,y):
    m = 10    # index of hilbert curve
    n = 1024    # number of boxes (2^m)
    
    xcopy = x
    ycopy = y
    
    d = 0
    n = 2 ** m
    
    s = ( n // 2 )
    
    while ( 0 < s ):
        if ( 0 <  ( abs ( xcopy ) & s ) ):
          rx = 1
        else:
          rx = 0
        if ( 0 < ( abs ( ycopy ) & s ) ):
          ry = 1
        else:
          ry = 0
        d = d + s * s * ( ( 3 * rx ) ^ ry )
        xcopy, ycopy = rot(s, xcopy, ycopy, rx, ry )
        s = ( s // 2 )
    
    return d

#*****************************************************************************#

# load in phi psi csv
name = 'wt_pH7_300K_water'
data = pd.read_csv(name + '_phipsi.csv')
data.set_index('Unnamed: 0',inplace=True) # index is the time frames!

# transform and round data to integer values into pixel space
#   - adding 180 because our lowest phi/psi value possible is -180 and we
#       want lowest value to be zero.
#   - dividing by 1023 because we are using order 10 (0-1023 is 1024)
transformed_data=data.apply(lambda x: np.round((x+180)/(360/1023),decimals=0))
rounded_data = transformed_data.apply(np.int64)

# combine phi psi values into one column
combined_data = pd.DataFrame(index=rounded_data.index)
for i in [0,2,4,6,8,10,12,14,16,18]:
    combined_data['AA'+str(i)]=rounded_data.iloc[:,i:i+2].values.tolist()

# convert 2d into 1d
hilbert_data = np.zeros((19986, 10))
for i in range(19986):
    for j in range(10):
        hilbert_data[i, j] = xy2d(combined_data.iloc[i,j][0],combined_data.iloc[i,j][1])

# add index and column titles to hilbert data
hilbert_data=pd.DataFrame(hilbert_data,index=combined_data.index,columns=combined_data.columns)

# save
hilbert_data.to_csv('hd_' + name + '.csv')


