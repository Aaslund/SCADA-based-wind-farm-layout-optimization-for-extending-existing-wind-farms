# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:07:42 2025

@author: erica
"""

# Python 3 program for the 
# haversine formula
import math


def haversine(lat1, lon1, lat2, lon2):
    # distance between latitudes and longitudes
    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0

    # convert to radians
    lat1 = lat1 * math.pi / 180.0
    lat2 = lat2 * math.pi / 180.0

    # apply formulae
    a = (pow(math.sin(dLat / 2), 2) +
         pow(math.sin(dLon / 2), 2) *
         math.cos(lat1) * math.cos(lat2))
    rad = 6371 * 1e3  # Earth radius in meters
    c = 2 * math.asin(math.sqrt(a))
    
    # Calculate the distance
    distance = rad * c
    
    # Determine the sign based on the longitude difference
    if dLon < 0:
        distance = -distance
    elif dLat < 0:
        distance = -distance   
    return distance
# Driver code
# if __name__ == "__main__":
# 	lat1 = 38.391061
# 	lon1 = 21.692759
# 	lat2 = 38.395245
# 	lon2 = 21.692759
# 	
# 	print(haversine(lat1, lon1,lat2, lon2), "K.M.")

# This code is contributed 
# by ChitraNayal
