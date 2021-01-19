#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 18:12:17 2021

@author: chashi
"""

if __name__ == "__main__":
    start = int(input('Start:'))
    end = int(input('End:'))
    
    for i in range(start, end+1):
        sum = 0
        for j in str(i):
            # print(j)
            sum += int(j)**3;
        if sum == int(i):
            print(i)
        #print(sum)
            