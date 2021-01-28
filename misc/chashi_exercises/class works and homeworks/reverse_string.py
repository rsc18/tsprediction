#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:34:03 2021

@author: chashi
"""

def reverse_string(string):
    return string[::-1]

if __name__ == "__main__":
    s = input()
    print(reverse_string(s))
    