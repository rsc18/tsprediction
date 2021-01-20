#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:35:46 2021

@author: chashi
"""
def rem_white_space(s):
    l = s.split(" ")
    return ''.join(l)
if __name__ == "__main__":
    s = input()
    print(rem_white_space(s))
    