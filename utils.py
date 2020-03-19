# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 20:13:14 2020

@author: dougl
"""

import string


def col_num_to_alpha(n, zero_indexed=True):
    ''' col_num_to_alpha(28) = 'AC' '''
    if zero_indexed:
        n = n + 1
    string = ''
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        string = chr(65 + remainder) + string
    return string


def col_alpha_to_num(col, zero_indexed=True):
    ''' col_alpha_to_num('AC') = 28 '''
    num = 0
    for c in col:
        if c in string.ascii_letters:
            num = num * 26 + (ord(c.upper()) - ord('A')) + 1
    return num - 1 if zero_indexed else num