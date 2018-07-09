# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import random
import numpy as np
import math

def mz_map(y_true, y_pred, rel_threshold=0):

    def _to_list(x):
        if isinstance(x, list):
            return x
        return [x]

    print("((((((((((((((((((((((((((((((((((")
    print("y_true", y_true, type(y_true))
    print("((((((((((((((((((((((((((((((((((")
    print("y_pred", y_pred, type(y_pred))
    print("((((((((((((((((((((((((((((((((((")
    print("rel_threshold", rel_threshold)

    s = 0.
    y_true = _to_list(np.squeeze(y_true).tolist())
    y_pred = _to_list(np.squeeze(y_pred).tolist())
    c = list(zip(y_true, y_pred))
    random.shuffle(c)
    c = sorted(c, key=lambda x:x[1], reverse=True)
    ipos = 0
    for j, (g, p) in enumerate(c):
        if g > rel_threshold:
            ipos += 1.
            s += ipos / ( j + 1.)
    if ipos == 0:
        s = 0.
    else:
        s /= ipos
    return s
