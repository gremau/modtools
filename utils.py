import numpy as np

def trimdates_dict(d, stdt, enddt):
    d_trim = {}
    for k in d.keys():
        d_trim[k] = d[k].loc[np.logical_and(
            d[k].index >= stdt, d[k].index < enddt)].copy()
    return d_trim
