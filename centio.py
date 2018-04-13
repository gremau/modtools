"""
Functions for file loading, transforming, and output of Century/DayCent files
"""

import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path
import pdb

def load_binlist( fpathname, previous_bin_range=None ) :
    """
    Load a specified list100 output file (*.lis) which is derived from .bin
    DayCent or Century output. Return a pandas DataFrame object.

    Args:
        fpathname (str) : path and filename of desired century (.lis) file
        previous_bin_range: range of index values to remove from previous
                            binary file (DayCent -e extends previous run)
    Return:
        df   : pandas DataFrame    
    """

    print('Parsing ' + fpathname)

    # Parse fixed width file
    df = pd.read_fwf(fpathname , skiprows=( 1, ), index_col='time',
            header=0, na_values=['NaN', 'NAN', 'INF', '-INF'])
    # Remove previous bin range based on index, and index==0.0 (introduced
    # by reading files with indexes beyond 9999).
    if previous_bin_range is not None:
        test1 = np.logical_and(df.index >= previous_bin_range[0],
                df.index <= previous_bin_range[1])
        # Values over 10000 become zero - remove
        test2 = df.index==0.0
        df = df.loc[~np.logical_or(test1, test2), :]

    df['year'] = np.floor(df.index)
    df['month'] = np.round((df.index - np.floor(df.index))*12)
    df.month.loc[df.month==0] = 12
    df.year.loc[df.month==12] = df.year.loc[df.month==12] - 1
       
    return df

def load_dcout( fpathname, skipr=0, noheader=False, tgmonth=False ) :
    """
    Load a specified daycent output file (*.out) and return a pandas
    DataFrame object.

    Args:
        fpathname (str) : path and filename of desired daycent (.out) file
        skipr : number of lines to skip at head of file
        noheader : Create a new header to replace abnormal or missing one
        tgmonth : Add a month column if needed
    Return:
        df   : pandas DataFrame    
    """
    fpath = Path(fpathname)
    if fpath.is_file():
        print('Parsing ' + fpathname)

        if noheader:
            # Open file, skip lines if asked and get first line of data
            # Then create a column name array
            with open(fpathname, 'r') as f:
                for _ in range(skipr):
                    next(f)
                cols = f.readline().rstrip('\n')
            cols = [x for x in cols.split(' ') if x is not '']
            cols = ['time', 'dayofyr'] + ["c{:02d}".format(x)
                    for x in range(len(cols) - 2)]
            skipr = skipr - 1

        else:
            with open(fpathname, 'r') as f:
                for i in range(skipr+1):
                    cols = f.readline().rstrip('\n')
                cols = [x for x in cols.split(' ') if x is not '']
    
        if tgmonth:
            cols.insert(1, 'month')

        # Parse fixed width file
        df = pd.read_fwf(fpathname , skiprows=skipr + 1, header=None,
                names=cols, na_values=['NaN', 'NAN', 'NA', 'INF', '-INF'])
       
        return df
    else:
        print('File' + fpathname + ' does not exist')
        return pd.DataFrame()


def lisindex_dt( idx, startyr=None ) :
    """
    Convert a list100 index in decimal year format to datetime index (if
    given a startyear in datetime range), or period index (default, works with
    far future or past values). Note that the convention for list100 output
    is that year.00 is actually december the prior year, so this is adjusted
    to make dates compatible with DayCent .out dates (when converted)

    Args:
        idx: decimal year index from a Century/DayCent .lis or .out file
        startyr: None, or integer year if placing new index in datetime range. 
                 Note that datetime range is from 1677 to 2262.
    Returns:
        newidx: Datetime or period index
    """
    df = pd.DataFrame({'year':np.floor(idx).astype(int),
        'month':np.round((idx - np.floor(idx))*12).astype(int)})
    df['day'] = 1
    df.loc[df.month==0, 'month'] = 12
    df.loc[df.month==12, 'year'] = df.loc[df.month==12, 'year'] - 1
    # Make datetime index if idx is in proper range or startyear given
    if min(idx)>1677 and max(idx)<2262 :
        newidx = pd.to_datetime(df) + pd.offsets.MonthEnd(0)
    elif startyr is not None :
        offset = df.year.iloc[0] - startyr
        df.year = df.year - offset
        newidx = pd.to_datetime(df) + pd.offsets.MonthEnd(0)
    # Otherwise make a period index
    else:
        ymd = df.iloc[0].loc[['year', 'month', 'day']]
        start = [str(s) for s in ymd.astype(int)]
        start = '-'.join(start)
        ymd2 = df.iloc[-1].loc[['year', 'month', 'day']]
        end = [str(s) for s in ymd2.astype(int)]
        end = '-'.join(end)
        newidx = pd.period_range(start, end, freq='M')
    return newidx


def dcindex_ydoy_dt( df, startyr=None ) :
    """
    Convert a daycent index in year + dayofyear format to datetime index (if
    given a startyear in datetime range), or period index (default, works with
    far future or past values).

    Args:
        df: dataframe from Day/Century with a 'time' and 'dayofyr' column
        startyr: None, or integer year if placing new index in datetime range. 
                 Note that datetime range is from 1677 to 2262.
    Returns:
        newidx: Datetime or period index
    """
    
    df_c = df.copy()
    # Make datetime index if idx is in proper range or startyear given
    if min(df.time)>1677 and max(df.time)<2262 :
        df_c['year'] = df_c.time.astype(int).astype(str)
        df_c['doy'] = df_c.dayofyr.astype(str)
        df_c['ts'] = df_c.year + df_c.doy
        newidx = pd.to_datetime(df_c.ts.values, format='%Y%j')
    elif startyr is not None:
        offset = df_c.time.iloc[0] - startyr
        df_c.time = df_c.time - offset
        df_c['year'] = df_c.time.astype(int).astype(str)
        df_c['doy'] = df_c.dayofyr.astype(str)
        df_c['ts'] = df_c.year + df_c.doy
        newidx = pd.to_datetime(df_c.ts.values, format='%Y%j')
    else:
        styear = str(int(df_c.time.iloc[0]))
        endyear = str(int(df_c.time.iloc[-1]))
        df_c['doy'] = df_c.dayofyr.astype(str)
        start = pd.to_datetime('1900' + df_c.doy[0], format='%Y%j')
        end = pd.to_datetime('1901' + df_c.doy.iloc[-1], format='%Y%j')
        start = dt.date.strftime(start, '%Y-%m-%d').replace('1900', styear)
        end = dt.date.strftime(end, '%Y-%m-%d').replace('1901', endyear)
        newidx = pd.period_range(start, end, freq='D')
    # Note: Daycent dates are off by about 1 day a century due to some
    # error in leapyear calculations. To deal with this we find the leapdays
    # in the new index and remove the necessary number (randomly)
    discrepancy = len(newidx) - df.shape[0]
    if discrepancy > 0:
        is_leap_day = np.logical_and(newidx.month == 2, newidx.day == 29)
        leapdays = np.where(is_leap_day)[0]
        rmleaps = np.random.choice(leapdays, discrepancy, replace=False)
        rmleaplabels = newidx[rmleaps]
        newidx = newidx.drop(rmleaplabels)
    #day366 = np.add(leapdays, 366 - (31+28))
    return newidx

def dcindex_ymo_dt( df, startyr=None ) :
    """
    Convert a daycent index in year + month format to datetime index (if
    given a startyear in datetime range), or period index (default, works with
    far future or past values).

    Args:
        df: dataframe from Day/Century with a 'time' and 'dayofyr' column
        startyr: None, or integer year if placing new index in datetime range. 
                 Note that datetime range is from 1677 to 2262.
    Returns:
        newidx: Datetime or period index
    """
    
    df_c = df.copy()
    # Make datetime index if idx is in proper range or startyear given
    if min(df.time)>1677 and max(df.time)<2262 :
        df_c['year'] = df_c.time.astype(int).astype(str)
        df_c['month'] = df_c.month.astype(str)
        df_c['ts'] = df_c.year + df_c.month
        newidx = pd.to_datetime(df_c.ts, format='%Y%m') + pd.offsets.MonthEnd(0)
    elif startyr is not None:
        offset = df_c.time.iloc[0] - startyr
        df_c.time = df_c.time - offset
        df_c['year'] = df_c.time.astype(int).astype(str)
        df_c['month'] = df_c.month.astype(str)
        df_c['ts'] = df_c.year + df_c.month
        newidx = pd.to_datetime(df_c.ts, format='%Y%m') + pd.offsets.MonthEnd(0)
    else:
        styear = str(int(df_c.time.iloc[0]))
        endyear = str(int(df_c.time.iloc[-1]))
        df_c['month'] = df_c.month.astype(str)
        start = pd.to_datetime('1900' + df_c.month[0],
                format='%Y%m') + pd.offsets.MonthEnd(0)
        end = pd.to_datetime('1901' + df_c.month.iloc[-1],
                format='%Y%m') + pd.offsets.MonthEnd(0)
        start = dt.date.strftime(start, '%Y-%m-%d').replace('1900', styear)
        end = dt.date.strftime(end, '%Y-%m-%d').replace('1901', endyear)
        newidx = pd.period_range(start, end, freq='M')
    return newidx


def dcindex_y_dt( df, startyr=None ) :
    """
    Convert a daycent index in year format to datetime index (if
    given a startyear in datetime range), or period index (default, works with
    far future or past values).

    Args:
        df: dataframe from Day/Century with a 'time' and 'dayofyr' column
        startyr: None, or integer year if placing new index in datetime range. 
                 Note that datetime range is from 1677 to 2262.
    Returns:
        newidx: Datetime or period index
    """
    
    df_c = df.copy()
    # Make datetime index if idx is in proper range or startyear given
    if min(df.time)>1677 and max(df.time)<2262 :
        df_c['year'] = df_c.time.astype(int).astype(str)
        newidx = pd.to_datetime(df_c.year, format='%Y') + pd.offsets.YearEnd(0)
    elif startyr is not None:
        offset = df_c.time.iloc[0] - startyr
        df_c.time = df_c.time - offset
        df_c['year'] = df_c.time.astype(int).astype(str)
        #df_c['month'] = df_c.month.astype(str)
        #df_c['ts'] = df_c.year + df_c.month
        newidx = pd.to_datetime(df_c.year, format='%Y') + pd.offsets.YearEnd(0)
    else:
        styear = str(int(df_c.time.iloc[0]))
        endyear = str(int(df_c.time.iloc[-1]))
        start = pd.to_datetime('1900', format='%Y') + pd.offsets.YearEnd(0)
        end = pd.to_datetime('1901', format='%Y') + pd.offsets.YearEnd(0)
        start = dt.date.strftime(start, '%Y-%m-%d').replace('1900', styear)
        end = dt.date.strftime(end, '%Y-%m-%d').replace('1901', endyear)
        newidx = pd.period_range(start, end, freq='A')
    return newidx

def get_daycent_sim(path, siten, simn, branchn, startyear=None,
        previous_bin_range=None):
    d = {'bin':load_binlist(path + '{0}.out/{1}/{0}_{1}_{2}.lis'.format(
            siten, simn, branchn), previous_bin_range=previous_bin_range),
        'summ':load_dcout(path + '{0}.out/{1}/summary_{1}_{2}.out'.format(
            siten, simn, branchn)),
        'bio':load_dcout(path + '{0}.out/{1}/bio_{1}_{2}.out'.format(
            siten, simn, branchn)),
        'resp':load_dcout(path + '{0}.out/{1}/resp_{1}_{2}.out'.format(
            siten, simn, branchn)),
        'nflux':load_dcout(path + '{0}.out/{1}/nflux_{1}_{2}.out'.format(
            siten, simn, branchn)),
        'soilc':load_dcout(path + '{0}.out/{1}/soilc_{1}_{2}.out'.format(
            siten, simn, branchn)),
        'sysc':load_dcout(path + '{0}.out/{1}/sysc_{1}_{2}.out'.format(
            siten, simn, branchn)),
        'swc':load_dcout(path + '{0}.out/{1}/vswc_{1}_{2}.out'.format(
            siten, simn, branchn), noheader=True),
        'soiltavg':load_dcout(path + '{0}.out/{1}/soiltavg_{1}_{2}.out'.format(
            siten, simn, branchn), noheader=True),
        'soiltmax':load_dcout(path + '{0}.out/{1}/soiltmax_{1}_{2}.out'.format(
            siten, simn, branchn), noheader=True),
        'soiltmin':load_dcout(path + '{0}.out/{1}/soiltmin_{1}_{2}.out'.format(
            siten, simn, branchn), noheader=True),
        'soilco2':load_dcout(path + '{0}.out/{1}/co2_{1}_{2}.out'.format(
            siten, simn, branchn), noheader=True, skipr=1), #replace odd header
        'tgmonth':load_dcout(path + '{0}.out/{1}/tgmonth_{1}_{2}.out'.format(
            siten, simn, branchn), tgmonth=True),
        'ysumm':load_dcout(path + 
            '{0}.out/{1}/year_summary_{1}_{2}.out'.format(siten,simn, branchn)),
        'sip':pd.read_csv(path + '{0}.out/{1}/dc_sip_{1}_{2}.csv'.format(
            siten, simn, branchn))}
    d['bin'].index = lisindex_dt(d['bin'].index, startyr=startyear)
    d['tgmonth'].index = dcindex_ymo_dt(d['tgmonth'], startyr=startyear)
    d['ysumm'].index = dcindex_y_dt(d['ysumm'], startyr=startyear)
        
    dayidx = dcindex_ydoy_dt(d['summ'], startyr=startyear)
    dailytables = ['summ', 'bio', 'resp', 'nflux', 'soilc', 'sysc',
            'swc','soiltavg', 'soiltmax', 'soiltmin', 'soilco2', 'sip']
        
    for t in dailytables:
        d[t].index = dayidx

    return d
