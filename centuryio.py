import numpy as np
import pandas as pd
import datetime as dt
import pdb

def load_centlis( fpathname ) :
    """
    Load a specified century list100 output file (*.lis) and return a pandas
    DataFrame object.

    Args:
        fpathname (str) : path and filename of desired century (.lis) file
    Return:
        df   : pandas DataFrame    
    """

    print('Parsing ' + fpathname)

    # Parse fixed width file
    df = pd.read_fwf(fpathname , skiprows=( 1, ), index_col='time',
            header=0, na_values=['NaN', 'NAN', 'INF', '-INF'])

    df['year'] = np.floor(df.index)
    df['month'] = np.round((df.index - np.floor(df.index))*12)
    df.month.loc[df.month==0] = 12
    df.year.loc[df.month==12] = df.year.loc[df.month==12] - 1
       
    return df

def load_dcout( fpathname, skipr=0, noheader=False ) :
    """
    Load a specified daycent output file (*.out) and return a pandas
    DataFrame object.

    Args:
        fpathname (str) : path and filename of desired daycent (.out) file
        skipr : number of lines to skip at head of file
    Return:
        df   : pandas DataFrame    
    """

    print('Parsing ' + fpathname)

    if noheader:
        with open(fpathname, 'r') as f:
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

    # Parse fixed width file
    df = pd.read_fwf(fpathname , skiprows=skipr + 1, header=None, names=cols,
            na_values=['NaN', 'NAN', 'NA', 'INF', '-INF'])
       
    return df


def dcindex_decyr( idx, startyr=None ) :
    """
    Convert a daycent index in decimal year format to datetime index (if
    given a startyear in datetime range), or period index (default, works with
    far future or past values).

    Args:
        idx: decimal year index from a Century/DayCent .lis or .out file
        startyr: None, or integer year if placing new index in datetime range. 
                 Note that datetime range is from 1677 to 2262.
    Returns:
        newidx: Datetime or period index
    """
    df = pd.DataFrame({'year':np.floor(idx),
        'month': np.round((idx - np.floor(idx)) * 12)})
    df['day'] = 1
    df.loc[df.month==0, 'month'] = 12
    df.loc[df.month==12, 'year'] = df.loc[df.month==12, 'year'] - 1
    if startyr is not None:
        offset = df.year.iloc[0] - startyr
        df.year = df.year - offset
        newidx = pd.to_datetime(df) + pd.offsets.MonthEnd(0)
    else:
        ymd = df.iloc[0].loc[['year', 'month', 'day']]
        start = [str(s) for s in ymd.astype(int)]
        start = '-'.join(start)
        ymd2 = df.iloc[df.shape[0]-1].loc[['year', 'month', 'day']]
        end = [str(s) for s in ymd2.astype(int)]
        end = '-'.join(end)
        newidx = pd.period_range(start, end, freq='M')
    return newidx


def dcindex_ydoy( df, startyr=None ) :
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
    if startyr is not None:
        offset = df_c.time.iloc[0] - startyr
        df_c.time = df_c.time - offset
        df_c.year = df_c.time.astype(int).astype(str)
        df_c['doy'] = df_c.dayofyr.astype(str)
        df_c['ts'] = df_c.year + df_c.doy
        newidx = pd.to_datetime(df_c.ts, format='%Y%j')
    else:
        styear = str(int(df_c.time.iloc[0]))
        endyear = str(int(df_c.time.iloc[df_c.shape[0]-1]))
        df_c['doy'] = df_c.dayofyr.astype(str)
        start = pd.to_datetime('1900' + df_c.doy[0], format='%Y%j')
        end = pd.to_datetime('1901' + df_c.doy[df_c.shape[0]-1], format='%Y%j')
        start = dt.date.strftime(start, '%Y-%m-%d').replace('1900', styear)
        end = dt.date.strftime(end, '%Y-%m-%d').replace('1901', endyear)
        newidx = pd.period_range(start, end, freq='D')
    return newidx

