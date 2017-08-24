import numpy as np
import pandas as pd
import os
import datetime as dt

def load_prism_daily(fname):
    df = pd.read_csv(fname, skiprows=10, index_col='Date', parse_dates=True)
    cols = df.columns
    cols = [c.split(' ')[0] for c in cols]
    df.columns = cols
    df.ppt = df.ppt/10 # convert mm to cm
    return df

def make_wth_prism(sitelist, prism_path, wth_path):
    """
    Make sure incoming precip data are in cm
    """
    prismfiles = [f for f in os.listdir(prism_path) if '.csv' in f]

    for sitename in sitelist:
        matchfile = [f for f in prismfiles if sitename in f]
        if len(matchfile) > 1:
            raise ValueError('Multiple {0} files in {1}'.format(sitename,
                prism_path))
        else:
            df = load_prism_daily(os.path.join(prism_path, matchfile[0]))
        # build_wth will put together the file in a correct format
        wth = build_wth(df)
        print(wth.head())
        # Use wth file to get monthly site.100 parameters
        site100 = build_site100(wth)
        # Change output format of ppt column and write to file
        wth['ppt'] = wth['ppt'].map(lambda x:'{0:.3}'.format(x))
        wth.to_csv(os.path.join(wth_path, sitename + '.wth'), sep='\t',
                index=False, header=False)
        site100['var'] = site100['var'].map(lambda x:'{0:.3}'.format(x))
        site100.to_csv(os.path.join(wth_path, sitename + '.100clim'), sep='\t',
                index=False, header=False)


def build_wth(df):
    wth = pd.DataFrame()
    wth['day'] = df.index.day
    wth['month'] = df.index.month
    wth['year'] = df.index.year
    wth['doy'] = df.index.dayofyear
    wth['tmax'] = df.tmax.values
    wth['tmin'] = df.tmin.values
    wth['ppt'] = df.ppt.values
    wth['std1'] = -999
    wth['std2'] = -999
    return wth

def build_site100(wth):
    gby = wth.groupby('month')
    # Mean monthly precip and std
    pptsum = pd.DataFrame(gby.sum().ppt/30)
    pptsum.columns = ['var']
    pptsum['param'] = pptsum.index.map(lambda x:"'PRECIP({})'".format(x))
    pptstd = pd.DataFrame(gby.std().ppt)
    pptstd.columns = ['var']
    pptstd['param'] = pptstd.index.map(lambda x:"'PRCSTD({})'".format(x))
    pptskw = pd.DataFrame(gby.skew().ppt)
    pptskw.columns = ['var']
    pptskw['param'] = pptskw.index.map(lambda x:"'PRCSKW({})'".format(x))
    # Mean tmax tmin
    meantmax = pd.DataFrame(gby.mean().tmax)
    meantmax.columns = ['var']
    meantmax['param'] = meantmax.index.map(lambda x:"'TMX2M({})'".format(x))
    meantmin = pd.DataFrame(gby.mean().tmin)
    meantmin.columns = ['var']
    meantmin['param'] = meantmin.index.map(lambda x:"'TMN2M({})'".format(x))
    site100 = pd.concat([pptsum, pptstd, pptskw,meantmin, meantmax])
    return(site100)
