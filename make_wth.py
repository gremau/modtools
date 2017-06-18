import numpy as np
import pandas as pd
import os
import datetime as dt

def load_prism_daily(fname):
    df = pd.read_csv(fname, skiprows=10, index_col='Date', parse_dates=True)
    return df

def make_wth_prism(sitelist, prism_path, wth_path):
    pvars = ['ppt', 'tmax', 'tmin']
    for sitename in sitelist:
        d = {v:load_prism_daily(os.path.join(prism_path, 'PRISM_' + v +
            '_stable_4km_19810101_20101231_' + sitename + '.csv')) for
            v in pvars}
        d['ppt'] = d['ppt']/10 # convert mm to cm
        # build_wth will put together the file in a correct format
        wth = build_wth(d)
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


def build_wth(d):
    wth = pd.DataFrame()
    wth['day'] = d['ppt'].index.day
    wth['month'] = d['ppt'].index.month
    wth['year'] = d['ppt'].index.year
    wth['doy'] = d['ppt'].index.dayofyear
    wth['tmax'] = d['tmax'].values
    wth['tmin'] = d['tmin'].values
    wth['ppt'] = d['ppt'].values
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
