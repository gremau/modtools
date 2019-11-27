import numpy as np
import pandas as pd
import os
import datetime as dt
from IPython.core.debugger import set_trace

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

def fill_site100(sitelist, site100_path, clim100_path, modelscenario=None):
    """
    Copy the data from the 100clim file into the site.100 file
    """
    if modelscenario is not None:
        site100_files = [f for f in os.listdir(site100_path) if '.100' in f
                and modelscenario in f]
        clim100_files = [f for f in os.listdir(clim100_path) if '.100clim' in f
                and modelscenario in f]
    else:
        site100_files = [f for f in os.listdir(site100_path) if '.100' in f]
        clim100_files = [f for f in os.listdir(clim100_path) if '.100clim' in f]
    # check if target directory exists... if not, create it.
    outdir = os.path.join(site100_path, 'new')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for sitename in sitelist:
        sitefile = [f for f in site100_files if sitename in f]
        climfile = [f for f in clim100_files if sitename in f]
        if len(sitefile) > 1:
            raise ValueError('Multiple {0} files in {1}'.format(sitename,
                site100_path))
        if len(climfile) > 1:
            raise ValueError('Multiple {0} files in {1}'.format(sitename,
                clim100_path))

        # Create a new file to write to
        outfile =  os.path.join(outdir, sitefile[0])
        # Open the files and read in lines (base new site.100 on old one)
        with open(os.path.join(clim100_path, climfile[0]), 'r') as fw:
            with open(os.path.join(site100_path, sitefile[0]), 'r') as fs:
                with open(outfile, 'w') as fout:
                    outlines = fs.readlines()
                    climlines = fw.readlines()
                    # Copy over precipitation
                    outlines[2:26] = climlines[0:24]
                    # Copy over temperature
                    outlines[38:62] = climlines[36:60]
                    # Write outfile
                    fout.writelines(outlines)

def make_wth_ushcn(sitelist, ushcn_pathname, out_pathname):
    """
    Create wth and site.100 files for USHCN data
    Make sure incoming precip data are in cm
    """
    # Read in USHCN file, creating datetime index from columns
    df = pd.read_csv(ushcn_pathname, parse_dates={'Date':[2,1,0]})
    df.set_index(df.Date, inplace=True)
    # Rename and recalculate columns
    df = df.rename(columns={"min.temp": "tmin", "max.temp": "tmax",
        "precip":"ppt"})
    for sitename in sitelist:
        # build_wth will put together the file in a correct format
        wth = build_wth(df)
        print(wth.head())
        # Use wth file to get monthly site.100 parameters
        site100 = build_site100(wth)
        # Change output format of ppt column and write to file
        wth['ppt'] = wth['ppt'].map(lambda x:'{0:.3}'.format(x))
        wth.to_csv(os.path.join(out_pathname, sitename + '.wth'), sep='\t',
                index=False, header=False)
        site100['var'] = site100['var'].map(lambda x:'{0:.3}'.format(x))
        site100.to_csv(os.path.join(out_pathname, sitename + '.100clim'),
                sep='\t',index=False, header=False)

def make_wth_ghcnd(sitelist, ghcnd_pathname, out_pathname):
    """
    Create wth and site.100 files for GHCND data
    Make sure incoming precip data are in cm
    """
    # Read in GHCND file, creating datetime index from columns
    df = pd.read_csv(ghcnd_pathname, parse_dates=['DATE'])
    df.set_index(df.DATE, inplace=True)
    # Rename and recalculate columns
    df = df.rename(columns={"TMIN": "tmin", "TMAX": "tmax",
        "PRCP":"ppt"})
    for sitename in sitelist:
        # build_wth will put together the file in a correct format
        wth = build_wth(df)
        print(wth.head())
        # Use wth file to get monthly site.100 parameters
        site100 = build_site100(wth)
        # Change output format of ppt column and write to file
        wth['ppt'] = wth['ppt'].map(lambda x:'{0:.3}'.format(x))
        wth.to_csv(os.path.join(out_pathname, sitename + '.wth'), sep='\t',
                index=False, header=False, float_format='%g')
        site100['var'] = site100['var'].map(lambda x:'{0:.3}'.format(x))
        site100.to_csv(os.path.join(out_pathname, sitename + '.100clim'),
                sep='\t',index=False, header=False)


def make_wth_loca(sitelist, loca_path, wth_path,
        modelname=r'HadGEM2-ES', scenario=r'rcp45',
        rmbefore='2018-01-01 12:00'):
    """
    Create wth and site.100 files for LOCA downscaled data
    """
    # Read in loca file
    locafile = pd.read_csv(loca_path, parse_dates=[0], index_col=0)
    # Remove some rows
    rmtest = locafile.index < rmbefore
    locafile = locafile.iloc[~rmtest,:]
    # Parse out correct GCM model and scenario
    modeltest = locafile.variable.str.contains(modelname)
    scenariotest = locafile.variable.str.contains(scenario)
    select = np.logical_and(modeltest, scenariotest)
    loca_m_s = locafile.loc[select, :]
    # Find the precipitation, tmax, and tmin rows
    ppttest = loca_m_s.variable.str.contains('^pr_')
    tmaxtest = loca_m_s.variable.str.contains('^tasmax_')
    tmintest = loca_m_s.variable.str.contains('^tasmin_')

    for sitename in sitelist:
        # First parse out the sites ppt, tmax, tmin data into a dataframe
        df = pd.concat([loca_m_s.loc[ppttest, sitename],
            loca_m_s.loc[tmaxtest, sitename],
            loca_m_s.loc[tmintest, sitename]],axis=1)
        df.columns = ['ppt', 'tmax', 'tmin']
        # Divide precip by 10 to get cm
        df.ppt = df.ppt/10

        # build_wth will put together the file in a correct format
        wth = build_wth(df)
        print(wth.head())
        # Use wth file to get monthly site.100 parameters
        site100 = build_site100(wth)
        # Change output format of ppt column and write to file
        wth['ppt'] = wth['ppt'].map(lambda x:'{0:.3}'.format(x))
        wth['tmax'] = wth['tmax'].map(lambda x:'{0:.3}'.format(x))
        wth['tmin'] = wth['tmin'].map(lambda x:'{0:.3}'.format(x))
        wth.to_csv(os.path.join(wth_path, modelname + '_' + scenario + '_' +
            sitename + '.wth'), sep='\t', index=False, header=False)
        site100['var'] = site100['var'].map(lambda x:'{0:.3}'.format(x))
        site100.to_csv(os.path.join(wth_path, modelname + '_' + scenario + '_' +
            sitename + '.100clim'), sep='\t', index=False, header=False)

def make_wth_dailymet(site, df_d, prism_file, wth_path, fmodifier=None):
    """
    Take daily prism data for a site and replace with local daily met data
    where available. Make sure incoming precip data are in cm
    """
    import matplotlib.pyplot as plt
    df = load_prism_daily(prism_file)
    # Replace some columns with measured data
    #df.tmax.plot()
    repvars = ['Rain_cm_Tot_sum', 'AirTC_Max_max', 'AirTC_Min_min']
    prismvars = ['ppt', 'tmax', 'tmin']
    for i, repvar in enumerate(repvars):
        repidx = ~np.isnan(df_d[repvar])
        df.loc[repidx.index[repidx],prismvars[i]] = df_d.loc[repidx, repvar]

    # build_wth will put together the file in a correct format
    wth = build_wth(df)
    print(wth.head())
    # Use wth file to get monthly site.100 parameters
    site100 = build_site100(wth)
    # Change output format of ppt column and write to file
    wth['ppt'] = wth['ppt'].map(lambda x:'{0:.3}'.format(x))
    if fmodifier is not None:
        outfilename1 = site + '_' + fmodifier + '.wth'
        outfilename2 = site + '_' + fmodifier + '.100clim'
    else:
        outfilename1 = site + '.wth'
        outfilename2 = site + '.100clim'
    wth.to_csv(os.path.join(wth_path, outfilename1), sep='\t',
            index=False, header=False)
    site100['var'] = site100['var'].map(lambda x:'{0:.3}'.format(x))
    site100.to_csv(os.path.join(wth_path, outfilename2), sep='\t',
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
