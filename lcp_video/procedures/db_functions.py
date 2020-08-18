#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 06:48:58 2019

@author: philipa
"""
from logging import debug,info,warn,error
from pdb import pm

def alias_from_basename_variant(basename,variant):
    """Defines 'alias' from 'basename' and 'variant' according to rules."""
    # Simplify basename.
    if '-PC0' in basename: # special case from 2019.03
        alias = basename.split('-PC0')[0]
    elif '-PC1' in basename: # special case from 2019.03
        alias = basename.split('-PC1')[0]
    elif '-Buffer' in basename or 'offB' in variant:
        alias = '-'.join([basename.split('-Buffer')[0],'Buffer'])
        if 'RNA' in basename and (beamtime == '2019.06' or beamtime == '2019.03'):
            alias = 'RNA-Buffer'       
    elif '-static' in basename:
        alias = basename.split('-static')[0]
    elif '-ramp' in basename:
        alias = basename.split('-ramp')[0]
        if variant != '':
            alias = '-'.join([alias,variant])
    else:
        if variant != '':
            alias = '-'.join([basename,variant])
        else:
            alias = basename                
    # extract 'off' and terminating 'T' from alias
    if '-off' in alias: 
        alias = '-'.join([alias.split('-off')[0],alias.split('-off')[1].split('T')[0]])        
    return alias

def create_database(db_name):
    """Creates LOGFILES, DATASETS, and SAMPLES tables."""
    create_LOGFILES(db_name)
    create_DATASETS(db_name)
    create_SAMPLES(db_name)

def create_DATASETS(db_name):
    """If not exists: create DATASETS table with hard-coded keys and dtypes."""
    table_name = 'DATASETS'
    terms = ['datetime_first TEXT PRIMARY KEY NOT NULL',
             'datetime_last TEXT',
             'dataset TEXT',
             'alias TEXT',
             'conditions TEXT',
             'repeat INTEGER',
             'mb REAL',
             'ms REAL',
             'method TEXT',
             'header TEXT',
             'N_entries INTEGER',
             'delays TEXT',
             'temperatures TEXT',
             'powers TEXT',
             'Qcheck TEXT']
    descriptor = ', '.join(terms)
    db_table_create(db_name,table_name,descriptor)

def create_LOGFILES(db_name):
    """If not exists: create LOGFILES table with hard-coded keys and dtypes."""
    table_name = 'LOGFILES'
    terms = ['datetime_first TEXT PRIMARY KEY NOT NULL',
             'datetime_last TEXT',
             'logfile TEXT',
             'N_entries INTEGER',
             'N_nan INTEGER',
             'N_xi INTEGER',
             'N_xt INTEGER',
             'N_lt INTEGER',
             'Qcheck TEXT',
             'uploaded TEXT']
    descriptor = ', '.join(terms)
    db_table_create(db_name,table_name,descriptor)

def create_SAMPLES(db_name):
    """If not exists: create SAMPLES table with hard-coded keys and dtypes."""
    table_name = 'SAMPLES'
    terms = ['alias TEXT PRIMARY KEY NOT NULL',
             'sample TEXT',
             'chemical_formula TEXT',
             'MW REAL',
             'N INTEGER']
    descriptor = ', '.join(terms)
    db_table_create(db_name,table_name,descriptor)

def data_from_logfile(logfile):
    """Returns 'data' as array, 'keys' and 'dtypes' as lists, and 'header' as 
    string from logfile."""
    from numpy import array
    lines = open(logfile,'r').readlines()
    data = []
    N_header = 0
    for line in lines:
        if line.startswith("#"):
            headerline = line
            N_header += 1
        else:
            fields = line[:-1].split('\t') # Omit eol character before split
            data.append(fields)
    header = lines[:N_header-1]
    
    keys = keys_from_headerline(headerline)
    dtypes = dtypes_from_data(data)
    data = array(data)
    return data,keys,dtypes,header
    
def datetime_from_mccd(filename):
    """Returns ['creation_datetime','acquire_datetime', 'header_datetime']. The 
    'creation_time' must be determined before opening the *.mccd file to read 
    its header; else the 'creation_time' returns the time the file was opened. """
    from os.path import getmtime
    from datetime import datetime, timedelta, timezone
    from dateutil.parser import parse
    import mmap
    
    creation_time = getmtime(filename) 
    creation_datetime = timestamp_to_datetime(creation_time) 
    
    Central=timezone(timedelta(hours=-5))    
    with open(filename,"rb") as f:
        # try to map to reduce any overhead to read file.
        content = mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) #2048+512
        f.close()
    
    t0 = parse("1970-01-01 00:00:00+0000")    
    fileparam_offset = content.find(b'MarCCD X-ray Image File')
    fileparam = content[fileparam_offset:fileparam_offset+1024]
    acquire_timestamp = fileparam[320:352] 
    header_timestamp = fileparam[352:384] 

    months = int(acquire_timestamp[:2])
    days = int(acquire_timestamp[2:4])
    hours = int(acquire_timestamp[4:6])
    mins = int(acquire_timestamp[6:8])
    year = int(acquire_timestamp[8:12])
    seconds = int(acquire_timestamp[13:15])
    microseconds = int(acquire_timestamp[16:22])
    t = datetime(year,months,days,hours,mins,seconds,microseconds,tzinfo=Central)
    acquire_time = (t-t0).total_seconds()  
    acquire_datetime = timestamp_to_datetime(acquire_time)
    
    months = int(header_timestamp[:2])
    days = int(header_timestamp[2:4])
    hours = int(header_timestamp[4:6])
    mins = int(header_timestamp[6:8])
    year = int(header_timestamp[8:12])
    seconds = int(header_timestamp[13:15])
    microseconds = int(header_timestamp[16:22])
    t = datetime(year,months,days,hours,mins,seconds,microseconds,tzinfo=Central)
    header_time = (t-t0).total_seconds()  
    header_datetime = timestamp_to_datetime(header_time)
    
    return [creation_datetime, acquire_datetime, header_datetime]    
 
def datetime_to_timestamp(datetime,timezone=None):
    """Convert a datetime string to number of seconds since 1 Jan 1970 00:00 
    UTC date: e.g. "2016-01-27 12:24:06.302724692-08"
    """
    from dateutil.parser import parse
    from numpy import nan
    
    try:
        t = parse(datetime)
        if t.tzinfo is None:
            if timezone is None:
                from dateutil.tz import tzlocal
                from datetime import datetime
                timezone = datetime.now(tzlocal()).tzname()
            t = parse(datetime+timezone)
        t0 = parse("1970-01-01 00:00:00+0000")
        T = (t-t0).total_seconds()
    except Exception as msg: error("timestamp: %r: %s" % (msg,datetime)); T = nan
    return T   

def db_query(db_name, db_commands):
    """Executes list of 'db_commands' and returns results in a 'data' array."""
    from  sqlite3 import connect
    
    if type(db_commands) == str:
        db_commands = [db_commands]
    with connect(db_name) as db:
        cursor = db.cursor()
        data = []
        for command in db_commands:
            cursor.execute(command)
            data.append(cursor.fetchall())
        db.commit()
    return data

def db_table_create(db_name,table_name,descriptor):
    """If table_name doesn't exist create it according to 'keys-dypes' descriptor."""
    db_command = '''CREATE TABLE IF NOT EXISTS "{}" ({})'''.format(table_name,descriptor)
    db_query(db_name, db_command)

def db_table_delete(db_name,table_name):
    """If table_name exists, delete it."""
    db_command = ('DROP TABLE IF EXISTS "{}"'.format(table_name))
    db_query(db_name, db_command)

def db_table_extract_key(db_name,table_name,key,key_sort):
    """This function extracts from 'table_name' the column of data specified 
    by 'key' and returns the contents as a numpy array sorted in the same order
    as key_sort."""
    from numpy import array
    
    db_command = ('SELECT {!s} from {!r} ORDER by {!s}'.format(key,table_name,key_sort))
    data = db_query(db_name, db_command)[0]
    result = []
    for entry in data:
        result.append(entry[0])
    return array(result)

def db_table_insert_keys_data(db_name,table_name,keys,data):
    """Insert into table rows from data into columns specified by keys."""
    from numpy import array
    
    if type(keys) == str: keys = [keys]
    keys = str(keys)[1:-1]
    data = array(data)
    db_commands = []
    for row in data:
        values = str(list(row))[1:-1]
        db_commands.append(('INSERT into "{}" ({}) VALUES ({})'.format(table_name,keys,values)))
    db_query(db_name, db_commands)
    
def db_table_append_keys_values(db_name,table_name,keys,values):
    """If not exists, insert values into table in columns specified by keys."""
    if type(keys) == str: keys = [keys]
    if type(values) == str: values = [values]
    keys = str(keys)[1:-1]
    values = str(values)[1:-1]
    db_command = ('INSERT or IGNORE into "{}" ({}) VALUES ({})'.format(table_name,keys,values))
    db_query(db_name, db_command)
    
def db_table_append_row(db_name,table_name,values):
    """Insert values into table; must have values for every column."""
    if type(values) == str: values = [values]
    values = str(values)[1:-1]
    db_command = ('INSERT into "{}" VALUES ({})'.format(table_name,values))
    db_query(db_name, db_command)

def db_tables_list(db_name):
    """Returns a list of tables found in db_name."""
    db_command = ('SELECT name FROM sqlite_master WHERE type="table"')
    data = db_query(db_name, db_command)[0]
    result = []
    for entry in data:
        result.append(entry[0])
    return result

def db_table_update_column(db_name,table_name,key,values,key_ref,values_ref):
    """Update key values in table where key_ref = values_ref."""
    db_commands = []
    for i in range(len(values)):
        value = values[i]
        value_ref = values_ref[i]
        db_commands.append(('UPDATE "{}" SET {}={} WHERE {}={}'.format(table_name,key,value,key_ref,value_ref)))
    db_query(db_name, db_commands)

def db_xray_image_names(db_name,topdir,dataset):
    """Returns a list fully-qualified filenames for x-ray images in dataset."""
    files = db_table_extract_key(db_name,dataset,'file','mccd_time')
    filenames = []
    for file in files:
        filenames.append('/'.join([topdir,'/'.join(dataset.split('/')[1:-1]),'xray_images',file]))
    return filenames
    
def dtypes_from_data(data):
    """Returns dtypes determined from first row of values in data."""
    def containsAny(str, char):
        for c in char:
            if c in str: return 1;
        return 0;
    # determine dtypes = []
    row = data[0]
    dtypes = []
    char = ['_',':','.','nan']
    for field in row:
        if containsAny(field,char[:2]):
            dtypes.append('TEXT')
        elif containsAny(field,char[2:]):
            dtypes.append('REAL')
        else:
            dtypes.append('INTEGER')
    return dtypes

def exclude():
    """Returns a list of patterns to exclude from a search. Add
    terms as required."""
    exclude = ['*/alignment*', 
               '*/trash*',
               '*/_Archived*',
               '*/backup*', 
               '*/Commissioning*', 
               '*/Test*', 
               '*/.AppleDouble*',
               '*LaserX*',
               '*LaserZ*',
               '*Copy*',
               '*._*',
               '*.DS_Store*']
    return exclude

def find(topdir, name=[], exclude=[]):
    """A list of files found starting at 'topdir' that match the patterns given 
    by 'name', excluding those matching the patterns given by 'exclude'."""
    def glob_to_regex(pattern):
        return "^"+pattern.replace(".", "\.").replace("*", ".*").replace("?", ".")+"$"
    try:
        from scandir import walk
    except ImportError:
        from os import walk
    import re
    if type(name) == str:
        name = [name]
    if type(exclude) == str:
        exclude = [exclude]
    name = [re.compile(glob_to_regex(pattern)) for pattern in name]
    exclude = [re.compile(glob_to_regex(pattern)) for pattern in exclude]

    file_list = []
    for (directory, subdirs, files) in walk(topdir):
        for file in files:
            pathname = directory+"/"+file
            match = any([pattern.match(pathname) for pattern in name]) and\
                not any([pattern.match(pathname) for pattern in exclude])
            if match:
                file_list += [pathname]
    return file_list

def info_from_header(header):
    """Returns mb, ms, sequence extracted from header in logfile."""
    from numpy import nan
    
    # If present in header information, extract mb, ms, and sequence 
    mb = nan
    ms = nan
    sequence = ''
    header = str(header).replace(' ','') # remove spaces before splitting
    if 'mb=' in header: mb = float(header.split('mb=')[1].split(',')[0])
    if 'ms=' in header: ms = float(header.split('ms=')[1].split(',')[0])
    if 'sequence=' in header: sequence = header.split('sequence=')[1].split(',')[0]
    return mb,ms,sequence

def info_from_logfile(logfile):
    """Returns 'basename', 'variants' (or 'conditions'), 'repeat' and 'method', 
    as deduced from logfile."""
    from analysis_functions import unique
    data,keys,dtypes,header = data_from_logfile(logfile)

    # Define method from logfile and keys.
    method = 'static'
    if 'ramp' in logfile: method = 'Tramp'
    if 'delay' in keys: method = 'pump_probe'
    
    # Extract repeat and basename from logfile
    folder = logfile.split('/')[-2] # One level up from logfilename
    repeat = folder.split('-')[-1]
    basename = '-'.join(folder.split('-')[:-1]) # Remove repeat from basename

    # Find unique names in filenames.
    filenames = data[:,keys.index('file')]
    names = []
    for filename in filenames:
        try:
            names.append(filename.split(folder)[1].split('_')[1])
        except:
            print('failed to to parse {}'.format(filename))
            exit
    variants = unique(names)
    if len(variants) == len(data):
        if 'PC0' in variants:
            variants = ['PCO']
        elif 'PC1' in variants:
            variants = ['PC1']
        elif 'PC2' in variants:
            variants = ['PC2']
        else:
            variants = ['']        
    
    return basename,variants,repeat,method

def keys_from_headerline(headerline):
    """Returns keys determined from last header line in data after executing 
    hard-coded key_replace."""
    # Replace key names according to key_replace list
    key_replace = [['date time','datetime'],
                   ['Temperature','T_set'],
                   ['temperature','T_obs'],
                   ['Repeat','repeat'],
                   ['Delay','delay']]
    for key in key_replace:
        headerline = headerline.replace(key[0],key[1])
    
    # Generate keys from modified headerline
    keys = headerline[1:-1].split('\t')
    
    # Rename one or more appearances of repeat to 'repeat1', 'repeat2', etc.
    i = 1
    for key in keys:
        if key == 'repeat':
            keys[keys.index('repeat')] = 'repeat'+str(i)
            i+=1
    return keys

def logfile_Qcheck(logfile,tolerance = 1.0):
    """Performs quality check on 'logfile'; compares datetime_log and 
    datetime_image; determines number of xray_images, xray_traces, 
    laser_traces; returns datetime_first, datetime_last, N_entries, N_nan, 
    N_xi, N_xt, N_lt, Qcheck."""
    from pathlib import Path
    from numpy import where,nan
    from os.path import getmtime

    data,keys,dtypes,header = data_from_logfile(logfile)
    N_entries = len(data)
    N_nan = len(where('nan' == data)[0])
    
    datetime_index = keys.index('datetime')
    datetimes = data[:,datetime_index]
    datetime_first = datetimes[0]
    datetime_last = datetimes[-1]
    
    file_index = keys.index('file')
    xray_images = data[:,file_index]
        
    folder = Path(logfile).parent.joinpath('xray_traces')
    xtrace_files = list(folder.glob('*.trc'))
    N_xt = len(xtrace_files)
    
    folder = Path(logfile).parent.joinpath('laser_traces')
    ltrace_files = list(folder.glob('*.trc'))
    N_lt = len(ltrace_files)
    
    folder = Path(logfile).parent.joinpath('xray_images')
    image_files = list(folder.glob('*.mccd'))
    N_xi = len(image_files)
    
    log_ts = [] # log file timestamps
    for datetime in datetimes:
        log_ts.append(datetime_to_timestamp(datetime,timezone=None))
    
    missing = 0
    image_ts = [] # image file timestamps
    for image in xray_images:
        filename = folder.joinpath(image)
        try:
            image_ts.append(getmtime(filename))
        except:
            image_ts.append(nan)
            missing += 1
    
    if (len(image_ts) > missing+2): #ensure nan_timediff_statistics can be calculated
        mean,slope,sigma,residual_max,residual_min,index_max,index_min = nan_timediff_statistics(log_ts,image_ts)
        if (residual_max < tolerance) and (abs(residual_min) < tolerance):
            Qcheck = 'passed, mean={:0.3f}, slope={:0.2e}, sigma={:0.3f}, residual_max={:0.3f}, residual_min={:0.3f}, index_max={}, index_min={}, missing {} *.mccd files'.format(mean,slope,sigma,residual_max,residual_min,index_max,index_min,missing)
        else:
            Qcheck = 'failed, mean={:0.3f}, slope={:0.2e}, sigma={:0.3f}, residual_max={:0.3f}, residual_min={:0.3f}, index_max={}, index_min={}, missing {} *.mccd files'.format(mean,slope,sigma,residual_max,residual_min,index_max,index_min,missing)
        if N_xi > len(xray_images):
            Qcheck += ', number of xray_images exceeds entries in log file'
        if N_entries > 0:
            if N_xt/N_entries != int(N_xt/N_entries):
                Qcheck += ', mismatch in number of xray_traces'
            if N_lt/N_entries != int(N_lt/N_entries):
                Qcheck += ', mismatch in number of laser_traces'
    else:
        Qcheck = 'failed, found {} logfile entries and {} xray_images'.format(len(log_ts),N_xi)

    return datetime_first,datetime_last,N_entries,N_nan,N_xi,N_xt,N_lt,Qcheck

def logfile_to_DATASETS(db_name,logfile):
    """Extracts dataset(s) from logfile; uploads corresponding info into 
    DATASETS and creates corresponding dataset tables."""
    from analysis_functions import unique
    from pathlib import Path
    from numpy import array,column_stack
    from time import time
    
    t0 = time()
    root = logfile.split(beamtime)[0]
    image_folder = str(Path(logfile).parent.joinpath('xray_images'))
    
    # Gather information for DATASETS
    basename,variants,repeat,method = info_from_logfile(logfile)
    data,keys,dtypes,header = data_from_logfile(logfile)
    mb,ms,sequence = info_from_header(header)    

    xray_images = data[:,keys.index('file')]

    # Merge started_time, finished_time, and mccd_time with data; identify xray_images that exist
    started_index = keys.index('started')
    finished_index = keys.index('finished')   
    started_time= []
    finished_time = []
    mccd_time = []
    exists = []
    for i in range(len(data)):
        started_time.append(datetime_to_timestamp(data[i,started_index]))
        finished_time.append(datetime_to_timestamp(data[i,finished_index]))
        image = data[i,keys.index('file')]
        filename = '/'.join([image_folder,image])
        try:
            mccd_datetime = datetime_from_mccd(filename)[1] # Select acquire option.
            mccd_time.append(datetime_to_timestamp(mccd_datetime))
            exists.append(True)
        except:
            mccd_time.append('')
            exists.append(False)
    exists = array(exists)
    # When converting *_time lists to arrays, ensure dtype is same as data for merging.
    started_time = array(started_time,dtype=data.dtype) 
    finished_time = array(finished_time,dtype=data.dtype)
    mccd_time = array(mccd_time,dtype=data.dtype) 
    data = column_stack((data,started_time)) 
    data = column_stack((data,finished_time))
    data = column_stack((data,mccd_time))
    keys += ['started_time','finished_time','mccd_time']
    dtypes += ['REAL','REAL','REAL']
    
    # When 'delay' is embedded in a 'sequence() string, replace string with 'delay' (found in 2019 logfiles).
    if 'delay' in keys and 'Sequence' in data[0,keys.index('delay')]:
        for i in range(len(data)):
            data[i,keys.index('delay')] = data[i,keys.index('delay')].split('delay=')[1].split(')')[0]
        
    # Define keylist and dtypelist for dataset in desired order.
    keys0 = ['outlier','datetime','started_time','finished_time','mccd_time']
    dtypes0 = ['TEXT','TEXT','REAL','REAL','REAL']
    keys1 = keys[keys.index('file'):-3] # keys beyond 'file', ignoring three added columns
    dtypes1 = []
    for key in keys1:
        dtypes1.append(dtypes[keys.index(key)])
    keys2 = ['Id','Ib','Ibx','Iby','xscope','lscope','T_act']
    dtypes2 = ['REAL','REAL','REAL','REAL','REAL','REAL','REAL']
    keylist = keys0+keys1+keys2
    dtypelist = dtypes0+dtypes1+dtypes2
    descriptor = table_descriptor(keylist,dtypelist)
 
    keys_sub = keylist[1:keylist.index('Id')]
    indices = []
    for key in keys_sub:
        indices.append(keys.index(key))   
    
    # Upload information into DATASETS and datasets. 
    for variant in variants:
        conditions = '' # To be implemened in 2019.
        
        # Determine 'alias' and 'dataset' from 'basename' and 'variant'
        alias = alias_from_basename_variant(basename,variant)
        dataset = '/'.join(logfile.split(root)[1].split('/')[:-1]+[alias])
                
        # 'select' is  subset of data corresponding to variant that exists
        select = []
        if variant != '':
            search_term = variant+'_'
            for image in xray_images: 
                select.append(search_term in image)
            select = array(select)
            N_entries = sum(select)
        else:
            N_entries = len(data)
            select = exists
        select = select & exists
        N_missing = N_entries - sum(select)
        
        # Strip '-PC*' from 'alias' before inserting into SAMPLES table.
        table_name = 'SAMPLES'
        value = alias
        if '-PC' in alias:
            value = alias.split('-PC')[0]
        key = 'alias'
        db_table_append_keys_values(db_name,table_name,key,value)
        
        # Define values for DATASETS table.
        delays = []
        temperatures = []
        powers = []
        if 'delay' in keys: delays = unique(data[:,keys.index('delay')][select])
        if 'T_set' in keys: temperatures = unique(data[:,keys.index('T_set')][select])
        if 'power' in keys: powers = unique(data[:,keys.index('power')][select])
        datetime_first = data[:,keys.index('datetime')][select][0]
        datetime_last = data[:,keys.index('datetime')][select][-1]
        Qcheck = 'N_missing={},'.format(N_missing)
        
        # Upload values into DATASETS table.
        table_name = 'DATASETS'
        values = [datetime_first,
                      datetime_last,
                      dataset,
                      alias,
                      conditions,
                      str(repeat),
                      str(mb),
                      str(ms),
                      method,
                      str(header),
                      str(N_entries),
                      str(delays),
                      str(temperatures),
                      str(powers),
                      Qcheck]
        db_table_append_row(db_name,table_name,values)
        
        # Create dataset table
        table_name = dataset
        db_table_create(db_name,table_name,descriptor)
        
        # Upload values into dataset
        data_sub = []
        for i in range(sum(select)):
            data_sub.append(data[select][i,indices]) 
        db_table_insert_keys_data(db_name,table_name,keys_sub,data_sub)
    print('{:0.2f} sec to load dataset(s) from {}'.format(time()-t0,logfile))  
    return

def logfile_to_LOGFILES(db_name,logfile):
    """Uploads relevant information into the LOGFILES table of db_name."""
    from time import time
    
    t0 = time()    
    table_name = 'LOGFILES'
    beamtime = topdir.split('/')[-1]
    logfilename = beamtime+logfile.split(topdir)[-1]
    datetime_first,datetime_last,N_entries,N_nan,N_xi,N_xt,N_lt,Qcheck = logfile_Qcheck(logfile)       
    uploaded = ''
    values = [datetime_first,
                  datetime_last,
                  logfilename,
                  str(N_entries),
                  str(N_nan),
                  str(N_xi),
                  str(N_xt),
                  str(N_lt),
                  Qcheck,
                  uploaded]
    db_table_append_row(db_name,table_name,values)
    t1 = time()
    print('{:0.2f} sec to characterize and upload {} into LOGFILES.'.format(t1-t0,logfile.split('/')[-1]))  

def nan_timediff_statistics(log_ts,image_ts):
    """Returns mean, slope, sigma, residual_max, residual_min, index_max, 
    index_min of time difference (image_ts - log_ts). The mean corresponds to 
    the linear-least-squares fit difference at the start of the dataset. """
    from numpy import array,nanmax,nanmin,nansum,isnan,sqrt,where
    
    x = array(log_ts) - log_ts[0]
    y = array(image_ts) - log_ts
    xsum = nansum(x)
    x2sum = nansum(x**2)
    ysum = nansum(y)
    xysum = nansum(x*y)
    N = len(x)-sum(isnan(y))
    delta = N*x2sum - xsum**2
    mean = (x2sum*ysum - xsum*xysum)/delta
    slope = (N*xysum -xsum*ysum)/delta
    residual = y-mean-slope*x
    residual_max = nanmax(residual)
    residual_min = nanmin(residual)
    sigma = sqrt(nansum(residual**2)/(N-2))
    index_max = where(residual == residual_max)[0][0]
    index_min = where(residual == residual_min)[0][0]
    return mean,slope,sigma,residual_max,residual_min,index_max,index_min
      
def table_descriptor(keylist,dtypelist):
    """Returns a table descriptor that combines information from keylist 
    and dtypelist."""
    descriptor = ''
    i = 0
    for key in keylist:
        descriptor = descriptor + str(key)+' '+ str(dtypelist[i]) + ', '
        i+=1
    descriptor = descriptor[:-2] 
    return descriptor
    
def timestamp_to_datetime(timestamp):
    """Converts timestamp to datetime string."""
    from datetime import datetime
    from dateutil.tz import tzlocal
    import pytz
    
    timeUTC = datetime.utcfromtimestamp(timestamp)
    timeLocal = pytz.utc.localize(timeUTC).astimezone(tzlocal())
    return timeLocal.strftime("%Y-%m-%d %H:%M:%S.%f%z")  

# z_Functions below are under development, or depricated.


def z_upload_logfiles_datasets(db_name,create_new = False):
    logfiles = find(topdir, '*.log', exclude())
    if create_new:
        create_database(db_name)
    for logfile in logfiles:
        print('Uploading {}'.format(logfile))
        logfile_to_LOGFILES(db_name,logfile)
        try:
            logfile_to_DATASETS(db_name,logfile)
        except:
            print('FAILED to upload dataset')
    return

def z_findfiles_in_subfolder_with_pattern(logfile,subfolder,pattern):
    """Returns [files], [datetimes] found with 'pattern' in 'subfolder'; 
    subfolder is in the same folder as 'logfile'; the lists are sorted."""
    from os.path import getmtime
    from numpy import argsort,array
    from time import time
    from pathlib import Path
    
    t0 = time()
    folder = Path(logfile).parent.joinpath(subfolder)
    timestamps = []
    files = list(folder.glob(pattern))
    for file in files:
        timestamps.append(getmtime(file))
    sort_order = argsort(timestamps)
    files = array(files)[sort_order]
    timestamps = array(timestamps)[sort_order]
    datetimes = []
    for timestamp in timestamps:
        datetimes.append(timestamp_to_datetime(timestamp))
    print('{:0.2f} sec to find and sort {} "{}" files according to timestamp.'.format(time()-t0, len(files), pattern))
    return files,datetimes



if __name__ == "__main__": # example for testing
    """ """
    #db_name = '/Mirror/Femto/C/All Projects/APS/Data Analysis/SAXS-WAXS/database/SAXS_WAXS_database.sqlite'
    #db_name = '/Mirror/Femto/C/All Projects/APS/Data Analysis/SAXS-WAXS/database/test1.sqlite'
    #beamtime = '2019.06'
    #mccd_root = '//femto-data/C/Data/' # when connected to femto-data via nfs
    #mccd_root = '/Volumes/data-2/'
    #tpkl_root = '//femto/C/Data/'
    #topdir = mccd_root+beamtime
    #topdir = '/Volumes/data-2/2019.06'
    #topdir = '/Volumes/data-1/2019.03'
    
    
    #logfile ='//femto-data/C/Data/2019.06/WAXS/Reference/Reference-2/Reference-2.log'
    #beamtime = logfile.split('Data/')[1].split('/')[0]
    #topdir = logfile.split(beamtime)[0]+beamtime
    


    
    
 
        
        
    
    
    
