# src/quicc_dynavis/io.py
import os
import fnmatch
import numpy as np
import pandas as pd
import fnmatch

def get_parameters(filepath,output):
    """
    Function to extract input  values form parameter file:
    
    Args:
    ---------
    filepath: Path to the parameter file.
    output: Shall the resutls be printed use: 'print'
    
    Returns:
    ---------
    Ek: Ekman number
    Pm: magnetic Prandtl number
    Pr: Prandtl number
    q: Roberts number (if other nondim is used)
    Ra: Rayleigh number
    Ro: Rosby number (if other nondim is used)
    """
    f = open(filepath, 'r')
    Ek = 0;Pm=0;Ra=0;Pr=0;q=0;Ro=0
    for line in f:
        if '<ekman>' in line:
            ss = line.find('>')
            ll = line.rfind('<')
            Ek = float(line[ss+1:ll])
        if '<magnetic_prandtl>' in line:
            ss = line.find('>')
            ll = line.rfind('<')
            Pm = float(line[ss+1:ll])
        if '<rayleigh>' in line:
            ss = line.find('>')
            ll = line.rfind('<')
            Ra = float(line[ss+1:ll])
        if '<prandtl>' in line:
            ss = line.find('>')
            ll = line.rfind('<')
            Pr = float(line[ss+1:ll])
        if '<roberts>' in line:
            ss = line.find('>')
            ll = line.rfind('<')
            q = float(line[ss+1:ll])
        if '<rossby>' in line:
            ss = line.find('>')
            ll = line.rfind('<')
            Ro = float(line[ss+1:ll])
    f.close()
    # Here comes the output:
    if output=='print':	    
        print('#################################')
        print('## Input parameters ##')
        print('Ekman:',Ek)
        print('Rayleigh:',Ra)
        print('magnetic Prandtl:',Pm)
        print('Prandtl:',Pr)
        print('magnetic Ekman:', Ek/Pm)
        print('convective Rosby:', np.sqrt(Ek*Ra/Pr))
    	 
    return(Ek,Pm,Pr,q,Ra,Ro) 

def read_spectra(filepath):
    """
    Reading a single kinetic or magnetic spectra file
    
    Args:
    ---------
    filepath= Path to the spectra file.
    
    Returns:
    ---------
    lm: l or m (depending on type of spectra)
    lmtot: total energy at l or m (depending on degree or order)
    lmtor: toroidal energy at l or m (depending on degree or order)
    lmpol: poloidal energy at l or m (depending on degree or order)
    time: time stamp of the spectra   
    """

    path, filename = os.path.split(filepath)
    f = open(filepath, 'r')
    lm = []
    lmtot = []
    lmtor = []
    lmpol = []

    gateline_1 = f.readline()
    aa_1 = f.readline()
    bb_1 = f.readline()
    cc_1 = f.readline()
    cc = cc_1.split()
    time = np.double(cc[2])
    dd = f.readline().split()
    kinEtot= dd[2]; kinEtor= dd[3]; kinEpol= dd[4]
    aa_1 = f.readline()
    bb_1 = f.readline()


    # Getting rid of the extra line in l-spectra header
    if (filename[0:9] == 'kinetic_l') or (filename[0:10] == 'magnetic_l'):
        aa_1 = f.readline()

    for line in f:
        line = line.strip()       #Creating lines
        columns = line.split()    #Splitting into colums 
        lm.append(np.double(columns[0]))
        lmtot.append(np.double(columns[1]))
        lmtor.append(np.double(columns[2]))
        lmpol.append(np.double(columns[3]))

    f.close()    

    lm = np.asarray(lm)
    lmtot= np.asarray(lmtot)
    lmtor = np.asarray(lmtor)
    lmpol = np.asarray(lmpol)    
    return(lm,lmtot,lmtor,lmpol,time)
    

def read_single_spectrum(folderpath, spec_type='kinetic', which='last'):
    """
    Return the last/first/selected single spectrum from folder.

    Returns:
        l, m, ltot, ltor, lpol, mtot, mtor, mpol, time
    """
    indexList = ['_m*','_l*']
    for h in range(2):
        pattern = spec_type + indexList[h]
        FileList = []
        for path, subdirs, files in sorted(os.walk(folderpath)):
            if fnmatch.fnmatch(path,'*run*'):
                for name in files:
                    if fnmatch.fnmatch(name, pattern):
                        FileList.append(os.path.join(path, name))

        FileList = sorted(FileList)
        if not FileList:
            raise FileNotFoundError(f'No files matching {pattern} in {folderpath}')

        if which == 'last':
            chosen_file = FileList[-1]
        elif which == 'first':
            chosen_file = FileList[0]
        elif isinstance(which, int):
            chosen_file = FileList[which]
        else:
            raise ValueError("`which` must be 'last', 'first', or an integer index")

        lm_vals, ltot_vals, ltor_vals, lpol_vals, time = read_spectra(chosen_file)

        if h == 0:
            m = lm_vals + 1
            mtot = ltot_vals
            mtor = ltor_vals
            mpol = lpol_vals
        else:
            l = lm_vals
            ltot = ltot_vals
            ltor = ltor_vals
            lpol = lpol_vals

    return l, m, ltot, ltor, lpol, mtot, mtor, mpol, time

def avgSpectra_new(folderpath,spec_type,start_time,stop_time):
    """
    Averaging kinetic or magnetic spectra in time
    Note: this reads BOTH l and m (degree AND order)
    
    Args:
    ---------
    folderpath= Path to all the runfolders (only works in my ordering)
    spec_type= flag for the type oof spectra either 'magnetic' or 'kinetic'
    start_time= from what time on 
    stop_time= unitl ehen shall the average be calcualted 
            --> use something extrem if average to the end (say 100 or so)

    Returns:
    ---------
    l: spherical harmonic degree l  (magnetic or kinetic)
    l: spherical harmonic order m  (magnetic or kinetic)
    ltot: total energy at l (magnetic or kinetic)
    ltor: toroidal energy at l(magnetic or kinetic)
    lpol: poloidal energy at l (magnetic or kinetic)
    mtot: total energy in at m(magnetic or kinetic)
    mtor: toroidal energy at m(magnetic or kinetic)
    mpol: poloidal energy at m(magnetic or kinetic)
 
    """
    indexList=['_m*','_l*']
    for h in range(0,2):
        pattern=spec_type+indexList[h]
        FileList=[]
        for path, subdirs, files in sorted(os.walk(folderpath)):
            if fnmatch.fnmatch(path, '*run*'):
            #if fnmatch(path,'*run*'):
                for name in files:
                    if fnmatch.fnmatch(name, pattern):
                        FileList.append(os.path.join(path, name))
        #print('FileList') 
        #print(len(FileList))   
        lm,lmtot,lmtor,lmpol,time = read_spectra(FileList[-1])
        df_lmtot = pd.DataFrame()
        df_lmtor = pd.DataFrame()
        df_lmpol = pd.DataFrame()

        #print('Start reading data:')
        for i in range(0,len(FileList)):
            lm,lmtot,lmtor,lmpol,time = read_spectra(FileList[i])
            #print('shapes of read spectra:')
            #print(FileList[i])
            #print(np.shape(lmtot))
            if (time > start_time) & (time < stop_time):
                dum_lmtot = pd.DataFrame(lmtot)
                dum_lmtor = pd.DataFrame(lmtor)
                dum_lmpol = pd.DataFrame(lmpol)
                df_lmtot= pd.concat([df_lmtot, dum_lmtot],axis=1)
                df_lmtor= pd.concat([df_lmtor, dum_lmtor],axis=1)
                df_lmpol= pd.concat([df_lmpol, dum_lmpol],axis=1)
        if h == 0:
            m=lm+1
            mtot =  df_lmtot.mean(axis=1)
            mtor =  df_lmtor.mean(axis=1)
            mpol =  df_lmpol.mean(axis=1)
        elif h==1:
            l=lm
            ltot =  df_lmtot.mean(axis=1)
            ltor =  df_lmtor.mean(axis=1)
            lpol =  df_lmpol.mean(axis=1)
            

    return(l,m,ltot,ltor,lpol,mtot,mtor,mpol)


# TODO: later add read_timeseries() and read_field_snapshot()
