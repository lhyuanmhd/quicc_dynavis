# src/quicc_dynavis/io.py
import os
import fnmatch
import numpy as np


def read_spectra(filepath):
    """
    Read a single kinetic or magnetic spectra file from QuICC.

    Returns:
        lm, lmtot, lmtor, lmpol, time
    """
    with open(filepath, 'r') as f:
        lm = []
        lmtot = []
        lmtor = []
        lmpol = []

        # Skip headers
        for _ in range(4):
            f.readline()
        cc = f.readline().split()
        time = float(cc[2])
        f.readline()
        f.readline()

        # Skip extra line for l-spectra
        filename = os.path.basename(filepath)
        if filename.startswith('kinetic_l') or filename.startswith('magnetic_l'):
            f.readline()

        for line in f:
            columns = line.strip().split()
            lm.append(float(columns[0]))
            lmtot.append(float(columns[1]))
            lmtor.append(float(columns[2]))
            lmpol.append(float(columns[3]))

    return np.array(lm), np.array(lmtot), np.array(lmtor), np.array(lmpol), time


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

def F_avgSpectra_new(folderpath,spec_type,start_time,stop_time):
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
            if fnmatch(path,'*run*'):
                for name in files:
                    if fnmatch(name, pattern):
                        FileList.append(os.path.join(path, name))
        print('FileList') 
        print(len(FileList))   
        lm,lmtot,lmtor,lmpol,time = F_read_Spectra(FileList[-1])
        df_lmtot = pd.DataFrame()
        df_lmtor = pd.DataFrame()
        df_lmpol = pd.DataFrame()

        #print('Start reading data:')
        for i in range(0,len(FileList)):
            lm,lmtot,lmtor,lmpol,time = F_read_Spectra(FileList[i])
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
