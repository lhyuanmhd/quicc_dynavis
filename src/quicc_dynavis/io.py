# src/quicc_dynavis/io.py
import os
import fnmatch
import numpy as np
import pandas as pd
import fnmatch

#--------------Parameters and Resolutions -----------#
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


def get_resolution(filepath,output):
    """
    Function to extract resolition from parameter file:
    
    Args:
    ---------
    filepath: Path to the parameter file.
    output: Shall the resutls be printed use: 'print'
    
    Returns:
    ---------
    n,l,m max values of n,l,m
    """
    f = open(filepath, 'r')
    n = 0;l=0;m=0;
    for line in f:
        if '<dim1D>' in line:
            ss = line.find('>')
            ll = line.rfind('<')
            n = int(line[ss+1:ll])
        if '<dim2D>' in line:
            ss = line.find('>')
            ll = line.rfind('<')
            l = int(line[ss+1:ll])
        if '<dim3D>' in line:
            ss = line.find('>')
            ll = line.rfind('<')
            m = int(line[ss+1:ll])
        if '<transform>' in line:
            break
    f.close()
    if output=='print':
        print('#################################')
        print('Resolution:')
        print('n:',n,'l:',l,'m:',m)	    
    return(n,l,m)

#-------------- Timeseries -----------#

def F_read_energyQCC(filepath):
    """
    Function to read the energy.dat (and entrophy.dat-files ourput)
       
    Args:
    ---------
    filepath= Path to the energy or enstrophy file.
    
    Returns:
    ---------
    istep: number of the timestep
    time: array with time
    Etot: total Energy or Enstrophy (kin or mag)
    Etor: toroidal Energy or Enstrophy (kin or mag)
    Epol: poloidal Energy or Enstrophy (kin or mag)
    """

    f = open(filepath, 'r')
        # initiaalize the all variables as lists:

    istep = []
    time = []
    Etot = []
    Etor = []
    Epol = []

    # Reading some unsed dummy lines from the file header
    gateline_1 = f.readline()
    aa_1 = f.readline()
    bb_1 = f.readline()

    i = 0
    for line in f:
        line = line.strip()       #Creating lines
        columns = line.split()    #Splitting into colums 

        istep.append(np.double(i))
        time.append(np.double(columns[0]))
        Etot.append(np.double(columns[1]))
        Etor.append(np.double(columns[2]))
        Epol.append(np.double(columns[3]))
        i = i+1

    f.close()

    #converting to nbumpy arrays:
    istep = np.asarray(istep)
    time = np.asarray(time)
    Etot= np.asarray(Etot)
    Etor = np.asarray(Etor)
    Epol = np.asarray(Epol)
    return(istep,time,Etot,Etor,Epol)

def F_read_dipolarity(filepath):
    """
    Function to read the dipolarity.dat-files ourput
       
    Args:
    ---------
    filepath= Path to the dipolarity file.
    
    Returns:
    ---------
    time: array with time
    fdip: array CMB dipolarity 
    g10: array with Gauss coefficients: coefficient g10 
    g11: array with Gauss coefficients: real part of the g11
    h11: array with Gauss coefficients: imaginary part of the g11

    """
    f = open(filepath, 'r')
    time = []
    fdip = []
    g10 = []
    g11 = []
    h11 = []
   
    gateline_1 = f.readline()
    aa_1 = f.readline()
    bb_1 = f.readline() 
    #bb_1 = f.readline()

    for line in f:
        line = line.strip()       #Creating lines
        columns = line.split()    #Splitting into colums 

        time.append(np.double(columns[0]))
        fdip.append(np.double(columns[1]))
        g10.append(np.double(columns[2]))
        g11.append(np.double(columns[3]))
        h11.append(np.double(columns[4]))

    f.close()
    time = np.asarray(time)
    fdip = np.asarray(fdip)
    g10 = np.asarray(g10)
    g11 = np.asarray(g11)
    h11 = np.asarray(h11)
    return(time,fdip,g10,g11,h11)

def F_read_Nusselt(filepath):
    """
    Function to read the nusselt.dat-files ourput
       
    Args:
    ---------
    filepath= Path to the Nusselt number file.
    
    Returns:
    ---------
    time: array with time
    Nu: Nusselt number as function of time 
    """

    f = open(filepath, 'r')
    time = []
    Nu = []

    gateline_1 = f.readline()
    aa_1 = f.readline()
    bb_1 = f.readline() 
    for line in f:
        line = line.strip()       #Creating lines
        columns = line.split()    #Splitting into colums 

        time.append(np.double(columns[0]))
        Nu.append(np.double(columns[1]))
    f.close()
    Nu = np.asarray(Nu)
    time = np.asarray(time)
    return(time,Nu)

def F_conc_timeseries(RunFolders,filetype):
    """
    Function to conncatenate the time series of different runs (e.g after restart))
   
    Note: For a smooth use the folders should be structured as I do it!!!
    
    
    Args:
    ---------
    RunFolders: Python list with all folders that should be condiered:
                for exampele with my folder organisation one can get this through: 
                    RunFolders =  sorted(glob.iglob(StartFolder+'/run*'))
    filetype: string-flag that encodes the time series that should be concatenated;
                - Energy timeseries: 'kinE' or 'magE' 
                - Dipolarity: 'Dip'
                - Enstrophy (Dissipation): 'kinDis' or 'magDis'
                - Nusselt number: Nusselt
        Depending on the Filetype-flag different routines are called.

    Returns:
    ---------
    time: array with time
    
    Rest depends on the filetype:
    -'kinE' or 'magE': 
        Etot:  total  energy (mag or kin)
        Etor:  toroidal energy (mag or kin)
        Epol:  poloidal  energy (mag or kin)
    -'Dip': 
        fdip: array CMB dipolarity 
        g10: array with Gauss coefficients: coefficient g10 
        g11: array with Gauss coefficients: real part of the g11
        h11: array with Gauss coefficients: imaginary part of the g11
    -'Nusselt':
        Nu: nusselt number
    -'kinDis' or 'magDis': 
        Dtot:  total Enstophy (mag or kin) 
        Dtor:  toroidal Enstophy (mag or kin)
        Dpol:  poloidal Enstophy (mag or kin)
    
    """

    if filetype == 'kinE':
        filename = '/kinetic_energy.dat'
    elif filetype == 'magE':
        filename = '/magnetic_energy.dat'
    elif filetype == 'Nusselt':
        filename = '/nusselt.dat'
    elif filetype == 'Dip':
        filename = '/magnetic_dipolarity.dat'
    elif filetype == 'kinDis':
        filename = '/kinetic_enstrophy.dat'    
    elif filetype == 'magDis':
        filename = '/magnetic_enstrophy.dat'

    num_runs = len(RunFolders)
    time = np.array([])

    if (filetype == 'kinE') or (filetype == 'magE'):
        Etot = np.array([])
        Etor = np.array([])
        Epol = np.array([])
    elif filetype == 'Nusselt':
        Nu = np.array([])
    elif filetype =='Dip':
        fdip = np.array([])
        g10 = np.array([])
        g11 = np.array([])
        h11 = np.array([])
    elif (filetype == 'kinDis') or (filetype == 'magDis'):
        Dtot = np.array([])
        Dtor = np.array([])
        Dpol = np.array([])

    for i in range(0,num_runs):
        if i == 0:
            lastval = 0
        else:
            if  np.shape(time)[0]==0:
                lastval=0
            else:
                lastval = time[-1]

        #print(filepath+runs[i]+filetype)i
        if (filetype == 'kinE') or (filetype == 'magE'):
            if os.path.exists(RunFolders[i]+'/kinetic_energy.dat') == True:
                dum,tt,eEtot,eEtor,eEpol = F_read_energyQCC(RunFolders[i]+'/'+filename)
                start_index= np.argmax(tt>lastval)
                time = np.concatenate((time,tt[start_index:]))
                Etot = np.concatenate((Etot,eEtot[start_index:]))
                Etor = np.concatenate((Etor,eEtor[start_index:]))
                Epol = np.concatenate((Epol,eEpol[start_index:]))
        elif filetype == 'Nusselt':
            if os.path.exists(RunFolders[i]+'/nusselt.dat') == True:
                tt,nNu = F_read_Nusselt(RunFolders[i]+'/'+filename)
                start_index= np.argmax(tt>lastval)
                time = np.concatenate((time,tt[start_index:]))
                Nu = np.concatenate((Nu,nNu[start_index:]))
        elif filetype == 'Dip':
            if os.path.exists(RunFolders[i]+'/magnetic_dipolarity.dat') == True:
                tt,ffdip,gg10,gg11,hh11 = F_read_dipolarity(RunFolders[i]+'/'+filename)
                start_index= np.argmax(tt>lastval)
                time = np.concatenate((time,tt[start_index:]))
                fdip = np.concatenate((fdip,ffdip[start_index:]))
                g10 = np.concatenate((g10,gg10[start_index:]))
                g11 = np.concatenate((g11,gg11[start_index:]))
                h11 = np.concatenate((h11,hh11[start_index:]))
        elif (filetype == 'kinDis') or (filetype == 'magDis'):
            if os.path.exists(RunFolders[i]+'/magnetic_enstrophy.dat') == True:
                dum2,tt,dDtot,dDtor,dDpol = F_read_energyQCC(RunFolders[i]+'/'+filename)
                start_index= np.argmax(tt>lastval)
                time = np.concatenate((time,tt[start_index:]))
                Dtot = np.concatenate((Dtot,dDtot[start_index:]))
                Dtor = np.concatenate((Dtor,dDtor[start_index:]))
                Dpol = np.concatenate((Dpol,dDpol[start_index:]))

    if (filetype == 'kinE') or (filetype == 'magE'):
        return(time,Etot,Etor,Epol)
    elif filetype == 'Nusselt':
        return(time,Nu)
    elif filetype == 'Dip':
        return(time,fdip,g10,g11,h11)
    elif (filetype == 'kinDis') or (filetype == 'magDis'):
        return(time,Dtot,Dtor,Dpol)
    

#-------------- Spectra -----------#
       
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


# TODO:  read_field_snapshot()


