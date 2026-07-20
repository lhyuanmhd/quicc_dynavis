#src/quicc_dynavis/io.py

import os
import fnmatch
import numpy as np
import pandas as pd
import fnmatch
import re

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
            ss = line.find('<ekman>') + len('<ekman>')
            ll = line.find('</ekman>')
            Ek = float(line[ss:ll].strip())
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
    
    # Calculating missing nondim numbers:
    # if q not given and Pm and Pr are given, set q=Pm/pr
    if (q==0) & (Pm!=0) & (Pr!=0):
        q= Pm/Pr

    # Here comes the output:
    if output=='print':	    
        print('#################################')
        print('## -- Input parameters -- ##')
        print('Ekman:   ',Ek)
        print('Rayleigh:',Ra)
        print('magnetic Prandtl:',Pm)
        print('Prandtl: ',Pr)
        print('magnetic Ekman:', Ek/Pm)
        print('convective Rosby:', np.sqrt(Ek*Ra/Pr))
        print('#---------------------------------#')
    	 
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


# def get_boundary_conditions(filepath, output='none'):
#     """
#     Extract boundary condition settings from parameter file.

#     Args:
#     ---------
#     filepath: Path to the parameter file.
#     output: 'print' to display the results.

#     Returns:
#     ---------
#     bc_magnetic   : Magnetic boundary condition (e.g., insulating or conducting)
#     bc_temperature: Temperature boundary condition (e.g., fixed_flux or fixed_temperature)
#     bc_velocity   : Velocity boundary condition (e.g., no_slip or stress_free)
#     """
#     bc_magnetic = bc_temperature = bc_velocity = "undefined"
    
#     with open(filepath, 'r') as f:
#         for line in f:
#             if '<magnetic>' in line:
#                 ss, ll = line.find('>'), line.rfind('<')
#                 bc_magnetic = line[ss+1:ll].strip()
#             elif '<temperature>' in line:
#                 ss, ll = line.find('>'), line.rfind('<')
#                 bc_temperature = line[ss+1:ll].strip()
#             elif '<velocity>' in line:
#                 ss, ll = line.find('>'), line.rfind('<')
#                 bc_velocity = line[ss+1:ll].strip()

#     if output == 'print':
#         #print('#################################')
#         print('## Boundary Conditions ##')
#         print('Magnetic    BC:', bc_magnetic)
#         print('Temperature BC:', bc_temperature)
#         print('Velocity    BC:', bc_velocity)
    
#     return bc_magnetic, bc_temperature, bc_velocity



def get_boundary_conditions(filepath, output='none'):
    """
    Extract boundary condition settings from parameter file.

    Returns only the inner text, e.g.
        insulating
        fixed_flux
        no_slip
    """

    bc_magnetic = "undefined"
    bc_temperature = "undefined"
    bc_velocity = "undefined"

    # Regular expression pattern for <tag>value</tag>
    pattern = re.compile(r'<(\w+)>\s*([^<]+?)\s*</\1>')

    with open(filepath, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                tag, value = match.groups()
                value = value.strip()

                if tag == "magnetic":
                    bc_magnetic = value
                elif tag == "temperature":
                    bc_temperature = value
                elif tag == "velocity":
                    bc_velocity = value

    if output == 'print':
        print('## Boundary Conditions ##')
        print('Magnetic    BC:', bc_magnetic)
        print('Temperature BC:', bc_temperature)
        print('Velocity    BC:', bc_velocity)

    return bc_magnetic, bc_temperature, bc_velocity


def get_framework_and_setup_info(filepath, output='none'):
    """
    Extract timestepping scheme, timestep type, boundary scheme, and model setup flags
    from a QUICC parameters.cfg file.

    Returns:
        timestep (float)
        timestep_type ('adaptive' or 'fixed')
        scheme (str) - time integration scheme
        boundary_scheme (str)
        split_equation (bool)
    """
    timestep = None
    scheme = None
    boundary_scheme = None
    split_equation = False

    current_section = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Detect which section we are in
            if '<framework>' in line:
                current_section = 'framework'
            elif '</framework>' in line:
                current_section = None
            elif '<setup>' in line:
                current_section = 'setup'
            elif '</setup>' in line:
                current_section = None
            elif '<model>' in line:
                current_section = 'model'
            elif '</model>' in line:
                current_section = None

            # --- Framework: timestepping info ---
            if current_section == 'framework':
                if '<timestep>' in line:
                    ss, ll = line.find('>'), line.rfind('<')
                    try:
                        timestep = float(line[ss+1:ll])
                    except ValueError:
                        timestep = None
                elif '<scheme>' in line and '</scheme>' in line:
                    ss, ll = line.find('>'), line.rfind('<')
                    scheme = line[ss+1:ll].strip()

            # --- Setup: boundary scheme ---
            elif current_section == 'setup':
                if '<scheme>' in line and '</scheme>' in line:
                    ss, ll = line.find('>'), line.rfind('<')
                    boundary_scheme = line[ss+1:ll].strip()

            # --- Model: equation split ---
            elif current_section == 'model':
                if '<split_equation>' in line:
                    ss, ll = line.find('>'), line.rfind('<')
                    value = line[ss+1:ll].strip().lower()
                    split_equation = (value == 'on')

    # Interpret timestep meaning
    timestep_type = 'adaptive' if timestep == -1 else 'fixed'

    if output == 'print':
        print('#################################')
        print('## Framework and Setup Info ##')
        print(f'Timestep type:     {timestep_type}')
        if timestep != -1 and timestep is not None:
            print(f'Fixed timestep:    {timestep}')
        print(f'Time scheme:       {scheme}')
        print(f'Boundary scheme:   {boundary_scheme}')
        print(f'Split equation:    {"on" if split_equation else "off"}')

    return timestep, timestep_type, scheme, boundary_scheme, split_equation



#----------------------------------------
# 3 NEW — print_simulation_summary()
#----------------------------------------
def print_simulation_summary(filepath):
    """
    Print a clean summary of simulation configuration from parameters.cfg.
    Combines physical parameters, numerical schemes, and boundary info.
    """
    print('========================================')
    print(' Simulation Configuration Summary ')
    print('========================================')
    # Physical parameters
    Ek, Pm, Pr, q, Ra, Ro = get_parameters(filepath, output='none')
    print('\n--- Physical Parameters ---')
    print(f"Ekman number:          {Ek}")
    print(f"Rayleigh number:       {Ra}")
    print(f"Prandtl number:        {Pr}")
    print(f"Magnetic Prandtl:      {Pm}")
    print(f"Magnetic Ekman:        {Ek/Pm:.3e}" if Pm != 0 else "Magnetic Ekman:  undefined")
    print(f"Convective Rossby:     {np.sqrt(Ek*Ra/Pr):.3e}" if (Ek>0 and Ra>0 and Pr>0) else "Convective Rossby: undefined")
    print(f"Roberts number (q):    {q}")
    if Ro != 0:
        print(f"Rossby number:         {Ro}")
    
    #boundary conditions
    bc_magnetic, bc_temperature, bc_velocity = get_boundary_conditions(filepath, output='none')
    print('\n--- Boundary Conditions ---')
    print(f"Velocity BC:           {bc_velocity}")
    print(f"Magnetic BC:           {bc_magnetic}")
    print(f"Temperature BC:        {bc_temperature}")

    # Framework and setup
    timestep, timestep_type, scheme, boundary_scheme, split_eq = get_framework_and_setup_info(filepath, output='none')
    print('\n--- Numerical Setup ---')
    print(f"Timestep type:         {timestep_type}")
    if timestep != -1 and timestep is not None:
        print(f"Δt (fixed timestep):   {timestep}")
    print(f"Time integration:      {scheme}")
    print(f"Boundary scheme:       {boundary_scheme}")
    print(f"Equation splitting:    {'on' if split_eq else 'off'}")

    print('\n========================================')
    print('            End of Summary                 ')
    print('========================================')




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

# def F_conc_timeseries(RunFolders,filetype):
#     """
#     Function to conncatenate the time series of different runs (e.g after restart))
   
#     Note: For a smooth use the folders should be structured as I do it!!!
    
    
#     Args:
#     ---------
#     RunFolders: Python list with all folders that should be condiered:
#                 for exampele with my folder organisation one can get this through: 
#                     RunFolders =  sorted(glob.iglob(StartFolder+'/run*'))
#     filetype: string-flag that encodes the time series that should be concatenated;
#                 - Energy timeseries: 'kinE' or 'magE' 
#                 - Dipolarity: 'Dip'
#                 - Enstrophy (Dissipation): 'kinDis' or 'magDis'
#                 - Nusselt number: Nusselt
#         Depending on the Filetype-flag different routines are called.

#     Returns:
#     ---------
#     time: array with time
    
#     Rest depends on the filetype:
#     -'kinE' or 'magE': 
#         Etot:  total  energy (mag or kin)
#         Etor:  toroidal energy (mag or kin)
#         Epol:  poloidal  energy (mag or kin)
#     -'Dip': 
#         fdip: array CMB dipolarity 
#         g10: array with Gauss coefficients: coefficient g10 
#         g11: array with Gauss coefficients: real part of the g11
#         h11: array with Gauss coefficients: imaginary part of the g11
#     -'Nusselt':
#         Nu: nusselt number
#     -'kinDis' or 'magDis': 
#         Dtot:  total Enstophy (mag or kin) 
#         Dtor:  toroidal Enstophy (mag or kin)
#         Dpol:  poloidal Enstophy (mag or kin)
    
#     """

#     if filetype == 'kinE':
#         filename = '/kinetic_energy.dat'
#     elif filetype == 'magE':
#         filename = '/magnetic_energy.dat'
#     elif filetype == 'Nusselt':
#         filename = '/nusselt.dat'
#     elif filetype == 'Dip':
#         filename = '/magnetic_dipolarity.dat'
#     elif filetype == 'kinDis':
#         filename = '/kinetic_enstrophy.dat'    
#     elif filetype == 'magDis':
#         filename = '/magnetic_enstrophy.dat'

#     num_runs = len(RunFolders)
#     time = np.array([])

#     if (filetype == 'kinE') or (filetype == 'magE'):
#         Etot = np.array([])
#         Etor = np.array([])
#         Epol = np.array([])
#     elif filetype == 'Nusselt':
#         Nu = np.array([])
#     elif filetype =='Dip':
#         fdip = np.array([])
#         g10 = np.array([])
#         g11 = np.array([])
#         h11 = np.array([])
#     elif (filetype == 'kinDis') or (filetype == 'magDis'):
#         Dtot = np.array([])
#         Dtor = np.array([])
#         Dpol = np.array([])

#     for i in range(0,num_runs):
#         if i == 0:
#             lastval = 0
#         else:
#             if  np.shape(time)[0]==0:
#                 lastval=0
#             else:
#                 lastval = time[-1]

#         #print(filepath+runs[i]+filetype)i
#         if (filetype == 'kinE') or (filetype == 'magE'):
#             if os.path.exists(RunFolders[i]+'/kinetic_energy.dat') == True:
#                 dum,tt,eEtot,eEtor,eEpol = F_read_energyQCC(RunFolders[i]+'/'+filename)
#                 #start_index= np.argmax(tt>lastval)
#                 mask = tt > lastval
#                 if np.any(mask):
#                     start_index = np.argmax(mask)
#                 else:
#                     # No new time values; skip this file safely
#                     continue
#                 time = np.concatenate((time,tt[start_index:]))
#                 Etot = np.concatenate((Etot,eEtot[start_index:]))
#                 Etor = np.concatenate((Etor,eEtor[start_index:]))
#                 Epol = np.concatenate((Epol,eEpol[start_index:]))
#         elif filetype == 'Nusselt':
#             if os.path.exists(RunFolders[i]+'/nusselt.dat') == True:
#                 tt,nNu = F_read_Nusselt(RunFolders[i]+'/'+filename)
#                 start_index= np.argmax(tt>lastval)
#                 time = np.concatenate((time,tt[start_index:]))
#                 Nu = np.concatenate((Nu,nNu[start_index:]))
#         elif filetype == 'Dip':
#             if os.path.exists(RunFolders[i]+'/magnetic_dipolarity.dat') == True:
#                 tt,ffdip,gg10,gg11,hh11 = F_read_dipolarity(RunFolders[i]+'/'+filename)
#                 start_index= np.argmax(tt>lastval)
#                 time = np.concatenate((time,tt[start_index:]))
#                 fdip = np.concatenate((fdip,ffdip[start_index:]))
#                 g10 = np.concatenate((g10,gg10[start_index:]))
#                 g11 = np.concatenate((g11,gg11[start_index:]))
#                 h11 = np.concatenate((h11,hh11[start_index:]))
#         elif (filetype == 'kinDis') or (filetype == 'magDis'):
#             if os.path.exists(RunFolders[i]+'/magnetic_enstrophy.dat') == True:
#                 dum2,tt,dDtot,dDtor,dDpol = F_read_energyQCC(RunFolders[i]+'/'+filename)
#                 start_index= np.argmax(tt>lastval)
#                 time = np.concatenate((time,tt[start_index:]))
#                 Dtot = np.concatenate((Dtot,dDtot[start_index:]))
#                 Dtor = np.concatenate((Dtor,dDtor[start_index:]))
#                 Dpol = np.concatenate((Dpol,dDpol[start_index:]))

#     if (filetype == 'kinE') or (filetype == 'magE'):
#         return(time,Etot,Etor,Epol)
#     elif filetype == 'Nusselt':
#         return(time,Nu)
#     elif filetype == 'Dip':
#         return(time,fdip,g10,g11,h11)
#     elif (filetype == 'kinDis') or (filetype == 'magDis'):
#         return(time,Dtot,Dtor,Dpol)
    
#update  F_conc_timeseries to inculde temE
def F_conc_timeseries(RunFolders, filetype):
    """
    Function to conncatenate the time series of different runs (e.g after restart))
    (Original docstring unchanged; add temE support)

    New:
    - filetype == 'temE' reads temperature_energy.dat and returns (time, Etot)
      because it has no toroidal/poloidal components.
    """

    if filetype == 'kinE':
        filename = '/kinetic_energy.dat'
    elif filetype == 'magE':
        filename = '/magnetic_energy.dat'
    elif filetype == 'temE':
        filename = '/temperature_energy.dat'
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
    elif filetype == 'temE':
        Etot = np.array([])
    elif filetype == 'Nusselt':
        Nu = np.array([])
    elif filetype == 'Dip':
        fdip = np.array([])
        g10 = np.array([])
        g11 = np.array([])
        h11 = np.array([])
    elif (filetype == 'kinDis') or (filetype == 'magDis'):
        Dtot = np.array([])
        Dtor = np.array([])
        Dpol = np.array([])

    for i in range(0, num_runs):
        if i == 0:
            lastval = 0
        else:
            if np.shape(time)[0] == 0:
                lastval = 0
            else:
                lastval = time[-1]

        # --- kinetic / magnetic energy (original) ---
        if (filetype == 'kinE') or (filetype == 'magE'):
            if os.path.exists(RunFolders[i] + '/kinetic_energy.dat') == True:
                dum, tt, eEtot, eEtor, eEpol = F_read_energyQCC(RunFolders[i] + '/' + filename)

                mask = tt > lastval
                if np.any(mask):
                    start_index = np.argmax(mask)
                else:
                    continue

                time = np.concatenate((time, tt[start_index:]))
                Etot = np.concatenate((Etot, eEtot[start_index:]))
                Etor = np.concatenate((Etor, eEtor[start_index:]))
                Epol = np.concatenate((Epol, eEpol[start_index:]))

        # --- NEW: temperature energy ---
        elif filetype == 'temE':
            if os.path.exists(RunFolders[i] + '/temperature_energy.dat') == True:
                # Expect a reader that returns (tt, Etot) OR similar.
                # If you already have a different reader name, replace the call below.
                tt, eEtot =F_read_Nusselt(RunFolders[i] + '/' + filename)

                mask = tt > lastval
                if np.any(mask):
                    start_index = np.argmax(mask)
                else:
                    continue

                time = np.concatenate((time, tt[start_index:]))
                Etot = np.concatenate((Etot, eEtot[start_index:]))

        # --- Nusselt (original) ---
        elif filetype == 'Nusselt':
            if os.path.exists(RunFolders[i] + '/nusselt.dat') == True:
                tt, nNu = F_read_Nusselt(RunFolders[i] + '/' + filename)
                start_index = np.argmax(tt > lastval)
                time = np.concatenate((time, tt[start_index:]))
                Nu = np.concatenate((Nu, nNu[start_index:]))

        # --- Dipolarity (original) ---
        elif filetype == 'Dip':
            if os.path.exists(RunFolders[i] + '/magnetic_dipolarity.dat') == True:
                tt, ffdip, gg10, gg11, hh11 = F_read_dipolarity(RunFolders[i] + '/' + filename)
                start_index = np.argmax(tt > lastval)
                time = np.concatenate((time, tt[start_index:]))
                fdip = np.concatenate((fdip, ffdip[start_index:]))
                g10 = np.concatenate((g10, gg10[start_index:]))
                g11 = np.concatenate((g11, gg11[start_index:]))
                h11 = np.concatenate((h11, hh11[start_index:]))

        # --- Enstrophy (original) ---
        elif (filetype == 'kinDis') or (filetype == 'magDis'):
            if os.path.exists(RunFolders[i] + '/magnetic_enstrophy.dat') == True:
                dum2, tt, dDtot, dDtor, dDpol = F_read_energyQCC(RunFolders[i] + '/' + filename)
                start_index = np.argmax(tt > lastval)
                time = np.concatenate((time, tt[start_index:]))
                Dtot = np.concatenate((Dtot, dDtot[start_index:]))
                Dtor = np.concatenate((Dtor, dDtor[start_index:]))
                Dpol = np.concatenate((Dpol, dDpol[start_index:]))

    if (filetype == 'kinE') or (filetype == 'magE'):
        return (time, Etot, Etor, Epol)
    elif filetype == 'temE':
        return (time, Etot)
    elif filetype == 'Nusselt':
        return (time, Nu)
    elif filetype == 'Dip':
        return (time, fdip, g10, g11, h11)
    elif (filetype == 'kinDis') or (filetype == 'magDis'):
        return (time, Dtot, Dtor, Dpol)


#-------------- Spectra -----------#
import os
import fnmatch
import numpy as np


def read_spectra(filepath):
    """
    Reading a single kinetic, magnetic, or temperature spectra file.
    
    Args:
    ---------
    filepath: str
        Path to the spectra file.
    
    Returns:
    ---------
    lm: array
        l or m (depending on type of spectra)
    lmtot: array
        total energy at l or m
    lmtor: array or None
        toroidal energy (if available)
    lmpol: array or None
        poloidal energy (if available)
    time: float
        time stamp of the spectra   
    """

    path, filename = os.path.split(filepath)
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Determine file type
    is_temperature = ('temperature_l' in filename) or ('temperature_m' in filename)
    is_kinetic = ('kinetic_l' in filename) or ('kinetic_m' in filename)
    is_magnetic = ('magnetic_l' in filename) or ('magnetic_m' in filename)

    # Extract time
    for line in lines:
        if line.startswith("# time:"):
            time = float(line.split()[2])
            break
    else:
        raise ValueError(f"No time info found in {filename}")

    # Skip header lines
    data_start = 0
    for i, line in enumerate(lines):
        if not line.startswith("#"):
            data_start = i
            break

    # Initialize containers
    lm, lmtot, lmtor, lmpol = [], [], [], []

    # Read data
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) == 0:
            continue

        # Temperature: only l and total
        if is_temperature:
            if len(parts) < 2:
                continue
            lm.append(float(parts[0]))
            lmtot.append(float(parts[1]))
        else:
            # Kinetic or Magnetic: total, toroidal, poloidal
            if len(parts) < 4:
                continue
            lm.append(float(parts[0]))
            lmtot.append(float(parts[1]))
            lmtor.append(float(parts[2]))
            lmpol.append(float(parts[3]))

    lm = np.array(lm)
    lmtot = np.array(lmtot)
    lmtor = np.array(lmtor) if lmtor else None
    lmpol = np.array(lmpol) if lmpol else None

    return lm, lmtot, lmtor, lmpol, time

import os, re, glob
from pathlib import Path

def _extract_int(pattern: str, s: str, default=-1) -> int:
    """
    Extract first integer using regex pattern with one capturing group.
    """
    m = re.search(pattern, s)
    return int(m.group(1)) if m else default

#def _pick_run_dir(folderpath: str | Path, which: str = "last") -> Path:
from typing import Union

def _pick_run_dir(folderpath: Union[str, Path], which: str = "last") -> Path:
    """
    Pick a run directory under folderpath. Uses run number if present (run3, run12, ...).
    Fallback: modification time.
    """
    folderpath = Path(folderpath)
    run_dirs = [p for p in folderpath.rglob("*") if p.is_dir() and re.search(r"run\d+", p.name)]
    if not run_dirs:
        raise FileNotFoundError(f"No run* directories found under {folderpath}")

    # sort by run number (preferred), fallback to mtime
    def run_key(p: Path):
        rn = _extract_int(r"run(\d+)", p.name, default=-1)
        if rn >= 0:
            return (0, rn)   # (has_run_number, run_number)
        return (1, p.stat().st_mtime)

    run_dirs = sorted(run_dirs, key=run_key)

    if which == "last":
        return run_dirs[-1]
    elif which == "first":
        return run_dirs[0]
    elif isinstance(which, int):
        return run_dirs[which]
    else:
        raise ValueError("`which` must be 'last', 'first', or an integer index")

def _pick_spectrum_file(run_dir: Path, spec_type: str, index_tag: str, which: str = "last") -> Path:
    """
    Pick spectrum file inside one run directory.
    Example filename you mentioned: magnetic_n_spectrum8000.dat
    We sort by the integer after 'spectrum'.
    """
    # old logic: pattern = spec_type + '_m*' or spec_type + '_l*'
    # actual filenames: magnetic_l_spectrum8000.dat
    # keep it flexible: accept spec_type + index_tag and then anything, but require 'spectrum<number>'.
    pattern = str(run_dir / f"{spec_type}{index_tag}*spectrum*.dat")
    files = [Path(p) for p in glob.glob(pattern)]
    if not files:
        raise FileNotFoundError(f"No spectrum files matching {pattern}")

    def spec_key(p: Path):
        # Sort by spectrum index number; fallback to mtime if not found
        sn = _extract_int(r"spectrum(\d+)", p.name, default=-1)
        if sn >= 0:
            return (0, sn)
        return (1, p.stat().st_mtime)

    files = sorted(files, key=spec_key)

    if which == "last":
        return files[-1]
    elif which == "first":
        return files[0]
    elif isinstance(which, int):
        return files[which]
    else:
        raise ValueError("`which` must be 'last', 'first', or an integer index")

def read_single_spectrum(folderpath, spec_type="kinetic", which="last"):
    """
    Read a single spectrum from the *selected run* (last/first/index),
    and within that run read the last/first/index spectrum file.

    Returns:
        l, m, ltot, ltor, lpol, mtot, mtor, mpol, time
    """
    # Old code assumed two files: spec_type + '_m*' and spec_type + '_l*'
    indexList = ["_m", "_l"]  # we'll add wildcard in picker

    l = m = None
    ltot = ltor = lpol = None
    mtot = mtor = mpol = None
    time = None

    run_dir = _pick_run_dir(folderpath, which=which)

    for h, tag in enumerate(indexList):
        # tag like "_m" -> pass as "_m" and picker uses f"{spec_type}{tag}*spectrum*.dat"
        chosen_file = _pick_spectrum_file(run_dir, spec_type, tag, which="last" if which in ("last","first") else which)

        lm_vals, ltot_vals, ltor_vals, lpol_vals, time = read_spectra(str(chosen_file))

        if h == 0:
            m = lm_vals
            mtot = ltot_vals
            mtor = ltor_vals
            mpol = lpol_vals
        else:
            l = lm_vals
            ltot = ltot_vals
            ltor = ltor_vals
            lpol = lpol_vals

    return l, m, ltot, ltor, lpol, mtot, mtor, mpol, time

# def read_single_spectrum(folderpath, spec_type='kinetic', which='last'):
#     """
#     Return the last/first/selected single spectrum from folder.
# 
#     spec_type: 'kinetic', 'magnetic', or 'temperature'
# 
#     Returns:
#         l, m, ltot, ltor, lpol, mtot, mtor, mpol, time
#         (Note: for temperature spectra, ltor/lpol/mtor/mpol = None)
#     """
#     indexList = ['_m*', '_l*']
#     l = m = None
#     ltot = ltor = lpol = None
#     mtot = mtor = mpol = None
#     time = None
# 
#     for h in range(2):
#         pattern = spec_type + indexList[h]
#         FileList = []
#         for path, subdirs, files in sorted(os.walk(folderpath)):
#             if fnmatch.fnmatch(path, '*run*'):
#                 for name in files:
#                     if fnmatch.fnmatch(name, pattern):
#                         FileList.append(os.path.join(path, name))
# 
#         FileList = sorted(FileList)
#         if not FileList:
#             raise FileNotFoundError(f'No files matching {pattern} in {folderpath}')
# 
#         if which == 'last':
#             chosen_file = FileList[-1]
#         elif which == 'first':
#             chosen_file = FileList[0]
#         elif isinstance(which, int):
#             chosen_file = FileList[which]
#         else:
#             raise ValueError("`which` must be 'last', 'first', or an integer index")
# 
#         lm_vals, ltot_vals, ltor_vals, lpol_vals, time = read_spectra(chosen_file)
# 
#         if h == 0:
#             m = lm_vals
#             mtot = ltot_vals
#             mtor = ltor_vals
#             mpol = lpol_vals
#         else:
#             l = lm_vals
#             ltot = ltot_vals
#             ltor = ltor_vals
#             lpol = lpol_vals
# 
#     return l, m, ltot, ltor, lpol, mtot, mtor, mpol, time


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



def read_single_n_spectrum(folderpath, spec_type='kinetic', which='last', plot=False):
    """
    Read a single n-spectrum (for kinetic, magnetic, or temperature field).

    Returns:
        n_vals, l_vals, e_tot_2d, e_tor_2d, e_pol_2d, time
    where e_*_2d has shape (n, l)

    If plot=True, creates a scatter (or pcolormesh) plot of total energy vs (n, l).
    """
    import re
    import fnmatch
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    pattern = spec_type + '_n*'
    FileList = []
    for path, subdirs, files in sorted(os.walk(folderpath)):
        if fnmatch.fnmatch(path, '*run*'):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    FileList.append(os.path.join(path, name))

    FileList = sorted(FileList)
    if not FileList:
        raise FileNotFoundError(f'No files matching {pattern} in {folderpath}')

    # --- choose file ---
    if which == 'last':
        chosen_file = FileList[-1]
    elif which == 'first':
        chosen_file = FileList[0]
    elif isinstance(which, int):
        chosen_file = FileList[which]
    else:
        raise ValueError("`which` must be 'last', 'first', or an integer index")

    with open(chosen_file, 'r') as f:
        lines = f.readlines()

    # --- locate the '# n     Total data' headers ---
    section_indices = []
    for i, line in enumerate(lines):
        if re.match(r'#\s*n\s+Total', line):
            section_indices.append(('total', i))
        elif re.match(r'#\s*n\s+Toroidal', line):
            section_indices.append(('tor', i))
        elif re.match(r'#\s*n\s+Poloidal', line):
            section_indices.append(('pol', i))
    section_indices.append(('end', len(lines)))  # mark end

    # Extract time if present
    time = None
    for line in lines:
        if line.startswith("# time:"):
            try:
                time = float(line.split()[2])
            except Exception:
                pass
            break

    # --- helper to parse section ---
    def parse_section(start, end):
        block = []
        for line in lines[start:end]:
            if not line.strip().startswith("#") and line.strip():
                vals = np.fromstring(line, sep=' ')
                if len(vals) > 1:
                    block.append(vals)
        if not block:
            return None, None, None
        data = np.array(block)
        n_vals = data[:, 0]
        e_matrix = data[:, 1:]  # shape (n, l)
        l_vals = np.arange(1, e_matrix.shape[1] + 1)
        return n_vals, l_vals, e_matrix

    # --- parse sections ---
    n_tot = l_tot = e_tot = n_tor = l_tor = e_tor = n_pol = l_pol = e_pol = None
    for idx, (stype, start_line) in enumerate(section_indices[:-1]):
        next_line = section_indices[idx + 1][1]
        if stype == 'total':
            n_tot, l_tot, e_tot = parse_section(start_line + 1, next_line)
        elif stype == 'tor':
            n_tor, l_tor, e_tor = parse_section(start_line + 1, next_line)
        elif stype == 'pol':
            n_pol, l_pol, e_pol = parse_section(start_line + 1, next_line)

    # --- temperature only has total data ---
    if spec_type == 'temperature':
        e_tor = e_pol = None

    # --- optional scatter plot ---
    if plot and e_tot is not None:
        n_grid, l_grid = np.meshgrid(l_tot, n_tot)
        plt.figure(figsize=(7, 5), dpi=150)
        sc = plt.scatter(l_grid, n_grid, c=e_tot, s=10, cmap='viridis', norm='log')
        plt.colorbar(sc, label='Energy')
        plt.xlabel(r'Spherical degree $l$')
        plt.ylabel(r'Radial mode $n$')
        plt.title(f'{spec_type.capitalize()} total energy $E(n, l)$ at t={time:.2e}')
        plt.yscale('log')
        plt.xscale('log')
        plt.tight_layout()
        plt.show()

    return n_tot, l_tot, e_tot, e_tor, e_pol, time
