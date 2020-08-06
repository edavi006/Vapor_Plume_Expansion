"""
This script loads in an eos and calculates an adiabat from a given shock
state.

This code also takes an initial volume of shocked material that will
expand as a vapor plume and calculates a cooling rate at different
volumes. (spherical plume)

Order of operations
1. Load in isentrope
2. Calculate Planar expansion velocity - assume this is close to spherical
    expansion velocity
3. Calculate time profiles of P, rho, T and total time to ambient "pressure"
     or system closure (below the triple point)
4. Calculate MFP (skin depth) of the expanding cloud
5. Dullemond radiative cooling time for each "timestep" (when does radiative
    cooling become dominant for this initial volume)


"""

import pylab as py
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy as sp
import statistics as stat
from scipy.optimize import curve_fit
from scipy import interpolate
from matplotlib import rc
import eostable as eos

########Plot Parameters begin############3
#These control font size in plots.
params = {'legend.fontsize': 10,
         'axes.labelsize': 10,
         'axes.titlesize':10,
         'xtick.labelsize':10,
         'ytick.labelsize':10}
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
plt.rcParams.update(params)
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['xtick.minor.size'] = 2
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['ytick.minor.size'] = 2
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['axes.linewidth']= 1

plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['lines.dashed_pattern'] = [6, 6]
plt.rcParams['lines.dashdot_pattern'] = [3, 5, 1, 5]
plt.rcParams['lines.dotted_pattern'] = [1, 3]
plt.rcParams['errorbar.capsize'] = 3
plt.rcParams['lines.scale_dashes'] = False
plt.rcParams['legend.fancybox'] = False
plt.rcParams['legend.framealpha'] = None
plt.rcParams['legend.edgecolor'] = 'inherit'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100

plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['font.family']='Times New Roman'
plt.rcParams['figure.figsize']=5,4

########Plot Parameters finish############3

#User parameter inputs
#Enotropy value
S_cho=7000 #J/K/kg

#Assume spherical expansion for volume
#Input initial compressed radius
ri=1 #km
tp=5.2 #Pa, typical for silicates, change if different material

#Load in model

MODELNAME = 'Forsterite-ANEOS-SLVTv1.0G1'
# read EOS table      
forstEOS = eos.extGADtable()
forstEOS.readStdGadget('NEW-GADGET-STD-NOTENSION.TXT') # reads P-V-T-S-U-cs
forsthug = eos.EOShugoniot()
forsthug.T=np.loadtxt('NEW-SESAME-HUG.TXT',delimiter=',',skiprows=3,usecols=[0]) # K
forsthug.rho=np.loadtxt('NEW-SESAME-HUG.TXT',delimiter=',',skiprows=3,usecols=[1]) * 1000 # To kg/m^3
forsthug.P=np.loadtxt('NEW-SESAME-HUG.TXT',delimiter=',',skiprows=3,usecols=[2]) # Gpa
forsthug.U=np.loadtxt('NEW-SESAME-HUG.TXT',delimiter=',',skiprows=3,usecols=[3]) # MJ/kg
forsthug.S=np.loadtxt('NEW-SESAME-HUG.TXT',delimiter=',',skiprows=3,usecols=[4]) *(10**6) # J/K/kg
forsthug.up=np.loadtxt('NEW-SESAME-HUG.TXT',delimiter=',',skiprows=3,usecols=[5]) #km/s
forsthug.us=np.loadtxt('NEW-SESAME-HUG.TXT',delimiter=',',skiprows=3,usecols=[6]) #km/s



#Units are in CGS so replace with SI
#S is in erg/K/g
forstEOS.S=forstEOS.S *(10**(-4)) #J/K/kg
# rho is cm/g^3
forstEOS.rho=forstEOS.rho *(10**(3)) #kg/m^3
#P is dynes/cm2
forstEOS.P=forstEOS.P *(10**(-10))#GPa
#T is K, do nothing
#Cs, sound sped is cm/s
forstEOS.cs=forstEOS.cs *(10**(-5)) 


#Find shock state, and eos index for given entropy
#First entropy index
Si_min=min(min(np.where(S_cho<forstEOS.S)))
Si_max=max(min(np.where(S_cho>forstEOS.S)))
temp1=abs(forstEOS.S[Si_min]-S_cho)
temp2=abs(forstEOS.S[Si_max]-S_cho)
S_ind=Si_min #Set index to first one
if temp1 > temp2: #if temp2 is smaller use that one instead
    S_ind=Si_max

#Same as above except for hugoniot index
Si_min=min(min(np.where(S_cho<forsthug.S)))
Si_max=max(min(np.where(S_cho>forsthug.S)))
temp1=abs(forsthug.S[Si_min]-S_cho)
temp2=abs(forsthug.S[Si_max]-S_cho)
Sh_ind=Si_min #Set index to first one
if temp1 > temp2: #if temp2 is smaller use that one instead
    Sh_ind=Si_max

#Similar to above, finding peak pressure index of eos # i may not need this one.
Pi_max=max(min(np.where(forsthug.P[Sh_ind]>forstEOS.P[S_ind,:])))
P_ind=Pi_max #Set index to first one



#Print the Shock state
print('The Shock State for chosen entropy = ',S_cho,' J/K/kg')
print('Peak Pressure = ',forsthug.P[Sh_ind], ' GPa')
print('Peak Temperature = ',forsthug.T[Sh_ind], ' K')
print('Peak Density = ',forsthug.rho[Sh_ind], ' Kg/m^3')
print('Peak Shock Velocity = ',forsthug.us[Sh_ind], ' km/s')
print('Peak Particle Velocity = ',forsthug.up[Sh_ind], ' km/s')

print(forstEOS.P[S_ind,P_ind])

#Isentrope state variables and sound speed
#S_isen=forstEOS.S[S_ind]
#P_isen=forstEOS.P[S_ind,:]
#T_isen=forstEOS.T[S_ind,:]
#cs_isen=forstEOS.cs[S_ind,:]
#rho_isen=forstEOS.rho[:]
#Make 1d interpolations for vs and volume (1/rho)

#Interpolated isentrope function for riemann integral, function of P
rho_s=interpolate.interp1d(forstEOS.P[S_ind,:]*(10**9),1/(forstEOS.rho[:]*forstEOS.cs[S_ind,:]*1000))


#Using a Riemann integral for isentropic planar expansion

up_exp = np.zeros(P_ind+1) #Set up array, the calc outputs km/s
for i in range(P_ind):
    up_exp[P_ind-i]= (forsthug.up[Sh_ind]*1000 - integrate.quad(rho_s,forsthug.P[Sh_ind]*(10**9),forstEOS.P[S_ind,P_ind-i]*(10**9))[0])/1000
#There is some wonkiness due to the no tension areas of the EOS, but this calculation asymptotes pretty quick (by like 10^-2 GPa)
#We deal with this by finding when changes go crazy - giant jumps - the velocity normals out by
#extremely low pressures, so I call that the "final" expansion velocity
# For this calc, there seemed only to be one "Jump", so the following lines find the jump and average the two sides over the jump
# even it out

count=0
index_track=np.zeros(P_ind)
for i in range(P_ind):
    if abs(up_exp[P_ind-i] - up_exp[P_ind-i -1]) > up_exp[P_ind-i]*2:
        index_track[count] = int(P_ind-i)
        count=count + 1
for i in range(1,P_ind):
    if abs(up_exp[i+1] - up_exp[i]) > up_exp[i]*2:
        index_track[count] = int(i)
        count=count + 1
#print(count)
temp_ind=min(np.where(index_track>0))
ind_tra=index_track[temp_ind]


#now that the jump is found, get the two sides, and set the jump to an average of both sides
#If count is great than 1, than do the following
if count >1:
    up_ave=(up_exp[int(ind_tra[0])]+up_exp[int(ind_tra[1])])/2
    for i in range(int(ind_tra[1]),int(ind_tra[0])):
        up_exp[i]=up_ave
        
#up_exp also doesnt set the zero index so
up_exp[0]=up_exp[1]

 ###################
#Given an initial radius sphere, how long does it take to expand to the "Final" radius?
#Take the triple point as the final pressure
Pi_max=max(min(np.where(tp*(10**(-9))>forstEOS.P[S_ind,:])))
Ptp_ind=Pi_max #Index of triple point

#Volume ratio is the density at the shocked pressure over the released TP pressure
VR=forstEOS.rho[P_ind]/forstEOS.rho[Ptp_ind]
#So final radius (assuming sphere v = 4/3 pi r^3)     vf/vi = rf^3/ri^3 so rf = ri * (vf/vi)^(1/3)
rf=ri*(VR**(1/3)) #km

#The way this calc works, is that the parcel is allowed to keep expanding until the final radius is met
# until the final radius is met, the parcel is inertially trapped integrate for time to volume at each
# density index so we get a cool time profile
#We use the above lines to scale the volume increase for each step.
rf_scaled = np.zeros(P_ind-Ptp_ind)
time_dt= np.zeros(P_ind-Ptp_ind)
total_time= np.zeros(P_ind-Ptp_ind)

#Fill in radius array
for i in range(P_ind-Ptp_ind):
    VR_scaled=forstEOS.rho[P_ind-i]/forstEOS.rho[Ptp_ind]
    rf_scaled[i]=ri*(VR_scaled**(1/3))
#Calculate time to get to each step
for i in range(P_ind-Ptp_ind):
    if i == 0:
        time_dt[0] = (rf_scaled[(P_ind-Ptp_ind-1)-0]-ri)/up_exp[P_ind-0]
    if i > 0:
        time_dt[i] = (rf_scaled[(P_ind-Ptp_ind-1)-i]-rf_scaled[(P_ind-Ptp_ind-1)-i+1])/up_exp[P_ind-i]
#Calc integrated time now

total_time[0] = time_dt[0]
for i in range(1,np.size(time_dt)):
    total_time[i] = total_time[i-1] + time_dt[i]

#####Radiative cooling####
# we have total energy of the vapor plume from aneos
#Need to input some initial volume/mass
#t_radiative = E/L

#####PLOTTING###
##P-UP###
plt.figure()
plt.plot(forsthug.up,forsthug.P,'-',color='black',label='ANEOS Hugoniot')
plt.plot(up_exp,forstEOS.P[S_ind,:P_ind+1],'-',color='blue',label='release curve')

plt.semilogy()

plt.ylabel('Pressure (GPa)')
plt.xlabel('Particle Velocity (km/s)')
plt.legend(loc='best', fontsize='x-small',numpoints=1,scatterpoints=1)

plt.xlim(0,20)
plt.ylim(0.0001,1000)

##P-rho###
plt.figure()
plt.plot(forsthug.rho,forsthug.P,'-',color='black',label='ANEOS Hugoniot')
plt.plot(forstEOS.rho[:P_ind+1],forstEOS.P[S_ind,:P_ind+1],'-',color='blue',label='release curve')

plt.semilogy()

plt.ylabel('Pressure (GPa)')
plt.xlabel('Density (kg/m^3)')
plt.legend(loc='best', fontsize='x-small',numpoints=1,scatterpoints=1)

#plt.xlim(0,20)
plt.ylim(0.0001,1000)

##P-T###
plt.figure()
plt.plot(forsthug.T,forsthug.P,'-',color='black',label='ANEOS Hugoniot')
plt.plot(forstEOS.T[S_ind,:P_ind+1],forstEOS.P[S_ind,:P_ind+1],'-',color='blue',label='release curve')

plt.semilogy()

plt.ylabel('Pressure (GPa)')
plt.xlabel('Temperature (K)')
plt.legend(loc='best', fontsize='x-small',numpoints=1,scatterpoints=1)

plt.xlim(0,15000)
plt.ylim(0.0001,1000)

##P-time###
plt.figure()
plt.plot(total_time[::-1],forstEOS.P[S_ind,Ptp_ind:P_ind],'-',color='black',label='Decompression Time Profile')


plt.semilogy()

plt.ylabel('Pressure (GPa)')
plt.xlabel('time (s)')
plt.legend(loc='best', fontsize='x-small',numpoints=1,scatterpoints=1)

#plt.xlim(0,15000)
plt.ylim(0.0001,1000)

##T-time###
plt.figure()
plt.plot(total_time[::-1],forstEOS.T[S_ind,Ptp_ind:P_ind],'-',color='black',label='Decompression Time Profile')


#plt.semilogy()

plt.ylabel('Temperature (K)')
plt.xlabel('time (s)')
plt.legend(loc='best', fontsize='x-small',numpoints=1,scatterpoints=1)

#plt.xlim(0,15000)
#plt.ylim(0.0001,1000)

##rho-time###
plt.figure()
plt.plot(total_time[::-1],forstEOS.rho[Ptp_ind:P_ind],'-',color='black',label='Decompression Time Profile')
plt.semilogy()

plt.ylabel('Density (kg/m^3)')
plt.xlabel('time (s)')
plt.legend(loc='best', fontsize='x-small',numpoints=1,scatterpoints=1)

#plt.xlim(0,15000)
#plt.ylim(0.0001,1000)

##up-time###
plt.figure()
plt.plot(total_time[::-1],up_exp[Ptp_ind:P_ind],'-',color='black',label='Decompression Time Profile')
#plt.semilogy()

plt.ylabel('Expansion Velocity (km/s)')
plt.xlabel('time (s)')
plt.legend(loc='best', fontsize='x-small',numpoints=1,scatterpoints=1)

#plt.xlim(0,15000)
#plt.ylim(0.0001,1000)

##up-time###
plt.figure()
plt.plot(total_time[::-1],rf_scaled[:],'-',color='black',label='Decompression Time Profile')
#plt.semilogy()

plt.ylabel('Radius (km)')
plt.xlabel('time (s)')
plt.legend(loc='best', fontsize='x-small',numpoints=1,scatterpoints=1)

#plt.xlim(0,15000)
#plt.ylim(0.0001,1000)

plt.show()


