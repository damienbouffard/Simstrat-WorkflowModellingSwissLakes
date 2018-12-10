# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 10:02:06 2017

@author: gaudarad
@description: Function necessary to execute the script "Simstrat.py"
"""

import os
import csv, json, codecs, ftplib
from datetime import datetime, timedelta
import calendar
import unidecode, re
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from scipy import stats
import plotly.offline as py, plotly.graph_objs as go
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pyperclip, imageio
np.seterr(all='ignore')

#logFile = open('Functions.log','w')
def writelog(txt):
    logFile.write(txt)
    print(txt,end='')

def toFloat(string):
    try:
        return float(string)
    except:
        return float('NaN')

#Simplify name to be usable everywhere
def simple(name):
    return unidecode.unidecode(name.replace(' ','').replace('.','').replace(':','-').replace('\'',''))

#Get the number of days between two dates
def daydiff(t,tref):
    return ((t-tref).days + (t-tref).seconds/24/3600)

#Round a datetime object to any time lap in seconds (default: 1 minute)
def roundTime(t,roundTo=60,floor=False,ceil=False):
    seconds = (t.replace(tzinfo=None)-t.min).seconds
    rounding = (seconds+(0 if floor else roundTo if ceil else roundTo/2))//roundTo * roundTo
    return (t + timedelta(0,rounding-seconds,-t.microsecond))

#Compute the averages over a given time period in seconds (default: 1 day)
def timeAverage(tdata,data,period=24*60*60):
    t_round = [roundTime(t,period) for t in tdata]
    t_ceil = np.array([roundTime(t,period,ceil=True) for t in tdata])
    data_np = np.array(data)
    data_avg = [np.nan]*len(data)
    for tr in np.unique(t_round):
        data_avg[t_round.index(tr)] = np.nanmean(data_np[t_ceil==tr])
    return data_avg

#Return the day-of-year from a time vector, ignoring the 29th of February (combines it with 28th of Feb)
#Returned values range from 0 to 364 (1st of Jan to 31st of Dec)
def doy365(time):
    if type(time)!=list: time=[time]
    year = np.array([t.year for t in time])
    leapYear = np.logical_and(year%4==0,np.logical_or(year%100!=0,year%400==0))
    doy = np.array([t.timetuple().tm_yday for t in time])
    doy[np.logical_and(leapYear,doy>59)] = doy[np.logical_and(leapYear,doy>59)]-1
    return (doy-1)

#Function to get latitude from X,Y CH1903+ coordinates
def CH1903ptoLat(X,Y):
    y = float(X-2.6E6)/1E6
    x = float(Y-1.2E6)/1E6
    lat = (16.9023892 + 3.238272*x - 0.270978*y**2 - 0.002528*x**2 - 0.0447*y**2*x - 0.014*x**3)*100/36
    return lat

#Function to convert coordinates
def CH1903ptoWGS84(coord):
    coord_new = []
    for i in range(len(coord)):
        x = (float(coord[i][0])-2.6E6)/1E6
        y = (float(coord[i][1])-1.2E6)/1E6
        lon = 100/36*(2.6779094 + 4.728982*x + 0.791484*x*y + 0.1306*x*y**2 - 0.0436*x**3)
        lat = 100/36*(16.9023892 + 3.238272*y - 0.270978*x**2 - 0.002528*y**2 - 0.0447*x**2*y - 0.014*y**3)
        coord_new.append([lon,lat])
    return coord_new

#Function to compute freshwater density for given vectors of temperature and salinity
def waterDensity(T,S=None):
    if S is None: S=[0]*len(T)
    T,S = (np.array(T),np.array(S))
    rho = 1000*(0.9998395+T*(6.7914e-5+T*(-9.0894e-6+T*(1.0171e-7+T*(-1.2846e-9+T*(1.1592e-11+T*(-5.0125e-14))))))+(8.181e-4+T*(-3.85e-6+T*(4.96e-8)))*S)#Density [kg/m3]
    return list(rho)

#Function to convert salinity from µS/cm to ‰ (equivalent to ppt or PSU)
def cond2sal(cond):
    K1 =  0.0120
    K2 = -0.2174
    K3 = 25.3283
    K4 = 13.7714
    K5 = -6.4788
    K6 =  2.5842
    R = np.array(cond)/53087
    sal = K1 + K2*R**0.5 + K3*R + K4*R**1.5 + K5*R**2 + K6*R**2.5
    sal = np.round(sal,3)
    return (list(sal) if (type(cond) is list) else sal)

#Function to compute the clear sky solar radiation (source: Heat Flux Analyzer, unless otherwise specified)
def clearSkySolRad(time,pair,vap,lat):
    if type(time) is not list: time=[time]
    vap = np.array(vap)
    lat = lat*np.pi/180
    hr = np.array([t.hour+t.minute/60+t.second/3600 for t in time])
    doy = doy365(time) + hr/24
    doy_winter = doy+10
    doy_winter[doy_winter>=365.24] = doy_winter[doy_winter>=365.24]-365.24
    phi = np.arcsin(-0.39779*np.cos(2*np.pi/365.24*doy_winter)) #Declination of the sun (Wikipedia)
    cosZ = np.sin(lat)*np.sin(phi)+np.cos(lat)*np.cos(phi)*np.cos(np.pi/12*(hr-12.5)) #Cosine of the solar zenith angle (Wikipedia); 13.5 is the average solar noon (13h30)
    cosZ[cosZ<0] = 0
    m = 35*cosZ*(1244*cosZ**2+1)**-0.5 #Air mass thickness coefficient
    G = np.interp(doy,[15,105,196,288],[2.7,2.95,2.77,2.71]) #Empirical constant for latitudes 40-50°
    Td = (243.5*np.log(vap/6.112))/(17.67-np.log(vap/6.112))+33.8 #Dew point temperature [°C]
    pw = np.exp(0.1133-np.log(G+1)+0.0393*(1.8*Td+32)) #Precipitable water
    Tw = 1-0.077*(pw*m)**0.3 #Attenuation coefficient for water vapour
    Ta = 0.935**m #Attenuation coefficient for aerosols
    TrTpg = 1.021-0.084*(m*(0.000949*pair+0.051))**0.5 #Attenuation coefficient for Rayleigh scattering and permanent gases
    Ieff = 1353*(1+0.034*np.cos(2*np.pi/365.24*doy)) #Effective solar constant
    return Ieff*cosZ*TrTpg*Tw*Ta

def midValue(vec):
    mid = (np.array(vec)[:-1]+np.array(vec)[1:])/2
    if type(vec) is list: mid = list(mid)
    return mid

#Function to compute a vector of surface areas for given depth vector and lake properties dictionary
def getAreas(z,lakeProperties):
    if 'Bathymetry' in lakeProperties: #Bathymetry available: interpolate
        areas = np.interp(np.abs(z),lakeProperties['Bathymetry']['Depth [m]'],lakeProperties['Bathymetry']['Area [m2]'])
    else: #No bathymetry available: just use surface and bottom, and interpolate
        areas = np.interp(np.abs(z),[0,lakeProperties['Max depth [m]']],[lakeProperties['Surface [km2]']*1E6,0])
    return areas

#Function to compute the mean temperature for a given temperature profile
def meanTemperature(z,areas,T):
    z,areas,T = (np.array(z),np.array(areas),np.array(T))
    Tavg = sum(midValue(areas*T)*np.abs(np.diff(z)))/sum(midValue(areas)*np.abs(np.diff(z))) #Mean temperature [°C]
    return round(Tavg,2)

#Function to compute the heat content for a given temperature profile
def heatContent(z,areas,T,S=None):
    if S is None: S=[0]*len(z)
    z,areas,T,S = (np.array(z),np.array(areas),np.array(T),np.array(S))
    rho = waterDensity(T,S)
    hc = sum(4183*midValue(areas*rho*T)*np.abs(np.diff(z))) #Heat content [J] (e.g. Weinberger and Vetter 2014)
    if hc!=0: hc=np.round(hc,3-int(np.log10(np.abs(hc)))) #Round to 4 significant figures
    return hc

#Function to compute the Schmidt stability for a given temperature-salinity profile
def schmidtStability(z,areas,T,S=None):
    if S is None: S=[0]*len(z)
    z,areas,T,S = (np.abs(z),np.array(areas),np.array(T),np.array(S))
    rho = waterDensity(T,S)
    volume = sum(midValue(areas)*np.abs(np.diff(z))) #Lake volume from bathymetry [m3]
    zv = 1/volume*sum(midValue(z*areas)*np.abs(np.diff(z))) #Centre of volume [m]
    St = 9.81/max(areas)*sum(midValue((z-zv)*rho*areas)*np.abs(np.diff(z))) #Schmidt stability [J/m^2] (e.g. Kirillin and Shatwell 2016)
    if St!=0: St=np.round(St,3-int(np.log10(np.abs(St)))) #Round to 4 significant figures
    return St

#Function to compute the start and end dates of stratification for a given temperature-salinity array, given a Schmidt stability threshold
def getStratificationStartEnd(t,z,areas,T,S=None):
    Stmin = 10
    if S is None: S=[0]*len(z)
    z,T,S = (np.abs(z),np.array(T),np.array(S))
    St_vol = [schmidtStability(z,areas,Tp,S)/max(z) for Tp in T]
    tstart = []
    tend = []
    stratified = False if St_vol[0]<Stmin else True
    k=1
    while k<len(t):
        if stratified:
            while St_vol[k]>=Stmin:
                k=k+1
                if k==len(t): return (tstart, tend)
            if (len(tstart)==0 or (len(tstart)>0 and t[k]-tstart[-1]>timedelta(60))): #Make sure this is the real end of the stratification season
                tend.append(t[k])
            else: #False end: look for next stratification start during this year
                tstart = tstart[:-1]
            stratified = False
        else:
            while St_vol[k]<Stmin:
                k=k+1
                if k==len(t): return (tstart, tend)
            tstart.append(t[k])
            stratified = True
        k=k+1    
    return (tstart,tend)

#Function to compute the start and end dates of ice cover for a given time series of ice thickness
def getIceCoverStartEnd(t,ice):
    ice = np.array(ice)
    tstart = []
    tend = []
    iceCovered = True if ice[0]>0 else False
    k=1
    while k<len(t):
        if iceCovered:
            while ice[k]>0:
                k=k+1
                if k==len(t): return (tstart, tend)
            tend.append(t[k])
            iceCovered = False
        else:
            while ice[k]==0:
                k=k+1
                if k==len(t): return (tstart, tend)
            if (len(tend)==0 or (len(tend)>0 and t[k]-tend[-1]>timedelta(60))): #Make sure this is a new ice season
                tstart.append(t[k])
            else: #False end (not the last one for this ice season)
                tend = tend[:-1]
            iceCovered = True
        k=k+1
    return (tstart,tend)

#Function to convert a temperature to a hexadecimal color for KML file
def colorT(T):
    if T is None or np.isnan(T):
        R = hex(75)[2:4]
        G = hex(75)[2:4]
        B = hex(75)[2:4]
        trans = '33'
    elif T<4:
        R = hex(int(round(min(max(0,255-63.75*T),255))))[2:4]
        G = hex(int(round(min(max(0,255-18.75*T),255))))[2:4]
        B = hex(int(round(min(max(0,255-6.75*T),255))))[2:4]
        trans = 'dd'
    else:
        R = hex(int(round(min(max(0,9.5*(T-4)),255))))[2:4]
        G = hex(int(round(min(max(0,180-7*(T-4)),255))))[2:4]
        B = hex(int(round(min(max(0,228-11.4*(T-4)),255))))[2:4]
        trans = 'dd'
    return ('%s%2s%2s%2s' % (trans,B,G,R)).replace(' ','0')

#Replace the missing values in a time series with the day-of-year means from the available values
def completeData(time,val):
    doy = doy365(time)
    val = np.array(val)
    valdoy = np.array([np.nanmean(val[doy==d]) for d in range(366)])
    val[np.isnan(val)] = valdoy[doy[np.isnan(val)]]
    return list(val)

#Replace the NaNs in vector y:
#   -Linearly interpolate for gaps smaller than "tgap" items
#   -Set larger gaps and left- and right-side values to "valdef" (to the mean value if valdef='mean')
def interpNaN(x,y,tgap,valdef=np.nan):
    if type(x[0]) is datetime:
        x = [calendar.timegm(t.timetuple()) for t in x]
    x = np.array(x)
    if np.any(np.diff(x)<0):
        raise Exception('Error: x-values for interpolation must be increasing. Interpolation not performed.')
    y = np.array(y)
    nans = np.isnan(y)
    #Find out length of NaN-series
    nanlims = np.diff(np.hstack(([0],nans*1,[0])))
    nanlen_short = np.where(nanlims<0)[0]-np.where(nanlims>0)[0]
    nanlen = np.zeros(len(nans))
    idx = 0
    wait = True
    for k in range(len(nans)):
        if nans[k]:
            nanlen[k] = nanlen_short[idx]
            wait = False
        elif not wait:
            idx = idx+1
            wait = True
    nans_rep = np.logical_and(nanlen>0,nanlen<tgap)
    y[nans_rep] = np.interp(x[nans_rep],x[~nans_rep],y[~nans_rep])
    if valdef=='mean': valdef=np.nanmean(y)
    nans = np.isnan(y)
    nans[0:list(nans).index(False)] = [False]*list(nans).index(False) #Don't replace the left-side NaNs
    y[nans] = valdef
    return y.tolist()


#Read the CTD profiles at a given location
def readObs(path,XY):
    fid = open(os.path.join(path,'CTD.csv'))
    obsrd = csv.reader(fid,delimiter=';')
    next(obsrd)
    Observations = {}
    Observations['Time'] = []
    Observations['Depth [m]'] = []
    Observations['Temperature [°C]'] = []
    Observations['Salinity [ppt]'] = []
    for row in obsrd:
        coord = [float(row[3].split(' / ')[0])+2E6,float(row[3].split(' / ')[1])+1E6]
        dist = (coord[0]-XY[0])**2+(coord[1]-XY[1])**2
        if dist<1E6:
            Observations['Time'].append(datetime.strptime(row[5],'%d.%m.%Y %H:%M'))
            Observations['Depth [m]'].append(float(row[6]))
            val = toFloat(row[8]) #Temperature: in degree Celsius
            Observations['Temperature [°C]'].append(val)
            val = cond2sal(toFloat(row[7])) #Salinity: convert from µS/cm to ‰ (equivalent to ppt or PSU)
            Observations['Salinity [ppt]'].append(val)
    fid.close()
    if len(Observations['Time'])==0:
        return None
    writelog('\tObservations: %s - %s, from %g to %g m (nb profiles: %d)\n' % (datetime.strftime(Observations['Time'][0],'%d.%m.%Y'),datetime.strftime(Observations['Time'][-1],'%d.%m.%Y'),min(Observations['Depth [m]']),max(Observations['Depth [m]']),len(np.unique(Observations['Time'])))) 
    return Observations

#Write two text files containing a table of temperature observations and of salinity observations
def writeObs(Lake,Observations,Ice=None):
    lakeName = simple(Lake['Name'])
    fidT = open(os.path.join('Observations',lakeName+'_T.txt'),'w')
    fidS = open(os.path.join('Observations',lakeName+'_S.txt'),'w')
    time = np.array(Observations['Time'])
    time_unique = list(sorted(set(time)))
    depths = np.array(Observations['Depth [m]'])
    depths_unique = list(sorted(set(depths)))
    z_rem = []
    for z in depths_unique: #Remove depths which appear rarely
        if sum(depths==z)/len(time_unique) < 0.4:
            z_rem.append(z)
    [depths_unique.remove(z) for z in z_rem];
    fidT.write('Time,' + ','.join([('%g' % -z) for z in depths_unique]) + '\n')
    fidS.write('Time,' + ','.join([('%g' % -z) for z in depths_unique]) + '\n')
    Temp = np.array(Observations['Temperature [°C]'])
    Sal = np.array(Observations['Salinity [ppt]'])
    for t in time_unique:
        lineT = datetime.strftime(t,'%Y-%m-%d %H:%M')
        lineS = datetime.strftime(t,'%Y-%m-%d %H:%M')
        for z in depths_unique:
            if any((time==t) & (depths==z)):
                lineT = lineT + ',' + ('%.2f' % Temp[(time==t) & (depths==z)][0])
                lineS = lineS + ',' + ('%.3f' % Sal[(time==t) & (depths==z)][0])
            else:
                lineT = lineT + ',' + 'nan'
                lineS = lineS + ',' + 'nan'
        fidT.write(lineT+'\n')
        fidS.write(lineS+'\n')
    if Ice is not None:
        fidIce = open(os.path.join('Observations',lakeName+'_IceH.txt'),'w')
        time = np.array(Ice['Time'])
        time_unique = list(sorted(set(time)))
        fidIce.write('Time,0\n')
        iceH = np.array(Ice['Ice thickness [m]'])
        for t in time_unique:
            fidIce.write(datetime.strftime(t,'%Y-%m-%d %H:%M')+','+('%.3f' % iceH[time==t][0])+'\n')

#Read the ice thickness at a given location
def readIce(path,XY):
    fid = open(os.path.join(path,'Ice.csv'))
    obsrd = csv.reader(fid,delimiter=';')
    next(obsrd)
    Observations = {}
    Observations['Time'] = []
    Observations['Ice thickness [m]'] = []
    for row in obsrd:
        coord = [float(row[1])+2E6,float(row[2])+1E6]
        dist = (coord[0]-XY[0])**2+(coord[1]-XY[1])**2
        if dist<1E6:
            Observations['Time'].append(datetime.strptime(row[3],'%d.%m.%Y %H:%M'))
            val = toFloat(row[4]) #Ice thickness: in meters
            Observations['Ice thickness [m]'].append(val) #Ice thickness [m]
    fid.close()
    if len(Observations['Time'])==0:
        return None
    writelog('\tIce thickness observations: %s - %s (nb values: %d)\n' % (datetime.strftime(Observations['Time'][0],'%d.%m.%Y'),datetime.strftime(Observations['Time'][-1],'%d.%m.%Y'),len(np.unique(Observations['Time']))))
    return Observations

#Read the Secchi depths at a given location, convert them to absorption coefficient
def readSecchi(path,XY):
    fid = open(os.path.join(path,'Secchi.csv'))
    obsrd = csv.reader(fid,delimiter=';')
    next(obsrd)
    Observations = {}
    Observations['Time'] = []
    Observations['Depth [m]'] = []
    Observations['Data'] = []
    for row in obsrd:
        coord = [float(row[1])+2E6,float(row[2])+1E6]
        dist = (coord[0]-XY[0])**2+(coord[1]-XY[1])**2
        if dist<1E6:
            Observations['Time'].append(datetime.strptime(row[3],'%d.%m.%Y %H:%M'))
            Observations['Depth [m]'].append(0)
            val = toFloat(row[4]) #Secchi depth: in meters
            Observations['Data'].append(round(1.7/val,3)) #Absorption coefficient [m^-1]
    fid.close()
    if len(Observations['Time'])==0:
        return None
    writelog('\tSecchi depth: %s - %s (nb values: %d)\n' % (datetime.strftime(Observations['Time'][0],'%d.%m.%Y'),datetime.strftime(Observations['Time'][-1],'%d.%m.%Y'),len(np.unique(Observations['Time']))))
    return {'Absorption [m^-1]': Observations}


#Read the meteorological data for given stations
def readMeteo(path,stations):
    #Define the meteo variables
    pars = json.load(codecs.open('Meteo.json','r','utf-8'))
    par_names = [par['Name'] for par in pars]
    par_ids = [par['Short name'] for par in pars]
    Meteo = {}
    #Read the files
    for station in stations:
        fid = open(os.path.join(path,station+'_data.txt'))
        meteord = csv.reader(fid,delimiter=';')
        Meteo[station] = {}
        for row in meteord:
            if len(row)<=1:
                continue
            elif row[0]=='stn':
                par_index = [par_ids.index(par) for par in row[2:]]
                for par_id in row[2:]:
                    par_name = par_names[par_ids.index(par_id)]
                    Meteo[station][par_name] = {}
                    Meteo[station][par_name]['Time'] = []
                    Meteo[station][par_name]['Data'] = []
            else:
                for ip,par_val in enumerate(row[2:]):
                    par_name = par_names[par_index[ip]]
                    tformat = ('%Y%m%d%H' if len(row[1])==10 else '%Y%m%d')
                    time = datetime.strptime(row[1],tformat)
                    val = toFloat(par_val)
                    Meteo[station][par_name]['Time'].append(time)
                    Meteo[station][par_name]['Data'].append(val)
        for par in Meteo[station]:
            time = np.array(Meteo[station][par]['Time'])
            tvalid = time[np.invert(np.isnan(Meteo[station][par]['Data']))]
            if par=='Air temperature [°C]': #Trim the observations period to remove missing values on the extremities
                trange = np.logical_and(time>=tvalid[0],time<=tvalid[-1])
                Meteo[station][par]['Time'] = list(np.array(Meteo[station][par]['Time'])[trange])
                Meteo[station][par]['Data'] = list(np.array(Meteo[station][par]['Data'])[trange])
            writelog('\t\tStation %s: %23s: %s - %s (max gap: %.1f days)\n' % (station,par,datetime.strftime(tvalid[0],'%d.%m.%Y'),datetime.strftime(tvalid[-1],'%d.%m.%Y'),max(np.diff(tvalid)).days))
        fid.close()
        for line in open(os.path.join(path,station+'_legend.txt')):
            if line[0:3]==station:
                Meteo[station]['Elevation [m]'] = float(line[157:])
                break
    return Meteo

#Complete the basis meteo station with data from other stations to obtain more complete and longer time series
def combineMeteo(data,lakeElevation,t_ref,t_end):
    #Define the meteo variables
    pars = json.load(codecs.open('Meteo.json','r','utf-8'))
    par_names = [par['Name'] for par in pars]
    Meteo = {}
    for par in par_names:
        for station in data:
            if par in data[station]:
                if par not in Meteo: #Parameter not yet there: just add it
                    Meteo[par] = dict(data[station][par])
                    if par=='Air temperature [°C]': #Correct temperature for elevation 
                        Meteo[par]['Data'] = list(np.array(Meteo[par]['Data'])-0.0065*(lakeElevation-data[station]['Elevation [m]']))
                    writelog('\t\t%s: Using data from station %s.' % (par,station))
                else: #Parameter already there: complete with additional data
                    #First add missing times and values
                    time_add = sorted(list(set(data[station][par]['Time'])-set(Meteo[par]['Time'])))
                    if len(time_add)>0:
                        bool_bef = (np.array(data[station][par]['Time'])<min(Meteo[par]['Time']))
                        ib = len(bool_bef)-np.argmax(bool_bef[::-1]) #Index of last time before old time
                        if not any(bool_bef): ib=0
                        bool_aft = (np.array(data[station][par]['Time'])>max(Meteo[par]['Time']))
                        ia = np.argmax(bool_aft)+1 #Index of first time after old time
                        if not any(bool_aft): ia=len(bool_aft)
                        time_add_np = np.array(time_add)
                        bool_mid = np.in1d(data[station][par]['Time'][ib:ia],time_add_np[np.logical_and(time_add_np>min(Meteo[par]['Time']),time_add_np<max(Meteo[par]['Time']))],assume_unique=True)
                        bool_add = np.logical_or(bool_bef,bool_aft)
                        bool_add = np.logical_or(bool_add,np.concatenate(([False]*ib,bool_mid,[False]*(len(bool_aft)-ia))))
                        #bool_add = np.in1d(data[station][par]['Time'],time_add,assume_unique=True)
                        val_add = list(np.array(data[station][par]['Data'])[bool_add])
                        #val_add = [data[station][par]['Data'][i] for i in range(len(data[station][par]['Time'])) if data[station][par]['Time'][i] in time_add]
                        if par=='Air temperature [°C]': #Correct temperature for elevation
                            val_add = list(np.array(val_add)-0.0065*(lakeElevation-data[station]['Elevation [m]']))
                        Meteo[par]['Time'].extend(time_add)
                        Meteo[par]['Data'].extend(val_add)
                    #Secondly replace NaN values
                    time_valid = np.array(Meteo[par]['Time'])[np.invert(np.isnan(Meteo[par]['Data']))]
                    time_valid_new = np.array(data[station][par]['Time'])[np.invert(np.isnan(data[station][par]['Data']))]
                    time_replace = sorted(list(set(time_valid_new)-set(time_valid)))
                    for t in time_replace:
                        val_replace = data[station][par]['Data'][data[station][par]['Time'].index(t)]
                        Meteo[par]['Data'][Meteo[par]['Time'].index(t)] = val_replace
                    writelog(' Completing with data from station %s.' % station)
                    time_valid = np.array(Meteo[par]['Time'])[np.invert(np.isnan(Meteo[par]['Data']))]
        if par in Meteo:
            writelog('\n')
            Meteo[par]['Time'],Meteo[par]['Data'] = [list(tpl) for tpl in zip(*sorted(zip(Meteo[par]['Time'],Meteo[par]['Data'])))]
            tvalid = np.array(Meteo[par]['Time'])[np.invert(np.isnan(Meteo[par]['Data']))]
            writelog('\t\t\t%23s %s - %s (max gap: %.1f days)\n' % (' '*len(par),datetime.strftime(tvalid[0],'%d.%m.%Y'),datetime.strftime(tvalid[-1],'%d.%m.%Y'),max(np.diff(tvalid)).days))
        else:
            writelog('\t\t%s: No data.' % par)
            #if par=='Cloud cover [%]':                
            #    Meteo[par] = {'Time':[t_ref,t_end],'Data':[50,50]}
            writelog('\n')
    return Meteo

#Correct the meteo data, adapt it to Simstrat, and interpolate missing values if possible
def adaptMeteo(data,lake):
    #Convert wind speed and angle to wind in X- and Y-directions
    theta = np.array(data['Wind direction [°]']['Data'])
    theta_avg = np.arctan2(np.mean(np.sin(theta)),np.mean(np.cos(theta)))
    if data['Wind speed [m/s]']['Time'][0]!=data['Wind direction [°]']['Time'][0] or len(data['Wind speed [m/s]']['Time'])!=len(data['Wind direction [°]']['Time']):
        writelog('\t\tThe timeframes of wind speed and direction do not match. They will be homogenized (time-intensive).\n')
        theta = [theta[it] if tw in data['Wind direction [°]']['Time'] else theta_avg for it,tw in enumerate(data['Wind speed [m/s]']['Time'])]
    theta = [th if not np.isnan(th) else theta_avg for th in theta]
    wind = data['Wind speed [m/s]']['Data']
    wind = [0 if ws<0 else ws for ws in wind]
    wind = [np.nan if ws>20 else ws for ws in wind]
    data['Wind direction [°]']['Time'] = data['Wind speed [m/s]']['Time']
    data['Wind speed [m/s]']['Data'] = [-wind[k]*np.sin(theta[k]*np.pi/180) for k in range(len(wind))]
    data['Wind X [m/s]'] = data.pop('Wind speed [m/s]')
    data['Wind direction [°]']['Data'] = [-wind[k]*np.cos(theta[k]*np.pi/180) for k in range(len(wind))]
    data['Wind Y [m/s]'] = data.pop('Wind direction [°]')
    #Adapt units and perform basic checks
    data['Vapour pressure [mbar]'] = data.pop('Vapour pressure [hpa]')
    data['Vapour pressure [mbar]']['Data'] = [np.nan if val<1.0 else val for val in data['Vapour pressure [mbar]']['Data']]
    data['Solar radiation [W/m^2]']['Data'] = [0 if val<0 else val for val in data['Solar radiation [W/m^2]']['Data']]
    data['Precipitation [mm]']['Data'] = [0 if val<0 else val*0.001 for val in data['Precipitation [mm]']['Data']]
    data['Precipitation [m/hr]'] = data.pop('Precipitation [mm]')
    #Complete cloud cover data
    if 'Cloud cover [%]' not in data:
        data['Cloud cover [%]'] = {'Time':data['Solar radiation [W/m^2]']['Time'],'Data':[np.nan]*len(data['Solar radiation [W/m^2]']['Time'])}
    else:
        data['Cloud cover [%]']['Data'] = interpNaN(data['Cloud cover [%]']['Time'],data['Cloud cover [%]']['Data'],24)
    if any(np.isnan(data['Cloud cover [%]']['Data'])):
        #Estimate cloudiness based on ratio between measured and theoretical solar radiation
        writelog('\t\tEstimating missing cloud cover data based on theoretical and measured solar radiation.\n')
        cssr = clearSkySolRad(data['Solar radiation [W/m^2]']['Time'],1013.25*np.exp((-9.81*0.029*lake['Properties']['Elevation [m]'])/(8.314*283.15)),data['Vapour pressure [mbar]']['Data'],CH1903ptoLat(lake['X [m]'],lake['Y [m]']))
        #Better estimate by using daily averages, giving daily values for cloud cover (at noon)
        cssr = np.array(timeAverage(data['Solar radiation [W/m^2]']['Time'],cssr,24*60*60))
        solrad = np.array(timeAverage(data['Solar radiation [W/m^2]']['Time'],data['Solar radiation [W/m^2]']['Data'],24*60*60))
        cloud = 100*(1-solrad/(0.9*cssr))
        cloud[cloud<0] = 0
        cloud = interpNaN(data['Solar radiation [W/m^2]']['Time'],cloud,len(cloud))
        tsol = [calendar.timegm(t.timetuple()) for t in data['Solar radiation [W/m^2]']['Time']]
        nans = np.isnan(data['Cloud cover [%]']['Data'])
        tnans = np.array([calendar.timegm(t.timetuple()) for t in data['Cloud cover [%]']['Time']])[nans]
        cc = np.array(data['Cloud cover [%]']['Data'])
        cc[nans] = np.interp(tnans,tsol,cloud)
        data['Cloud cover [%]']['Data'] = list(cc)
    #Adapt units and perform basic checks
    data['Cloud cover [%]']['Data'] = [cc*0.01 if not np.isnan(cc) else 0.5 for cc in data['Cloud cover [%]']['Data']]
    data['Cloud cover [-]'] = data.pop('Cloud cover [%]')
    #Data-patching to fill the gaps
    for var in data:
        if any(np.isnan(data[var]['Data'])):
            if var in ['Air temperature [°C]','Solar radiation [W/m^2]','Vapour pressure [mbar]']: #Seasonal variables: first interpolate small gaps, then complete larger gaps with day-of-year means
                data[var]['Data'] = interpNaN(data[var]['Time'],data[var]['Data'],2*24)
                data[var]['Data'] = completeData(data[var]['Time'],data[var]['Data'])
            elif var in ['Wind X [m/s]','Wind Y [m/s]']: #Random variables (wind): interpolate small gaps, and complete larger gaps with overall mean (same in X- and Y-directions)
                data[var]['Data'] = interpNaN(data[var]['Time'],data[var]['Data'],7*24,np.nanmean(wind)/np.sqrt(2))
            elif var in ['Precipitation [m/hr]']: #Random variables (precipitation): interpolate small gaps, and complete larger gaps with overall mean
                data[var]['Data'] = interpNaN(data[var]['Time'],data[var]['Data'],7*24,np.nanmean(data[var]['Data']))
    return data

#Read the meteorological data for a given station
def readHydro(path,station):
    #Define the hydro variables
    par_ids = ['Q','T']
    par_names = ['Flowrate [m^3/s]','Temperature [°C]']
    #Read the files
    Hydro = {}
    for par in range(len(par_ids)):
        if (par==1 and not os.path.exists(os.path.join(path,par_ids[par]+'_'+station+'_Stundenmittel.asc'))):
            writelog('\t\tStation %s - No temperature data in folder %s\\\n' % (station,path))
            break
        fid = open(os.path.join(path,par_ids[par]+'_'+station+'_Stundenmittel.asc'))
        hydrord = csv.reader(fid,delimiter=';')
        Hydro['Station'] = station
        Hydro[par_names[par]] = {}
        Hydro[par_names[par]]['Time'] = []
        Hydro[par_names[par]]['Data'] = []
        for row in hydrord:
            if len(row)<3:
                continue
            else:
                #for ind in range(1,len(row)):
                tformat = '%Y.%m.%d %H'
                time = datetime.strptime(row[1][0:13],tformat)
                val = toFloat(row[2])
                Hydro[par_names[par]]['Time'].append(time)
                Hydro[par_names[par]]['Data'].append(val)
        tvalid = np.array(Hydro[par_names[par]]['Time'])[np.invert(np.isnan(Hydro[par_names[par]]['Data']))]
        writelog('\t\tStation %s - %s: %s - %s (max gap: %.1f days)\n' % (station,par_names[par],datetime.strftime(tvalid[0],'%d.%m.%Y'),datetime.strftime(tvalid[-1],'%d.%m.%Y'),max(np.diff(tvalid)).days))
        fid.close()
    return Hydro

#Complete the hydro stations having a shorter time series and aggregate all inflows in a single one
def combineInflow(data,tmin,tmax):
    time_nb = np.linspace(calendar.timegm(tmin.timetuple()),calendar.timegm(tmax.timetuple()),num=daydiff(tmax,tmin)*24+1)
    time = [datetime.fromtimestamp(t) for t in time_nb]
    Q = []
    for k in range(len(data)):
        t_nb = [calendar.timegm(t.timetuple()) for t in data[k]['Flowrate [m^3/s]']['Time']]
        Q.append(np.interp(time_nb,t_nb,data[k]['Flowrate [m^3/s]']['Data'],left=np.nan,right=np.nan))
        #First interpolate small gaps, then replace missing flowrate values by the average on that day-of-year
        if any(np.isnan(Q[-1])):
            writelog('\t\tFlowrate data from %s does not cover the entire time series. Completing based on available data.\n' % ('station '+data[k]['Station'] if 'Station' in data[k] else 'upstream lake'))
            Q[-1] = interpNaN(time,Q[-1],5*24)
            Q[-1] = completeData(time,Q[-1])
    T = []
    for k in range(len(data)):
        if 'Temperature [°C]' not in data[k]: continue
        t_nb = [calendar.timegm(t.timetuple()) for t in data[k]['Temperature [°C]']['Time']]
        T.append(np.interp(time_nb,t_nb,data[k]['Temperature [°C]']['Data'],left=np.nan,right=np.nan))
        #Replace missing temperature values by the average on that day-of-year
        if any(np.isnan(T[-1])):
            writelog('\t\tTemperature data from %s does not cover the entire time series. Completing based on available data.\n' % ('station '+data[k]['Station'] if 'Station' in data[k] else 'upstream lake'))
            T[-1] = completeData(time,T[-1])
    S = []
    for k in range(len(data)):
        if 'Salinity [ppt]' not in data[k]: continue
        t_nb = [calendar.timegm(t.timetuple()) for t in data[k]['Salinity [ppt]']['Time']]
        S.append(np.interp(time_nb,t_nb,data[k]['Salinity [ppt]']['Data'],left=np.nan,right=np.nan))
        #Replace missing salinity values by the average on that day-of-year
        if any(np.isnan(S[-1])):
            writelog('\t\tSalinity data from %s does not cover the entire time series. Completing based on available data.\n' % ('station '+data[k]['Station'] if 'Station' in data[k] else 'upstream lake'))
            S[-1] = completeData(time,S[-1])
    if Q==[]:
        writelog('\t\tNo flowrate data for any inflow. Inflows and outflows will be ignored.\n')
        return None
    if T==[]:
        writelog('\t\tNo temperature data for any inflow. Inflows and outflows will be ignored.\n')
        return None
    Q_final = [sum([Q[k][t] for k in range(len(Q))]) for t in range(len(time_nb))]
    Q_final = [0 if qf<0 else qf for qf in Q_final]
    T_final = [(sum([T[k][t]*Q[k][t] for k in range(len(T))])/Q_final[t] if Q_final[t]>0 else 0) for t in range(len(time_nb))]
    T_final = [0 if tf<0 else tf for tf in T_final]
    S_final = [(sum([S[k][t]*Q[k][t] for k in range(len(S))])/Q_final[t] if Q_final[t]>0 else 0) for t in range(len(time_nb))]
    S_final = [0 if tf<0 else tf for tf in S_final]
    Inflow = {'Flowrate [m^3/s]':{'Time':time,'Data':Q_final},'Temperature [°C]':{'Time':time,'Data':T_final},'Salinity [ppt]':{'Time':time,'Data':S_final}}
    return Inflow

#Combine the fixed inflows with different depths/times
def combineInflowFixed(data,tmin,tmax):
    time_nb = np.linspace(calendar.timegm(tmin.timetuple()),calendar.timegm(tmax.timetuple()),num=daydiff(tmax,tmin)*24+1)
    time = [datetime.fromtimestamp(t) for t in time_nb]
    zmax = max([inflow['Inflow depth [m]'] for inflow in data])
    Q = []
    for k in range(len(data)):
        t_nb = [calendar.timegm(t.timetuple()) for t in data[k]['Flowrate [m^3/s]']['Time']]
        Q.append(np.interp(time_nb,t_nb,data[k]['Flowrate [m^3/s]']['Data'],left=np.nan,right=np.nan))
        #First interpolate small gaps, then replace missing flowrate values by the average on that day-of-year
        if any(np.isnan(Q[-1])):
            writelog('\t\tFlowrate data from upstream lake does not cover the entire time series. Completing based on available data.\n')
            Q[-1] = interpNaN(time,Q[-1],5*24)
            Q[-1] = completeData(time,Q[-1])
    T,S = [],[]
    for k in range(len(data)):
        t_nb = [calendar.timegm(t.timetuple()) for t in data[k]['Temperature [°C]']['Time']]
        T.append(np.array([list(np.interp(time_nb,t_nb,Ti,left=np.nan,right=np.nan)) for Ti in data[k]['Temperature [°C]']['Data']]))
        #Replace missing temperature values by the average on that day-of-year
        for it in range(len(T[-1])):
            if any(np.isnan(T[-1][it])):
                if it==0: writelog('\t\tTemperature data from upstream lake does not cover the entire time series. Completing based on available data.\n')
                T[-1][it] = completeData(time,T[-1][it])
        t_nb = [calendar.timegm(t.timetuple()) for t in data[k]['Salinity [ppt]']['Time']]
        S.append(np.array([list(np.interp(time_nb,t_nb,Si,left=np.nan,right=np.nan)) for Si in data[k]['Salinity [ppt]']['Data']]))
        #Replace missing salinity values by the average on that day-of-year
        for it in range(len(S[-1])):
            if any(np.isnan(S[-1][it])):
                if it==0: writelog('\t\tSalinity data from upstream lake does not cover the entire time series. Completing based on available data.\n')
                S[-1][it] = completeData(time,S[-1][it])
    if Q==[]:
        writelog('\t\tNo flowrate data for any inflow. Inflows and outflows will be ignored.\n')
        return None
    if T==[]:
        writelog('\t\tNo temperature data for any inflow. Inflows and outflows will be ignored.\n')
        return None
    Q_final,T_final,S_final = [[]]*len(data),[[]]*len(data),[[]]*len(data)
    z = np.arange(np.max(np.concatenate([inflow['Temperature [°C]']['Depth [m]'] for inflow in data]))+2)[::-1]
    for k in range(len(data)):
        for it in range(len(time)):
            Q_final[k].append(np.interp(z,[0,zmax-1,zmax],[Q[k][it]/(zmax-.5),Q[k][it]/(zmax-.5),0]))
            T_final[k].append(np.interp(z,data[k]['Temperature [°C]']['Depth [m]'],[T[k][iz][it] for iz in range(len(T[k]))]))
            S_final[k].append(np.interp(z,data[k]['Salinity [ppt]']['Depth [m]'],[S[k][iz][it] for iz in range(len(S[k]))]))
        Q_final[k],T_final[k],S_final[k] = np.array(Q_final[k]),np.array(T_final[k]),np.array(S_final[k])
    T_final = sum([Q_final[k]*T_final[k] for k in range(len(data))])
    S_final = sum([Q_final[k]*S_final[k] for k in range(len(data))])
    Q_final = sum(Q_final)
    Inflow = {'Flowrate [m^3/s]':{'Time':time,'Data':Q_final},'Temperature [°C]':{'Time':time,'Data':T_final},'Salinity [ppt]':{'Time':time,'Data':S_final},'Depths [m]':-z}
    return Inflow

#Get the longest possible timeframe of the model based on available input data
def getTimeframe(data):
    tframe = []
    for d in data:
        if d==None: continue
        for var in d:
            if 'Time' in d[var] and 'Data' in d[var]:
                tval = [d[var]['Time'][idx] for idx in range(len(d[var]['Time'])) if not np.all(np.isnan(d[var]['Data'][idx]))]
                tframe.append([var,tval[0],tval[-1]])
    return tframe

#Write the bathymetry file for Simstrat
def writeBathy(bathy,file):
    fid = open(file,'w',encoding='utf-8')
    fid.write('%s    %s\n' % ('Depth [m]','Area [m^2]'))
    for k in range(len(bathy['Depth [m]'])):
        fid.write('%6.1f    %9.0f\n' % (-abs(bathy['Depth [m]'][k]),bathy['Area [m^2]'][k]))
    fid.close()

#Create a proxy temperature profile at given depths, day-of-year and elevation
def getTemperatureProfile(z,doy,elevation):
    zp =  [0, 10, 20, 30, 40, 50,100,150,200,300]
    #Typical temperature profile at 500m elevation
    Tp500 = [[5.5,5.5,5.0,5.0,5.0,4.5,4.5,4.5,4.5,4.5], #~Jan 1st
             [8.,6.0,5.0,5.0,5.0,4.5,4.5,4.5,4.5,4.5], #~Apr 1st
             [20.,18.,14.,8.0,6.0,4.5,4.5,4.5,4.5,4.5], #~Jul 1st
             [9.5,9.5,9.0,8.0,7.0,5.0,4.5,4.5,4.5,4.5], #~Oct 1st
             [5.5,5.5,5.0,5.0,5.0,4.5,4.5,4.5,4.5,4.5]] #~Dec 31st
    #Typical temperature profiles at 1500m elevation
    Tp1500 = [[0.0,2.5,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0], #~Jan 1st
              [0.0,2.5,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0], #~Apr 1st
              [14.,9.0,6.0,4.5,4.0,4.0,4.0,4.0,4.0,4.0], #~Jul 1st
              [8.0,8.0,7.0,6.0,5.0,4.0,4.0,4.0,4.0,4.0], #~Oct 1st
              [0.0,2.5,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0]] #~Dec 31st
    T500 = np.concatenate([np.interp([doy],[0,91,182,273,365],[Tp500[0][k],Tp500[1][k],Tp500[2][k],Tp500[3][k],Tp500[4][k]]) for k in range(len(zp))])
    T1500 = np.concatenate([np.interp([doy],[0,91,182,273,365],[Tp1500[0][k],Tp1500[1][k],Tp1500[2][k],Tp1500[3][k],Tp1500[4][k]]) for k in range(len(zp))])
    T = np.concatenate([np.interp([elevation],[500,1500],[T500[k],T1500[k]]) for k in range(len(zp))])
    T = np.interp(z,zp,T)
    return T

#Write the initial conditions file for Simstrat
def writeInitialConditions(obs,time,file,elevation,maxDepth):
    doy = doy365(time)
    if obs is None:
        writelog('\tNo observation available. A proxy/default profile is used as initial conditions.\n')
        z = np.array([0,10,20,30,40,50,100,150,200,300])
        z = np.append(z[z<maxDepth],maxDepth)
        T = getTemperatureProfile(z,doy[0],elevation) #Proxy temperature profile at given day-of-year and altitude
        S = [0.15]*len(z) #Proxy salinity profile at given day-of-year and altitude
    else:
        #Find two temperature profiles that can be used to build the initial conditions
        if min(obs['Time'])>time or max(obs['Time'])<time:
            writelog('\tModel start time falls out of observations time range. The closest observations will be interpolated at the day-of-year of the start time.\n')
            if min(obs['Time'])>time:
                if min(obs['Time']).timetuple().tm_yday>time.timetuple().tm_yday:
                    time = datetime(min(obs['Time']).year+1,time.month,time.day,time.hour,time.minute)
                else:
                    time = datetime(min(obs['Time']).year,time.month,time.day,time.hour,time.minute)
            elif max(obs['Time'])<time:
                if max(obs['Time']).timetuple().tm_yday<time.timetuple().tm_yday:
                    time = datetime(max(obs['Time']).year-1,time.month,time.day,time.hour,time.minute)
                else:
                    time = datetime(max(obs['Time']).year,time.month,time.day,time.hour,time.minute)
        if (not any([t for t in obs['Time'] if t<time])) or (not any([t for t in obs['Time'] if t>time])):
            writelog('\tThe observation time span is too short. A proxy/default profile is used as initial conditions.\n')
            z = np.array([0,10,20,30,40,50,100,150,200,300])
            z = np.append(z[z<maxDepth],maxDepth)
            T = getTemperatureProfile(z,doy[0],elevation) #Proxy temperature profile at given day-of-year and altitude
            S = [0.15]*len(z) #Proxy salinity profile at given day-of-year and altitude
        else:
            t1 = max([t for t in obs['Time'] if t<time])
            t1_nb = calendar.timegm(t1.timetuple())
            t2 = min([t for t in obs['Time'] if t>time])
            t2_nb = calendar.timegm(t2.timetuple())
            time_nb = calendar.timegm(time.timetuple())
            if (t2_nb-t1_nb)>100*24*3600 and (t2_nb-time_nb)>14*24*3600 and (time_nb-t1_nb)>14*24*3600:
                writelog('\tThere are more than three months between the profiles to interpolate to get the initial conditions on %s. A proxy/default profile is used as initial conditions.\n' % datetime.strftime(time,'%d.%m.%Y'))
                z = np.array([0,10,20,30,40,50,100,150,200,300])
                z = np.append(z[z<maxDepth],maxDepth)
                T = getTemperatureProfile(z,doy[0],elevation) #Proxy temperature profile at given day-of-year and altitude
                S = [0.15]*len(z) #Proxy salinity profile at given day-of-year and altitude
            else: #Profiles are close enough to the required date
                #Sort profiles in order of increasing depths
                z1 = [z for i,z in enumerate(obs['Depth [m]']) if t1==obs['Time'][i]]
                T1 = [T for i,T in enumerate(obs['Temperature [°C]']) if t1==obs['Time'][i]]
                S1 = [S for i,S in enumerate(obs['Salinity [ppt]']) if t1==obs['Time'][i]]
                z1,T1,S1 = [list(tpl) for tpl in zip(*sorted(zip(z1,T1,S1)))]
                z2 = [z for i,z in enumerate(obs['Depth [m]']) if t2==obs['Time'][i]]
                T2 = [T for i,T in enumerate(obs['Temperature [°C]']) if t2==obs['Time'][i]]
                S2 = [S for i,S in enumerate(obs['Salinity [ppt]']) if t2==obs['Time'][i]]
                z2,T2,S2 = [list(tpl) for tpl in zip(*sorted(zip(z2,T2,S2)))]
                z = list(sorted(set(z1+z2))) #Depths appearing in either one of the profiles
                #Interpolate the temperature and salinity on z and at the model initial time
                T = [(((t2_nb-time_nb)*np.interp(zk,z1,T1) + (time_nb-t1_nb)*np.interp(zk,z2,T2)) / (t2_nb-t1_nb)) for zk in z]
                S = [(((t2_nb-time_nb)*np.interp(zk,z1,S1) + (time_nb-t1_nb)*np.interp(zk,z2,S2)) / (t2_nb-t1_nb)) for zk in z]
                #Enforce the first depth to be zero (this sets the initial water level in Simstrat)
                if (z[0]!=0):
                    writelog('\tFirst depth for initial conditions is not zero. Adding initial conditions at the surface with the same conditions as at the first depth.\n')
                    z = [0] + z
                    T = [np.nan if all(np.isnan(T)) else [Tv for Tv in T if not np.isnan(Tv)][0]] + T
                    S = [np.nan if all(np.isnan(S)) else [Sv for Sv in S if not np.isnan(Sv)][0]] + S
    if all(np.isnan(T)) or T==[]:
        writelog('\tNo valid temperature data found for the initial conditions! A proxy profile will be used.\n')
        T = getTemperatureProfile(z,doy,elevation)
    if all(np.isnan(S)) or S==[]:
        writelog('\tNo valid salinity data found for the initial conditions! A uniform profile will be used.\n')
        S[:] = [0.15]*len(z)
    #Write initial conditions file
    fid = open(file,'w',encoding='utf-8')
    fid.write('%s    %s    %s    %s    %s    %s    %s\n' % ('Depth [m]','U [m/s]','V [m/s]','T [°C]','S [‰]','k [J/kg]','eps [W/kg]'))
    for k in range(len(z)):
        if not np.isnan(T[k]):
            if np.isnan(S[k]): S[k]=np.nanmean(S)
            fid.write('%7.2f    %7.3f    %7.3f    %7.3f    %7.3f    %6.1e    %6.1e\n' % (-abs(z[k]),0,0,T[k],S[k],3E-6,5E-10))
    fid.close()
    return (z,S)

#Write the grid file for simstrat
def writeGrid(maxDepth,res,file):
    nGrid = np.ceil(abs(maxDepth/res))
    if nGrid>1000:
        writelog('\tDepth grid limited to 1000 points.\n')
        nGrid = 1000
    fid = open(file,'w',encoding='utf-8')
    fid.write('Number of grid points\n')
    fid.write('%d\n' % nGrid)
    fid.close()
    return nGrid

#Write the output depths file for Simstrat
def writeOutputDepths(depths,file):
    fid = open(file,'w',encoding='utf-8')
    fid.write('Depths [m]\n')
    for z in -np.abs(depths):
        fid.write('%.2f\n' % z)
    fid.close()
    return list(-np.abs(depths))

#Write the output times file for Simstrat
def writeOutputTimes(ndt,file):
    fid = open(file,'w',encoding='utf-8')
    fid.write('Number of time steps\n')
    fid.write('%d\n' % np.floor(ndt))
    fid.close()
    return np.floor(ndt)

def writeAbsorption(data,tref,file):
    fid = open(file,'w',encoding='utf-8')
    fid.write('Time [d] (1.col)    z [m] (1.row)    Absorption [m-1] (rest)\n')
    z_abs = set([abs(z) for z in data['Absorption [m^-1]']['Depth [m]']])
    fid.write('%d\n' % len(z_abs))
    fid.write('-1         ' + ' '.join(['%5.2f' % -z for z in z_abs]) + '\n')
    for t in data['Absorption [m^-1]']['Time']:
        fid.write('%10.4f' % daydiff(t,tref))
        for z in z_abs:
            ind = np.logical_and(np.array(data['Absorption [m^-1]']['Time'])==t,np.abs(data['Absorption [m^-1]']['Depth [m]'])==z)
            if sum(ind)>1:
                raise Exception('Error: time %s seems to be repeated in the Secchi data; check the source file.' % datetime.strftime(t,"%d.%m.%Y %H:%M"))
            fid.write(' %5.3f' % np.array(data['Absorption [m^-1]']['Data'])[ind])
        fid.write('\n')
    fid.close()

#Write the forcing file for Simstrat
def writeForcing(data,tref,tmodel,file,version):
    vars = ['Wind X [m/s]','Wind Y [m/s]','Air temperature [°C]','Solar radiation [W/m^2]','Vapour pressure [mbar]','Cloud cover [-]']
    if version.startswith('2.'): vars = vars + ['Precipitation [m/hr]']
    vars_short = ['u [m/s]','v [m/s]','Tair [°C]','sol [W/m2]','vap [mbar]','cloud [-]']
    if version.startswith('2.'): vars_short = vars_short + ['rain [m/hr]']
    if version.startswith('1.'): data.pop('Precipitation [m/hr]')
    #Interpolate on a unified time series and reorder
    time = []
    for var in data:
        time = time + data[var]['Time']
    time = set(time)
    time = sorted(time, key=lambda d: datetime.strftime(d,'%Y-%m-%d-%H-%M'))
    time = [t for t in time if (t>=tmodel[0] and t<=tmodel[1])]
    time_nb = [calendar.timegm(t.timetuple()) for t in time]
    data_int = []
    for var in vars:
        tvar_nb = [calendar.timegm(t.timetuple()) for t in data[var]['Time']]
        data_int.append(np.interp(time_nb,tvar_nb,data[var]['Data'],left=np.nan,right=np.nan))
        if var=='Cloud cover [-]': #Set missing cloud cover data as default of 50% so that simulation is possible
            data_int[-1] = [0.5 if np.isnan(cc) else cc for cc in data_int[-1]]
        if var=='Precipitation [m/hr]': #Set missing precipitation data to the average so that simulation is possible
            pmean = np.nanmean(data_int[-1])
            data_int[-1] = [pmean if np.isnan(p) else p for p in data_int[-1]]
    #Write
    fid = open(file,'w',encoding='utf-8')
    fid.write('%10s    ' % 'Time [d]')
    fid.write(' '.join(['%8s' % var for var in vars_short]) + '\n')
    k=0
    for t in range(len(time)):
        if any(np.isnan([val[t] for val in data_int])): k=k+1; continue
        fid.write('%10.4f    ' % daydiff(time[t],tref))
        fid.write(' '.join(['%8.4f' % val[t] for val in data_int]))
        fid.write('\n')
    fid.close()

#Write the inflow file for Simstrat (gravity inflow)
def writeInflow(data,tref,fileQ,fileT,fileS,S_inflow):
    fid = open(fileQ,'w',encoding='utf-8')
    fid.write('%10s %10s\n' % ('Time [d]','Q_in [m3/s]'))
    fid.write('%10d\n' % 1)
    fid.write('-1          %8.2f' % 0)
    fid.write('\n')
    for it in range(len(data['Flowrate [m^3/s]']['Time'])):
        if not np.isnan(data['Flowrate [m^3/s]']['Data'][it]):
            fid.write('%10.4f' % daydiff(data['Flowrate [m^3/s]']['Time'][it],tref))
            fid.write(' %9.4f' % data['Flowrate [m^3/s]']['Data'][it])
            fid.write('\n')
    fid.close()
    
    fid = open(fileT,'w',encoding='utf-8')
    fid.write('%10s %10s\n' % ('Time [d]','T_in [°C]'))
    fid.write('%10d\n' % 1)
    fid.write('-1          %8.2f' % 0)
    fid.write('\n')
    for it in range(len(data['Temperature [°C]']['Time'])):
        if not np.isnan(data['Temperature [°C]']['Data'][it]):
            fid.write('%10.4f' % daydiff(data['Temperature [°C]']['Time'][it],tref))
            fid.write(' %9.4f' % data['Temperature [°C]']['Data'][it])
            fid.write('\n')
    fid.close()
    
    fid = open(fileS,'w',encoding='utf-8')
    fid.write('%10s %10s\n' % ('Time [d]','S_in [‰]'))
    fid.write('%10d\n' % 1)
    fid.write('-1          %8.2f' % 0)
    fid.write('\n')
    for it in range(len(data['Temperature [°C]']['Time'])):
        if not np.isnan(data['Salinity [ppt]']['Data'][it]):
            fid.write('%10.4f' % daydiff(data['Temperature [°C]']['Time'][it],tref))
            fid.write(' %9.4f' % (data['Salinity [ppt]']['Data'][it] if (S_inflow is None) else S_inflow)) #River inflow salinity
            fid.write('\n')
    fid.close()

#Write the inflow file for Simstrat (fixed-depth inflow)
def writeInflowFixed(data,tref,fileQ,fileT,fileS):
    fid = open(fileQ,'w',encoding='utf-8')
    fid.write('%10s %10s %10s\n' % ('Time [d]','Depth [m]','Q_in [m2/s]'))
    fid.write('%10d\n' % len(data['Depths [m]']))
    fid.write('-1        '+' '.join(['%9.2f' % z for z in data['Depths [m]']]))
    fid.write('\n')
    for it in range(len(data['Flowrate [m^3/s]']['Time'])):
        fid.write('%10.4f ' % daydiff(data['Flowrate [m^3/s]']['Time'][it],tref))
        fid.write(' '.join(['%9.4f' % Q for Q in data['Flowrate [m^3/s]']['Data'][it]]))
        fid.write('\n')
    fid.close()
    
    fid = open(fileT,'w',encoding='utf-8')
    fid.write('%10s %10s %10s\n' % ('Time [d]','Depth [m]','T_in [°C*m2/s]'))
    fid.write('%10d\n' % len(data['Depths [m]']))
    fid.write('-1        '+' '.join(['%9.2f' % z for z in data['Depths [m]']]))
    fid.write('\n')
    for it in range(len(data['Temperature [°C]']['Time'])):
        fid.write('%10.4f ' % daydiff(data['Temperature [°C]']['Time'][it],tref))
        fid.write(' '.join(['%9.4f' % T for T in data['Temperature [°C]']['Data'][it]]))
        fid.write('\n')
    fid.close()
    
    fid = open(fileS,'w',encoding='utf-8')
    fid.write('%10s %10s %10s\n' % ('Time [d]','Depth [m]','S_in [‰*m2/s]'))
    fid.write('%10d\n' % len(data['Depths [m]']))
    fid.write('-1        '+' '.join(['%9.2f' % z for z in data['Depths [m]']]))
    fid.write('\n')
    for it in range(len(data['Salinity [ppt]']['Time'])):
        fid.write('%10.4f ' % daydiff(data['Salinity [ppt]']['Time'][it],tref))
        fid.write(' '.join(['%9.4f' % S for S in data['Salinity [ppt]']['Data'][it]]))
        fid.write('\n')
    fid.close()

#Write the outflow file for Simstrat
def writeOutflow(data,outflow_z,tref,file):
    fid = open(file,'w',encoding='utf-8')
    fid.write('%10s %10s\n' % ('Time [d]','Q_out [m2/s]'))
    if outflow_z==0: #Write zero outflow (case of overflowing lake)
        fid.write('%10d\n' % 1)
        fid.write('-1          %8.2f\n' % -1)
        for t in range(len(data['Flowrate [m^3/s]']['Time'])):
            fid.write('%10.4f' % daydiff(data['Flowrate [m^3/s]']['Time'][t],tref))
            fid.write(' %8.3f' % 0)
            fid.write('\n')
    else: #Spread the outflow over the given depth (case of basin outflowing into another basin)
        outflow_z = max(2,outflow_z) #Min 2m
        fid.write('%10d\n' % 3)
        fid.write('-1         %8.2f %8.2f %8.2f\n' % (-outflow_z,-outflow_z+1,0))
        for t in range(len(data['Flowrate [m^3/s]']['Time'])):
            fid.write('%10.4f' % daydiff(data['Flowrate [m^3/s]']['Time'][t],tref))
            Qs = data['Flowrate [m^3/s]']['Data'][t]/(outflow_z-0.5)
            fid.write(' %8.3f %8.3f %8.3f' % (0,Qs,Qs))
            fid.write('\n')
    fid.close()

#Get the values for outflow from the model results
def getValForOutflow(lakeName,var,outflow_z,version):
    delim = ',' if version.startswith('2.') else None
    end = None if version.startswith('2.') else -1
    if os.path.exists(os.path.join('Simstrat',lakeName,'Results',var+'_out.dat')):
        first = True
        t,val = [],[]
        for line in open(os.path.join('Simstrat',lakeName,'Results',var+'_out.dat')):
            if first:
                z = [float(v) for v in line.split(delim)[1:end]]
                first = False
            else:
                t.append(float(line.split(delim)[0]))
                val.append([float(v) for v in line.split(delim)[1:end]])
        z = np.abs(z)
        val = np.array(val)
    else:
        writelog('No data found for variable %s! Outflow into another basin will not be considered.' % var)
        return None
    val_out = []
    if outflow_z==0: #Case of outflow as river
        z_out = None
        for it in range(len(t)):
            val_out.append(np.mean(val[it,z<=1])) #Use mean over first meter as outflow value
    else: #Case of outflow over given depth
        z_out = z[z<=outflow_z]
        for it in range(len(t)):
            val_out.append(val[it,[(zk in z_out) for zk in z]])
        val_out = [[val_out[it][iz] for it in range(len(t))] for iz in range(len(z_out))]
    return t,z_out,val_out

#Get the values of the model parameters for a given lake
def getParVal(Lake,retrievePar,version='1.6',mode=''):
    lakeName = simple(Lake['Name'])
    par = {}
    #First set all parameters to default
    par['lat'] = round(CH1903ptoLat(Lake['X [m]'],Lake['Y [m]']),1)
    par['p_air'] = round(1013.25*np.exp((-9.81*0.029*Lake['Properties']['Elevation [m]'])/(8.314*283.15)),0)
    par['a_seiche'] = round(0.0017*np.sqrt(Lake['Properties']['Surface [km2]']),3)
    par['q_nn'] = 1.10
    par['f_wind'] = 1.00
    par['c10'] = 1.00
    par['cd'] = 0.002
    par['hgeo'] = 0.08 if 'Geothermal flux [W/m2]' not in Lake['Properties'] else Lake['Properties']['Geothermal flux [W/m2]']
    par['k_min'] = 1e-9
    par['p_radin'] = 1.00
    par['p_windf'] = 1.00
    par['beta_sol'] = 0.35
    par['albsw'] = 0.09
    par['salsource'] = max(1e-6,min(round((Lake['Properties']['Surface [km2]']/500)**2,6),2.0))
    if version.startswith('2.'):
        par['p_albedo'] = 1
        par['freez_temp'] = 0.01
        par['snow_temp'] = 2
    #Then check if values should and can be retrieved from a previous calibration
    if retrievePar and os.path.exists(os.path.join('PEST',lakeName,'simstrat_calib.par')):
        if mode!='calib': writelog('\tModel parameters: using values from previous calibration.\n')
        else: writelog('\t\tModel parameters: using values from previous calibration as initial values for calibration.\n')
        with open(os.path.join('PEST',lakeName,'simstrat_calib.par')) as fres:
            results = fres.readlines()[1:]
        for res in results:
            if (res.split()[0] in par):
                par[res.split()[0]] = float(res.split()[1])
    else:
        if mode!='calib': writelog('\tModel parameters: using default values.\n')
        else :writelog('\t\tModel parameters: using default values as initial values for calibration.\n')
    #Finally add parameters minimum and maximum (used for calibration)
    if mode=='calib':
        par['lat'] = [par['lat'],par['lat']-1,par['lat']+1]
        par['p_air'] = [par['p_air'],par['p_air']-30,par['p_air']+30]
        par['a_seiche'] = [par['a_seiche'],0.0001,0.05]
        par['q_nn'] = [par['q_nn'],0.70,1.30]
        par['f_wind'] = [par['f_wind'],0.30,1.25]
        par['c10'] = [par['c10'],0.80,1.25]
        par['cd'] = [par['cd'],0.001,0.003]
        par['hgeo'] = [par['hgeo'],0.00,0.20]
        par['k_min'] = [par['k_min'],1e-12,1e-6]
        par['p_radin'] = [par['p_radin'],0.90,1.10]
        par['p_windf'] = [par['p_windf'],0.80,1.25]
        par['beta_sol'] = [par['beta_sol'],0.20,0.40]
        par['albsw'] = [par['albsw'],0.05,0.15]
        par['salsource'] = [par['salsource'],1e-9,2.00]
        if version.startswith('2.'):
            par['p_albedo'] = [par['p_albedo'],0.80,1.25]
            par['freez_temp'] = [par['freez_temp'],0.00,0.05]
            par['snow_temp'] = [par['snow_temp'],1,3]
    return par

#Write the parameter file for Simstrat
def writeParFile(Lake,trange,tref,dt,grid,z_out,t_out,file,gravityInflow,version):
    lakeName = simple(Lake['Name'])
    par = getParVal(Lake,True,version)
    fid = open(file,'w',encoding='utf-8')
    if version.startswith('1.'):
        fid.write('\n')
        fid.write(os.path.join(lakeName,'InitialConditions.dat')+'\n')
        fid.write(os.path.join(lakeName,'Grid.dat')+'\n')
        fid.write(os.path.join(lakeName,'Bathymetry.dat')+'\n')
        fid.write(os.path.join(lakeName,'Forcing.dat')+'\n')
        fid.write(os.path.join(lakeName,'Absorption.dat')+'\n')
        fid.write(os.path.join(lakeName,'Results','')+'\n')
        fid.write(os.path.join(lakeName,'z_out.dat')+'\n')
        fid.write(os.path.join(lakeName,'t_out.dat')+'\n')
        fid.write(os.path.join(lakeName,'Qin.dat')+'\n')
        fid.write(os.path.join(lakeName,'Qout.dat')+'\n')
        fid.write(os.path.join(lakeName,'Tin.dat')+'\n')
        fid.write(os.path.join(lakeName,'Sin.dat')+'\n')
        fid.write('\n')
        fid.write('%d          Timestep dt [s]\n' % round(dt))
        fid.write('%5d        Start time [d] (%s)\n' % (daydiff(trange[0],tref),datetime.strftime(trange[0],'%Y.%m.%d')))
        fid.write('%5d        End time [d] (%s)\n' % (daydiff(trange[1],tref),datetime.strftime(trange[1],'%Y.%m.%d')))
        fid.write('\n')
        fid.write('1            Turbulence model (1:k-epsilon, 2:MY)\n')
        fid.write('2            Stability function (1:constant, 2:quasi-equilibrium)\n')
        fid.write('1            Flux condition (0:Dirichlet condition, 1:no-flux)\n')
        fid.write('3            Forcing (1:Wind+Temp+SolRad, 2:(1)+Vap, 3:(2)+Cloud, 4:Wind+HeatFlux+SolRad)\n')
        fid.write('0            Use filtered wind to compute seiche energy (0/default:off, 1:on) (if 1:on, one more column is needed in forcing file)\n')
        fid.write('2            Seiche normalization (1:max N^2, 2:integral)\n')
        fid.write('3            Wind drag model (1/default:constant, 2:ocean (increasing), 3:lake (Wüest and Lorke 2003))\n')
        fid.write('%d            Inflow placement (0/default:manual, 1:density-driven)\n' % (1 if gravityInflow else 0))
        fid.write('0            Pressure gradients (0:off, 1:Svensson 1978, 2:?)\n')
        fid.write('1            Enable salinity transport (0:off, 1/default:on)\n')
        fid.write('0            Display simulation (0:off, 1:when data is saved, 2:at each iteration, 3:extra display)\n')
        fid.write('0            Display diagnose (0:off, 1:standard display, 2:extra display)\n')
        fid.write('10           Averaging data\n')
        fid.write('\n')
        fid.write('%.1f         lat [°] Latitude for Coriolis parameter\n' % par['lat'])
        fid.write('%.0f          p_air [mbar] Air pressure\n' % par['p_air'])
        fid.write('%.4f       a_seiche [-] Fraction of wind energy to seiche energy\n' % par['a_seiche'])
        fid.write('%.3f        q_NN Fit parameter for distribution of seiche energy\n' % par['q_nn'])
        fid.write('%.2f         f_wind [-] Fraction of forcing wind to wind at 10m (W10/Wf)\n' % par['f_wind'])
        fid.write('%.2f         C10 [-] Wind drag: constant value (if wind drag model is 1:lazy) or scaling factor (otherwise)\n' % par['c10'])
        fid.write('%.3f        CD [-] Bottom friction coefficient\n' % par['cd'])
        fid.write('%.3f         Hgeo [W/m2] Geothermal heat flux\n' % par['hgeo'])
        fid.write('%1.0e        k_min [J/kg] Minimal value for TKE\n' % par['k_min'])
        fid.write('%.3f        p_radin Fit parameter for absorption of IR radiation from sky\n' % par['p_radin'])
        fid.write('%.3f        p_windf Fit parameter for convective and latent heat fluxes\n' % par['p_windf'])
        fid.write('%.2f         beta_sol [-] Fraction of short-wave radiation directly absorbed as heat\n' % par['beta_sol'])
        fid.write('%.2f         albsw [-] Albedo for reflection of short-wave radiation\n' % par['albsw'])
        fid.write('%1.2e     salsource [ppt*m^3/s]\n' % par['salsource'])
    elif version.startswith('2.'):
        setup = {}
        setup['Input'] = {}
        setup['Input']['Initial conditions'] = os.path.join(lakeName,'InitialConditions.dat')
        setup['Input']['Grid'] = grid
        setup['Input']['Morphology'] = os.path.join(lakeName,'Bathymetry.dat')
        setup['Input']['Forcing'] = os.path.join(lakeName,'Forcing.dat')
        setup['Input']['Absorption'] = os.path.join(lakeName,'Absorption.dat')
        setup['Input']['Inflow'] = os.path.join(lakeName,'Qin.dat')
        setup['Input']['Outflow'] = os.path.join(lakeName,'Qout.dat')
        setup['Input']['Inflow temperature'] = os.path.join(lakeName,'Tin.dat')
        setup['Input']['Inflow salinity'] = os.path.join(lakeName,'Sin.dat')
        setup['Output'] = {}
        setup['Output']['Path'] = os.path.join(lakeName,'Results','')
        setup['Output']['Depths'] = os.path.join(lakeName,'z_out.dat')#z_out
        setup['Output']['OutputDepthReference'] = 'surface'
        setup['Output']['Times'] = os.path.join(lakeName,'t_out.dat')#t_out
        setup['Simulation'] = {}
        setup['Simulation']['Timestep s'] = 300
        setup['Simulation']['Start d'] = daydiff(trange[0],tref)
        setup['Simulation']['End d'] = daydiff(trange[1],tref)
        setup['Simulation']['DisplaySimulation'] = 0
        setup['ModelConfig'] = {}
        setup['ModelConfig']['MaxLengthInputData'] = 1000
        setup['ModelConfig']['CoupleAED2'] = False
        setup['ModelConfig']['TurbulenceModel'] = 1
        setup['ModelConfig']['StabilityFunction'] = 2
        setup['ModelConfig']['FluxCondition'] = 1
        setup['ModelConfig']['Forcing'] = 3
        setup['ModelConfig']['UseFilteredWind'] = False
        setup['ModelConfig']['SeicheNormalization'] = 2
        setup['ModelConfig']['WindDragModel'] = 3
        setup['ModelConfig']['InflowPlacement'] = (1 if gravityInflow else 0)
        setup['ModelConfig']['PressureGradients'] = 0
        setup['ModelConfig']['IceModel'] = 1
        setup['ModelConfig']['SnowModel'] = 1
        setup['ModelParameters'] = {}
        for p in par.keys():
            setup['ModelParameters'][p] = par[p]
        json.dump(setup,fid,indent=4,ensure_ascii=False)
    else:
        raise Exception('Model (Simstrat) version must start with "1." or "2.".')
    fid.close()

#Write the parameter file for PEST
def writePESTParFile(Lake,modelExe,parFile,tref,par_calib,useSal,useIce,nCPU,file,version):
    lakeName = simple(Lake['Name'])
    par_calib = [par.lower() for par in par_calib]
    par = getParVal(Lake,False,version,mode='calib')
    setup = {}
    setup['files'] = {'model': modelExe, \
                    'configFile': parFile, \
                    'obsFile_T': os.path.join('Observations',lakeName+'_T.txt'), \
                    'refDate': datetime.strftime(tref,'%Y.%m.%d'), \
                    'pestDir': os.path.join('PEST',lakeName), \
                    'configFile_out': parFile, \
                    'results_out': os.path.join('Simstrat',lakeName,'Results')}
    if useSal: setup['files']['obsFile_S'] = os.path.join('Observations',lakeName+'_S.txt')
    if useIce: setup['files']['obsFile_IceH'] = os.path.join('Observations',lakeName+'_IceH.txt')
    setup['PEST'] = {'nCPU': nCPU}
    setup['parameters'] = {p: (par[p][0] if p not in par_calib else par[p]) for p in par.keys()}
    with open(file,'w',encoding='utf-8') as fid:
        json.dump(setup,fid,indent=4,ensure_ascii=False)


#Create the plots specific to each lake
def createPlots(Lake,t_ref,var,version,plotly=False):
    buttons = ['sendDataToCloud','zoom2d','pan2d','select2d','lasso2d','zoomIn2d','zoomOut2d','autoScale2d','resetScale2d','hoverClosestCartesian','hoverCompareCartesian','toggleSpikelines']
    lakeName = simple(Lake['Name'])
    delim = ',' if version.startswith('2.') else None
    end = None if version.startswith('2.') else -1
    
    fileID = 'NN' if (var=='N2' and version.startswith('2.')) else 'T' if (var=='heat' or var=='Schmidt') else 'IceH' if var=='ice' else var
    typ = 'processed' if (var=='heat' or var=='Schmidt' or var=='ice') else 'raw'
    variable = 'temperature' if var=='T' else 'salinity' if var=='S' else 'thermal diffusivity' if var=='nuh' else 'stratification' if var=='N2' else 'Schmidt stability' if var=='Schmidt' else 'heat content' if var=='heat' else 'ice cover' if var=='ice' else ''
    label_fig = 'Temperature [°C]' if var=='T' else 'Salinity [ppt]' if var=='S' else 'Thermal diffusivity [m$^2$/s]' if var=='nuh' else 'Brunt-Väisälä frequency (N$^2$) [s$^{-2}$]' if var=='N2' else 'Schmidt stability [J/m$^2$]' if var=='Schmidt' else 'Heat content [J]' if var=='heat' else 'Ice thickness [m]' if var=='ice' else ''
    label = 'Temperature [°C]' if var=='T' else 'Salinity [ppt]' if var=='S' else 'Thermal diffusivity [m<sup>2</sup>/s]' if var=='nuh' else 'Brunt-Väisälä frequency (N<sup>2</sup>) [s<sup>-2</sup>]' if var=='N2' else 'Schmidt stability [J/m<sup>2</sup>]' if var=='Schmidt' else 'Heat content [J]' if var=='heat' else 'Ice thickness [m]' if var=='ice' else ''
    ndec = 2 if fileID=='T' else 3 if fileID=='S' else 4 if fileID=='IceH' else 10 if fileID=='nuh' else 6
    
    (plots0,plot1,plot2,plot3,plots4,plots5,zout,plot6,Tsurf) = (None,None,None,None,[None],[None],None,None,None)
    
    if os.path.exists(os.path.join('Simstrat',lakeName,'Results',fileID+'_out.dat')):
        first = True
        t = []
        T = []
        for line in open(os.path.join('Simstrat',lakeName,'Results',fileID+'_out.dat')):
            if first:
                z = [float(val) for val in line.split(delim)[1:end]]
                first = False
            else:
                t.append(float(line.split(delim)[0]))
                T.append([float(val) for val in line.split(delim)[1:end]])
        T = np.round(T,ndec)
        if len(T)!=0:
            Tsurf = T[-1,-1]
            tdate = [t_ref+timedelta(tk) for tk in t]
            tdate = [roundTime(tk) for tk in tdate] #Round to nearest minute
            
            #Define time series for last year plot, and line plot for processed variables
            area = getAreas(z,Lake['Properties'])
            if typ=='raw':
                if var!='N2':
                    val = [T[k][-1] for k in range(len(tdate))] #Suface value
                else:
                    val = np.max(T,1) #Max value
            elif typ=='processed':
                if var=='heat':
                    val = [heatContent(z,area,T[k]) for k in range(len(tdate))] #Heat content [J]
                elif var=='Schmidt':
                    val = [schmidtStability(z,area,T[k]) for k in range(len(tdate))] #Schmidt stability [J/m2]
                elif var=='ice':
                    val = T
            val=np.array(val)
            
            if typ=='raw':
                #Contour plot
                #plt.rcParams['figure.figsize'] = 10,4
                plot0_img = lakeName+'_contour_'+var+'.png'
                plt.figure(figsize=(10,4))
                plt.contourf(tdate,z,np.transpose(T),100,cmap='rainbow')
                cbar = plt.colorbar()
                cbar.ax.set_ylabel(label_fig)
                plt.xlabel('Year')
                plt.ylabel('Depth [m]')
                plt.savefig(os.path.join('Plots',plot0_img),dpi=250)
                plt.clf(); plt.close()
                plot0_img = '<img src="'+os.path.join('static','templates','plots',plot0_img)+'" style=height:360px;margin-left:-23px>'
                if plotly:
                    data = [go.Contour(
                                z=np.transpose(T),
                                x=tdate,
                                y=z,
                                contours=dict(showlines=False),
                                ncontours=100,
                                colorscale='rainbow',
                                zmin=0 if var=='T' else 0 if var=='S' else 0,
                                zmax=25 if var=='T' else 0.4 if var=='S' else 0,
                                colorbar=dict(title=label,titleside='right'))
                    ]
                    layout = go.Layout(
                        xaxis=dict(title='Year'),
                        yaxis=dict(title='Depth [m]')
                    )
                    plot0_plotly = py.offline.plot(go.Figure(data=data,layout=layout),filename=os.path.join('Plots',lakeName+'_contour_'+var+'.html'),config={'showLink':False,'displaylogo':False,'modeBarButtonsToRemove':buttons},auto_open=False)
                    plot0_plotly = os.path.join('static','templates','plots',plot0_plotly.split(os.path.sep)[-1])
                    #plot0_plotly = re.sub('\.[0-9]{6}','',plot0_plotly.replace('width: 100%','width: 800px'))
                    plots0 = [plot0_img,plot0_plotly]
                else:
                    plots0 = [plot0_img,None]
            
                #Line plot
                plot1 = lakeName+'_line_'+var+'.png'
                plt.figure(figsize=(8,4))
                plt.plot(tdate,[T[k][0] for k in range(len(tdate))],color='xkcd:blue')
                plt.plot(tdate,val,color='xkcd:red')
                plt.xlim(tdate[0],tdate[-1])
                if var!='N2': plt.legend(['Lake bottom','Lake surface'],loc='upper left')
                plt.xlabel('Year')
                plt.ylabel(label_fig)
                plt.savefig(os.path.join('Plots',plot1),dpi=250)
                plt.clf(); plt.close()
                plot1 = '<img src="'+os.path.join('static','templates','plots',plot1)+'" style=height:360px;margin-left:-23px>'
                if plotly:
                    data = [go.Scatter(
                                x=tdate,
                                y=val,
                                name='Lake surface',
                                line=dict(color=('rgb(205,12,24)'))),
                            go.Scatter(
                                x=tdate,
                                y=[T[k][0] for k in range(len(tdate))],
                                name='Lake bottom',
                                line=dict(color=('rgb(22,96,167)')))
                    ]
                    layout = go.Layout(
                        xaxis=dict(title='Year',showgrid=False),
                        yaxis=dict(title=label,showgrid=False)
                    )
                    plot1 = py.plot(dict(data=data,layout=layout),config={'showLink':False,'displaylogo':False,'modeBarButtonsToRemove':buttons},include_plotlyjs=False,output_type='div')
                    plot1 = plot1.replace('width: 100%','width:800px').replace('height: 100%','height:480px')
                    plot1 = plot1.replace(':00:00"','"') #Remove minutes and seconds to reduce plot size
                
                #Last profile
                xmin = np.min(T[-1])
                n = (np.floor(np.log10(np.abs(xmin))) if xmin!=0 else 0)
                if n>0: n=n-1
                xmin = np.floor(xmin*10**-n)/10**-n
                xmax = np.max(T[-1])
                n = (np.floor(np.log10(np.abs(xmax))) if xmax!=0 else 0)
                if n>0: n=n-1
                xmax = np.ceil(xmax*10**-n)/10**-n
                plot2 = lakeName+'_profile_'+var+'.png'
                plt.figure(figsize=(5,6))
                plt.plot(T[-1],z)
                plt.xlim(xmin,xmax)
                plt.title('Last %s profile (%s)' % (variable,datetime.strftime(tdate[-1],'%d.%m.%Y')))
                plt.xlabel(label_fig)
                plt.ylabel('Depth [m]')
                plt.savefig(os.path.join('Plots',plot2),dpi=250)
                plt.clf(); plt.close()
                plot2 = '<img src="'+os.path.join('static','templates','plots',plot2)+'" style=height:640px;margin-left:50px>'
                if plotly:
                    data = [go.Scatter(
                                x=T[-1],
                                y=z)
                    ]
                    layout = go.Layout(
                        title=('Last %s profile (%s)' % (variable,datetime.strftime(tdate[-1],'%d.%m.%Y'))),
                        xaxis=dict(title=label,range=[xmin,xmax],showgrid=False),
                        yaxis=dict(title='Depth [m]',showgrid=False),
                        hovermode='y',
                        annotations=[dict(x=xmax-0.15*(xmax-xmin),y=0,text='Lake surface',bgcolor='#dddddd',showarrow=False,yanchor='bottom'),
                                     dict(x=xmax-0.15*(xmax-xmin),y=z[0],text='Lake bottom',bgcolor='#dddddd',showarrow=False)]
                    )
                    plot2 = py.plot(dict(data=data,layout=layout),config={'showLink':False,'displaylogo':False,'modeBarButtonsToRemove':buttons},include_plotlyjs=False,output_type='div')
                    plot2 = plot2.replace('width: 100%','width:400px').replace('height: 100%','height:640px')
            
            #Last year
            plot3 = lakeName+'_year_'+var+'.png'
            plt.figure(figsize=(8,4))
            tyear = [tk for tk in tdate if tk>(tdate[-1]-timedelta(200))]
            Tyear = val[np.searchsorted(tdate,tyear)]
            doy = doy365(tyear)+1
            doy_unique = doy[0]+np.linspace(1,len(np.unique(doy)),len(np.unique(doy)))
            doy_unique[doy_unique>365] = doy_unique[doy_unique>365]-365
            Tyear = np.round([np.nanmean(Tyear[doy==d]) for d in doy_unique],ndec)
            tyear = [tyear[0].date()+timedelta(k) for k in range(len(doy_unique))]
            plt.plot(tyear,Tyear,color='xkcd:black')
            
            tbef = [tk for tk in tdate if tk<=(tdate[-1]-timedelta(200))]
            Tbef = val[np.searchsorted(tdate,tbef)]
            doy = doy365(tbef)+1
            doy_unique = doy_unique[0] + np.linspace(1,365,365)
            doy_unique[doy_unique>365] = doy_unique[doy_unique>365]-365
            Tdoyavg = np.round([np.nanmean(Tbef[doy==d]) for d in doy_unique],ndec)
            Tdoymin = np.array([np.nanmin(Tbef[doy==d]) for d in doy_unique])
            Tdoymax = np.array([np.nanmax(Tbef[doy==d]) for d in doy_unique])
            tdoyyear = [tyear[0]+timedelta(k) for k in range(len(doy_unique))]
            plt.plot(tdoyyear,Tdoyavg,color='xkcd:grey',linestyle='--')
            plt.plot(tdoyyear,Tdoymin,color='xkcd:blue',linestyle='--')
            plt.plot(tdoyyear,Tdoymax,color='xkcd:red',linestyle='--')
            plt.xlim(tdoyyear[0],tdoyyear[-1])
            plt.legend(['Current year','Mean','Minimum','Maximum'],loc='upper right')
            plt.ylabel(label_fig)
            plt.savefig(os.path.join('Plots',plot3),dpi=250)
            plt.clf(); plt.close()
            ys = tdate[0].year
            ye = tdate[-1].year-1
            plot3 = '<img src="'+os.path.join('static','templates','plots',plot3)+'" style=height:360px;margin-left:-23px>'
            if plotly:
                data = [go.Scatter(
                            x=tdoyyear,
                            y=Tdoymin,
                            name=('Min %d-%d' % (ys,ye)),
                            line=dict(width=0.5,color=('rgb(24,24,255)'))),
                        go.Scatter(
                            x=tdoyyear,
                            y=Tdoymax,
                            name=('Max %d-%d' % (ys,ye)),
                            line=dict(width=0.5,color=('rgb(255,24,24)')),
                            fillcolor='rgb(200,200,240)',
                            fill='tonexty'),
                        go.Scatter(
                            x=tdoyyear,
                            y=Tdoyavg,
                            name=('Mean %d-%d' % (ys,ye)),
                            line=dict(width=0.5,color=('rgb(100,100,100)'))),
                        go.Scatter(
                            x=tyear,
                            y=Tyear,
                            name='Current year',
                            line=dict(color=('rgb(0,0,0)')))
                ]
                layout = go.Layout(
                    title=('Current evolution of the %s%s' % ('' if typ=='processed' else 'maximal ' if var=='N2' else 'surface ',variable)),
                    xaxis=dict(tickformat='%B %d',showgrid=False,ticks="outside"),
                    yaxis=dict(title=label,showgrid=False,range=[min(np.concatenate((Tyear,Tdoymin))),max(np.concatenate((Tyear,Tdoymax)))]),
                    legend=dict(x=0.9,y=1)
                )
                plot3 = py.plot(dict(data=data,layout=layout),config={'showLink':False,'displaylogo':False,'modeBarButtonsToRemove':buttons},include_plotlyjs=False,output_type='div')
                plot3 = plot3.replace('width: 100%','width:800px').replace('height: 100%','height:480px')
            
            #Processed variables: line plot
            if typ=='processed':
                plot1 = lakeName+'_line_'+var+'.png'
                plt.figure(figsize=(8,4))
                plt.plot(tdate,val)
                plt.xlim(tdate[0],tdate[-1])
                plt.xlabel('Year')
                plt.ylabel(label_fig)
                plt.savefig(os.path.join('Plots',plot1),dpi=250)
                plt.clf(); plt.close()
                plot1 = '<img src="'+os.path.join('static','templates','plots',plot1)+'" style=height:360px;margin-left:-23px>'
                if plotly:
                    data = [go.Scatter(
                                x=tdate,
                                y=val)
                    ]
                    layout = go.Layout(
                        xaxis=dict(title='Year',showgrid=False),
                        yaxis=dict(title=label,showgrid=False)
                    )
                    plot1 = py.plot(dict(data=data,layout=layout),config={'showLink':False,'displaylogo':False,'modeBarButtonsToRemove':buttons},include_plotlyjs=False,output_type='div')
                    plot1 = plot1.replace('width: 100%','width:800px').replace('height: 100%','height:480px')
                    plot1 = plot1.replace(':00:00"','"') #Remove minutes and seconds to reduce plot size
                
            #Statistics plots
            if var=='T':
                zstep = 2 if min(z)>-20 else 4 if min(z)>-40 else 10 if min(z)>=-120 else 20
                zout = -np.arange(0,abs(min(z)),zstep)[::-1]
                #Profile
                months = np.array([tk.month for tk in tdate])
                Tmonth = []
                Tmonth_std = []
                for mth in range(1,13):
                    Tmonth.append(np.round(np.mean(T[months==mth],axis=0),2))
                    Tmonth_std.append(np.round(np.std(T[months==mth],axis=0),2))
                Tmonth = np.array(Tmonth)[:,[zo in zout for zo in z]]
                Tmonth_std = np.array(Tmonth_std)[:,[zo in zout for zo in z]]
                plots4 = [None]
                if plotly:
                    for mth in range(1,13):
                        data = [go.Scatter(
                                    x=Tmonth[mth-1,],
                                    y=zout,
                                    error_x=dict(type='data',array=Tmonth_std[mth-1],visible=True))
                        ]
                        layout = go.Layout(
                            xaxis=dict(title=label,range=[0,np.max(Tmonth+Tmonth_std)],showgrid=False),
                            yaxis=dict(title='Depth [m]',showgrid=False),
                            margin=dict(l=75,r=75,b=75,t=25),
                            hovermode='y',
                            annotations=[dict(x=0.85*26,y=0,text='Lake surface',bgcolor='#dddddd',showarrow=False,yanchor='bottom'),
                                         dict(x=0.85*26,y=z[0],text='Lake bottom',bgcolor='#dddddd',showarrow=False)]
                        )
                        plots4.append(py.plot(dict(data=data,layout=layout),config={'showLink':False,'displaylogo':False,'modeBarButtonsToRemove':buttons},include_plotlyjs=False,output_type='div'))
                        plots4[-1] = plots4[-1].replace('width: 100%','width:600px; display:none').replace('height: 100%','height:480px')
                #Yearly course
                doys = doy365(tdate)
                Tdoy = []
                Tdoy_std = []
                for doy in range(365):
                    Tdoy.append(np.round(np.mean(T[doys==doy],axis=0),2))
                    Tdoy_std.append(np.round(np.std(T[doys==doy],axis=0),2))
                Tdoy = np.array(Tdoy)[:,[zo in zout for zo in z]]
                Tdoy_std = np.array(Tdoy_std)[:,[zo in zout for zo in z]]
                plots5 = []
                if plotly:
                    for iz in range(len(zout)):
                        data = [go.Scatter(
                                    x=np.linspace(1,365,365),
                                    y=Tdoy[:,iz]-Tdoy_std[:,iz],
                                    line=dict(width=0),
                                    hoverinfo='x'),
                                go.Scatter(
                                    name='',
                                    x=np.linspace(1,365,365),
                                    y=Tdoy[:,iz],
                                    error_y=dict(type='data',array=Tdoy_std[:,iz],visible=True,color='rgba(0,0,0,0)'),
                                    line=dict(color='rgb(31,119,180)'),
                                    fillcolor='rgba(68,68,68,0.3)',
                                    fill='tonexty',
                                    text=[datetime.strftime(datetime.strptime(str(doy),'%j'),'%B %d') for doy in range(1,366)]),
                                go.Scatter(
                                    x=np.linspace(1,365,365),
                                    y=Tdoy[:,iz]+Tdoy_std[:,iz],
                                    line=dict(width=0),
                                    fillcolor='rgba(68,68,68,0.3)',
                                    fill='tonexty',
                                    hoverinfo='x')
                        ]
                        layout = go.Layout(
                            xaxis=dict(title='Day of year',showgrid=False),
                            yaxis=dict(title=label,range=[0,np.max(Tdoy+Tdoy_std)],showgrid=False),
                            margin=dict(l=75,r=75,b=75,t=25),
                            showlegend=False
                        )
                        plots5.append(py.plot(dict(data=data,layout=layout),config={'showLink':False,'displaylogo':False,'modeBarButtonsToRemove':buttons},include_plotlyjs=False,output_type='div'))
                        plots5[-1] = plots5[-1].replace('width: 100%','width:800px; display:none').replace('height: 100%','height:480px')
                
                #Long-term trend
                if (tdate[-1].year-tdate[0].year)>30:
                    monthlyTrends = []
                    for mth in range(1,13): #Calculate slope coeff [°C/decade] for each month and depth
                        x = np.array([calendar.timegm(tk.timetuple())/3600/24/365/10 for tk in np.array(tdate)[months==mth]])
                        y = T[months==mth,:][:,[zo in zout for zo in z]]
                        monthlyTrends.append([np.round((np.mean(x*y[:,zk])-np.mean(x)*np.mean(y[:,zk]))/(np.mean(x**2)-np.mean(x)**2),2) for zk in range(len(zout))])
                    if plotly:
                        data = [go.Contour(
                            z=np.transpose(monthlyTrends),
                            x=np.linspace(1,12,12),
                            y=zout,
                            contours=dict(showlines=False),
                            ncontours=100,
                            zmin=-0.5,
                            zmax=0.5,
                            colorbar=dict(title='Temperature trend [°C/decade]',titleside='right'))
                        ]
                        layout = go.Layout(
                            xaxis=dict(title='Month',tickvals=np.linspace(1,12,12),ticktext=[calendar.month_abbr[m] for m in range(1,13)]),
                            yaxis=dict(title='Depth [m]'),
                            margin=dict(l=75,r=75,b=75,t=25)
                        )
                        plot6 = py.plot(dict(data=data,layout=layout),config={'showLink':False,'displaylogo':False,'modeBarButtonsToRemove':buttons},include_plotlyjs=False,output_type='div')
                        plot6 = plot6.replace('width: 100%','width:800px').replace('height: 100%','height:480px')
                
                #Temperature trend at bottom
                dt = np.mean(np.diff(t))
                doy = list(doy365(tdate))
                t1 = int(60/dt) #Start time: ignore first 2 months (model spin up)
                t2 = len(t)-1-doy[::-1].index(doy[t1]) #End time: same day-of-year as start time
                plt.plot(tdate[t1:t2],T[t1:t2,0])
                linreg = stats.linregress(t[t1:t2],T[t1:t2,0])
                signif = '***' if linreg.pvalue<0.001 else '**' if linreg.pvalue<0.01 else '*' if linreg.pvalue<0.05 else ''
                plt.plot(tdate[t1:t2],linreg.slope*np.array(t[t1:t2])+linreg.intercept,color='black',linestyle='dashed')
                plt.annotate('Linear trend%s: %s%.2f °C/decade' % (signif,'+' if linreg.slope>0 else '',linreg.slope*365*10),(tdate[int(len(t)/2)],min(T[t1:t2,0])))
                plt.ylabel(label_fig)
                #plt.savefig(os.path.join('Plots',lakeName+'_trend_'+var+'bottom.pdf'),format='pdf',bbox_inches='tight')
                #plt.savefig(os.path.join('Plots',lakeName+'_trend_'+var+'bottom.png'),dpi=250,bbox_inches='tight')
                plt.clf(); plt.close()
                #Evolution of yearly mean/minimum/maximum surface and bottom temperatures
                years_all = np.array([tk.year for tk in tdate])
                years = np.unique(years_all)[1:-1] #All full years
                Ts_avg,Ts_max,Ts_min = [],[],[]
                Tb_avg,Tb_max,Tb_min = [],[],[]
                for yr in years:
                    Ts_avg.append(np.mean(T[yr==years_all,-1]))
                    Ts_max.append(max(T[yr==years_all,-1]))
                    Ts_min.append(min(T[yr==years_all,-1]))
                    Tb_avg.append(np.mean(T[yr==years_all,0]))
                    Tb_max.append(max(T[yr==years_all,0]))
                    Tb_min.append(min(T[yr==years_all,0]))
                #plt.plot(years,Ts_max,'-o')
                #plt.plot(years,Ts_min,'-o')
                #plt.plot(years,Tb_max,'-o')
                #plt.plot(years,Tb_min,'-o')
                #plt.legend(['Yearly minimum at surface','Yearly maximum at bottom','Yearly minimum at bottom'],loc='lower right')
                #plt.ylabel(label_fig)
                #plt.savefig(os.path.join('Plots',lakeName+'_trend_'+var+'.eps'),format='eps',bbox_inches='tight')
                #plt.savefig(os.path.join('Plots',lakeName+'_trend_'+var+'.png'),dpi=250,bbox_inches='tight')
                #plt.clf(); plt.close()
                plt.plot(years,Ts_avg,'o')
                plt.plot(years,Tb_avg,'o')
                plt.legend(['Yearly mean at surface','Yearly mean at bottom'],loc='center right')
                for curve in [Ts_avg,Tb_avg]:
                    linreg = stats.linregress(years,curve)
                    signif = '***' if linreg.pvalue<0.001 else '**' if linreg.pvalue<0.01 else '*' if linreg.pvalue<0.05 else ''
                    plt.plot(years,linreg.slope*years+linreg.intercept,color='black',linestyle='dashed')
                    plt.annotate('Linear trend%s: %s%.2f °C/decade' % (signif,'+' if linreg.slope>0 else '',linreg.slope*10),(years[int(len(years)/2)],min(curve)-0.5))
                plt.ylim([np.floor(min(Tb_avg)-1),np.ceil(max(Ts_avg))])
                plt.ylabel(label_fig)
                plt.savefig(os.path.join('Plots',lakeName+'_trend_'+var+'.eps'),format='eps',bbox_inches='tight')
                plt.savefig(os.path.join('Plots',lakeName+'_trend_'+var+'.png'),dpi=250,bbox_inches='tight')
                plt.clf(); plt.close()
                #Evolution of start and end of stratification
                ts,te = getStratificationStartEnd(tdate,z,area,T)
                if (ts!=[] and te!=[]):
                    yr_ts = np.array([t.year for t in ts])
                    yr_te = np.array([t.year for t in te])
                    ts,te = doy365(ts),doy365(te)
                    yr_te[te<100] = yr_te[te<100]-1
                    te[te<100] = te[te<100]+365
                    plt.plot(yr_te,te,'o')
                    plt.plot(yr_ts,ts,'o')
                    plt.legend(['End of stratification','Start of stratification'],loc='center right')
                    for curve_x,curve_y in zip([yr_te,yr_ts],[te,ts]):
                        linreg = stats.linregress(curve_x,curve_y)
                        signif = '***' if linreg.pvalue<0.001 else '**' if linreg.pvalue<0.01 else '*' if linreg.pvalue<0.05 else ''
                        plt.plot(years,linreg.slope*years+linreg.intercept,color='black',linestyle='dashed')
                        plt.annotate('Linear trend%s: %s%.1f day/decade' % (signif,'+' if linreg.slope>0 else '',linreg.slope*10),(curve_x[int(len(curve_x)/2)],min(curve_y)-10))
                    plt.ylabel('Day of year')
                    plt.savefig(os.path.join('Plots',lakeName+'_trend_strat.eps'),format='eps',bbox_inches='tight')
                    plt.savefig(os.path.join('Plots',lakeName+'_trend_strat.png'),dpi=250,bbox_inches='tight')
                    plt.clf(); plt.close()
            
            if var=='heat':
                years_all = np.array([tk.year for tk in tdate])
                years = np.unique(years_all)[1:-1] #All full years
                hc_max,hc_avg,hc_min = [],[],[]
                for yr in years:
                    hc_max.append(max(val[yr==years_all]))
                    hc_avg.append(np.mean(val[yr==years_all]))
                    hc_min.append(min(val[yr==years_all]))
                plt.plot(years,hc_avg,color='grey')
                plt.fill_between(years,hc_min,hc_max,facecolor='xkcd:bluegrey',alpha=0.5)
                plt.ylabel(label_fig)
                plt.savefig(os.path.join('Plots',lakeName+'_trend_'+var+'.eps'),format='eps',bbox_inches='tight')
                plt.savefig(os.path.join('Plots',lakeName+'_trend_'+var+'.png'),dpi=250,bbox_inches='tight')
                plt.clf(); plt.close()
            if var=='Schmidt':
                years_all = np.array([tk.year for tk in tdate])
                years = np.unique(years_all)[1:-1] #All full years
                hc_max,hc_avg,hc_min = [],[],[]
                for yr in years:
                    hc_max.append(max(val[yr==years_all]))
                    hc_avg.append(np.mean(val[yr==years_all]))
                    hc_min.append(min(val[yr==years_all]))
                plt.plot(years,hc_avg,color='grey')
                plt.fill_between(years,hc_min,hc_max,facecolor='xkcd:bluegrey',alpha=0.5)
                plt.ylabel(label_fig)
                plt.savefig(os.path.join('Plots',lakeName+'_trend_'+var+'.eps'),format='eps',bbox_inches='tight')
                plt.savefig(os.path.join('Plots',lakeName+'_trend_'+var+'.png'),dpi=250,bbox_inches='tight')
                plt.clf(); plt.close()
            if var=='N2':
                #Evolution of yearly maximum N2
                years_all = np.array([tk.year for tk in tdate])
                years = np.unique(years_all)[1:-1] #All full years
                N2_max = []
                for yr in years:
                    N2_max.append(max(val[yr==years_all]))
                plt.plot(years,N2_max,'o')
                plt.legend(['Yearly maximum'],loc='upper left')
                linreg = stats.linregress(years,N2_max)
                signif = '***' if linreg.pvalue<0.001 else '**' if linreg.pvalue<0.01 else '*' if linreg.pvalue<0.05 else ''
                plt.plot(years,linreg.slope*years+linreg.intercept,color='black',linestyle='dashed')
                plt.annotate('Linear trend%s: %s%.4f s$^{-2}$/decade' % (signif,'+' if linreg.slope>0 else '',linreg.slope*10),(years[int(len(years)/3)],min(N2_max)))
                plt.ylabel(label_fig)
                plt.savefig(os.path.join('Plots',lakeName+'_trend_'+var+'.eps'),format='eps',bbox_inches='tight')
                plt.savefig(os.path.join('Plots',lakeName+'_trend_'+var+'.png'),dpi=250,bbox_inches='tight')
                plt.clf(); plt.close()
            if var=='ice':
                ts,te = getIceCoverStartEnd(tdate,val)
                if (ts!=[] and te!=[]):
                    #Evolution of start and end of ice cover
                    yr_ts = np.array([t.year for t in ts])
                    yr_te = np.array([t.year for t in te])
                    ts,te = doy365(ts),doy365(te)
                    yr_ts[ts>250] = yr_ts[ts>250]+1
                    ts[ts>250] = ts[ts>250]-365
                    plt.plot(yr_te,te,'o')
                    plt.plot(yr_ts,ts,'o')
                    plt.legend(['End of ice cover','Start of ice cover'],loc='center right')
                    plt.ylabel('Day of year')
                    plt.savefig(os.path.join('Plots',lakeName+'_trend_ice.eps'),format='eps',bbox_inches='tight')
                    plt.savefig(os.path.join('Plots',lakeName+'_trend_ice.png'),dpi=250,bbox_inches='tight')
                    plt.clf(); plt.close()
                    #Timing of ice cover through the years
                    if os.path.exists(os.path.join('Observations',lakeName+'_IceH.txt')):
                        iceObs = np.loadtxt(os.path.join('Observations',lakeName+'_IceH.txt'),dtype=bytes,skiprows=1,delimiter=',')
                        t_obs = [datetime.strptime(rec[0].decode(),'%Y-%m-%d %H:%M') for rec in iceObs]
                        h_obs = [toFloat(rec[1]) for rec in iceObs]
                        ts_obs,te_obs = getIceCoverStartEnd(t_obs,h_obs)
                        yr_te_obs = [t.year for t in te_obs]
                        ts_obs,te_obs = doy365(ts_obs),doy365(te_obs)
                    else:
                        iceObs = None
                    ts[ts<0] = ts[ts<0]+365
                    fig = plt.figure(figsize=(8,5))
                    ax = fig.add_axes([0,0,1,1])
                    if len(ts)>len(te): ts=ts[:-1]
                    for yi in range(len(ts)):
                        if te[yi]-ts[yi]>0:
                            ax.add_patch(plt.Rectangle([yr_te[yi]-0.4,ts[yi]],0.8,te[yi]-ts[yi]))
                        else: #Case of ice cover starting before December 31st
                            ax.add_patch(plt.Rectangle([yr_te[yi]-0.4,0],0.8,te[yi]))
                            ax.add_patch(plt.Rectangle([yr_te[yi]-0.4,ts[yi]],0.8,365-ts[yi]))
                        if ((iceObs is not None) and (yr_te[yi] in yr_te_obs)):
                            yi_obs = yr_te_obs.index(yr_te[yi])
                            if te_obs[yi_obs]-ts_obs[yi_obs]>0:
                                plt.plot([yr_te_obs[yi_obs],yr_te_obs[yi_obs]],[ts_obs[yi_obs],te_obs[yi_obs]],color='black')
                            else: #Case of ice cover starting before December 31st
                                plt.plot([yr_te_obs[yi_obs],yr_te_obs[yi_obs]],[0,te_obs[yi_obs]],color='black')
                                plt.plot([yr_te_obs[yi_obs],yr_te_obs[yi_obs]],[ts_obs[yi_obs],365],color='black')
                    plt.gca().xaxis.tick_top()
                    plt.tick_params(axis='both',which='both',top=False,labelsize=16)
                    plt.xlim(min(yr_te)-0.8,max(yr_te)+0.8)
                    plt.ylabel('Day of year',fontsize=16)
                    plt.ylim(0,365)
                    plt.gca().invert_yaxis()
                    plt.savefig(os.path.join('Plots',lakeName+'_timing_ice.eps'),format='eps',bbox_inches='tight')
                    plt.savefig(os.path.join('Plots',lakeName+'_timing_ice.png'),dpi=250,bbox_inches='tight')
                    plt.clf(); plt.close()
        else:
            writelog('\t'+Lake['Name']+': no results for '+variable+' (file '+fileID+'_out.dat)!\n')
    else:
        writelog('\t'+Lake['Name']+': no results for '+variable+' (file '+fileID+'_out.dat)!\n')
    
    return ([plots0,plot1,plot2,plot3,plots4,[plots5,zout],plot6],Tsurf)

#Create the plots for all lakes
def plotAllLakes(Lakes,t_ref,version,plotly=False):
    buttons = ['sendDataToCloud','zoom2d','pan2d','select2d','lasso2d','zoomIn2d','zoomOut2d','autoScale2d','resetScale2d','hoverClosestCartesian','hoverCompareCartesian','toggleSpikelines']
    #Retrieve model results
    delim = ',' if version.startswith('2.') else None
    end = None if version.startswith('2.') else -1
    z = -np.linspace(0,370,num=75)
    profilesT,profilesS = [],[]
    lakeNames,lakeDepths,lakeVolumes,lakeMeanDepths,lakeElevations = [],[],[],[],[]
    lakeSTs,lakeStratPeriod,lakeLastStratPeriod,lakeIcePeriod,lakeLastIcePeriod = [],[],[],[],[]
    tlake,lakeTmean,lakeTsurf,lakeTbott,lakeHC,lakeSC = [],[],[],[],[],[]
    nIceYears,nSimYears = [],[]
    for Lake in Lakes:
        try:
            t,T,S,ice = [],[],[],[]
            with open(os.path.join('Simstrat',simple(Lake['Name']),'Results','T_out.dat')) as fid:
                zlake = list(map(float,next(fid).split(delim)[1:end]))
                for line in fid:
                    t.append(float(line.split(delim)[0]))
                    T.append([float(val) for val in line.split(delim)[1:end]])
            with open(os.path.join('Simstrat',simple(Lake['Name']),'Results','S_out.dat')) as fid:
                next(fid)
                for line in fid:
                    S.append([float(val) for val in line.split(delim)[1:end]])
            if version.startswith('2.'):
                with open(os.path.join('Simstrat',simple(Lake['Name']),'Results','IceH_out.dat')) as fid:
                    next(fid)
                    for line in fid:
                        ice.append(float(line.split(delim)[1]))
            lakeNames.append(Lake['Name'])
            lakeVolumes.append(Lake['Properties']['Volume [km3]']*1E9 if Lake['Properties']['Volume [km3]'] is not None else np.nan)
            lakeMeanDepths.append((Lake['Properties']['Avg depth [m]'] if 'Avg depth [m]' in Lake['Properties'] else np.nan)) #Missing for some lakes...
            lakeElevations.append(Lake['Properties']['Elevation [m]'])
            lakeDepths.append(Lake['Properties']['Max depth [m]'])
            tdate = [t_ref+timedelta(tk) for tk in t]
            tdate = [roundTime(tk) for tk in tdate] #Round to nearest minute
            nSimYears.append(int((tdate[-1]-tdate[0]).days/365))
            profilesT.append(np.interp(z,zlake,T[-1],left=np.nan)) #Array of the last temperature profiles
            profilesS.append(np.interp(z,zlake,S[-1],left=np.nan)) #Array of the last salinity profiles
            area1 = getAreas(z,Lake['Properties'])
            area2 = getAreas(zlake,Lake['Properties'])
            last = (np.where(np.isnan(profilesT[-1]))[0][0] if np.any(np.isnan(profilesT[-1])) else len(profilesT[-1]))
            lakeSTs.append(schmidtStability(z[:last],area1[:last],profilesT[-1][:last],profilesS[-1][:last]))
            ts,te = getStratificationStartEnd(tdate,zlake,area2,T) #Only consider thermal stratification
            ts,te = doy365(ts),doy365(te)
            te[te<100] = te[te<100]+365 #Case of stratification ending after December 31st
            lakeStratPeriod.append([np.mean(ts),np.mod(np.mean(te),365)])
            lakeLastStratPeriod.append([ts[-1] if len(ts)>0 else np.nan,np.mod(te[-1] if len(te)>0 else np.nan,365)])
            if version.startswith('2.'):
                ts,te = getIceCoverStartEnd(tdate,ice)
                ts,te = doy365(ts),doy365(te)
                ts[ts<100] = ts[ts<100]+365 #Case of ice cover starting after December 31st
                lakeIcePeriod.append([np.mod(np.mean(ts),365),np.mean(te)])
                lakeLastIcePeriod.append([np.mod(ts[-1] if len(ts)>0 else np.nan,365),te[-1] if len(te)>0 else np.nan])
                nIceYears.append(len(ts))
                if nIceYears[-1]>nSimYears[-1]: nSimYears[-1]+=1
            else:
                lakeIcePeriod.append([np.nan,np.nan])
                lakeLastIcePeriod.append([np.nan,np.nan])
                nIceYears.append(np.nan)
            tlake.append(tdate)
            lakeTmean.append([meanTemperature(zlake,area2,Tp) for Tp in T]) #Time series of mean temperature
            lakeTsurf.append([Tp[-1] for Tp in T]) #Time series of surface temperature
            lakeTbott.append([Tp[0] for Tp in T]) #Time series of bottom temperature
            lakeHC.append([heatContent(zlake,area2,Tp,Sp) for Tp,Sp in zip(T,S)]) #Time series of heat content
            lakeSC.append([schmidtStability(zlake,area2,Tp,Sp) for Tp,Sp in zip(T,S)]) #Time series of Schmidt stability
        except:
            writelog('\tCouldn\'t extract results for %s...' % Lake['Name'])
    profilesT,profilesS,lakeNames,lakeDepths,lakeElevations = np.array(profilesT),np.array(profilesS),np.array(lakeNames),np.array(lakeDepths),np.array(lakeElevations)
    
    z_sort = np.argsort(lakeDepths)[::-1] #Sort by max depth
    h_sort = np.argsort(lakeElevations) #Sort by elevation
    #Generate the x-axis series
    gap = 0.2
    x = [-gap/2]
    xmid = []
    for p in range(len(profilesT)):
        xmid = xmid + [x[-1]+0.5+gap/2]
        x = x + [x[-1]+gap/2,x[-1]+1+gap/2,x[-1]+1+gap]
    x = x[1:]
    #Create the aggregated contour plot of temperature
    data_contour = []
    for iz in range(len(z)):
        data_contour.append([v for Tz in [[p[iz],p[iz],np.nan] for p in profilesT[z_sort]] for v in Tz])
    plt.figure(figsize=(16,10))
    plt.contourf(x,z,data_contour,100,cmap='rainbow')
    plt.gca().xaxis.tick_top()
    plt.tick_params(axis='both',which='both',top=False,labelsize=16)
    plt.xticks(xmid,lakeNames[z_sort],rotation=90)
    plt.ylabel('Depth [m]',fontsize=16)
    plt.tight_layout()
    cbar = plt.colorbar(cax=inset_axes(plt.gca(),width="30%",height="3%",loc=4,borderpad=3),orientation='horizontal',ticks=[4*v for v in range(7)])
    cbar.ax.set_ylabel('Temperature [°C]',rotation='horizontal',ha='left',y=1.5,fontsize=16)
    cbar.ax.tick_params(axis='x',labelsize=16)
    plt.savefig(os.path.join('Plots','All_LastProfile_T.eps'),format='eps')
    plt.savefig(os.path.join('Plots','All_LastProfile_T.png'),dpi=250)
    plt.clf(); plt.close()
    #Create the aggregated plot of stratification and ice period
    fig = plt.figure(figsize=(16,5))
    ax = fig.add_axes([0,0,1,1])
    plt.plot([min(x),max(x)],[0,365],alpha=0)
    kk = 0
    for k in h_sort: #Sort by elevation
        if not np.any(np.isnan(lakeStratPeriod[k])):
            if lakeStratPeriod[k][1]-lakeStratPeriod[k][0]>0:
                ax.add_patch(plt.Rectangle([x[3*kk],lakeStratPeriod[k][0]],np.diff(x)[3*kk],np.diff(lakeStratPeriod[k])[0],color='xkcd:coral'))
            else: #Case of stratification ending after December 31st
                ax.add_patch(plt.Rectangle([x[3*kk],0],np.diff(x)[3*kk],lakeStratPeriod[k][1],color='xkcd:coral'))
                ax.add_patch(plt.Rectangle([x[3*kk],lakeStratPeriod[k][0]],np.diff(x)[3*kk],365-lakeStratPeriod[k][0],color='xkcd:coral'))
            if lakeLastStratPeriod[k][1]-lakeLastStratPeriod[k][0]>0:
                plt.plot([xmid[kk],xmid[kk]],[lakeLastStratPeriod[k][0],lakeLastStratPeriod[k][1]],color='xkcd:maroon')
            else: #Case of stratification ending after December 31st
                plt.plot([xmid[kk],xmid[kk]],[0,lakeLastStratPeriod[k][1]],color='xkcd:maroon')
                plt.plot([xmid[kk],xmid[kk]],[lakeLastStratPeriod[k][0],365],color='xkcd:maroon')
        if not np.any(np.isnan(lakeIcePeriod[k])):
            if lakeIcePeriod[k][1]-lakeIcePeriod[k][0]>0:
                ax.add_patch(plt.Rectangle([x[3*kk],lakeIcePeriod[k][0]],np.diff(x)[3*kk],np.diff(lakeIcePeriod[k])[0],alpha=nIceYears[k]/nSimYears[k]))
                #ax.text(x[3*kk]+np.diff(x)[3*kk]/2,np.mean(lakeIcePeriod[k]),str(nIceYears[k])+'/'+str(nSimYears[k]),horizontalalignment='center',verticalalignment='center')
            else: #Case of ice cover starting before December 31st
                ax.add_patch(plt.Rectangle([x[3*kk],0],np.diff(x)[3*kk],lakeIcePeriod[k][1],alpha=nIceYears[k]/nSimYears[k]))
                ax.add_patch(plt.Rectangle([x[3*kk],lakeIcePeriod[k][0]],np.diff(x)[3*kk],365-lakeIcePeriod[k][0],alpha=nIceYears[k]/nSimYears[k]))
                #ax.text(x[3*kk]+np.diff(x)[3*kk]/2,lakeIcePeriod[k][1]/2,str(nIceYears[k])+'/'+str(nSimYears[k]),horizontalalignment='center',verticalalignment='center')
            if lakeLastIcePeriod[k][1]-lakeLastIcePeriod[k][0]>0:
                plt.plot([xmid[kk],xmid[kk]],[lakeLastIcePeriod[k][0],lakeLastIcePeriod[k][1]],color='xkcd:navy')
            else: #Case of ice cover starting before December 31st
                plt.plot([xmid[kk],xmid[kk]],[0,lakeLastIcePeriod[k][1]],color='xkcd:navy')
                plt.plot([xmid[kk],xmid[kk]],[lakeLastIcePeriod[k][0],365],color='xkcd:navy')
        kk = kk+1
    plt.gca().xaxis.tick_top()
    plt.tick_params(axis='both',which='both',top=False,labelsize=16)
    plt.xticks(xmid,lakeNames[h_sort],rotation=90)
    plt.xlim(min(x),max(x))
    plt.ylabel('Day of year',fontsize=16)
    plt.ylim(0,365)
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join('Plots','All_StratificationPeriod.eps'),format='eps',bbox_inches='tight')
    plt.savefig(os.path.join('Plots','All_StratificationPeriod.png'),dpi=250,bbox_inches='tight')
    plt.clf(); plt.close()
    
    #Create the aggregated plot of mean temperature, heat content, Schmidt stability
    lake_trend = []
    Tm_trend,Tm_trend_pval,Tm_trend_stderr = [],[],[]
    Ts_trend,Ts_trend_pval,Ts_trend_stderr = [],[],[]
    Tb_trend,Tb_trend_pval,Tb_trend_stderr = [],[],[]
    for k in range(len(lakeHC)):
        years_all = np.array([tk.year for tk in tlake[k]])
        years = np.unique(years_all)[1:-1] #All full years
        if years[-1]-years[0]>=20:
            Tm_yr,Ts_yr,Tb_yr = [],[],[]
            for yr in years:
                Tm_yr.append(np.mean(np.array(lakeTmean[k])[yr==years_all]))
                Ts_yr.append(np.mean(np.array(lakeTsurf[k])[yr==years_all]))
                Tb_yr.append(np.mean(np.array(lakeTbott[k])[yr==years_all]))
            plt.plot(years,Tm_yr)
            lake_trend.append(lakeNames[k])
            linreg = stats.linregress(years,Tm_yr)
            Tm_trend.append(linreg.slope)
            Tm_trend_pval.append(linreg.pvalue)
            Tm_trend_stderr.append(linreg.stderr)
            linreg = stats.linregress(years,Ts_yr)
            Ts_trend.append(linreg.slope)
            Ts_trend_pval.append(linreg.pvalue)
            Ts_trend_stderr.append(linreg.stderr)
            linreg = stats.linregress(years,Tb_yr)
            Tb_trend.append(linreg.slope)
            Tb_trend_pval.append(linreg.pvalue)
            Tb_trend_stderr.append(linreg.stderr)
        else:
            Tm_trend.append(np.nan)
            Tm_trend_pval.append(np.nan)
            Tm_trend_stderr.append(np.nan)
            Ts_trend.append(np.nan)
            Ts_trend_pval.append(np.nan)
            Ts_trend_stderr.append(np.nan)
            Tb_trend.append(np.nan)
            Tb_trend_pval.append(np.nan)
            Tb_trend_stderr.append(np.nan)
    plt.ylabel('Mean temperature [°C]')
    plt.savefig(os.path.join('Plots','All_MeanTemperatureEvolution.eps'),format='eps')
    plt.savefig(os.path.join('Plots','All_MeanTemperatureEvolution.png'),dpi=250)
    plt.clf(); plt.close()
    
    #Create the aggregated plot of heat content
    hc_trend,hc_trend_pval = [],[]
    for k in range(len(lakeHC)):
        years_all = np.array([tk.year for tk in tlake[k]])
        years = np.unique(years_all)[1:-1] #All full years
        if years[-1]-years[0]>30:
            hc_max,hc_avg,hc_min = [],[],[]
            for yr in years:
                hc_max.append(max(np.array(lakeHC[k])[yr==years_all]))
                hc_avg.append(np.mean(np.array(lakeHC[k])[yr==years_all]))
                hc_min.append(min(np.array(lakeHC[k])[yr==years_all]))
            plt.semilogy(years,hc_avg,color='grey')
            plt.fill_between(years,hc_min,hc_max,facecolor='xkcd:bluegrey',alpha=0.5)
            linreg = stats.linregress(years,hc_avg)
            hc_trend.append(linreg.slope)
            hc_trend_pval.append(linreg.pvalue)
    plt.ylabel('Heat content [J]')
    plt.savefig(os.path.join('Plots','All_HeatContentEvolution.eps'),format='eps')
    plt.savefig(os.path.join('Plots','All_HeatContentEvolution.png'),dpi=250)
    plt.clf(); plt.close()
    
    #Create the aggregated plot of Schmidt stability
    sc_trend,sc_trend_pval = [],[]
    for k in range(len(lakeHC)):
        years_all = np.array([tk.year for tk in tlake[k]])
        years = np.unique(years_all)[1:-1] #All full years
        if years[-1]-years[0]>30:
            sc_max,sc_avg,sc_min = [],[],[]
            for yr in years:
                sc_max.append(max(np.array(lakeSC[k])[yr==years_all]))
                sc_avg.append(np.mean(np.array(lakeSC[k])[yr==years_all]))
                sc_min.append(min(np.array(lakeSC[k])[yr==years_all]))
            plt.semilogy(years,sc_avg,color='grey')
            plt.fill_between(years,sc_min,sc_max,facecolor='xkcd:bluegrey',alpha=0.5)
            linreg = stats.linregress(years,sc_avg)
            sc_trend.append(linreg.slope)
            sc_trend_pval.append(linreg.pvalue)
    plt.ylabel('Schmidt stability [J/m$^2$]')
    plt.savefig(os.path.join('Plots','All_SchmidtStabilityEvolution.eps'),format='eps')
    plt.savefig(os.path.join('Plots','All_SchmidtStabilityEvolution.png'),dpi=250)
    plt.clf(); plt.close()
    
    #Create the aggregated plots of temperature trend
    fig = plt.figure(figsize=(16,5))
    ax = fig.add_axes([0,0,1,1])
    plt.plot([min(x),max(x)],[0,0],color='k',linestyle='--')
    kk = 0
    for k in h_sort: #Sort by elevation
        plt.plot([x[3*kk],x[3*kk+1]],[10*Tm_trend[k],10*Tm_trend[k]],color='k')
        ax.add_patch(plt.Rectangle([x[3*kk],10*Tm_trend[k]-1.96*10*Tm_trend_stderr[k]],np.diff(x)[3*kk],3.92*10*Tm_trend_stderr[k],color='k',alpha=0.5))
        kk = kk+1
    plt.gca().xaxis.tick_top()
    plt.tick_params(axis='both',which='both',top=False,labelsize=16)
    plt.xticks(xmid,lakeNames[h_sort],rotation=90)
    plt.xlim(min(x),max(x))
    plt.ylabel('Trend of yearly mean temperature [°C/decade]',fontsize=16)
    plt.ylim(-1,2)
    plt.savefig(os.path.join('Plots','All_TemperatureMeanTrend.eps'),format='eps',bbox_inches='tight')
    plt.savefig(os.path.join('Plots','All_TemperatureMeanTrend.png'),dpi=250,bbox_inches='tight')
    plt.clf(); plt.close()
    fig = plt.figure(figsize=(16,5))
    ax = fig.add_axes([0,0,1,1])
    plt.plot([min(x),max(x)],[0,0],color='k',linestyle='--')
    kk = 0
    for k in h_sort: #Sort by elevation
        plt.plot([x[3*kk],x[3*kk+1]],[10*Ts_trend[k],10*Ts_trend[k]],color='k')
        ax.add_patch(plt.Rectangle([x[3*kk],10*Ts_trend[k]-1.96*10*Ts_trend_stderr[k]],np.diff(x)[3*kk],3.92*10*Ts_trend_stderr[k],color='xkcd:coral',alpha=0.5))
        kk = kk+1
    plt.gca().xaxis.tick_top()
    plt.tick_params(axis='both',which='both',top=False,labelsize=16)
    plt.xticks(xmid,lakeNames[h_sort],rotation=90)
    plt.xlim(min(x),max(x))
    plt.ylabel('Trend of yearly surface temperature [°C/decade]',fontsize=16)
    plt.ylim(-1,2)
    plt.savefig(os.path.join('Plots','All_TemperatureSurfTrend.eps'),format='eps',bbox_inches='tight')
    plt.savefig(os.path.join('Plots','All_TemperatureSurfTrend.png'),dpi=250,bbox_inches='tight')
    plt.clf(); plt.close()
    fig = plt.figure(figsize=(16,5))
    ax = fig.add_axes([0,0,1,1])
    plt.plot([min(x),max(x)],[0,0],color='k',linestyle='--')
    kk = 0
    for k in h_sort: #Sort by elevation
        plt.plot([x[3*kk],x[3*kk+1]],[10*Tb_trend[k],10*Tb_trend[k]],color='k')
        ax.add_patch(plt.Rectangle([x[3*kk],10*Tb_trend[k]-1.96*10*Tb_trend_stderr[k]],np.diff(x)[3*kk],3.92*10*Tb_trend_stderr[k],alpha=0.5))
        kk = kk+1
    plt.gca().xaxis.tick_top()
    plt.tick_params(axis='both',which='both',top=False,labelsize=16)
    plt.xticks(xmid,lakeNames[h_sort],rotation=90)
    plt.xlim(min(x),max(x))
    plt.ylabel('Trend of yearly bottom temperature [°C/decade]',fontsize=16)
    plt.ylim(-1,2)
    plt.savefig(os.path.join('Plots','All_TemperatureBottTrend.eps'),format='eps',bbox_inches='tight')
    plt.savefig(os.path.join('Plots','All_TemperatureBottTrend.png'),dpi=250,bbox_inches='tight')
    plt.clf(); plt.close()
    fig = plt.figure(figsize=(16,5))
    ax = fig.add_axes([0,0,1,1])
    plt.plot([min(x),max(x)],[0,0],color='k',linestyle='--')
    kk = 0
    for k in h_sort: #Sort by elevation
        plt.plot([x[3*kk],x[3*kk+1]],[10*Ts_trend[k],10*Ts_trend[k]],color='k')
        ax.add_patch(plt.Rectangle([x[3*kk],10*Ts_trend[k]-1.96*10*Ts_trend_stderr[k]],np.diff(x)[3*kk],3.92*10*Ts_trend_stderr[k],color='xkcd:coral',alpha=0.5))
        plt.plot([x[3*kk],x[3*kk+1]],[10*Tb_trend[k],10*Tb_trend[k]],color='k')
        ax.add_patch(plt.Rectangle([x[3*kk],10*Tb_trend[k]-1.96*10*Tb_trend_stderr[k]],np.diff(x)[3*kk],3.92*10*Tb_trend_stderr[k],alpha=0.5))
        kk = kk+1
    plt.gca().xaxis.tick_top()
    plt.tick_params(axis='both',which='both',top=False,labelsize=16)
    plt.xticks(xmid,lakeNames[h_sort],rotation=90)
    plt.xlim(min(x),max(x))
    plt.ylabel('Trend of yearly temperatures [°C/decade]',fontsize=16)
    plt.ylim(-1,2)
    plt.savefig(os.path.join('Plots','All_TemperatureSurfBottTrend.eps'),format='eps',bbox_inches='tight')
    plt.savefig(os.path.join('Plots','All_TemperatureSurfBottTrend.png'),dpi=250,bbox_inches='tight')
    plt.clf(); plt.close()
    
    #Create the plot surface temperature vs elevation
    data_surf = np.array([p[0] for p in profilesT])
    if plotly:
        data = [go.Scatter(
                    x=lakeElevations[z_sort],
                    y=data_surf[z_sort],
                    mode='markers',
                    text=lakeNames[z_sort])
                ]
        layout = go.Layout(
                    hovermode='closest',
                    xaxis=dict(title='Elevation [m]',showgrid=False),
                    yaxis=dict(title='Surface temperature [°C]',showgrid=False),
                )
        py.plot(dict(data=data,layout=layout),config={'showLink':False,'displaylogo':False,'modeBarButtonsToRemove':buttons},filename='elevation.html')#,include_plotlyjs=False,output_type='div')
    
    #Create the plot Schmidt stability vs mean depth
    if plotly:
        data = [go.Scatter(
                    x=lakeDepths[z_sort],
                    y=np.abs(lakeSTs)[z_sort],
                    mode='markers',
                    text=lakeNames[z_sort])
                ]
        layout = go.Layout(
                    hovermode='closest',
                    xaxis=dict(title='Lake depth [m]',showgrid=False),
                    yaxis=dict(title='Schmidt stability [J/m<sup>2</sup>]',showgrid=False),
                )
        py.plot(dict(data=data,layout=layout),config={'showLink':False,'displaylogo':False,'modeBarButtonsToRemove':buttons},filename='schmidt.html')#,include_plotlyjs=False,output_type='div')
    
    #Create the plot of rmse and correlation coefficients
    fid = open('CalibrationResults.csv')
    obsrd = csv.reader(fid,delimiter=';')
    next(obsrd), next(obsrd)
    label,rmse,corr = [],[],[]
    for row in obsrd:
        if sum([el!='' for el in row])>1:
            label.append(row[0])
            rmse.append(float(row[-2]))
            corr.append(float(row[-1]))
    fid.close()
    plt.figure(figsize=(7,5))
    plt.scatter(corr,rmse)
    plt.xlabel('Correlation coefficient [-]')
    plt.xlim(0.925,1.0)
    plt.ylabel('Root mean square error [°C]')
    plt.ylim(0.0,2.0)
    for k in range(len(label)):
        plt.annotate(k+1,(corr[k],rmse[k]))
    plt.annotate('\n'.join([str(k+1)+': '+label[k] for k in range(len(label))]),(1.001,0),annotation_clip=False,fontsize=7)
    plt.savefig(os.path.join('Plots','All_CorrVsRMSE.eps'),format='eps',bbox_inches='tight')
    plt.savefig(os.path.join('Plots','All_CorrVsRMSE.png'),bbox_inches='tight')
    plt.clf(); plt.close()
    if plotly:
        data = [go.Scatter(
                    x=np.round(corr,3),
                    y=np.round(rmse,2),
                    mode='markers',
                    text=label)
                ]
        layout = go.Layout(
                    hovermode='closest',
                    xaxis=dict(title='Correlation coefficient [-]',showgrid=False),
                    yaxis=dict(title='Root mean square error [°C]',showgrid=False),
                )
        py.plot(dict(data=data,layout=layout),config={'showLink':False,'displaylogo':False,'modeBarButtonsToRemove':buttons},filename='performance.html')#,include_plotlyjs=False,output_type='div')

    return lake_trend,[Tm_trend,Tm_trend_pval],[Ts_trend,Ts_trend_pval],[Tb_trend,Tb_trend_pval],[hc_trend,hc_trend_pval],[sc_trend,sc_trend_pval]

#Get start and end time of model
def modelPeriod(lakeName,t_ref,version):
    if version.startswith('1.'):
        k=0
        for line in open(os.path.join('Simstrat',lakeName+'.par')):
            k=k+1
            if k==16: tstart = t_ref + timedelta(float(line.split()[0]))
            if k==17: tend = t_ref + timedelta(float(line.split()[0]))
    elif version.startswith('2.'):
        par = json.load(codecs.open(os.path.join('Simstrat',lakeName+'.par'),'r','utf-8'))
        tstart = t_ref + timedelta(par['Simulation']['Start d'])
        tend = t_ref + timedelta(par['Simulation']['End d'])
    return (tstart,tend)

#Get start and end time of calibration
def calibPeriod(lakeName,t_ref):
    k=0
    for line in open(os.path.join('PEST',lakeName,'simstrat_tout.dat')):
        k=k+1
        if k==1: continue
        elif k==2: tstart=t_ref+timedelta(float(line))
        else: tend=t_ref+timedelta(float(line))
    return (tstart,tend)

#Get results of calibration
def calibResults(Lake):
    lakeName = simple(Lake['Name'])
    #Summarize the calibration results
    par = []
    val = []
    #Get names and values of calibrated parameter
    first = True
    for line in open(os.path.join('PEST',lakeName,'simstrat_calib.par')):
        if first:
            first = False
            continue
        par.append(line.split()[0])
        val.append(line.split()[1])
    #Compute root mean square errors (RMSE) for all the observation groups
    obsgrp=''
    k=-1
    nobs=np.array([])
    rmse=np.array([])
    for line in open(os.path.join('PEST',lakeName,'simstrat_calib.res')):
        line = line.split()
        if line[0]=='Name': continue
        if line[1]!=obsgrp:
            obsgrp=line[1]
            k=k+1
            nobs = np.append(nobs,0)
            rmse = np.append(rmse,0)
        nobs[k] = nobs[k]+1
        rmse[k] = rmse[k]+float(line[4])**2
    nobs = list(nobs)
    rmse = list(np.sqrt(rmse/nobs))
    #Get correlation coefficient
    for line in open(os.path.join('PEST',lakeName,'simstrat_calib.rec')):
        if line.strip().startswith('Correlation coefficient'):
            corrcoeff = float(line.split()[-1])
    if (('rmse' not in locals()) or ('corrcoeff' not in locals())):
        raise Exception('The calibration record file is incomplete. Calibration seems to have failed. Check files "simstrat_calib.rec" and "simstrat_calib.rmr".')
    return (nobs,par,val,rmse,corrcoeff)

#Create the plot of temperature residuals
def createResidualsPlots(Lake,t_ref):
    lakeName = simple(Lake['Name'])
    with open(os.path.join('PEST',lakeName,'simstrat_tout.dat')) as fid:
        tout = fid.readlines()[1:]
    tout = np.array([t_ref+timedelta(float(t)) for t in tout])
    with open(os.path.join('PEST',lakeName,'simstrat_zout.dat')) as fid:
        zout = fid.readlines()[1:]
    zout = np.array(sorted([float(z) for z in zout]))
    res = np.empty((len(tout),len(zout)))*np.nan
    for line in open(os.path.join('PEST',lakeName,'simstrat_calib.res')):
        line = line.split()
        if line[0]=='Name': continue
        if line[0].lower().startswith('t'): #Only consider temperature
            ind_t = int(line[0].split('_')[1])-1
            ind_z = int(line[0].split('_')[2])-1
            res[ind_t,ind_z] = -float(line[4])
    maxres = np.nanmax(abs(res))
    if maxres>10:
        ind = np.where(abs(res)==maxres)
        writelog('\tMaximum absolute residual is above 10 °C (on %s at %g m). There may be erroneous data in the observation file.\n' % (datetime.strftime(tout[ind[0][0]],'%d.%m.%Y'),zout[ind[1][0]]))
    plt.contourf(tout,zout,np.transpose(res),cmap='coolwarm',norm=mpl.colors.Normalize(vmin=-maxres,vmax=maxres))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Residual Tmodel-Tobs [°C]')
    plt.xlabel('Time')
    plt.ylabel('Depth [m]')
    plt.savefig(os.path.join('Plots',lakeName+'_residuals.png'),dpi=250)
    plt.clf(); plt.close()
    for t in range(len(res)): #Interpolate the residuals through depth
        if sum(~np.isnan(res[t]))>1:
            res[t] = np.interp(zout,zout[~np.isnan(res[t])],res[t][~np.isnan(res[t])])
    plt.contourf(tout,zout,np.transpose(res),cmap='coolwarm',norm=mpl.colors.Normalize(vmin=-maxres,vmax=maxres))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Residual Tmodel-Tobs [°C]')
    plt.xlabel('Time')
    plt.ylabel('Depth [m]')
    plt.savefig(os.path.join('Plots',lakeName+'_residuals_interp.png'),dpi=250)
    plt.clf(); plt.close()

#Write the main calibration results to a file
def writeCalibResults(Lake,t_ref,calib,par_calib,useSal,first=True,version='1.6'):
    lakeName = simple(Lake['Name'])
    if first:
        fidres = open('CalibrationResults.csv','w')
        par = list(getParVal(Lake,False,version).keys())
        fidres.write('Lake;Calibration period;'+';'.join(par)+(';'+'Root mean square error [°C]')+((';'+'Root mean square error [ppt]') if useSal else '')+';'+'Correlation coefficient'+'\n')
        fidres.write(';;'+';'.join([('Calibrated' if p in par_calib else 'Fixed') for p in par])+';;\n')
    else:
        fidres = open('CalibrationResults.csv','a')
    fidres.write(Lake['Name'])
    if calib:
        (tstart_calib,tend_calib) = calibPeriod(lakeName,t_ref)
        (_,_,val,rmse,corrcoeff) = calibResults(Lake)
        fidres.write(';'+str(tstart_calib.year)+'-'+str(tend_calib.year))
        fidres.write(';'+';'.join(val))
        for k in range(len(rmse)):
            fidres.write(';'+str(rmse[k]))
        fidres.write(';'+str(corrcoeff))
    fidres.write('\n')
    fidres.close()


def meteoInfo(Lake,meteoD):
    #Meteo data
    pars = json.load(codecs.open('Meteo.json','r','utf-8'))
    par_names = [par['Name'] for par in pars]
    par_ids = [par['Short name'] for par in pars]
    meteoStations = Lake['Data']['Weather station']
    if type(meteoStations)==str: meteoStations = [meteoStations]
    meteoData = {}
    meteoNames = {}
    for stn in meteoStations:
        meteoData[stn] = []
        for line in open(os.path.join(meteoD,stn+'_legend.txt')):
            if line[0:3]==stn:
                meteoData[stn].append(par_names[par_ids.index(line[47:55])].split(' [')[0])
                meteoNames[stn] = line[10:46].strip()
    return (meteoNames,meteoData)

def hydroInfo(Lake,hydroD):
    #Hydro data
    inflowStations = Lake['Data']['Hydro-inflow station']
    if type(inflowStations) is str: inflowStations=[inflowStations]
    inflowNames = {}
    inflowData = {}
    inflowType = 'River'
    for stn in inflowStations:
        inflowData[stn] = []
        if os.path.exists(os.path.join(hydroD,'Q_'+stn+'_Stundenmittel.asc')):
            inflowData[stn].append('Flowrate')
            for line in open(os.path.join(hydroD,'Q_'+stn+'_Stundenmittel.asc')):
                if line!='\n' and line[0]!='*':
                    inflowNames[stn] = line[13:].split()[0]
                    if inflowNames[stn][-1]==',': inflowNames[stn]=inflowNames[stn][:-1]
                    break
        if os.path.exists(os.path.join(hydroD,'T_'+stn+'_Stundenmittel.asc')):
            inflowData[stn].append('Temperature')
            if stn not in inflowNames:
                for line in open(os.path.join(hydroD,'T_'+stn+'_Stundenmittel.asc')):
                    if line!='\n' and line[0]!='*':
                        inflowNames[stn] = re.split(r' \d',line[13:])[0].split(',')[0]
                        break
    if len(inflowNames)==0:
        if 'Hydro-inflow lake' in Lake['Data']:
            inflowLakes = Lake['Data']['Hydro-inflow lake']
            if type(inflowLakes) is str: inflowLakes=[inflowLakes]
            inflowType = 'Lake'
            inflowNames = {'Lake'+str(k):inflowLakes[k] for k in range(len(inflowLakes))}
            inflowData = {}
    return (inflowType,inflowNames,inflowData)

#Write the metadata file
def writeMetadata(Lake,t_ref,meteoD,hydroD,dt,dz,version):
    lakeName = simple(Lake['Name'])
    #Get info about the meteo and inflow data used
    (meteoNames,meteoData) = meteoInfo(Lake,meteoD)
    (inflowType,inflowNames,inflowData) = hydroInfo(Lake,hydroD)
    #Write metadata file
    fid = open(os.path.join('Metadata',lakeName+'.txt'),'w',encoding='utf-8')
    fid.write('Data was generated by Simstrat v%s, a one-dimensional hydrodynamic lake model developed at Eawag. Simstrat was initally described by Goudsmit et al. (2002) and improved by Schmid and Köster (2016). Simstrat v%s is available at https://github.com/Eawag-AppliedSystemAnalysis/Simstrat/releases/tag/v%s.\n\n' % (version,version,version))
    fid.write('%s - for meteorological forcing, data from the MeteoSwiss stations %s was used; ' % (Lake['Name'],', '.join([stn for stn in meteoData])))
    if len(inflowNames)==0: fid.write('and no hydrological forcing was applied (lake is considered as a closed basin).\n\n')
    elif inflowType=='River': fid.write('for hydrological forcing, data from the FOEN station(s) %s was used.\n\n' % (', '.join([stn for stn in inflowData])))
    elif inflowType=='Lake': fid.write('for hydrological forcing, data from the upstream lake(s) %s was used.\n\n' % (', '.join([inflowData[stn] for stn in inflowData])))
    fid.write('The result files contain model output at a spatial resolution of %.1f m and temporal resolution of %.1f hr, over the whole depth and simulation period for each lake. Each result file contains a text-readable table (rows: time, columns: depths) for one of the modeled physical property (temperature, salinity, Brunt-Väisälä frequency, vertical diffusivity). In the results files, time is given as days since the reference date %s, and depth as (negative) meters below the initial lake surface.\n\n' % (dz,dt/3600,datetime.strftime(t_ref,'%d.%m.%Y')))
    fid.close()

#Write the HTML and metadata file
def writeHTML(Lake,t_ref,meteoD,hydroD,calib,plots,version):
    lakeName = simple(Lake['Name'])
    
    #Get info about the meteo and inflow data used
    (meteoNames,meteoData) = meteoInfo(Lake,meteoD)
    (inflowType,inflowNames,inflowData) = hydroInfo(Lake,hydroD)

    #Retrieve model start and end times
    (tstart,tend) = modelPeriod(lakeName,t_ref,version)
    
    #Write HTML file
    fid = open(os.path.join('Web','html',lakeName+'.html'),'w',encoding='utf-8')
    fid.write('<!DOCTYPE html>\n')
    fid.write('<html>\n')
    fid.write('\t<head>\n')
    fid.write('\t\t<title>Simstrat - %s</title>\n' % Lake['Name'])
    fid.write('\t\t<meta charset="utf-8">\n')
    fid.write('\t\t<meta http-equiv="X-UA-Compatible" content="IE=edge">\n')
    fid.write('\t\t<meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
    fid.write('\t\t<meta name="author" content="Adrien Gaudard, Eawag">\n')
    fid.write('\t\t<link href="static/bootstrap/dist/css/bootstrap.min.css" rel="stylesheet">\n')
    fid.write('\t\t<link href="static/css/chart.css" rel="stylesheet">\n')
    fid.write('\t\t<script src="static/lib/js/plotly-latest.min.js"></script>\n')
    fid.write('\t</head>\n')
    fid.write('\t<body>\n')
    fid.write('\t\t<nav class="navbar navbar-inverse navbar-fixed-top">\n')
    fid.write('\t\t\t<div class="container">\n')
    fid.write('\t\t\t\t<h2 style=margin-bottom:10px;color:white>Simstrat - 1D lake model</h2>\n')
    fid.write('\t\t\t\t<div class="navbar-header">\n')
    fid.write('\t\t\t\t\t<a class="navbar-brand" style=margin-right:10px href="/">Home</a>\n')
    fid.write('\t\t\t\t\t<a class="navbar-brand" style=margin-right:10px href="/lakes">Modeled lakes</a>\n')
    fid.write('\t\t\t\t\t<a class="navbar-brand" style=margin-right:10px href="/graphs">Graphical comparison</a>\n')
    fid.write('\t\t\t\t\t<a class="navbar-brand" style=margin-right:10px href="/about">About</a>\n')
    fid.write('\t\t\t\t\t<a class="navbar-brand" style=margin-right:10px href="/participate">Participate</a>\n')
    fid.write('\t\t\t\t</div>\n')
    fid.write('\t\t\t</div>\n')
    fid.write('\t\t</nav>\n')
    fid.write('\t\t<div style=margin-top:100px;margin-left:25px;max-width:800px>\n')
    fid.write('\t\t\t<br>\n')
    fid.write('\t\t\t<a style=float:right href="javascript:history.back()">Back to previous page</a>\n')
    fid.write('\t\t\t<h3 style=margin-bottom:25px>%s</h3>\n' % Lake['Name'])
    fid.write('\t\t\t<p>Model time frame: %s - %s</p>\n' % (datetime.strftime(tstart,'%d.%m.%Y'),datetime.strftime(tend,'%d.%m.%Y')))
    fid.write('\t\t\t<p>Weather stations (<a href="http://www.meteoswiss.admin.ch/" target=_top>MeteoSwiss</a>):</p>\n')
    fid.write('\t\t\t<ul>\n')
    for stn in meteoData:
        fid.write('\t\t\t<li>%s <a href="http://www.meteoswiss.admin.ch/home/measurement-and-forecasting-systems/land-based-stations/automatisches-messnetz.html?station=%s" target=_top>%s</a> (%s)</li>\n' % (meteoNames[stn],stn.lower(),stn,', '.join([par for par in meteoData[stn]])))
    fid.write('\t\t\t</ul>\n')
    if len(inflowNames)>0:
        fid.write('\t\t\t<p>Hydro-inflow %s (<a href="http://hydrodaten.admin.ch/en/" target=_top>Hydrodata-FOEN</a>):</p>\n' % ('station(s)' if inflowType=='River' else 'lake(s)'))
        fid.write('\t\t\t<ul>\n')
        for stn in inflowNames:
            if inflowType=='River': fid.write('\t\t\t<li>%s <a href="http://hydrodaten.admin.ch/en/%s.html" target=_top>%s</a> (%s)</li>\n' % (inflowNames[stn],stn,stn,', '.join([par for par in inflowData[stn]])))
            if inflowType=='Lake': fid.write('\t\t\t<li><a href="https://simstrat.eawag.ch/%s" target=_top>%s</a></li>\n' % (simple(inflowNames[stn]),inflowNames[stn]))
        fid.write('\t\t\t</ul>\n')
    if not calib:
        fid.write('\t\t\t<p>Model parameters were not calibrated.</p><br>\n')
    else:
        (tstart_calib,tend_calib) = calibPeriod(lakeName,t_ref)
        (_,_,_,rmse,corrcoeff) = calibResults(Lake)
        fid.write('\t\t\t<p>Model parameters were calibrated with <a href="http://pesthomepage.org/" target=_top>PEST</a> v15.0, using %s observations from %d to %d.</p><br>\n' % ('temperature'+(' and salinity' if len(rmse)>1 else ''),tstart_calib.year,tend_calib.year))
        #Model quality (qualitative assessment based on R^2 and temperature RMSE)
        MQ = (0 if (corrcoeff>=0.95 and rmse[0]<0.5) else 1 if (corrcoeff>=0.90 and rmse[0]<0.8) else 2 if (corrcoeff>=0.85 and rmse[0]<1.3) else 3)
        MQtext = ['Excellent','Good','Satisfactory','We\'re working on it...']
        MQcolor = ['#006600','#00cc99','#cc9900','#cc0000']
        fid.write('\t\t\t<p style=margin-left:400px>Model quality: <span style=color:%s><b>%s</b></span> (R<sup>2</sup> = %.3f, RMSE = %.2f°C%s)</p>\n' % (MQcolor[MQ],MQtext[MQ],corrcoeff,rmse[0],(' / %.2f ppt' % rmse[1]) if len(rmse)>1 else ''))
    fid.write('\t\t\t<span style=margin-right:200px>Download model setup/metadata/results:\n')
    fid.write('\t\t\t<select id="file" onchange="location = this.value;">\n')
    fid.write('\t\t\t\t<option disabled selected="selected">Choose file</option>\n')
    fid.write('\t\t\t\t<option value="static/templates/setup/%s.par">Model setup</option>\n' % lakeName)
    fid.write('\t\t\t\t<option value="static/templates/metadata/%s.txt">Metadata</option>\n' % lakeName)
    fid.write('\t\t\t\t<option value="static/templates/results/%s_T.dat">Temperature [°C]</option>\n' % lakeName)
    fid.write('\t\t\t\t<option value="static/templates/results/%s_S.dat">Salinity [‰]</option>\n' % lakeName)
    fid.write('\t\t\t\t<option value="static/templates/results/%s_N2.dat">Brunt-Väisälä frequency N&sup2; [s&#8315;&sup2;]</option>\n' % lakeName)
    fid.write('\t\t\t\t<option value="static/templates/results/%s_nuh.dat">Vertical diffusivity [m&sup2;/s]</option>\n' % lakeName)
    fid.write('\t\t\t</select></span>\n')
    fid.write('\t\t\t<p style=margin-left:600px>[Last updated: %s]</p>\n' % datetime.strftime(datetime.now(),'%d.%m.%Y %H:%M'))
    fid.write('\t\t\t<div class="tab">\n')
    fid.write('\t\t\t\t<button class="tablinks" onclick="openTab(event)" id="T">Temperature</button>\n')
    fid.write('\t\t\t\t<button class="tablinks" onclick="openTab(event)" id="S">Salinity (beta)</button>\n')
    fid.write('\t\t\t\t<button class="tablinks" onclick="openTab(event)" id="nuh">Thermal diffusivity</button>\n')
    fid.write('\t\t\t\t<button class="tablinks" onclick="openTab(event)" id="N2">Stratification</button>\n')
    fid.write('\t\t\t\t<button class="tablinks" onclick="openTab(event)" id="Schmidt">Schmidt stability</button>\n')
    fid.write('\t\t\t\t<button class="tablinks" onclick="openTab(event)" id="heat">Heat content</button>\n')
    fid.write('\t\t\t</div>\n')
    for var in plots.keys(): writeHTMLTab1(fid,var)
    for var in plots.keys(): writeHTMLPlotGroup(fid,var,plots[var])
    fid.write('\t\t\t<script>\n')
    fid.write('\t\t\t\tdocument.getElementById("T-actual_1").click(); //Default tab to display\n')
    fid.write('\t\t\t\tfunction openTab(evt) {\n')
    fid.write('\t\t\t\t\tvar i, id, tabcontent, tablinks;\n')
    fid.write('\t\t\t\t\tid = evt.currentTarget.id;\n')
    fid.write('\t\t\t\t\tid_global = id.split(\'_\')[0]\n')
    fid.write('\t\t\t\t\t//Hide all tabs\n')
    fid.write('\t\t\t\t\ttabcontent = document.getElementsByClassName("tabcontent");\n')
    fid.write('\t\t\t\t\tfor (i=0; i<tabcontent.length; i++) {\n')
    fid.write('\t\t\t\t\t\ttabcontent[i].style.display = "none";\n')
    fid.write('\t\t\t\t\t}\n')
    fid.write('\t\t\t\t\t//Remove all highlights of the buttons\n')
    fid.write('\t\t\t\t\ttablinks = document.getElementsByClassName("tablinks");\n')
    fid.write('\t\t\t\t\tfor (i=0; i<tablinks.length; i++) {\n')
    fid.write('\t\t\t\t\t\ttablinks[i].className = tablinks[i].className.replace(" active", "");\n')
    fid.write('\t\t\t\t\t}\n')
    fid.write('\t\t\t\t\tdocument.getElementById(id_global+"-tab").style.display = "block"; //Display tab\n')
    fid.write('\t\t\t\t\tdocument.getElementById(id.split("-")[0]).className += " active"; //Highlight first-level button\n')
    fid.write('\t\t\t\t\tif (id.indexOf("-")>-1) { //Highlight second-level button\n')
    fid.write('\t\t\t\t\t\tel = document.getElementById(id_global+"_0")\n')
    fid.write('\t\t\t\t\t\tk = 0\n')
    fid.write('\t\t\t\t\t\twhile(el) {\n')
    fid.write('\t\t\t\t\t\t\tel.className += " active";\n')
    fid.write('\t\t\t\t\t\t\tk = k+1\n')
    fid.write('\t\t\t\t\t\t\tel = document.getElementById(id_global+"_"+k.toString())\n')
    fid.write('\t\t\t\t\t\t}\n')
    fid.write('\t\t\t\t\t} else {\n')
    fid.write('\t\t\t\t\t\tif (id=="nuh") {;\n')
    fid.write('\t\t\t\t\t\t\tdocument.getElementById(id+"-history_1").click();\n')
    fid.write('\t\t\t\t\t\t} else {\n')
    fid.write('\t\t\t\t\t\t\tdocument.getElementById(id+"-actual_1").click();\n')
    fid.write('\t\t\t\t\t\t}\n')
    fid.write('\t\t\t\t\t}\n')
    fid.write('\t\t\t\t}\n')
    fid.write('\t\t\t</script>\n')
    if len(plots['T'][4])>1 and all(plots['T'][4][1:]): #if there are statistics plots (T-profiles)
        fid.write('\t\t\t<script>\n')
        fid.write('\t\t\t\tvar selectMonth = document.getElementById("selectMonth");\n')
        fid.write('\t\t\t\tvar profile = document.getElementById("statsProfile");\n')
        fid.write('\t\t\t\tselectMonth.onchange = function() {\n')
        for mth in range(1,13):
            fid.write('\t\t\t\t\t%sif (selectMonth.value==%d) {\n' % ('} else ' if mth>1 else '',mth))
            for mth2 in range(1,13):
                plot_id = plots['T'][4][mth2].split('id=')[-1].split()[0]
                fid.write('\t\t\t\t\t\tdocument.getElementById(%s).style.display = "%s"\n' % (plot_id,'block' if mth2==mth else 'none'))
        fid.write('\t\t\t\t\t}\n')
        fid.write('\t\t\t\t}\n')
        fid.write('\t\t\t\tdocument.getElementById("selectMonth").selectedIndex = %d\n' % 0)
        fid.write('\t\t\t</script>\n')
    if plots['T'][5][0] and all(plots['T'][5][0]): #if there are statistics plots (T-courses)
        fid.write('\t\t\t<script>\n')
        fid.write('\t\t\t\tvar selectDepth = document.getElementById("selectDepth");\n')
        fid.write('\t\t\t\tvar course = document.getElementById("statsCourse");\n')
        fid.write('\t\t\t\tselectDepth.onchange = function() {\n')
        for iz in range(len(plots['T'][5][1])):
            fid.write('\t\t\t\t\t%sif (selectDepth.value==%d) {\n' % ('} else ' if iz>0 else '',iz))
            for iz2 in range(len(plots['T'][5][1])):
                plot_id = plots['T'][5][0][iz2].split('id=')[-1].split()[0]
                fid.write('\t\t\t\t\t\tdocument.getElementById(%s).style.display = "%s"\n' % (plot_id,'block' if iz2==iz else 'none'))
        fid.write('\t\t\t\t\t}\n')
        fid.write('\t\t\t\t}\n')
        fid.write('\t\t\t\tdocument.getElementById("selectDepth").selectedIndex = %d\n' % (len(plots['T'][5][1])-1))
        fid.write('\t\t\t</script>\n')
    fid.write('\t\t</div>\n')
    fid.write('\t</body>\n')
    fid.write('</html>\n')
    fid.close()

def writeHTMLTab1(fid,var):
    if var=='ice': return
    fid.write('\t\t\t<div id="%s-tab" class="tabcontent">\n' % var)
    fid.write('\t\t\t\t<div class="tab">\n')
    fid.write('\t\t\t\t\t<button class="tablinks" onclick="openTab(event)" id="%s-history_0">Historical model</button>\n' % var)
    if var!='nuh': fid.write('\t\t\t\t\t<button class="tablinks" onclick="openTab(event)" id="%s-actual_0">Current situation</button>\n' % var)
    if var=='T': fid.write('\t\t\t\t\t<button class="tablinks" onclick="openTab(event)" id="%s-stats_0">Statistics</button>\n' % var)
    fid.write('\t\t\t\t</div>\n')
    fid.write('\t\t\t</div>\n')

def writeHTMLPlotGroup(fid,var,plots):
    if var=='ice': return
    k=1
    if True: #Tab "historical model"
        fid.write('\t\t\t<div id="%s-history-tab" class="tabcontent">\n' % var)
        fid.write('\t\t\t\t<div class="tab">\n')
        fid.write('\t\t\t\t\t<button class="tablinks" onclick="openTab(event)" id="%s-history_%d">Historical model</button>\n' % (var,k)) #Button "historical model"
        if var!='nuh': fid.write('\t\t\t\t\t<button class="tablinks" onclick="openTab(event)" id="%s-actual_%d">Current situation</button>\n' % (var,k)) #Button "current situation"
        if var=='T': fid.write('\t\t\t\t\t<button class="tablinks" onclick="openTab(event)" id="%s-stats_%d">Statistics</button>\n' % (var,k)) #Button "statistics"
        fid.write('\t\t\t\t</div>\n')
        if plots[0]:
            fid.write('\t\t\t\t'+plots[0][0]+'\n')
            if plots[0][1]:
                fid.write('\t\t\t\t<a style=float:right href="%s" target=_top>View interactive contour plot (loading may be slow)</a><br>\n' % plots[0][1])
        if plots[1]:
            if var!='nuh' and var!='N2': fid.write('\t\t\t\t'+plots[1]+'\n')
        fid.write('\t\t\t</div>\n')
        k=k+1
    if var!='nuh': #Tab "current situation"
        fid.write('\t\t\t<div id="%s-actual-tab" class="tabcontent">\n' % var)
        fid.write('\t\t\t\t<div class="tab">\n')
        fid.write('\t\t\t\t\t<button class="tablinks" onclick="openTab(event)" id="%s-history_%d">Historical model</button>\n' % (var,k)) #Button "historical model"
        if var!='nuh': fid.write('\t\t\t\t\t<button class="tablinks" onclick="openTab(event)" id="%s-actual_%d">Current situation</button>\n' % (var,k)) #Button "current situation"
        if var=='T': fid.write('\t\t\t\t\t<button class="tablinks" onclick="openTab(event)" id="%s-stats_%d">Statistics</button>\n' % (var,k)) #Button "statistics"
        fid.write('\t\t\t\t</div>\n')
        if plots[3]: fid.write('\t\t\t\t'+plots[3]+'\n')
        if plots[2]: fid.write('\t\t\t\t'+plots[2]+'\n')
        fid.write('\t\t\t</div>\n')
        k=k+1
    if var=='T':  #Tab "statistics"
        fid.write('\t\t\t<div id="%s-stats-tab" class="tabcontent">\n' % var)
        fid.write('\t\t\t\t<div class="tab">\n')
        fid.write('\t\t\t\t\t<button class="tablinks" onclick="openTab(event)" id="%s-history_%d">Historical model</button>\n' % (var,k)) #Button "historical model"
        if var!='nuh': fid.write('\t\t\t\t\t<button class="tablinks" onclick="openTab(event)" id="%s-actual_%d">Current situation</button>\n' % (var,k)) #Button "current situation"
        if var=='T': fid.write('\t\t\t\t\t<button class="tablinks" onclick="openTab(event)" id="%s-stats_%d">Statistics</button>\n' % (var,k)) #Button "statistics"
        fid.write('\t\t\t\t</div>\n')
        if (len(plots[4])>1 and all(plots[4][1:])):
            fid.write('\t\t\t\t<br>\n')
            fid.write('\t\t\t\t<h4 style=display:inline>Average temperature profile in </h4>\n')
            fid.write('\t\t\t\t<select id="selectMonth" name="list">\n')
            for mth in range(1,13):
                fid.write('\t\t\t\t\t<option value=%d>%s</option>\n' % (mth,list(calendar.month_name)[mth]))
            fid.write('\t\t\t\t</select>\n')
            for mth in range(1,13):
                fid.write('\t\t\t\t\t'+plots[4][mth]+'\n')
        if plots[5][0]and all(plots[5][0]):
            fid.write('\t\t\t\t<br>\n')
            fid.write('\t\t\t\t<h4 style=display:inline>Average temperature course at </h4>\n')
            fid.write('\t\t\t\t<select id="selectDepth" name="list">\n')
            for iz in range(len(plots[5][1])):
                fid.write('\t\t\t\t\t<option value=%d>%d m</option>\n' % (iz,plots[5][1][iz]))
            fid.write('\t\t\t\t</select>\n')
            for iz in range(len(plots[5][1])):
                fid.write('\t\t\t\t\t'+plots[5][0][iz]+'\n')
        if plots[6]:
            fid.write('\t\t\t\t<br>\n')
            fid.write('\t\t\t\t<h4>Long-term monthly temperature trends (slope of linear regression) [°C/decade]</h4>\n')
            fid.write('\t\t\t\t'+plots[6]+'\n')
        fid.write('\t\t\t</div>\n')


#Write KML file (for display on the web-map)
def writeKML(Lakes,Tsurf,mode3D=False):
    fid = open(os.path.join('Web','simstrat.kml'),'w',encoding='utf-8')
    fid.write('<kml xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:atom="http://www.w3.org/2005/Atom" xmlns="http://www.opengis.net/kml/2.2">\n')
    fid.write('<Document>\n')
    fid.write('\t<name>Simstrat lake model</name>\n')
    for k in range(len(Lakes)):
        lakeName = simple(Lakes[k]['Name'])
        coord = Lakes[k]['GeometryCH1903+']
        coord = [s.split(',') for s in coord[coord.find('((')+2:coord.find('))')].split('),(')]
        coord = [[s.split() for s in xy] for xy in coord]
        coord = [CH1903ptoWGS84(s) for s in coord]
        coord = [[[round(s[0],4),round(s[1],4)] for s in xy] for xy in coord]
        fid.write('\t<Placemark id="kml_lake_%d">\n' % k)
        fid.write('\t\t<description><![CDATA[\n')
        fid.write('<h5><a href="https://simstrat.eawag.ch/%s" target=_parent>%s</a></h5>\n' % (lakeName,Lakes[k]['Name']))
        fid.write('<p>Elevation: %.0f m<br>Depth: %g m</p>\n' % (Lakes[k]['Properties']['Elevation [m]'],Lakes[k]['Properties']['Max depth [m]']))
        if Tsurf[k] is not None:
            fid.write('<p>Surface temperature: %.1f °C</p>\n' % Tsurf[k])
        fid.write(']]></description>\n')
        fid.write('\t\t<Style>\n')
        fid.write('\t\t\t<PolyStyle>\n')
        fid.write('\t\t\t\t<color>%s</color>\n' % colorT(Tsurf[k]))
        fid.write('\t\t\t</PolyStyle>\n')
        fid.write('\t\t\t<LineStyle>\n')
        fid.write('\t\t\t\t<color>%s</color>\n' % 'ff000000')
        fid.write('\t\t\t</LineStyle>\n')
        fid.write('\t\t</Style>\n')
        if not mode3D: fid.write('\t\t<MultiGeometry>\n') #3D mode doesn't support multigeometries
        fid.write('\t\t\t<Polygon>\n')
        if mode3D: fid.write('\t\t\t\t<altitudeMode>absolute</altitudeMode>\n')
        fid.write('\t\t\t\t<outerBoundaryIs>\n')
        for kk in range(len(coord)):
            if mode3D and kk>0: break #3D mode doesn't support the display of islands
            if kk>0: fid.write('\t\t\t\t<innerBoundaryIs>\n')
            fid.write('\t\t\t\t\t<LinearRing>\n')
            fid.write('\t\t\t\t\t\t<coordinates>%s</coordinates>\n' % str(coord[kk]).replace('[[','').replace(']]','').replace('], [','%s ' % ((','+str(Lakes[k]['Properties']['Elevation [m]'])) if mode3D else ' ')).replace(', ',','))
            fid.write('\t\t\t\t\t</LinearRing>\n')
            if kk==0: fid.write('\t\t\t\t</outerBoundaryIs>\n')
            if kk>0: fid.write('\t\t\t\t</innerBoundaryIs>\n')
        fid.write('\t\t\t</Polygon>\n')
        if not mode3D: fid.write('\t\t</MultiGeometry>\n')
        fid.write('\t</Placemark>\n')
    fid.write('</Document></kml>\n')
    fid.close()

def retrieveNewMeteoData(wdir,t):
    writelog('Updating the meteorological data with the latest data from the FTP server...\n')
    #Retrieve the new file from the FTP server
    ftp = ftplib.FTP('ftp.eawag.ch','anonymous','adrien.gaudard@eawag.ch')
    ftp.cwd('incoming/gaudarad')
    data_new = ''
    def store(txt):
        nonlocal data_new
        data_new += txt.decode('ansi')
    file = '.'.join(['VQCA44',datetime.strftime(t,'%Y%m%d0813'),'csv'])
    try:
        ftp.retrbinary('RETR %s' % file,store)
    except:
        raise IOError('File %s with updated meteorological data not found.' % file)
    ftp.quit()
    #Convert the text to a matrix of values
    data_new = data_new.split('\n')
    var_new = data_new[2].split()
    data_new = np.array([line.split() for line in data_new[4:-1]])
    time_new = [datetime.strptime(t,'%d.%m.%Y%H:%M') for t in np.core.defchararray.add(data_new[:,1],data_new[:,2])] #Convert time to datetime
    data_new[:,1] = [datetime.strftime(t,'%Y%m%d%H') for t in time_new]
    data_new = np.delete(data_new,2,1)
    
    for stn in np.unique(data_new[:,0]):
        stnData_new = data_new[data_new[:,0]==stn,:]
        with open(os.path.join(wdir,stn+'_data.txt'),'r') as fid:
            data_raw = fid.readlines()
        for k in range(len(data_raw)):
            line = data_raw[k].replace('\n','').split(';')
            if line[0]=='stn':
                var = line
                while data_raw[k+1][0:3]==stn:
                    k = k+1
                tformat = '%Y%m%d%H' if len(data_raw[k].split(';')[1])==10 else '%Y%m%d'
                time_last = datetime.strptime(data_raw[k].split(';')[1],tformat)
                if (time_new[0]-time_last)>timedelta(days=1):
                    tprev = t-timedelta(days=7)
                    writelog('\tMeteorological seems not to be up to date. Trying to update with older data file (%s).' % datetime.strftime(tprev,'%Y-%m-%d'))
                    retrieveNewMeteoData(wdir,tprev)
                    retrieveNewMeteoData(wdir,t)
                    return
                if time_new[-1]<=time_last: continue
                first = np.argmax(np.array(time_new)>time_last)
                for new in stnData_new[first:]:
                    idx = [(var_new.index(v) if v in var else []) for v in var]
                    add = [new[i] if i!=[] else '-' for i in idx]
                    data_raw.insert(k+1,';'.join(add)+'\n')
                    k = k+1
        with open(os.path.join(wdir,stn+'_data.txt'),'w') as fid:
            fid.write(''.join(data_raw))
    writelog(' %d stations updated.\n' % len(np.unique(data_new[:,0])))


def retrieveNewHydroData(wdir,Lakes):
    writelog('Updating the hydrological data with the latest data from the FTP server...\n')
    #Retrieve the new file from the FTP server
    ftp = ftplib.FTP('ftp.hydrodata.ch','eawag','fous60{avant')
    def store(txt):
        nonlocal data_new
        data_new += txt.decode('ansi')
    hydroFiles = os.listdir(wdir)
    
    stations_T = [f[2:6] for f in hydroFiles if f[0]=='T']
    for stn in stations_T:
        #Retrieve temperature data
        data_new = ''
        T_new = []
        try:
            files = ftp.nlst('%s/Log' % stn)
        except:
            writelog('\tStation %s: no FTP directory found for updated temperature data.' % stn)
            continue
        if len(files)>0: #Datafiles found
            files.sort()
            for file in files:
                ftp.retrbinary('RETR %s' % file,store)
            data_new = data_new.split('\r\n')
            k=2
            while data_new[k][0:11]=='# Parameter':
                data_vec = data_new[k].split('\t')
                if 'Temperatur' in data_vec or 'Wassertemperatur' in data_vec:
                    idx_T = int(data_vec[-1])
                    break
                k=k+1
            for line in data_new:
                if (len(line)>0 and line[0]!='#'): #Discard empty and header lines
                    line = line.split('\t')
                    if line[1][3:]=='00:00': #Keep only full hours
                        T_new.append([line[0]+' '+line[1][:-3],line[2*(idx_T-1)+2]])
        elif stn in ['2009','2105','2276','2432','2433','2457','2473']: #No PLC logger at these stations: datafiles are in another folder
            files = ftp.nlst('hydropro/%s' % stn)
            files = [file for file in files if (stn+'_03') in file]
            for file in files:
                ftp.retrbinary('RETR %s' % file,store)
            data_new = data_new.split('\r')
            for line in data_new[1:]:
                if (len(line)>0 and line[0]!=';'): #Discard empty and header lines
                    line = line.split('\t')
                    if line[1][3:]=='00': #Keep only full hours
                        T_new.append([line[0]+' '+line[1],line[2]])
        else: #No datafile found
            writelog('\tStation %s: no file found with updated temperature data.' % stn)
            continue
        #Complete the data file with the new data
        with open(os.path.join(wdir,'T_'+stn+'_Stundenmittel.asc'),'r') as fid:
            data_raw = fid.readlines()
        last = data_raw[-1].split(';')
        tlast = datetime.strptime(last[1][-16:],'%Y.%m.%d %H:%M')
        for v in T_new:
            nextDay=False
            if v[0][11:]=='24:00': #Correct bug in time
                v[0]=v[0][:11]+'00:00'
                nextDay=True
            t_new = datetime.strptime(v[0],'%d.%m.%Y %H:%M')
            if nextDay: t_new=t_new+timedelta(1)
            if t_new<=tlast: continue
            t_add = datetime.strftime(t_new,'%Y.%m.%d %H:%M')
            v_add = toFloat(v[1])
            if v_add>=0: #If non-NaN positive value
                data_raw.append(';'.join([stn,t_add+'-'+t_add,'%8.3f' % v_add])+'\n')
        with open(os.path.join(wdir,'T_'+stn+'_Stundenmittel.asc'),'w') as fid:
            fid.write(''.join(data_raw))
    
    stations_Q = [f[2:6] for f in hydroFiles if f[0]=='Q']
    #Map the station number to the "observation type" to get the correct file from the FTP server 
    stations_fileIDs = {'2009':'10', '2019':'10', '2034':'10', '2056':'10', '2068':'10', '2078':'13', '2084':'10', '2102':'10', '2104':'10', '2105':'13', '2109':'10', '2160':'10', '2276':'10', '2300':'12', '2307':'10', '2308':'10', '2312':'11', '2321':'10', '2368':'10', '2369':'10', '2372':'10', '2378':'10', '2412':'10', '2416':'10', '2426':'12', '2432':'10', '2433':'10', '2436':'11', '2447':'30', '2457':'10', '2458':'10', '2461':'10', '2469':'10', '2473':'32', '2477':'10', '2480':'11', '2481':'11', '2486':'10', '2488':'10', '2493':'10', '2605':'11', '2608':'10', '2629':'10', '2635':'10'}
    for stn in stations_Q:
        #Retrieve discharge data
        data_new = ''
        Q_new = []
        try:
            files = ftp.nlst('hydropro/%s' % stn)
        except:
            writelog('\tStation %s: no FTP directory found for updated discharge data.' % stn)
            continue
        files.sort(reverse=True)
        for file in files:
            if (('10_%s_02' % stations_fileIDs[stn]) in file):
                ftp.retrbinary('RETR %s' % file,store)
                break
        if len(files)==0 or data_new=='':
            writelog('\tStation %s: no file found with updated discharge data.' % stn)
            continue
        data_new = data_new.split('\r')
        for line in data_new[1:]:
            if (len(line)>0 and line[0]!=';'): #Discard empty and header lines
                line = line.split('\t')
                if line[1][3:]=='00': #Keep only full hours
                    Q_new.append([line[0]+' '+line[1],line[2]])
        #Complete the data file with the new data
        with open(os.path.join(wdir,'Q_'+stn+'_Stundenmittel.asc'),'r') as fid:
            data_raw = fid.readlines()
        last = data_raw[-1].split(';')
        tlast = datetime.strptime(last[1][-16:],'%Y.%m.%d %H:%M')
        for v in Q_new:
            nextDay=False
            if v[0][11:]=='24:00': #Correct bug in time
                v[0]=v[0][:11]+'00:00'
                nextDay=True
            t_new = datetime.strptime(v[0],'%d.%m.%Y %H:%M')
            if nextDay: t_new=t_new+timedelta(1)
            if t_new<=tlast: continue
            t_add = datetime.strftime(t_new,'%Y.%m.%d %H:%M')
            v_add = toFloat(v[1])
            if v_add>=0: #If non-NaN positive value
                data_raw.append(';'.join([stn,t_add+'-'+t_add,'%8.3f' % v_add])+'\n')
        with open(os.path.join(wdir,'Q_'+stn+'_Stundenmittel.asc'),'w') as fid:
            fid.write(''.join(data_raw))
    ftp.quit()
	
def create_gif(filenames, duration):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    output_file = 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
    imageio.mimsave(output_file, images, duration=duration)


def compareModelToRiverTemp(Lakes,Tsurf,t):
    ftp = ftplib.FTP('ftp.hydrodata.ch','eawag','fous60{avant')
    def store(txt):
        nonlocal data_new
        data_new += txt.decode('ansi')
    Lakes_Outflows = [(Lake['Name'],Lake['Data']['Hydro-outflow station']) for Lake in Lakes if Lake['Data']['Hydro-outflow station']!='']
    for lakeName,stn in Lakes_Outflows:
        #Retrieve temperature data
        data_new = ''
        try:
            files = ftp.nlst('%s/Log' % stn)
        except:
            continue
        if len(files)>0: #Datafiles found
            file = [f for f in files if datetime.strftime(t,'%Y%m%d') in f][0] #Find file with the correct date
            ftp.retrbinary('RETR %s' % file,store)
            data_new = data_new.split('\r\n')
            k=2
            idx_T = None
            while data_new[k][0:11]=='# Parameter': #Find index of temperature measurement
                data_vec = data_new[k].split('\t')
                if any('temperatur' in d.lower() for d in data_vec):
                    idx_T = int(data_vec[-1])
                    break
                k=k+1
            if idx_T is None:
                continue
            while data_new[k][0]=='#': #Skip the header
                k=k+1
            T_outflow = toFloat(data_new[k].split('\t')[2*(idx_T-1)+2]) #Read the first value
        elif stn in ['2009','2105','2276','2432','2433','2457','2473']: #No PLC logger at these stations: datafiles are in another folder
            files = ftp.nlst('hydropro/%s' % stn)
            file = [f for f in files if (stn+'_03') in f][0] #Find file with temperature data
            ftp.retrbinary('RETR %s' % file,store)
            data_new = data_new.split('\r')
            for line in data_new[1:]: #Find the correct date
                if (len(line)>0 and line[0]!=';'): #Discard empty, header lines
                    line = line.split('\t')
                    if line[1]!='24:00': #Discard time bugs
                        if datetime.strptime(line[0]+' '+line[1],'%d.%m.%Y %H:%M')>=t:
                            break
            T_outflow = toFloat(line[2]) #Read the temperature value
        else: #No datafile found
            continue
        #Compare to lake surface temperature
        Ts = [T for (lake,T) in Tsurf if lake==lakeName][0]
        print('%20s - Lake surface (model): %5.2f °C; Outflow (obs.): %5.2f °C'% (lakeName.split(':')[0],Ts,T_outflow))
    ftp.quit()

def createTableTrends(lakes,trend_Tm,trend_Ts,trend_Tb,trend_HC,trend_SC):
    table = 'Lake;Mean temperature [°C/decade];Surface temperature [°C/decade];Bottom temperature [°C/decade];Heat content [J/decade];Schmidt stability [J/m^2/decade]\n'
    for k in range(len(lakes)):
        table += lakes[k]+';'
        table += ('+' if trend_Tm[0][k]>0 else '')+('%.3f' % (trend_Tm[0][k]*10))+('***' if trend_Tm[1][k]<0.001 else '**' if trend_Tm[1][k]<0.01 else '*' if trend_Tm[1][k]<0.05 else '')+';'
        table += ('+' if trend_Ts[0][k]>0 else '')+('%.3f' % (trend_Ts[0][k]*10))+('***' if trend_Ts[1][k]<0.001 else '**' if trend_Ts[1][k]<0.01 else '*' if trend_Ts[1][k]<0.05 else '')+';'
        table += ('+' if trend_Tb[0][k]>0 else '')+('%.3f' % (trend_Tb[0][k]*10))+('***' if trend_Tb[1][k]<0.001 else '**' if trend_Tb[1][k]<0.01 else '*' if trend_Tb[1][k]<0.05 else '')+';'
        table += ('+' if trend_HC[0][k]>0 else '')+('%.2g' % (trend_HC[0][k]*10))+('***' if trend_HC[1][k]<0.001 else '**' if trend_HC[1][k]<0.01 else '*' if trend_HC[1][k]<0.05 else '')+';'
        table += ('+' if trend_SC[0][k]>0 else '')+('%.0f' % (trend_SC[0][k]*10))+('***' if trend_SC[1][k]<0.001 else '**' if trend_SC[1][k]<0.01 else '*' if trend_HC[1][k]<0.05 else '')+'\n'
    pyperclip.copy(table.replace(';;',';-;').replace(';\n',';-\n'))
    print('Table copied to clipboard.')

def createTableLakes(Lakes,t_ref,version):
    table = 'Lake;Volume [km3];Surface [km2];Max depth [m];Elevation [m];Weather station IDs (MeteoSwiss);Hydrological station IDs (FOEN); Model timeframe\n'
    for Lake in Lakes:
        lakeName = simple(Lake['Name'])
        (tstart,tend) = modelPeriod(lakeName,t_ref,version)
        table += Lake['Name']+('*' if os.path.exists(os.path.join('PEST',lakeName,'simstrat_calib.par')) else '')+';'
        table += ('%.3g' % Lake['Properties']['Volume [km3]'] if Lake['Properties']['Volume [km3]'] else '')+';'
        table += ('%.3g' % Lake['Properties']['Surface [km2]'] if Lake['Properties']['Surface [km2]'] else '')+';'
        table += ('%.0f' % Lake['Properties']['Max depth [m]'] if Lake['Properties']['Max depth [m]'] else '')+';'
        table += ('%.0f' % Lake['Properties']['Elevation [m]'] if Lake['Properties']['Elevation [m]'] else '')+';'
        table += str(Lake['Data']['Weather station']).replace('[','').replace(']','').replace('\'','')+';'
        table += str(Lake['Data']['Hydro-inflow station']).replace('[','').replace(']','').replace('\'','')+';'
        table += str(tstart.year)+'-'+str(tend.year)+';'
    pyperclip.copy(table.replace(';;',';-;').replace(';\n',';-\n'))
    print('Table copied to clipboard.')

#Create the table for the file lakes.html (works only with v2.0)
def createHTMLTableLakes(Lakes,t_ref):
    table = ''
    for Lake in Lakes:
        lakeName = simple(Lake['Name'])
        table += '<tr>\n'
        table += '  <td><a href="/%s">%s</a></td>\n' % (lakeName,Lake['Name'])
        table += '  <td>%s</td>\n' % Lake['Properties']['Type']
        table += '  <td></td>\n' if Lake['Properties']['Max depth [m]']==50 else ('  <td>%.1f</td>\n' % Lake['Properties']['Max depth [m]'])
        table += '  <td></td>\n' if Lake['Properties']['Volume [km3]'] is None else ('  <td>%g</td>\n' % Lake['Properties']['Volume [km3]'])
        table += '  <td>%g</td>\n' % Lake['Properties']['Surface [km2]']
        table += '  <td>%.0f</td>\n' % Lake['Properties']['Elevation [m]']
        setup = json.load(codecs.open(os.path.join('Simstrat',lakeName+'.par'),'r','utf-8'))
        table += '  <td>%d-2018</td>\n' % (t_ref+timedelta(setup['Simulation']['Start d'])).year
        table += '  <td>%s</td>\n' % ('Yes' if os.path.exists(os.path.join('PEST',lakeName)) else 'No')
        table += '</tr>\n'
    pyperclip.copy(table)
    print('Table copied to clipboard.')   

#Create the polar plot GIF
def createPolarGIF(Lakes,t_ref,t_end,version):
    delim = ',' if version.startswith('2.') else None
    LakesYears = []
    LakesTsurf = []
    for Lake in Lakes:
        try:
            t,T = [],[]
            with open(os.path.join('Simstrat',simple(Lake['Name']),'Results','T_out.dat')) as fid:
                next(fid) #Skip first line
                for line in fid:
                    t.append(t_ref+timedelta(float(line.split(delim)[0])))
                    T.append(float(line.split(delim)[-1]))
            LakesYears.append(np.array([tk.year for tk in t]))
            LakesTsurf.append(np.array(T))
        except:
            print('No results found for %s.' % Lake['Name'])
            LakesYears.append(np.array([np.nan]))
            LakesTsurf.append(np.array([np.nan]))
    LakesYears = np.array(LakesYears)
    LakesTsurf = np.array(LakesTsurf)
    yr_s,yr_e = t_ref.year,t_end.year
    cmap = mpl.colors.LinearSegmentedColormap.from_list('years',[((yr-yr_s)/(yr_e-yr_s),(yr_e-yr)/(yr_e-yr_s)/2,(yr_e-yr)/(yr_e-yr_s)) for yr in range(yr_s,yr_e)],yr_e-yr_s)
    norm = mpl.colors.Normalize(yr_s,yr_e-1)
    #sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    #sm.set_array([])
    Tbef = []
    images = []
    for yr in range(yr_s,yr_e):
        Tmax = [max(LakesTsurf[k][LakesYears[k]==yr],default=np.nan) for k in range(len(Lakes))]
        theta = np.arange(0,2*np.pi,2*np.pi/(len(Lakes)+2))
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_axes([0,0,1,1],polar=True)
        R = (yr-yr_s)/(yr_e-yr_s)
        G = (yr_e-yr)/(yr_e-yr_s)/2
        B = (yr_e-yr)/(yr_e-yr_s)
        ax.bar(theta[2:],Tmax,width=2*np.pi/len(Lakes),color=(R,G,B),alpha=0.6)
        for yrbef in range(yr_s,yr):
            R = (yrbef-yr_s)/(yr_e-yr_s)
            G = (yr_e-yr)/(yr_e-yr_s)/2
            B = (yr_e-yrbef)/(yr_e-yr_s)
            plt.plot(theta[2:],Tbef[yrbef-yr_s],'o',color=(R,G,B))
        ax.set_thetagrids(np.degrees(theta[2:]),range(1,len(Lakes)+1),frac=1.05)
        ax.set_ylim([10,30])
        ax.set_rgrids([15,20,25],angle=0)
        ax.set_yticklabels(['15°C','20°C','25°C'],{'weight':'bold'})
        ax.xaxis.grid(False)
        ax.annotate(str(yr),(0,10),horizontalalignment='center',backgroundcolor='w',fontsize=16)
        ax.annotate('\n'.join([('%d: %s' % (k+1,Lakes[k]['Name'])) for k in range(len(Lakes))]),(1.7*np.pi,49),annotation_clip=False)
        sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
        sm.set_array([])
        plt.colorbar(sm,orientation='horizontal',alpha=0.6,shrink=0.5)
        plt.savefig(os.path.join('Plots','PolarGIF','PolarPlot_Tmax_'+str(yr)+'.png'),dpi=100,bbox_inches='tight')
        plt.clf(); plt.close()
        Tbef.append(Tmax)
        images.append(imageio.imread(os.path.join('Plots','PolarGIF','PolarPlot_Tmax_'+str(yr)+'.png')))
    for k in range(20):
        images.append(images[-1])
    imageio.mimsave(os.path.join('Plots','PolarGIF','PolarPlot_Tmax.gif'),images,duration=0.25)

#Create the line plot GIFs
def createLineGIFs(Lakes,t_ref,t_end,version):
    delim = ',' if version.startswith('2.') else None
    for Lake in Lakes:
        try:
            t,T = [],[]
            with open(os.path.join('Simstrat',simple(Lake['Name']),'Results','T_out.dat')) as fid:
                next(fid) #Skip first line
                for line in fid:
                    t.append(t_ref+timedelta(float(line.split(delim)[0])))
                    T.append(float(line.split(delim)[-1]))
        except:
            print('No results found for %s.' % Lake['Name'])
        yrs = np.array([tk.year for tk in t])
        t = np.array(t)
        T = np.array(T)
        yr_s,yr_e = t_ref.year,t_end.year
        Tmean = []
        doy_all,Tyr_all = [],[]
        images = []
        for yr in range(yr_s,yr_e):
            if any(yrs==yr):
                tyr = t[yrs==yr]
                Tyr = T[yrs==yr]
                doy = [tk.timetuple().tm_yday+tk.timetuple().tm_hour/24 for tk in tyr.tolist()]
                Tmean.append(np.mean(Tyr))
                doy_all.append(doy)
                Tyr_all.append(Tyr)
            else:
                Tmean.append(np.nan)
                doy_all.append([np.nan])
                Tyr_all.append([np.nan])
        for yr in range(yr_s,yr_e):
            if any(yrs==yr):
                f,(ax1,ax2) = plt.subplots(1,2,gridspec_kw={'width_ratios':[2,1]},figsize=(10,5))
                ax1.plot(np.arange(yr_s,yr+1),np.array(Tmean[0:(yr-yr_s+1)])-np.nanmean(Tmean))
                ax1.plot([yr_s,yr_e],[0,0],'k:')
                ax1.set_xlabel('Year')
                ax1.set_xlim([yr_s,yr_e])
                ax1.set_ylabel('Mean surface temperature: anomaly relative to mean [°C]')
                ax1.set_ylim([-3,3])
                ax1.annotate(Lake['Name'],(yr_s+1,2.5),annotation_clip=False)
                ax2.set_xlabel('Day of year')
                ax2.set_xlim([0,365])
                ax2.set_ylabel('Surface temperature [°C]')
                ax2.set_ylim([0,30])
                for yrbef in range(yr_s,yr+1):
                    R = (yrbef-yr_s)/(yr_e-yr_s)
                    G = (yr_e-yr)/(yr_e-yr_s)/2
                    B = (yr_e-yrbef)/(yr_e-yr_s)
                    ax2.plot(doy_all[yrbef-yr_s],Tyr_all[yrbef-yr_s],color=(R,G,B),alpha=0.7,linewidth=0.5)
                plt.savefig(os.path.join('Plots','LineGIFs','LinePlot_Tsurf_'+simple(Lake['Name'])+'_'+str(yr)+'.png'),dpi=100,bbox_inches='tight')
                plt.clf(); plt.close()
                images.append(imageio.imread(os.path.join('Plots','LineGIFs','LinePlot_Tsurf_'+simple(Lake['Name'])+'_'+str(yr)+'.png')))
        for k in range(20):
            images.append(images[-1])
        imageio.mimsave(os.path.join('Plots','LineGIFs','LinePlot_Tsurf_'+simple(Lake['Name'])+'.gif'),images,duration=0.25)
