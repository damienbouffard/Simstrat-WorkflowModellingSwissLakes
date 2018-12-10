# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 06:45:02 2017

@author: Adrien Gaudard
@description: For all the lakes listed in "Lakes.json", this script does the following:
    - Read observational data and write a TXT file in "Observations/"
    - Read bathymetrical data and write the corresponding model input file
    - Read meteorological and hydrological data and write the corresponding model input files
    - Write a model input file for light absorption (based on lake trophic state)
    - Set-up the Simstrat model with default parameters
    - Run PEST calibration (if calib=True) (in "PEST/")
    - Run the Simstrat model (in "Simstrat/")
"""

from __future__ import print_function #Assuring compatibility if using Python 2.6+ instead of Python 3
import os, socket, platform
import shutil, subprocess
import json, codecs, pickle
import stat, time, datetime
import numpy as np
import Functions, PEST
np.seterr(all='ignore')
np.warnings.filterwarnings('ignore')

#JSON file defining the lakes to be modeled
LakesFile = 'Lakes.json'
Lakes = json.load(codecs.open(LakesFile,'r','utf-8'))

#Timeframe (start and end time)
t_ref = datetime.datetime(1981,1,1) #Start time of meteo data; reference date for Simstrat time
t_end = datetime.datetime.combine(datetime.date.today()-datetime.timedelta(days=datetime.date.today().weekday()),datetime.time())

#Input data
prepareData = True #Process the whole input data for the model
lakeD = 'LakeData' #Path to CTD and Secchi data
meteoD = 'MeteoData' #Path to meteorological data
hydroD = 'HydroData' #Path to hydrological data

#Model settings (Simstrat)
version = '2.0' #Model version (1.6, 1.8 or 2.0)
dt = 300 #Model time resolution (seconds)
dtout = 21600 #Output time resolution (seconds)
dz = 0.5 #Model depth resolution (meters)
dzout = 1 #Output depth resolution (meters)

#Calibration settings (PEST)
calib = False #Enable/disable calibration
useSal = False #Use salinity observations to calibrate the model
useIce = False #Use ice thickness observations to calibrate the model
par_calib = ['a_seiche','f_wind','p_radin','p_albedo'] # for p2 use 'p_windf' 
mode = 'BEOPEST' #'PEST' or 'BEOPEST'
ip = socket.gethostbyname(socket.gethostname()) #IP-address (or name) of computer (used if mode='BEOPEST')
arch = '64' if platform.machine().endswith('64') else '32' #Architecure (32- or 64-bit) of computer (used if mode='BEOPEST')
nCPU = 2*len(par_calib) #Number of CPUs for parallel com
if version.startswith('1.'): useIce = False

#Server update
updateServer = True

#Log file
logFile = open('Processing.log','w')
def writelog(txt):
    logFile.write(txt)
    print(txt,end='')
Functions.logFile = logFile
PEST.logFile = logFile

t_start = time.time()
#Update the meteorological data
Functions.retrieveNewMeteoData(meteoD,t_end)
Functions.retrieveNewHydroData(hydroD,Lakes)

#Lakes surface temperature for display
if os.path.exists(os.path.join('Web','Tsurf.rec')):
    with open(os.path.join('Web','Tsurf.rec'),'rb') as fid:
        Tsurf = pickle.load(fid)
else:
    Tsurf = [(Lake['Name'],None) for Lake in Lakes]

#Main loop
for Lake in Lakes:
    writelog(('%2d/%2d - '+Lake['Name']+'\n') % (Lakes.index(Lake)+1,len(Lakes)))
    lakeName = Functions.simple(Lake['Name'])
    lakeElevation = Lake['Properties']['Elevation [m]']
    lakeMaxDepth = Lake['Properties']['Max depth [m]']
    
    t_proc = time.time() #Pre-processing time
    if prepareData:
        ##Read and store the available data
        #Observations
        XY = [Lake['X [m]'],Lake['Y [m]']]
        Observations = Functions.readObs(lakeD,XY) #Temperature, salinity
        IceThickness = Functions.readIce(lakeD,XY) #Ice thickness
        if Observations is None:
            writelog('\tNo observation for temperature or salinity found.\n')
        else:
            if IceThickness is None: writelog('\tNo observation for ice thickness found.\n')
            Functions.writeObs(Lake,Observations,IceThickness) #Create observation files to be used during calibration
        #Bathymetry
        Bathymetry = {}
        if 'Bathymetry' in Lake['Properties']:
            Bathymetry['Depth [m]'] = Lake['Properties']['Bathymetry']['Depth [m]']
            Bathymetry['Area [m^2]'] = Lake['Properties']['Bathymetry']['Area [m2]']
        else:
            writelog('\tNo bathymetry data found. Using the lake surface and depth to assume a simple two-point profile.\n')
            Bathymetry['Depth [m]'] = [0,lakeMaxDepth]
            Bathymetry['Area [m^2]'] = [Lake['Properties']['Surface [km2]']*1E6,0]
        #Meteorological data
        writelog('\tMeteorological data\n')
        stations = Lake['Data']['Weather station']
        if len(stations)==0:
            writelog('\t\tError: No meteorological data is specified in "%s"\n.' % LakesFile)
            continue
        if type(stations)==str: stations=[stations]
        if not all([os.path.exists(os.path.join(meteoD,'%s_data.txt' % stn)) for stn in stations]):
            writelog('\t\tError: Not all of the meteo stations specified in "%s" are available in the folder "%s\\".\n' % (LakesFile,meteoD))
            continue
        Meteo = Functions.readMeteo(meteoD,stations) #Read data from the meteo stations
        MeteoCombined = Functions.combineMeteo(Meteo,lakeElevation,t_ref,t_end) #Combine in a single data set, prioritizing the first given stations
        MeteoForModel = Functions.adaptMeteo(MeteoCombined,Lake) #Convert the meteorological data to the model standard
        
        #Hydrological data
        writelog('\tHydrological data\n')
        flow = False #Will be True if inflow/outflow can be modeled
        gravityInflow = True #Will be False in the case of deeper inflow from another basin
        save_outflow = True if 'Hydro-outflow lake' in Lake['Data'] else False
        outflow_zmax = Lake['Data']['Hydro-outflow depth [m]'] if 'Hydro-outflow depth [m]' in Lake['Data'] else 0
        stations = Lake['Data']['Hydro-inflow station']
        Inflow = None
        if type(stations)==str: stations=[stations]
        if len(stations)==1 and stations[0]=='':
            writelog('\t\tNo inflow data is specified in "%s". Inflows and outflows will be ignored.\n' % LakesFile)
        elif not any([os.path.exists(os.path.join(hydroD,'Q_%s_Stundenmittel.asc' % stn)) for stn in stations]):
            writelog('\t\tNone of the hydro-inflow stations specified in "%s" are available in the folder "Hydro\\". Inflows and outflows will be ignored.\n' % LakesFile)
        else: #Potentially inflow from a river
            if not all([os.path.exists(os.path.join(hydroD,'Q_%s_Stundenmittel.asc' % stn)) for stn in stations]):
                writelog('\t\tNot all of the hydro-inflow stations specified in "%s" are available in the folder "Hydro\\". Missing stations will be ignored.\n' % LakesFile)
                stations = [stn for stn in stations if os.path.exists(os.path.join(hydroD,'Q_%s_Stundenmittel.asc' % stn))]
            Inflow = [[]]*len(stations)
            for k in range(len(stations)):
                Inflow[k] = Functions.readHydro(hydroD,stations[k])
            Inflow = Functions.combineInflow(Inflow,t_ref,t_end)
            if Inflow is not None:
                flow = True
        if not flow: #Potentially inflow from another basin
            if 'Hydro-inflow lake' in Lake['Data']:
                inflowLake = Lake['Data']['Hydro-inflow lake']
                if type(inflowLake) is str: inflowLake=[inflowLake]
                Inflow = []
                for k in range(len(inflowLake)):
                    writelog('\tReading inflow characteristics from the upstream lake(s)...\n')
                    with open(os.path.join(hydroD,'outflow_'+Functions.simple(inflowLake[k])+'.rec'),'rb') as fid:
                        inflow_t,inflow_z,inflow_Q,inflow_T,inflow_S,inflow_zmax = pickle.load(fid)
                    if not any([inflow_t,inflow_z,inflow_Q,inflow_T,inflow_S,inflow_zmax]):
                        writelog('\t\tNo inflow data from "%s". Inflows and outflows will be ignored.\n' % inflowLake)
                    elif inflow_zmax==0:
                        if k>0 and not gravityInflow: writelog('MISSING IMPLEMENTATION for both river and basin inflow!!')
                        gravityInflow = True
                        Inflow.append({'Flowrate [m^3/s]':{'Time':inflow_Q['Flowrate [m^3/s]']['Time'],'Data':inflow_Q['Flowrate [m^3/s]']['Data']},'Temperature [°C]':{'Time':inflow_t,'Data':inflow_T},'Salinity [ppt]':{'Time':inflow_t,'Data':inflow_S}})
                    else:
                        if k>0 and gravityInflow: writelog('MISSING IMPLEMENTATION for both river and basin inflow!!')
                        gravityInflow = False
                        Inflow.append({'Flowrate [m^3/s]':{'Time':inflow_Q['Flowrate [m^3/s]']['Time'],'Data':inflow_Q['Flowrate [m^3/s]']['Data']},'Temperature [°C]':{'Time':inflow_t,'Depth [m]':inflow_z,'Data':inflow_T},'Salinity [ppt]':{'Time':inflow_t,'Depth [m]':inflow_z,'Data':inflow_S},'Inflow depth [m]':inflow_zmax})
                if Inflow==[]: Inflow = None
                if Inflow is not None:
                    if gravityInflow:
                        Inflow = Functions.combineInflow(Inflow,t_ref,t_end)
                    else:
                        Inflow = Functions.combineInflowFixed(Inflow,t_ref,t_end)
                    if Inflow is not None:
                        flow = True
        if flow:
            #station = Lake['Data']['Hydro-outflow station']
            #if len(station)==0:
            #    writelog('\t\tNo outflow data is specified in "%s". Outflows will be ignored.\n' % LakesFile)
            #    Outflow = {'Flowrate [m^3/s]':{'Time':[t_ref,t_end],'Data':[0,0]}}
            #else:
            #    if os.path.exists(os.path.join(hydroD,'Q_%s_Stundenmittel.asc' % station)):
            #        Outflow = Functions.readHydro(hydroD,station)
            #    else:
            #        writelog('\t\tHydro-outflow station %s (specified in "%s") is not available in the folder "%s\\". Outflow will be ignored.\n' % (station,LakesFile,hydroD))
            #        Outflow = {}
            if outflow_zmax==0:
                Outflow = {'Flowrate [m^3/s]':{'Time':[t_ref,t_end],'Data':[0,0]}} #Zero outflow, lake will overflow
            else:
                Outflow = Inflow #Steady-state, inflow will be spread over outflow_zmax by Functions.writeOutflow()
        if not flow:
            Inflow = {'Flowrate [m^3/s]':{'Time':[t_ref,t_end],'Data':[0,0]},'Temperature [°C]':{'Time':[t_ref,t_end],'Data':[0,0]},'Salinity [ppt]':{'Time':[t_ref,t_end],'Data':[0,0]}} #Zero inflow
            Outflow = {'Flowrate [m^3/s]':{'Time':[t_ref,t_end],'Data':[0,0]}} #Zero outflow
        
        #Light absorption data
        Absorption = Functions.readSecchi(lakeD,XY)
        if Absorption is None:
            writelog('\tNo absorption data (Secchi depth) found. Computing absorption coefficient with a proxy based on trophic state.\n')
            if Lake['Properties']['Trophic state']=='Oligotrophic': ext=0.15
            elif Lake['Properties']['Trophic state']=='Eutrophic': ext=0.45
            else: ext=0.25
            if Lake['Properties']['Elevation [m]']>2000: ext=1.00
            Absorption = {'Absorption [m^-1]': {'Time':[t_ref,t_end],'Depth [m]':[1,1],'Data':[ext,ext]}}
        
        ##Check for the time frame of available data and set the model time frame
        inputData = [MeteoCombined,Absorption]
        if flow: inputData.extend([Inflow,Outflow])
        timeFrame = Functions.getTimeframe(inputData)
        modelTime = [t_ref,t_end]
        for tf in timeFrame:
            if any([tf[0].startswith(par) for par in ['Air temperature','Vapour pressure','Solar radiation','Wind']]):
                modelTime[0] = max(modelTime[0],tf[1])
                modelTime[1] = min(modelTime[1],tf[2])
        writelog('\tModel timeframe: %s - %s\n' % (datetime.datetime.strftime(modelTime[0],'%d.%m.%Y'),datetime.datetime.strftime(modelTime[1],'%d.%m.%Y')))
        
        ##Write the model input files
        #Create model folder
        Dir = os.path.join('Simstrat',lakeName)
        if os.path.exists(Dir):
            os.chmod(Dir,stat.S_IWUSR)
            shutil.rmtree(Dir)
            time.sleep(5)
        os.mkdir(Dir)
        #Initial conditions
        iniFile = os.path.join(Dir,'InitialConditions.dat')
        (z,S) = Functions.writeInitialConditions(Observations,modelTime[0],iniFile,lakeElevation,lakeMaxDepth)
        #Bathymetry
        bathyFile = os.path.join(Dir,'Bathymetry.dat')
        Functions.writeBathy(Bathymetry,bathyFile)
        #Grid
        gridFile = os.path.join(Dir,'Grid.dat')
        grid = Functions.writeGrid(abs(max(Bathymetry['Depth [m]'])),dz,gridFile)
        #Output depths
        zoutFile = os.path.join(Dir,'z_out.dat')
        z_out = Functions.writeOutputDepths(np.arange(0,abs(max(Bathymetry['Depth [m]']))+1E-3,dzout),zoutFile)
        #Output times
        toutFile = os.path.join(Dir,'t_out.dat')
        t_out = Functions.writeOutputTimes(round(dtout/dt),toutFile)
        #Absorption
        absFile = os.path.join(Dir,'Absorption.dat')
        Functions.writeAbsorption(Absorption,t_ref,absFile)
        #Forcing
        forcingFile = os.path.join(Dir,'Forcing.dat')
        Functions.writeForcing(MeteoForModel,t_ref,modelTime,forcingFile,version)
        #Inflow and outflow
        inflowQFile = os.path.join(Dir,'Qin.dat')
        inflowTFile = os.path.join(Dir,'Tin.dat')
        inflowSFile = os.path.join(Dir,'Sin.dat')
        outflowFile = os.path.join(Dir,'Qout.dat')
        if 'Salinity [ppt]' in Inflow: #Case of inflow from an upstream lake
            S_inflow = None
        else: #Compute the minimum salinity to use as river inflow concentration
            S_inflow = 0.0
            if Observations is not None:
                if not all(np.isnan(Observations['Salinity [ppt]'])):
                    S_inflow = np.nanmin(Observations['Salinity [ppt]'])
        if gravityInflow:
            Functions.writeInflow(Inflow,t_ref,inflowQFile,inflowTFile,inflowSFile,S_inflow)
        else:
            Functions.writeInflowFixed(Inflow,t_ref,inflowQFile,inflowTFile,inflowSFile)
        if Outflow!={}:
            Functions.writeOutflow(Outflow,outflow_zmax,t_ref,outflowFile)
        #Parameter file
        parFile = os.path.join('Simstrat',lakeName+'.par')
        Functions.writeParFile(Lake,modelTime,t_ref,dt,grid,z_out,t_out,parFile,gravityInflow,version)
        os.mkdir(os.path.join(Dir,'Results'))
    t_proc = time.time()-t_proc
    
    t_calibrun = time.time() #Calibration/run time
    calibrated = (os.path.exists(os.path.join('PEST',lakeName,'simstrat_calib.par')) and os.path.exists(os.path.join('PEST',lakeName,'simstrat_calib.rec')) and os.path.exists(os.path.join('PEST',lakeName,'simstrat_calib.res')))
    if calib: #Calibrate the model
        if Observations is None:
            writelog('\tNo observation available. Calibration is impossible.\n')
        elif modelTime[0]>=Observations['Time'][-1]:
            writelog('\tNo observation available within the model timeframe. Calibration is impossible.\n')
        else:
            writelog('\tCalibrating the model...\n')
            PESTparFile = os.path.join('PEST',lakeName+'.json')
            Functions.writePESTParFile(Lake,'simstrat_v%s.exe' % version,parFile,t_ref,par_calib,useSal,useIce,nCPU,PESTparFile,version)
            PEST.runPEST(PESTparFile,standalone=False,mode=mode,arch=arch,ip=ip,version=version)
            calibrated = True
    
    #Run the model
    writelog('\tRunning the model...\n')
    subprocess.Popen(os.path.join('..','simstrat_v%s.exe ' % version)+lakeName+'.par',shell=True,cwd=os.path.join('Simstrat')).wait()
    t_calibrun = time.time()-t_calibrun
    
    t_post = time.time()
    #Save the outflow to use as inflow for downstream basin
    if save_outflow:
        writelog('\tSaving outflow characteristics for the downstream lake...\n')
        if not flow:
            outflow_t,outflow_z,outflow_Q,outflow_T,outflow_S,outflow_zmax = None,None,None,None,None,None
        else:
            outflow_Q = Inflow
            outflow_t,outflow_z,outflow_T = Functions.getValForOutflow(lakeName,'T',outflow_zmax,version)
            _,_,outflow_S = Functions.getValForOutflow(lakeName,'S',outflow_zmax,version)
            outflow_t = [t_ref+datetime.timedelta(tk) for tk in outflow_t]
        with open(os.path.join(hydroD,'outflow_'+lakeName+'.rec'),'wb') as fid:
            pickle.dump((outflow_t,outflow_z,outflow_Q,outflow_T,outflow_S,outflow_zmax),fid)
    writelog('\tWriting the metadata file and the HTML file, creating the plots...\n')
    Functions.writeMetadata(Lake,t_ref,meteoD,hydroD,dt,dz,version)
    plots = {}
    plots['T'],Ts = Functions.createPlots(Lake,t_ref,'T',version,plotly=True)
    plots['S'],_ = Functions.createPlots(Lake,t_ref,'S',version,plotly=True)
    plots['nuh'],_ = Functions.createPlots(Lake,t_ref,'nuh',version,plotly=True)
    plots['N2'],_ = Functions.createPlots(Lake,t_ref,'N2',version,plotly=True)
    plots['Schmidt'],_ = Functions.createPlots(Lake,t_ref,'Schmidt',version,plotly=True)
    plots['heat'],_ = Functions.createPlots(Lake,t_ref,'heat',version,plotly=True)
    if version.startswith('2.'): plots['ice'],_ = Functions.createPlots(Lake,t_ref,'ice',version,plotly=True)
    Functions.writeHTML(Lake,t_ref,meteoD,hydroD,calibrated,plots,version)
    if calibrated: writelog('\tWriting the calibration results and creating the plot of residuals...\n')
    Functions.writeCalibResults(Lake,t_ref,calibrated,par_calib,useSal,(Lakes.index(Lake)==0),version)
    if calibrated: Functions.createResidualsPlots(Lake,t_ref)
    Functions.writeKML(Lakes,[T[1] for T in Tsurf],mode3D=True)
    
    t_post = time.time()-t_post
    writelog('\tTime needed: %.0f min (pre-processing) + %.0f min (%s) + %.0f min (post-processing)\n' % (t_proc/60,t_calibrun/60,'calibration+run' if (calib and calibrated) else 'run',t_post/60))
    
    Tsurf[Lakes.index(Lake)] = (Lake['Name'],Ts)
    with open(os.path.join('Web','Tsurf.rec'),'wb') as fid:
        pickle.dump(Tsurf,fid)
    
    #If last lake, create the plots for all lakes
    if (Lakes.index(Lake)+1)==len(Lakes):
        trend_lakes,trend_Tm,trend_Ts,trend_Tb,trend_hc,trend_sc = Functions.plotAllLakes(Lakes,t_ref,version)
    
    #Send files to the server and update the website
    if updateServer:
        subprocess.Popen('UpdateServer.bat %s' % lakeName,cwd='Web',shell=True)

writelog('Total time: %.1f hours\n' % ((time.time()-t_start)/3600))
logFile.close()
Functions.compareModelToRiverTemp(Lakes,Tsurf,t_end)
#Functions.createTableLakes(Lakes,t_ref,version)
#Functions.createTableTrends(trend_lakes,trend_Tm,trend_Ts,trend_Tb,trend_hc,trend_sc)
#Functions.createHTMLTableLakes(Lakes,t_ref)
#Functions.createPolarGIF(Lakes,t_ref,t_end,version)
#Functions.createLineGIFs(Lakes,t_ref,t_end,version)

#--------Code to add to server.py--------#
#addcode = ''
#for k in range(len(Lakes)):
#    lakeName = Functions.simple(Lakes[k]['Name'])
#    addcode = addcode + '\n' + '@app.route("/simstrat/%s")' % lakeName
#    addcode = addcode + '\n' + 'def simstrat_%s():' % lakeName.replace('-','')
#    addcode = addcode + '\n' + '\t' + 'return render_template("simstrat/%s.html",title=None,contents=None)' % lakeName
#    addcode = addcode + '\n'
