# -*- coding: utf-8 -*-
'''
Created on 6 Jun 2017
Last updated on 20 Sep 2018

@author: Adrien Gaudard
@description: Main file for the execution of PEST for Simstrat
'''

import os, glob, shutil, subprocess
import csv, json
import stat, time
from datetime import datetime

#logFile = open('PEST.log','w')
def writelog(txt):
    logFile.write(txt)
    print(txt,end='')

def checkJSON(x):
    
    if not 'files' in x:
        raise Exception('Missing \'files\' key in the PEST configuration file.')
    else:
        if not 'model' in x['files']:
            raise Exception('Missing \'model\' key (Simstrat executable file) under the \'files\' key in the PEST configuration file.')
        if not 'configFile' in x['files']:
            x['files']['configFile'] = 'simstrat.par'
            writelog('\t\tAssuming \"simstrat.par\" is the Simstrat configuration file. For another file, add the key \'configFile\' under the key \'files\' in the PEST configuration file.\n')
        if not 'pestDir' in x['files']:
            x['files']['pestDir'] = 'PEST'
            writelog('\t\tUsing \"PEST\" as the PEST working directory. For another directory, add the key \'pestDir\' under the key \'files\' in the PEST configuration file.\n')
        if not 'refDate' in x['files']:
            raise Exception('Missing \'refDate\' key (reference date corresponding to time 0 of the model, in format yyyy.mm.dd) under the \'files\' key in the PEST configuration file.')
    if not 'parameters' in x:
        raise Exception('Missing \'parameters\' key in the PEST configuration file.')
    if not 'PEST' in x:
        x['PEST'] = {'nCPU':1}
        writelog('\t\tSetting PEST up to run on only one CPU. For parallel computing, add the key \'PEST\', then under it add the key \'nCPU\' and give it a value in the PEST configuration file.\n')
    else:
        if not 'nCPU' in x['PEST']:
            x['PEST']['nCPU'] = 1
            writelog('\t\tSetting PEST up to run on only one CPU. For parallel computing, add the key \'nCPU\' under the key \'PEST\' and give it a value in the PEST configuration file.\n')
    return x


def checkParameters(x,mode,version):
    
    par_pest = [par for par in x['parameters'].keys()]
    if version.startswith('1.'):
        with open(x['files']['configFile']) as file:
            config = file.readlines()
        par_config = config[32:]
        par_config = [par.split()[1].lower() for par in par_config]
    elif version.startswith('2.'):
        with open(x['files']['configFile']) as file:
            config = json.load(file)
        par_config = config['ModelParameters'].keys()
    
    for par in par_config:
        if not par in par_pest:
            raise Exception('Parameter \"%s\" is in the Simstrat configuration file but not in the PEST configuration file.' % par)
    for par in par_pest:
        if not par in par_config:
            writelog('\t\tParameter \"%s\" is in the PEST configuration file but not in the Simstrat configuration file. It will be ignored.\n' % par)
        if mode.lower()=='beopest':
            parval = x['parameters'][par]
            if type(parval) is not list: parval=[parval]
            if any([v==0 for v in parval]):
                writelog('\t\tWARNING: Initial, min and/or max value of parameter \"%s\" is 0. This is likely to cause BEOPEST to crash.\n' % par)
    

def ControlFile(x,version):
    
    pestDir = x['files']['pestDir'] #PEST directory
    refDate = datetime.strptime(x['files']['refDate'],'%Y.%m.%d') #Reference time
    
    #Retrieve simulation period from configuration file
    if version.startswith('1.'):
        with open(x['files']['configFile']) as file:
            config = file.readlines()
        sim_t = [float(t.split()[0])*24*3600 for t in config[15:17]]
    elif version.startswith('2.'):
        with open(x['files']['configFile']) as file:
            config = json.load(file)
        sim_t = [t*24*3600 for t in [config['Simulation']['Start d'],config['Simulation']['End d']]]
    
    #Observations, in time x depth matrices
    obsFiles = [key for key in x['files'].keys() if key.startswith('obsFile')]
    if len(obsFiles)==0:
        raise ValueError('At least one observation file is needed for calibration. Add it/them in the PEST configuration file under the key \'files\' -> \'obsFile_X\', X being the observed variable, consistent with the naming of the model result file \'X_out.dat\'.')
    
    obsl = ['']*len(obsFiles)
    tobs = [[]]*len(obsFiles)
    zobs = [[]]*len(obsFiles)
    obsdat = [[]]*len(obsFiles)
    
    for k,var in enumerate(obsFiles):
        #Read observation files and extract time, depth, data
        fid = open(x['files'][var])
        fidread = csv.reader(fid)
        obsl[k] = var.split('_')[1]
        ncol = len(next(fidread))-1
        nrow = sum(1 for row in fidread)
        tobs[k] = [[]]*nrow
        obsdat[k] = [[]]*nrow
        fid.seek(0)
        for ir,row in enumerate(fidread):
            if ir==0:
                zobs[k] = [float(col) for col in row[1:]]
            else:
                obsdat[k][ir-1] = [[]]*ncol
                for ic,col in enumerate(row):
                    if ic==0:
                        td = datetime.strptime(col,'%Y-%m-%d %H:%M')-refDate #Time difference
                        tobs[k][ir-1] = td.days*24*3600 + td.seconds #Relative time [seconds]
                    else:
                        obsdat[k][ir-1][ic-1] = float(col)
        fid.close()
        
        #Remove measurements out of simulation period
        obsdat[k] = [obsdat[k][it] for it,t in enumerate(tobs[k]) if (t>=sim_t[0] and t<=sim_t[1])]
        tobs[k] = [t for t in tobs[k] if (t>=sim_t[0] and t<=sim_t[1])]
        if len(tobs[k])==0:
            writelog('\t\tNo %s observation data available for calibration (no observation during the model timeframe).' % obsl[k])
            continue
        if len(set(tobs[k]))<len(tobs[k]):
            raise Exception('Some times appear more than once in the observations file %s.' % x['files'][var])
        #Remove depths for which all values are missing
        nap = [sum([obsdat[k][it][iz]!=obsdat[k][it][iz] for it,t in enumerate(tobs[k])])/len(tobs[k]) for iz,z in enumerate(zobs[k])]
        obsdat[k] = [[obsdat[k][it][iz] for iz,z in enumerate(zobs[k]) if nap[iz]<1] for it,t in enumerate(tobs[k])]
        zobs[k] = [z for iz,z in enumerate(zobs[k]) if nap[iz]<1]
        if len(set(zobs[k]))<len(zobs[k]):
            raise Exception('Some depths appear more than once in the observations file %s.' % x['files'][var])
        
        
    zall = []
    [zall.extend(zl) for zl in zobs]
    zall = sorted(list(set(zall))) #All observation depths, unique and sorted [m]
    tall = []
    [tall.extend(tl) for tl in tobs]
    tall = sorted(list(set(tall))) #All observation times, unique and sorted [m]
    
    #Define parameter names, group, values, min, max
    parname = list(x['parameters'].keys())
    npar = len(parname)
    parval = list(x['parameters'].values())
    [parmin,parmax] = [list(parval),list(parval)]
    [partype,pargrp] = [['fixed']*npar,['none']*npar]
    for i,par in enumerate(parval):
        if type(par)==list:
            [parval[i], parmin[i], parmax[i]] = par
            if parmin[i]>parval[i]:
                parval[i] = parmin[i]
                writelog('\t\tThe starting value of the parameter %s is below its calibration range. Setting it to its minimum value.\n' % parname[i])
            elif parmax[i]<parval[i]:
                parval[i] = parmax[i]
                writelog('\t\tThe starting value of the parameter %s is above its calibration range. Setting it to its maximum value.\n' % parname[i])
            [partype[i],pargrp[i]] = ['none','fit']
        else: #Default: +/-20%
            [parmin[i], parmax[i]] = [par*0.8,par*1.2] if par>0 else [par*1.2,par*0.8] if par<0 else [0,1]
    
    nobs = sum([len(obs)*len(obs[0]) for obs in obsdat if obs!=[]]) #Total number of observations
    nnan = sum([sum([sum([obstz!=obstz for obstz in obst]) for obst in obs]) for obs in obsdat]) #Total number of NaNs

    #Write corresponding control file for PEST
    fid = open(pestDir+'/simstrat_calib.pst','w')
    fid.write('pcf\n')
    fid.write('* control data\n')
    fid.write('norestart estimation\n')
    fid.write('%d %d 1 0 %d\n' % (npar,nobs-nnan,len(obsdat)))
    fid.write('1 %d single nopoint 1 0 0\n' % len(obsFiles))
    fid.write('5.0 2.0 0.3 0.01 10\n')
    fid.write('5.0 5.0 0.001\n')
    fid.write('0.1\n')
    fid.write('20 0.005 4 3 0.01 3\n')
    fid.write('0 0 0\n')
    fid.write('* parameter groups\n')
    fid.write(' fit\trelative\t0.01\t0.00001\tswitch\t2.0\tparabolic\n')
    fid.write('* parameter data\n')
    for i in range(npar):
        if (parmin[i]==0 or parmax[i]==0 or abs(parmax[i]/parmin[i])>=1E4 or parmax[i]/parmin[i]<0):
            fid.write('%6s\t%s\t%s\t%10.4e\t%10.4e\t%10.4e\t%4s\t1.0\t0.0\t1\n' % (parname[i],partype[i],'relative',parval[i],parmin[i],parmax[i],pargrp[i]))            
        else:
            fid.write('%6s\t%s\t%s\t%10.4e\t%10.4e\t%10.4e\t%4s\t1.0\t0.0\t1\n' % (parname[i],partype[i],'factor',parval[i],parmin[i],parmax[i],pargrp[i]))
    fid.write('* observation groups\n')
    fid.write('obs%s\n' % '\nobs'.join(obsl))
    fid.write('* observation data\n')
    nanobs = [[] for k in range(len(obsdat))]
    for k in range(len(obsdat)):
        #Normalize the weight of observations: ratio (mean of first obs. variable)/(mean of current obs. variable)
        #weight = sum(sum(o for o in ov if o==o) for ov in obsdat[k])/len(obsdat[k])/len(obsdat[k][0])
        #weight = sum(sum(o for o in ov if o==o) for ov in obsdat[0])/len(obsdat[0])/len(obsdat[0][0])/weight
        #Normalize the weight of observations: ratio (max of first obs. variable)/(max of current obs. variable)
        if k==0: obsList0 = [val for vec in obsdat[k] for val in vec] #list (1D) of observations of first variable
        obsList = [val for vec in obsdat[k] for val in vec] #list (1D) of observations of current variable
        if all(v!=v for v in obsList): continue #if all NaNs, do nothing
        weight = max(v for v in obsList0 if v==v)/max(v for v in obsList if v==v) #ratio of max values
        #weight = max(max(o for o in ov if o==o) for ov in obsdat[k])
        #weight = max(max(o for o in ov if o==o) for ov in obsdat[0])/weight
        for itall,t in enumerate(tall):
            for izall,z in enumerate(zall):
                if (t in tobs[k] and z in zobs[k]):
                    it = list(tobs[k]).index(t)
                    iz = list(zobs[k]).index(z)
                    if obsdat[k][it][iz]==obsdat[k][it][iz]: #If not NaN
                        #Weight temperature obs. based on temperature gradient
                        #zp = np.array(zobs[k])[~np.isnan(obsdat[k][it])]
                        #Tp = np.array(obsdat[k][it])[~np.isnan(obsdat[k][it])]
                        #if len(zp)>1:
                        #    dTdz = np.interp(z,(np.array(zp[1:])+np.array(zp[:-1]))/2,np.abs(np.diff(Tp)/np.diff(zp)))
                        #else:
                        #    dTdz = 0
                        #fid.write('%s_%d_%d\t%12.4e\t%f\tobs\n' % (obsl[k],itall+1,izall+1,obsdat[k][it][iz],weight/(4*dTdz+1)))
                        fid.write('%s_%d_%d\t%12.4e\t%f\tobs%s\n' % (obsl[k],itall+1,izall+1,obsdat[k][it][iz],weight,obsl[k]))
                    else: #Store location of missing measurements so as not to write corresponding model observations
                        nanobs[k].extend([[itall,izall]])
    fid.write('* model command line\n')
    fid.write('simstrat.bat\n')
    fid.write('* model input/output\n')
    fid.write('simstrat_par.tpl\tsimstrat_PEST.par\n')
    for var in obsl:
        fid.write('simstrat_%s.ins\t%s_out.dat\n' % (var,os.path.join('results',var)))
    fid.write('* prior information\n')
    fid.close()
    
    #Write instructions file
    warnLength = False
    for k,var in enumerate(obsl):
        fid = open(os.path.join(pestDir,'simstrat_'+var+'.ins'),'w')
        fid.write('pif @\n')
        fid.write('l1\n') #Skip header
        if version.startswith('1.'): fid.write('l1\n') #Skip initial state
        for itall,t in enumerate(tall):
            strf = 'l1'
            if t in tobs[k]:
                for izall,z in enumerate(zall):
                    if z in zobs[k]:
                        if not ([itall,izall] in nanobs[k]):
                            if version.startswith('1.'): strf = strf+' [%s_%d_%d]%d:%d' % (var,itall+1,izall+1,12*izall+12,12*izall+23)
                            elif version.startswith('2.'): strf = strf+' @,@ !%s_%d_%d!' % (var,itall+1,izall+1)
                            if (12*izall+23)>=2000 and not warnLength:
                                writelog('\t\tWARNING: Some lines in the instruction file (%s) refer to locations exceeding 2000 characters. This cannot be handled by PEST and is likely to cause the calibration to crash. The number of observation depths should be reduced.\n' % os.path.basename(fid.name))
                                warnLength = True
                        else:
                            if version.startswith('2.'): strf = strf+' @,@'
            fid.write(strf+'\n')
        fid.close()
        
    #Write output depth and time files
    fid = open(os.path.join(pestDir,'simstrat_zout.dat'),'w')
    fid.write("output depths\n")
    for z in zall[::-1]:
        fid.write("%.2f\n" % z)
    fid.close()
    if len(zall)==1:
        writelog('\t\tWARNING: There is a single output depth in file \'simstrat_zout.dat\' (probably because there are observations only at one depth). This will be misunderstood by Simstrat and is likely to cause the calibration to crash.\n')
    fid = open(os.path.join(pestDir,'simstrat_tout.dat'),'w')
    fid.write("output times\n")
    for t in tall:
        fid.write("%.4f\n" % (t/3600/24))
    fid.close()
    

def TemplateFile(x,version):
    
    pestDir = x['files']['pestDir'] #PEST directory
    nCPU = x['PEST']['nCPU']
    
    if version.startswith('1.'):
        with open(x['files']['configFile'],'r') as file:
            config = file.readlines()
        #Correct path of input files
        for p in range(12):
            config[1+p] = os.path.relpath(os.path.join(os.path.dirname(os.path.realpath(x['files']['configFile'])),config[1+p]),pestDir)
            config[1+p] = os.path.join('..' if nCPU>1 else '',config[1+p])
        #Write results in a specific directory
        config[6] = os.path.join('results','') + '\n'
        #Write results only at the observation depths and times    
        config[7] = os.path.join('..' if nCPU>1 else '','simstrat_zout.dat' + '\n')
        config[8] = os.path.join('..' if nCPU>1 else '','simstrat_tout.dat' + '\n')
        #Replace the parameters by their corresponding key
        for p in range(len(x['parameters'])):
            ind = list(x['parameters'].keys()).index(config[32+p].split()[1].lower())
            config[32+p] = '#%10s# %s\n' % (list(x['parameters'].keys())[ind],' '.join(config[32+p].split()[1:]))
        config = ['ptf #\n'] + config
        with open(os.path.join(pestDir,'simstrat_par.tpl'),'w') as file:
            file.writelines(config)
    elif version.startswith('2.'):
        with open(x['files']['configFile'],'r') as file:
            config = json.load(file)
        #Correct path of input files
        for inputFile in config['Input']:
            if type(config['Input'][inputFile]) is str:
                config['Input'][inputFile] = os.path.relpath(os.path.join(os.path.dirname(os.path.realpath(x['files']['configFile'])),config['Input'][inputFile]),pestDir)
                config['Input'][inputFile] = os.path.join('..' if nCPU>1 else '',config['Input'][inputFile])
        #Write results in a specific directory
        config['Output']['Path'] = os.path.join('results','')
        #Write results only at the observation depths and times    
        config['Output']['Depths'] =  os.path.join('..' if nCPU>1 else '','simstrat_zout.dat')
        config['Output']['Times'] =  os.path.join('..' if nCPU>1 else '','simstrat_tout.dat')
        #Replace the parameters by their corresponding key
        for par in config['ModelParameters']:
            config['ModelParameters'][par] = ['%10s' % par]
        config_txt = json.dumps(config)
        config_txt = config_txt.replace('["','#').replace('"]','#')
        config_txt = config_txt.replace('}','\n}').replace(', ',',\n').replace('{','{\n')
        config_txt = 'ptf #\n' + config_txt
        with open(os.path.join(pestDir,'simstrat_par.tpl'),'w') as file:
            file.write(config_txt)


def BatchFile(x):
    
    pestDir = x['files']['pestDir'] #PEST directory
    nCPU = x['PEST']['nCPU']
    
    #Write batch file for PEST
    fid = open(os.path.join(pestDir,'simstrat.bat'),'w')
    modelPath = os.path.relpath(x['files']['model'],pestDir)
    if nCPU>1:
        modelPath = os.path.join('..',modelPath)
    fid.write(modelPath+' simstrat_PEST.par\n')
    fid.close()
    
    #Copy to all CPUs
    if nCPU>1:
        for k in range(nCPU):
            shutil.copy(os.path.join(pestDir,'simstrat.bat'),os.path.join(pestDir,'cpu'+str(k+1)))
        os.remove(os.path.join(pestDir,'simstrat.bat'))


def RunManagementFile(x):
    
    pestDir = x['files']['pestDir'] #PEST directory
    nCPU = x['PEST']['nCPU']
    
    #Write run management file for PEST
    fid = open(os.path.join(pestDir,'simstrat_calib.rmf'),'w')
    fid.write('prf\n')
    fid.write('%d 0 0.5 1\n' % nCPU)
    for k in range(nCPU):
        fid.write('\'cpu'+str(k+1)+'\' '+os.path.join('cpu'+str(k+1),'')+'\n')
    fid.write('%s\n' % ' '.join(['600']*nCPU))


def runPEST(configFile,standalone=True,mode='pest',arch='64',ip=None,port=None,version='1.6'):
    if standalone:
        global logFile
        logFile = open('PEST.log','w')
    
    if mode.lower()=='beopest' and port is None:
        port = '4004'
        #Kill running BEOPEST processes
        os.system('taskkill /F /IM beopest64.exe /T')
        os.system('taskkill /F /IM beopest32.exe /T')
    if mode.lower()=='pest':
        #Kill running PEST processes
        os.system('taskkill /F /IM pslave.exe /T')
        os.system('taskkill /F /IM ppest.exe /T')
    
    #Load PEST configuration
    x = json.load(open(configFile))
    #Basic checks
    x = checkJSON(x)
    checkParameters(x,mode,version)
        
    pestDir = x['files']['pestDir']
    nCPU = x['PEST']['nCPU']
    
    #Create folder structure
    if os.path.exists(pestDir):
        writelog('\t\tPEST directory already exists. Overwriting contents...\n')
        os.chmod(pestDir,stat.S_IWUSR)
        shutil.rmtree(pestDir)
        time.sleep(5)
    os.mkdir(pestDir)
    if nCPU>1:
        for k in range(nCPU):
            os.mkdir(os.path.join(pestDir,'cpu'+str(k+1)))
            os.mkdir(os.path.join(pestDir,'cpu'+str(k+1),'results'))
    else:
        os.mkdir(os.path.join(pestDir,'results'))
    
    #Write control and instructions file
    ControlFile(x,version)
    
    #Write template file based on configuration file
    TemplateFile(x,version)
    
    #Write batch file
    BatchFile(x)
    
    #Write run management file
    if nCPU>1:
        RunManagementFile(x)
    
    if mode.lower()=='pest': #Run PEST
        if nCPU==1:
            proc = subprocess.Popen('pest simstrat_calib',cwd=pestDir,shell=True)
        else:
            for k in range(nCPU):
                subprocess.Popen('START /B cmd /c echo simstrat.bat^> \"%temp%\\log.tmp\" ^& (pslave ^< \"%temp%\\log.tmp\") ^& del \"%temp%\\log.tmp',cwd=os.path.join(pestDir,'cpu'+str(k+1)),shell=True)
            proc = subprocess.Popen('ppest simstrat_calib /p1',cwd=pestDir,shell=True)
        (out,err) = proc.communicate() #Wait for PEST to finish running
    elif mode.lower()=='beopest': #Run BEOPEST
        if ip==None: raise Exception('IP of computer must be specified when using BEOPEST')
        if nCPU>=1:
           subprocess.Popen('beopest%s simstrat_calib /p1 /H :%s' % (arch,port),cwd=pestDir,shell=True)
           for k in range(nCPU):
               shutil.copy(os.path.join(pestDir,'simstrat_calib.pst'),os.path.join(pestDir,'cpu'+str(k+1)))
               shutil.copy(os.path.join(pestDir,'simstrat_par.tpl'),os.path.join(pestDir,'cpu'+str(k+1)))
               [shutil.copy(insFile,os.path.join(pestDir,'cpu'+str(k+1))) for insFile in glob.glob(os.path.join(pestDir,'*.ins'))]
               proc = subprocess.Popen('beopest%s simstrat_calib /H %s:%s' % (arch,ip,port),cwd=os.path.join(pestDir,'cpu'+str(k+1)),shell=True)
    (out,err) = proc.communicate() #Wait for BEOPEST to finish running
    
    #Display results
    try:
        with open(os.path.join(x['files']['pestDir'],'simstrat_calib.par'),'r') as file:
            results = file.readlines()[1:]
    except FileNotFoundError:
        raise Exception('%s' % out)
    x['results'] = {}
    par_opt = [results[i] for i in [i for i,x in enumerate([isinstance(a,list) for a in x['parameters'].values()]) if x]]
    writelog('\t\tResults:\n')
    for par in par_opt:
        par_s = par.split()
        writelog(('\t\t%14s = %8.5g\n' % (par_s[0],float(par_s[1]))))
        x['results'][par_s[0]] = float(par_s[1])
    
    #Record configuration and results
    #with open('PEST_'+datetime.now().strftime('%Y%m%d%H%M%S')+'.res','w') as file:
    #    json.dump(x,file,indent=4)
    
    #Rewrite a configuration file with the optimized parameter set
    if 'configFile_out' in x['files']:
        configFile_opt = x['files']['configFile_out']
    else:
        configFile_opt = os.path.join(x['files']['configFile'].split('/')[0],'simstrat_opt.par')
    refDir = os.path.dirname(configFile_opt)
    if 'results_out' in x['files']:
        resultsDir_opt = x['files']['results_out']
    else:
        resultsDir_opt = os.path.join(x['files']['configFile'].split('/')[0],'results_opt')
    configFile_opt = configFile_opt.replace('/',os.sep).replace('\\',os.sep)
    resultsDir_opt = resultsDir_opt.replace('/',os.sep).replace('\\',os.sep)
    if version.startswith('1.'):
        with open(x['files']['configFile'],'r') as file:
            config = file.readlines()
        config[6] = os.path.relpath(resultsDir_opt,refDir)+'\n'
        for p in range(len(x['parameters'])):
            ind = list(x['parameters'].keys()).index(config[32+p].split()[1].lower())
            par = list(x['parameters'].keys())[ind]
            val = list(x['parameters'].values())[ind]
            if par.lower() in list(x['results'].keys()):
                config[32+p] = ('%7g' % list(x['results'].values())[list(x['results'].keys()).index(par.lower())]) + ' '*6 + ' '.join(config[32+p].split()[1:]) + '\n'
            else:
                config[32+p] = str(val) + ' '*(13-len(str(val))) + ' '.join(config[32+p].split()[1:]) + '\n'
        with open(configFile_opt,'w') as file:
            file.writelines(config)
    elif version.startswith('2.'):
        with open(x['files']['configFile'],'r') as file:
            config = json.load(file)
        config['Output']['Path'] = os.path.relpath(resultsDir_opt,refDir)
        for par in config['ModelParameters']:
            if par.lower() in list(x['results'].keys()):
                ind = list(x['results'].keys()).index(par.lower())
                config['ModelParameters'][par] = list(x['results'].values())[ind]
            else:
                ind = list(x['parameters'].keys()).index(par.lower())
                config['ModelParameters'][par] = list(x['parameters'].values())[ind]
        with open(configFile_opt,'w') as file:
            json.dump(config,file,indent=4)
    writelog(('\t\tConfiguration file with optimal parameter set: %s\n' % configFile_opt))
    
    #Create folder for results with optimal parameter set
    if os.path.exists(resultsDir_opt):
        os.chmod(resultsDir_opt,stat.S_IWUSR)
        shutil.rmtree(resultsDir_opt)
        time.sleep(5)
    os.mkdir(resultsDir_opt)
    
    if standalone:
        #Run the model with initial configuration but best parameters
        subprocess.Popen(os.path.relpath(x['files']['model'],refDir) + ' ' + os.path.relpath(configFile_opt,refDir),shell=True,cwd=refDir).wait()
        writelog(('\t\tModel results using the optimized parameter set are in %s\n' % resultsDir_opt))
        logFile.close()