#!/usr/bin/env python
# coding: utf-8

def autocorr(tr,stalign_a=5,stalign_b=20):
        winlen = 15
        N = tr.stats.npts
        delta = tr.stats.delta
        yf = fft(tr.data)
        ayf = np.abs(yf)
        conlen = 15
        myf = np.convolve(ayf, np.ones((conlen,))/conlen, mode='same')
        yf = yf/myf
        xx = np.real(ifft(yf))
        
        stalign_a = int(stalign_a/delta)
        stalign_b = int(stalign_b/delta)
        winlen = stalign_b - stalign_a
        xxw = xx[stalign_a:stalign_b]
        acorr = np.correlate(xx,xxw,mode='full')
        acorr = acorr[winlen:]
        maxloc = np.argmax(abs(acorr))
        acorr = np.roll(acorr,-maxloc)
        
        tr.data = acorr
        return tr

from scipy import interpolate
import numpy as np
pPdP = np.load('./tables/pPdP30to330.npy')

ndepth = 301
ndis = 131
dep = np.linspace(30,330,ndepth)
dis = np.linspace(30,95,ndis)
fppdp = interpolate.interp2d(dis, dep, pPdP, kind='linear')


pPdP = np.load('./tables/tele30to330.npy')
ftelep = interpolate.interp2d(dis, dep, pPdP, kind='linear')



from collections import defaultdict
import random
import pickle
import numpy as np
import obspy
import distaz
from obspy import taup
from matplotlib import pyplot as plt
from scipy.signal import hilbert
from obspy.core.util import AttribDict
import os
from scipy.fftpack import fft,ifft
import glob
from scipy.signal import hilbert
from mpl_toolkits.basemap import Basemap
import pandas as pd
import distance
from obspy.core.stream import Stream
from obspy.signal.filter import bandpass
from obspy.io.sac import SACTrace
import time

trimb = 10
trima = 140
frqmin = 0.2
frqmax = 2.0
rsample = 10
endtime = 25.0
tpratio = 0.05
winlen = 15
envolope = 0
mindist = 30
maxdist = 90
mu = 2.0


eqdir = 'data_demo/'

eqs = open(''.join([eqdir,'event_list_pickle']),'rb')

eqs = pickle.load(eqs)
evid = []
model = taup.TauPyModel(model='ak135')
for ii in eqs:
    evid.append(ii['event_id'])
    
npoints = int(endtime*rsample+1)
evelist = glob.glob(eqdir+'/*a')
nevent = len(evelist) 


for ievent in range(0,nevent):  
    datasave_sub = defaultdict(list)
    start = time.time()
    ievent1 = ievent
    evname1 = evelist[ievent1].split('/')[-1]
    evidx1 = evid.index(evname1)
    evlat1 = eqs[evidx1]['latitude']
    evlon1 = eqs[evidx1]['longitude']
    evdep1 = eqs[evidx1]['depth']
    evtime1 = eqs[evidx1]['datetime']
    evmag1 = eqs[evidx1]['magnitude']  
    stalist1 = glob.glob('/'.join([eqdir,evname1,'*.BHZ']))
    stalist = [stal.split('/')[-1] for stal in stalist1]
    irissta = pd.read_csv('/'.join([eqdir,evname1,'station_event']),header=None,names=                   ('net','sta','loc','channel','lat','lon','ele','tmp1','datasrc','startt',                  'evlat','evlon','evdep','mg','tmp2','tmp3','tmp4','temp5'),na_filter=False)

    print ('the',ievent,'th event')
    print ('evinfo for eq 1:',evname1,evmag1,evlat1,evlon1,evdep1)

    nsta = len(stalist1)
    stapos = np.zeros((nsta,2))
    idx = 0
    staposcl = np.zeros((nsta,2))
    for ista in range(nsta):
        stanet = stalist[ista].split('.')[0]
        staname = stalist[ista].split('.')[1]
        stasub = irissta[(irissta['net']==stanet) & (irissta['sta']==staname)]
        stlat = stasub.iloc[0]['lat']
        stlon = stasub.iloc[0]['lon']
        dis1 = distaz.DistAz(evlat1,evlon1,stlat,stlon)
        stapos[ista,0] = stlat
        stapos[ista,1] = stlon
        if mindist<dis1.delta<maxdist:
            staposcl[idx,0] = stlat
            staposcl[idx,1] = stlon
            idx += 1
        
    if idx < 10:
        print (idx)
        continue
        
    strmacc1 = Stream()
    strmori = Stream()
    stalist1 = []


    idx = np.arange(nsta)
    for ista in range(len(idx)):
                
                stlat = stapos[idx[ista],0]#inv.get_coordinates(stid)['latitude']
                stlon = stapos[idx[ista],1]#inv.get_coordinates(stid)['longitude']
                

                dis1 = distaz.DistAz(evlat1,evlon1,stlat,stlon)


                if dis1.delta<mindist or dis1.delta>maxdist:
                    continue
                    
                trace = '/'.join([eqdir,evname1,stalist[idx[ista]]])
                strm = obspy.read(trace)
                tr = strm[0]
                tr.resample(rsample)
                tr.stats.coordinates=AttribDict({'latitude':stlat,'longitude':stlon,                                                 'elevation':0})
                
                parr = ftelep(dis1.delta,evdep1)[0]
                tr.stats.distance = dis1.delta#inc_angle
                tr.trim(evtime1+parr-trimb,evtime1+parr+trima,pad=True,fill_value=0)
                
                tr.stats.starttime = 0
                tr.detrend()
                tr.taper(tpratio)
                tr.data = bandpass(tr.data,frqmin,frqmax,tr.stats.sampling_rate,2,True)
                tr.normalize()
                strmori.append(tr.copy())
                
                tr = autocorr(tr)

                tr.taper(tpratio)
                tr.data = bandpass(tr.data,frqmin,frqmax,tr.stats.sampling_rate,2,True)
                
                
                
                if envolope:
                       tr.data = np.abs(hilbert(tr.data))
                tr.normalize()
                if np.isnan(tr.data).any():
                    continue
                strmacc1.append(tr.copy())
                stalist1.append(tr.id)

                reftime = fppdp(70,evdep1)
                ctime = fppdp(dis1.delta,evdep1)
                ctime = reftime-ctime
                data = np.roll(tr.data,int(np.round(ctime*rsample)))
                tr.data = data

                
    print ('finishing',time.time()-start )
    
    strmacc1.resample(rsample)
    if len(strmacc1) < 10:
        print ('small no. of traces:',len(strmacc1))
        continue
    

    refdismax = 70

    nsta = len(strmacc1)
    npts = strmacc1[0].stats.npts
    depstack = np.zeros(npts,)
    
#     print reflat,reflon,refdismax,len(strmacc1)
        
    reftime = fppdp(refdismax,evdep1)
#         data = abs(hilbert(strmacc1[0].data.copy()))
    data = strmacc1[0].data.copy()

    bu = np.zeros(npts,)
    phi = np.zeros((npts,),dtype=complex)
    for ii in range(0,nsta):
        refdis = strmacc1[ii].stats.distance
        ctime = fppdp(refdis,evdep1)
        ctime = reftime-ctime
        data[:len(strmacc1[ii].data)] = np.roll(strmacc1[ii].data,int(np.round(ctime*rsample)))
        phi = phi+np.exp(1j*np.angle(hilbert(data)))
        bu = bu+data

    aphi = (np.abs(phi)/nsta)**mu
    depstack = bu
    depstackpws = bu*aphi

    plt.figure(figsize=(15,4))
    depstack = (depstack-depstack.min())/(depstack.max()-depstack.min())
    depstackpws = (depstackpws-depstackpws.min())/(depstackpws.max()-depstackpws.min())
    plt.plot(depstack)
    plt.plot(depstackpws)
    plt.legend(['linear stacking','pws stacking'])
    plt.show()

