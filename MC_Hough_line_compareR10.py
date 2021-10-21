#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 00:03:35 2021

@author: rogerslc
"""
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as m3d
import math
from matplotlib import image
from matplotlib import pyplot

import os.path
from os import path

from scipy.signal import convolve as scipy_convolve

def randrange(n, vmin, vmax):
    return (vmax-vmin)*np.random.rand(n) + vmin


def Ry (R, phi, theta):
    return (R) * np.cos(phi)

def Rx (R, phi, theta):
    return (R) * np.sin(phi) * np.cos(theta)

def Rz (R, phi, theta):
    return (R) * np.sin(phi) * np.sin(theta)

def Distance (P1,P2,P3,U1,U2,U3):
    S_1=P2*U3-P3*U2
    S_2=P3*U1-P1*U3
    S_3=P1*U2-P2*U1
    Dist=np.sqrt(S_1**2+S_2**2+S_3**2)
    return Dist



low_lim=2
MaxTimefromMuon=.01

# path to save the file                                                                                                                                                           
savelocat=sys.argv[1]
#run numbers
start_run = int(sys.argv[2])
end_run   = int(sys.argv[3])



FidCut=178*2**-.5 #fiducial cut for x and y in mm                                                                                                                            
depth=510

numb=0
vars=np.arange(-205,205,100)
R=np.arange(-3500,3500,400)
width=430
height=430

angle_step=1

radius =180
allowedspread=35 #mm on either side of the track                                                                                                                             
allowedT=1
tracklenmin=120 #mm minimum accepted length of track within detector                                                                                                         
dedxmin=.017 #MeV/mm maximum dE/dx, determined from MC                                                                                                                       


diag_lenxy = int(np.ceil(np.sqrt(width * width + height * height)))   # max_dist                                                                                          
diag_lenzy = int(np.ceil(np.sqrt(depth * depth + height * height)))   # max_dist                                                                                          

rhos = np.linspace(-diag_lenxy, diag_lenxy, diag_lenxy * 2)
rhozed = np.linspace(-diag_lenzy, diag_lenzy, diag_lenzy * 2)


thetas = np.deg2rad(np.arange(-90, 90, angle_step))
num_thetas = len(thetas)


#For convolution with gausians in Fourier space                                                                                                                              
#First a 1-D  Gaussian                                                                                                                                                       
t = np.linspace(-10, 10, 30)
bump = np.exp(-0.1*t**2)
bump /= np.trapz(bump) # normalize the integral to 1                                                                                                                         

# make a 2-D kernel out of it                                                                                                                                                
kernel = bump[:, np.newaxis] * bump[np.newaxis, :]

#locatformat='/analysis/{}/hdf5/prod/v1.2.0/20191122/cdst/trigger2'

#MC muon data                                                                                                              
locatformat='/lustre/neu/data4/NEXT/NEXTNEW/MC/Background/NEXUS_NEXT_v1_05_02/Muons/esmeralda/output/'

num =0
probnum=0
ttlevts=0

for i in range(start_run, end_run):

    run_num=str(i)
    locat=locatformat
    fileformat=locat+'MuonGenerator_tracking_NEW_ICPR656-20191119_{}.h5'
    files = fileformat.format(run_num)       
    
    try:
        events_data=pd.read_hdf(files,key='/CHITS/highTh')
        line_data=pd.read_hdf(files,key='/Tracking/Tracks')
        
        MC_info=pd.read_hdf(files,key='/MC/hits')
        MC_particles=pd.read_hdf(files,key='/MC/particles')
                
    except FileNotFoundError:
        print('file not there')
        continue    
    except KeyError:
        print('KeyError')
        continue
    
    if path.isfile(savelocat+'/run'+run_num+'alleventscutinfo.h5')==True:
        print (files, 'already exists')
        continue
    
    outside_fid=line_data[(line_data.r_max > 197) | (line_data['z_min'] < 20) | (line_data['z_max'] > 490) & line_data.z_max < depth]
    ttlevts+=len(outside_fid['event'].unique())
    
    
    dedxall=[]
    pcline=[]
    linlen=[]
    munrg=[]
    muvts=[]
    mu_theta=[]
    mu_phi=[]
    xint=[]
    zint=[]

    truth_theta=[]
    truth_phi=[]
    truth_per=[]
    truth_dedx=[]
    truth_len=[]


    costhetas=[]
    costhetasall=[]

    for evts in outside_fid['event'].unique():
        print(run_num,evts)
        #check MC muon goes through detector                                                                                                                                 
        mixi=MC_particles[(MC_particles.event_id==evts) & (MC_particles.particle_id==1)]

        if len(mixi)==0:
            print('nonexistant',evts)
            continue
        MCdx=mixi.final_x.values-mixi.initial_x.values
        MCdy=mixi.final_y.values-mixi.initial_y.values
        MCdz=mixi.final_z.values-mixi.initial_z.values


        t=np.roots([(MCdx**2+MCdy**2)[0],(2*(mixi.initial_x.values*MCdx+mixi.initial_y.values*MCdy))[0],(mixi.initial_x.values**2+mixi.initial_y.values**2-((FidCut)**2))[0]])
        zintercept=(MCdz*t+mixi.initial_z.values)
        xintercept=(MCdx*t+mixi.initial_x.values)
        yintercept=(MCdy*t+mixi.initial_y.values)

        if any(np.iscomplex(t) == True):
            print("muon does no intersect fiducial volume",evts,t)
            continue        
        if all(zintercept <20 ) or all(zintercept >490):           
            print("muon does no intersect fiducial volume",evts,zintercept)
            continue


        #fiducial cuts                                                                                                                                                       
        events_dataplt=events_data[events_data['event']==evts][(events_data['Ec']>0) & (events_data.Z<=depth) & (events_data.Z>=20) ]
        events_dataplt=events_dataplt[(events_dataplt.X<FidCut) & (events_dataplt.X>-FidCut)]
        events_dataplt=events_dataplt[(events_dataplt.Y<FidCut) & (events_dataplt.Y>-FidCut)]


        ttl_nrg=sum(events_dataplt['Ec'])
        if ttl_nrg==0:
            print ("no energy deposited",ttl_nrg, run_num,evts)
            continue
        max_nrg=np.max(events_data[events_data['event']==evts]['Ec'])

        MCtheta=np.arctan2(-MCdz,-MCdx)
        MCphi=np.arctan2(np.sqrt(MCdx**2+MCdz**2),-MCdy)

        if MCphi>np.pi/2:
            MCtheta+=np.pi
            MCphi=np.pi-MCphi

        if MCtheta<0:
            MCtheta+=2*np.pi

        if MCtheta>2*np.pi:
            MCtheta-=2*np.pi

        #use this if doing full muon cuts 
        #if ttl_nrg < low_lim:                                                                                                                                               
            #print ('okay continuing',ttl_nrg,low_lim)                                                                                                                       
            #continue                                                                                                                                                        

        EmxEtl=max_nrg/ttl_nrg

        #collecting minimal data so can skip computations where wont be kept after cuts anyways
        if  (EmxEtl>.1) or (ttl_nrg<low_lim):
            print('EmxEtl is huge',EmxEtl,ttl_nrg)
            costhetasall.append(NaN)
            dedxall.append(NaN)
            pcline.append(NaN)
            linlen.append(NaN)
            munrg.append(ttl_nrg)
            muvts.append(evts)
            mu_theta.append(NaN)
            mu_phi.append(NaN)
            xint.append(NaN)
            zint.append(NaN)
            truth_theta.append(MCtheta)
            truth_phi.append(MCphi)
            truth_per.append(NaN)
            truth_dedx.append(NaN)
            truth_len.append(NaN)
            continue

        #now for accumlators to find best rhos and thetas
        accumulatorXY = np.zeros((2 * diag_lenxy, num_thetas), dtype=np.uint64)
        accumulatorZY = np.zeros((2 * diag_lenzy, num_thetas), dtype=np.uint64)
        accumulatorXZ = np.zeros((2 * diag_lenzy, num_thetas), dtype=np.uint64)


        for x,y,z in events_dataplt[['X','Y','Z']].values:
            posx=x
            posy=y
            #posz=z-depth/2                                                                                                                                                  
            posz=z*.97-depth/2 #to change from microsec to mm, the minus is just for a mid axis for hough trans                                                              
            tval=0
            for t in thetas:
                rhoXY=round(posx * np.cos(t) + posy * np.sin(t),0) + diag_lenxy #XY plane                                                                                    
                rhoZY=round(posz * np.cos(t) + posy * np.sin(t),0) + diag_lenzy #ZY plane   
                rhoXZ=round(posx * np.cos(t) + posz * np.sin(t),0) + diag_lenzy #XZ plane          
                accumulatorXY[int(rhoXY), tval] += 1
                accumulatorZY[int(rhoZY), tval] += 1
                accumulatorXZ[int(rhoXZ), tval] += 1
                tval+=1


        # Convolution: scipy's direct convolution mode spreads out NaNs                                                                                                      
        conv_accuXY = scipy_convolve(accumulatorXY, kernel, mode='same', method='direct')
        conv_accuZY = scipy_convolve(accumulatorZY, kernel, mode='same', method='direct')
        conv_accuXZ = scipy_convolve(accumulatorXZ, kernel, mode='same', method='direct')

        locatXY = np.unravel_index(np.argmax(conv_accuXY, axis=None), accumulatorXY.shape)
        locatZY = np.unravel_index(np.argmax(conv_accuZY, axis=None), accumulatorZY.shape)
        locatXZ = np.unravel_index(np.argmax(conv_accuXZ, axis=None), accumulatorXZ.shape)


        rhoXY = rhos[locatXY[0]]
        thetaXY = thetas[locatXY[1]]
        rhoZY=rhozed[locatZY[0]]
        thetaZY=thetas[locatZY[1]]
        rhoXZ=rhozed[locatXZ[0]]
        thetaXZ=thetas[locatXZ[1]]

        
        if (thetaZY ==0) or (thetaXY==0) or (thetaXZ==0):
            print ("need to implement catch for thetas =0",run_num,evts)
            continue

        z0=rhoZY/np.cos(thetaZY) +depth/2
        x0=rhoXY/np.cos(thetaXY)

        x1=rhoXZ/np.cos(thetaXZ)
        z1=rhoZY/np.cos(thetaZY)+depth/2

        z2=rhoXZ/np.cos(thetaXZ)+depth/2
        y2=rhoXY/np.cos(thetaXY)

        alpha1=np.arctan2(np.sqrt(np.tan(thetaXY)**2+np.tan(thetaZY)**2),1)
        beta1=np.arctan2(-np.tan(thetaZY),-np.tan(thetaXY))

        alpha2=np.arctan2(np.sqrt(np.tan(thetaZY)**2+np.tan(thetaXZ)**2*np.tan(thetaZY)**2),1)
        alpha3=np.arctan2(np.sqrt(np.tan(thetaXY)**2+np.tan(thetaXY)**2/np.tan(thetaXZ)**2),1)     



        if thetaXY*thetaXZ >0:
            beta2=thetaXZ+np.pi/2
            beta3=thetaXZ+np.pi/2

        if thetaXY*thetaXZ <0:
            beta2=thetaXZ-np.pi/2
            beta3=thetaXZ-np.pi/2    


        #there has to be a cleaner way to do this        
        if alpha1>np.pi/2:
            beta1+=np.pi
            alpha1=np.pi-alpha1
        if beta1<0:
            beta1+=2*np.pi
        if beta1>2*np.pi:
            beta1-=2*np.pi    

        if alpha2>np.pi/2:
            beta2+=np.pi
            alpha2=np.pi-alpha2
        if beta2<0:
            beta2+=2*np.pi
        if beta2>2*np.pi:
            beta2-=2*np.pi 

        if alpha3>np.pi/2:
            beta3+=np.pi
            alpha3=np.pi-alpha3
        if beta3<0:
            beta3+=2*np.pi
        if beta3>2*np.pi:
            beta3-=2*np.pi     

        
        #intersections for each set where they share a common axis
        x0=rhoXY/np.cos(thetaXY)
        z0=rhoZY/np.cos(thetaZY) 

        x1=rhoXZ/np.cos(thetaXZ)
        y1=rhoZY/np.sin(thetaZY)

        y2=rhoXY/np.sin(thetaXY)
        z2=rhoXZ/np.sin(thetaXZ)

        weightXY=conv_accuXY[int(diag_lenxy+rhoXY), int(90+np.rad2deg(thetaXY))]
        weightXZ=conv_accuXZ[int(diag_lenzy+rhoXZ), int(90+np.rad2deg(thetaXZ))]
        weightZY=conv_accuZY[int(diag_lenzy+rhoZY), int(90+np.rad2deg(thetaZY))]
        
        
        #finding set of points from each pair that projects onto 3rd plane's hough space for best 3d line definition

        if (np.abs(np.rad2deg(beta1)==180)) or (alpha1==0):
            weightXZ1=0
        else:
            Rpz=-z0/(np.sin(alpha1)*np.sin(beta1))
            Rpx=-x0/(np.sin(alpha1)*np.cos(beta1))
            xp0=Rx(Rpz,alpha1,beta1)+x0
            zp0=Rz(Rpx,alpha1,beta1)+z0
            thetxz=np.arctan(xp0/(zp0))
            roxz=xp0*np.cos(thetxz)
            weightXZ1=conv_accuXZ[int(diag_lenzy+rhoXZ), int(90+np.rad2deg(thetxz))] #put in roxz


        if (np.abs(np.rad2deg(beta2)==180)) or (alpha2==0):
            weightXY2=0
        else:
            Rpy=-y1/(np.cos(alpha2))
            Rpx=-x1/(np.sin(alpha2)*np.cos(beta2))
            xp1=Rx(Rpy,alpha1,beta1)+x1
            yp1=Ry(Rpx,alpha1,beta1)+y1
            thetxy=np.arctan(xp1/yp1)
            roxy=xp1*np.cos(thetxy)
            weightXY2=conv_accuXY[int(diag_lenxy+rhoXY), int(90+np.rad2deg(thetxy))]


        if (np.abs(np.rad2deg(beta3)==180)) or (alpha3==0):
            weightZY3=0
        else:
            Rpz=-z2/(np.sin(alpha3)*np.sin(beta3))
            Rpy=-y2/(np.cos(alpha3))
            yp2=Ry(Rpz,alpha3,beta3)+y2
            zp2=Rz(Rpy,alpha3,beta3)+z2
            thetzy=np.arctan((zp2)/yp2)
            rozy=zp2*np.cos(thetzy)
            weightZY3=conv_accuZY[int(diag_lenzy+rhoZY), int(90+np.rad2deg(thetzy))]

        ProdW1=weightXY*weightXZ1*weightZY    
        ProdW2=weightXY2*weightXZ*weightZY    
        ProdW3=weightXY*weightXZ*weightZY3

        WTs=[ProdW1,ProdW2,ProdW3]
        As=[alpha1,alpha2,alpha3]
        Bs=[beta1,beta2,beta3]
        xs=[x0,x1,0]
        ys=[0,y1,y2]
        zs=[z0,0,z2]


        #selecting the best set
        LOC=np.argmax(WTs)
        alpha=As[LOC]
        beta=Bs[LOC]

        Xint=xs[LOC]
        Yint=ys[LOC]    
        Zint=zs[LOC]

        bestx=Rx(R,alpha,beta)+Xint
        bestz=Rz(R,alpha,beta)+Zint+depth/2
        besty=Ry(R,alpha,beta)+Yint
        
        #MC truth stuff
        MCmag=np.sqrt(MCdx**2+MCdy**2+MCdz**2)

        MCunix=MCdx/MCmag
        MCuniy=MCdy/MCmag
        MCuniz=MCdz/MCmag

        dx=-np.sin(thetaXY)
        dy=np.cos(thetaXY)
        dz=-np.sin(thetaZY)

        mag=np.sqrt(dx**2+dy**2+dz**2)
        unix=dx/mag
        uniy=dy/mag
        uniz=dz/mag

        dot=unix*MCunix+uniy*MCuniy+uniz*MCuniz





        events_dataplt['cross']=Distance((events_dataplt.X-Xint),(events_dataplt.Z-Zint-depth/2),(events_dataplt.Y-Yint),Rx(1,alpha,beta),Rz(1,alpha,beta),Ry(1,alpha,beta))
        events_dataplt['MCcross']=Distance((events_dataplt.X-Xint),(events_dataplt.Z-Zint-depth/2),(events_dataplt.Y-Yint),Rx(1,MCphi,MCtheta),Rz(1,MCphi,MCtheta),Ry(1,MCphi,MCtheta))


        MClinelike=events_dataplt[(np.abs(events_dataplt.MCcross)<=allowedspread)]
        linelike=events_dataplt[(np.abs(events_dataplt.cross)<=allowedspread)]
        MClen=np.sqrt((np.min(MClinelike.X)-np.max(MClinelike.X))**2+(np.min(MClinelike.Y)-np.max(MClinelike.Y))**2+(np.min(MClinelike.Z)-np.max(MClinelike.Z))**2)
        linelen=np.sqrt((np.min(linelike.X)-np.max(linelike.X))**2+(np.min(linelike.Y)-np.max(linelike.Y))**2+(np.min(linelike.Z)-np.max(linelike.Z))**2)
        MCper=sum(MClinelike['Ec'])/ttl_nrg
        nrgonline=sum(linelike['Ec'])/ttl_nrg
        MCdedx=sum(MClinelike['Ec'])/linelen
        nrgovrlen=sum(linelike['Ec'])/linelen        
        
        #collecting all the data for saving at end
        costhetasall.append(dot[0])
        dedxall.append(nrgovrlen)  
        pcline.append(nrgonline)
        linlen.append(linelen)
        munrg.append(ttl_nrg)
        muvts.append(evts)
        mu_theta.append(beta)
        mu_phi.append(alpha)
        xint.append(x0)
        zint.append(z0)
        truth_theta.append(MCtheta)
        truth_phi.append(MCphi)
        truth_per.append(MCper)
        truth_dedx.append(MCdedx)
        truth_len.append(MClen)

        #2d plots
        '''xdataplot=events_dataplt.X
        ydataplot=events_dataplt.Z
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot()
        ax.set_xlabel('X')
        ax.plot([mixi.initial_x.values[0],mixi.final_x.values[0]],[mixi.initial_z.values[0],mixi.final_z.values[0]],linewidth=3, color='limegreen',label='True')
        ax.plot(bestx,bestz,color='magenta',linewidth=3,label='best')
        ax.plot([FidCut,FidCut],[-depth,depth],color='k')
        ax.plot([-FidCut,-FidCut],[-depth,depth],color='k')
        ax.plot([-width/2,width/2],[490,490],color='k')
        ax.plot([-width/2,width/2],[20,20],color='k')
        ax.set_ylabel('Z')
        ax.legend(fontsize=20)
        ax.set_xlim(-215, 215)
        ax.set_ylim(0, 500)
        colmap = cm.ScalarMappable(cmap=cm.jet)
        ax.scatter(xdataplot,ydataplot,c=cm.jet(events_dataplt['Ec']/np.max(events_dataplt['Ec'])))


        xdataplot=events_dataplt.X
        ydataplot=events_dataplt.Y
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot()
        ax.set_xlabel('X')
        ax.plot([mixi.initial_x.values[0],mixi.final_x.values[0]],[mixi.initial_y.values[0],mixi.final_y.values[0]],linewidth=3, color='limegreen',label='True')
        ax.plot(bestx,besty,color='magenta',linewidth=3,label='best')
        ax.plot([FidCut,FidCut],[-depth,depth],color='k')
        ax.plot([-FidCut,-FidCut],[-depth,depth],color='k')
        ax.plot([-width/2,width/2],[-FidCut,-FidCut],color='k')
        ax.plot([-width/2,width/2],[FidCut,FidCut],color='k')
        ax.set_ylabel('Y')
        ax.legend(fontsize=20)
        ax.set_xlim(-215, 215)
        ax.set_ylim(-215, 215)
        colmap = cm.ScalarMappable(cmap=cm.jet)
        ax.scatter(xdataplot,ydataplot,c=cm.jet(events_dataplt['Ec']/np.max(events_dataplt['Ec'])))


        xdataplot=events_dataplt.Z
        ydataplot=events_dataplt.Y
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot()
        ax.set_xlabel('Z')
        ax.plot([mixi.initial_z.values[0],mixi.final_z.values[0]],[mixi.initial_y.values[0],mixi.final_y.values[0]],linewidth=3, color='limegreen',label='True')
        ax.plot(bestz,besty,color='magenta',linewidth=3,label='best')
        ax.plot([0,depth],[-FidCut,-FidCut],color='k')
        ax.plot([0,depth],[FidCut,FidCut],color='k')
        ax.plot([20,20],[-FidCut,FidCut],color='k')
        ax.plot([490,490],[-FidCut,+FidCut],color='k')
        ax.set_ylabel('Y')
        ax.set_xlim(0, 500)
        ax.set_ylim(-215, 215)
        colmap = cm.ScalarMappable(cmap=cm.jet)
        ax.legend(fontsize=20)
        ax.scatter(xdataplot,ydataplot,c=cm.jet(events_dataplt['Ec']/np.max(events_dataplt['Ec'])))'''
        
        #hough transform convolved plots                                                                                                                                       
        '''fig = plt.figure(figsize=(9,9))
        ax = fig.add_subplot(111)
        ax.imshow(np.log(1+conv_accuXY), cmap='jet',extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), diag_lenxy, -diag_lenxy])
        ax.plot([-90,90],[rhoXY,rhoXY],color='white')
        ax.plot([np.rad2deg(thetaXY),np.rad2deg(thetaXY)],[-200,200],color='white')
        ax.scatter(np.rad2deg(thetxy),roxy,color='white')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Hough trans_conv XY')
        ax.set_xlabel('theta (degrees)')
        ax.set_ylabel('Distance')
        plt.ylim(200,-200)
        #fig.savefig(savelocat+'hough_convolved'+run_num+'_'+str(evts)+'XY.png')
                                                                                                                                       \

        fig = plt.figure(figsize=(9,9))
        ax = fig.add_subplot(111)
        ax.imshow(np.log(1+conv_accuZY), cmap='jet',extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), diag_lenzy, -diag_lenzy])
        ax.plot([-90,90],[rhoZY,rhoZY],color='white')
        ax.scatter(np.rad2deg(thetzy),rozy,color='white')
        ax.plot([np.rad2deg(thetaZY),np.rad2deg(thetaZY)],[-width/2,width/2],color='white')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Hough trans_conv ZY')
        ax.set_xlabel('theta (degrees)')
        #plt.xlim(-5,5)
        plt.ylim(200,-200)
        ax.set_ylabel('Distance')
        #fig.savefig(savelocat+'hough_convolved'+run_num+'_'+str(evts)+'ZY.png')

        fig = plt.figure(figsize=(9,9))
        ax = fig.add_subplot(111)
        ax.imshow(np.log(1+conv_accuXZ), cmap='jet',extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), diag_lenzy, -diag_lenzy])
        ax.plot([-90,90],[rhoXZ,rhoXZ],color='white')
        ax.scatter(np.rad2deg(thetxz),roxz,color='white')
        ax.plot([np.rad2deg(thetaXZ),np.rad2deg(thetaXZ)],[-width/2,width/2],color='white')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Hough trans_conv XZ')
        ax.set_xlabel('theta (degrees)')
        #plt.xlim(-5,5)
        plt.ylim(200,-200)
        ax.set_ylabel('Distance')
        #fig.savefig(savelocat+'hough_convolved'+run_num+'_'+str(evts)+'ZY.png')'''
        
        #3d plot                                                                                                                                                             
        '''xdataplot=events_dataplt.X
        ydataplot=events_dataplt.Z
        zdataplot=events_dataplt.Y

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.plot([mixi.initial_x.values[0],mixi.final_x.values[0]],[mixi.initial_z.values[0],mixi.final_z.values[0]],[mixi.initial_y.values[0],mixi.final_y.values[0]],linewidth=3, color='limegreen')
        ax.plot(bestx,bestz,besty,color='magenta',linewidth=3,label='best')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_xlim(-215, 215)
        ax.set_zlim(-215, 215)
        ax.set_ylim(0, 500)
        colmap = cm.ScalarMappable(cmap=cm.jet)
        ax.scatter(xdataplot,ydataplot,zdataplot,c=cm.jet(events_dataplt['Ec']/np.max(events_dataplt['Ec'])))
        #cb = fig.colorbar(colmap)                                                                                                                                           
        #fig.savefig(savelocat+'3Dplot'+run_num+'_'+str(evts)+'.png')'''

    #save for each run of events, things tend to run out of memory and crash
    data_out4 = savelocat+'/run'+run_num+'alleventscutinfo.h5'
    pd.DataFrame({'dEdx':dedxall,
                  'perconline':pcline,
                  'linelength':linlen,
                  'muenergy':munrg,
                  'eventnum':muvts,
                  'beta':mu_theta,
                  'alpha':mu_phi,
                  'xintercept':xint,
                  'zintercept':zint,
                  'MCtheta':truth_theta,
                  'MCphi':truth_phi,
                  'MCperconlin':truth_per,
                  'MCdEdx':truth_dedx,
                  'MClength':truth_len,
                  'costheta':costhetasall}).to_hdf(data_out4,'dEdx')
        
    
    


