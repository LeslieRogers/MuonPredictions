import numpy as np
import pandas as pd
try:
    import cPickle as pickle
except ImportError:
    import pickle
import scipy
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline

NumToRun=1000

#musDatamtn=pd.read_hdf('/n/holystore01/LABS/guenette_lab/Users/lrogers/Proposal/CombinedProposalOutputs.h5','Muons')
musDatamtn=pd.read_hdf('/n/holystore01/LABS/guenette_lab/Users/lrogers/Proposal/CombinedProposalInterpOutputs.h5','Muons')
#musDatamtn['perc']=musDatamtn.apply(lambda row: len(row.FinalMuons)/NumToRun, axis=1)
#musDatamtn=musDatamtn[musDatamtn.X!=1295]

#save interpolatations for proposal output for all X,Y for each energy

#musDatamtn=musDatamtn[(musDatamtn.X>-600)&(musDatamtn.X<600)]
#musDatamtn=musDatamtn[(musDatamtn.Y>-600)&(musDatamtn.Y<600)]
#musDatamtn=musDatamtn[np.cos(np.deg2rad(musDatamtn.Theta))>=.4]


energies=musDatamtn.Energy.unique()
#print(energies)
energies.sort()
#print(len(energies))

#energies=[5e5,6e5,7e5,8e5,9e5,10e5]
#energies=[100,300,400,450,500,550,600,650,700,650,800,900,1000,2000,4000,6000] #in MeV  

for LOC in range(0,len(energies)):
    r=0


    NRG=energies[LOC]
    #print(NRG)
    MUSinterp=musDatamtn[musDatamtn.Energy==NRG]
    



    kX=MUSinterp.X.unique()
    kX.sort()
    lY=MUSinterp.Y.unique()
    lY.sort()


    Xs=[]
    Ys=[]
    Percs=[]
    for X in kX:
        for Y in lY:
            percsum=0
            multi=0
            

            if len(MUSinterp[(MUSinterp.X==X)&(MUSinterp.Y==Y)])==0:
                Xs.append(X)
                print("missing",X,Y)
                Percs.append(0) #need to rerun anything missing
                Ys.append(Y)
                continue
            perc=MUSinterp[(MUSinterp.X==X)&(MUSinterp.Y==Y)].SurvivalPercent.values[0]
            
            Xs.append(X)
            Percs.append(perc)
            Ys.append(Y)
    #print(X,Y,perc)

    df=pd.DataFrame(Xs, columns = ['X'])
    df['Y']=Ys
    df['perc']=Percs

    percsarray=np.array(df.perc)
    percsarray = percsarray.reshape(len(df.X.unique()), len(df.Y.unique()))
    
    #print (np.unique(df.X),np.unique(df.Y),percsarray)
    PercentMuons=scipy.interpolate.RectBivariateSpline(np.unique(df.X),np.unique(df.Y),percsarray,s=0,kx=3, ky=3)
    

    with open('/n/holystore01/LABS/guenette_lab/Users/lrogers/Proposal/Proposal_Muons_interpolator'+str(NRG*10**-3)+'GeV.pkl', 'wb') as f:
        pickle.dump(PercentMuons, f)
