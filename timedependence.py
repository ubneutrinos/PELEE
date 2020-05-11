import sys, os
import numpy as np
import pandas as pd
import math

class TimePlotter:

    # potON  -> csv file containing run -> POT info for on-beam
    # potOFF -> csv file containing run -> Trigger info for off-beam
    def __init__(self,potON,potOF,dfON,dfOF):

        self._potON = pd.read_csv(potON,delimiter=' ')
        self._potOF = pd.read_csv(potOF,delimiter=' ')

        self._potON = self._potON.sort_values(by=['run'])
        self._potOF = self._potOF.sort_values(by=['run'])

        print ('POT for run 4953 is ',((self._potON).query('run==4953'))['POT'].values)
        print ('E1D for run 4953 is ',((self._potON).query('run==4953'))['E1D'].values)

        self._runMIN = self._potON['run'].min()
        self._runMAX = self._potON['run'].max()

        if (self._potOF['run'].min() < self._runMIN):
            self._runMIN = self._potOF['run'].min()

        if (self._potOF['run'].max() > self._runMAX):
            self._runMAX = self._potOF['run'].max()
        
        self._dfON = dfON
        self._dfOF = dfOF

        #print (self._potON.columns.values, " with %i entries"%(self._potON.shape[0]))
        #print (self._potOF.columns.values)

    def QueryByRun(self,NRUN,QUERY,RMIN=None,RMAX=None,ONBEAMONLY=False):

        # get subset of data passing the query
        dfON = (self._dfON).query(QUERY)
        dfOF = (self._dfOF).query(QUERY)

        rMIN = self._runMIN
        rMAX = self._runMAX
        if (RMIN != None):
            rMIN = RMIN
        if (RMAX != None):
            rMAX = RMAX
        
        RUNbins = np.arange(rMIN,rMAX,NRUN)

        # events (ON) / 1e18 POT
        selON_v = []
        # events (OF) / 1e18 POT
        selOF_v = []
        # events (ON) / 1e18 POT (error)
        selON_e = []
        # events (OF) / 1e18 POT (error)
        selOF_e = []
        # on-off scaled
        selNU_v = []
        selNU_e = []
        # POT stored
        POT_v = []
        # triggers (ON)
        E1D_v = []
        # triggers (OF)
        EXT_v = []
        # run number
        RUN_v = []
        
        # for each run-interval, compute passing events, POT, and return binned results
        for n in range(len(RUNbins)-1):

            rMIN = RUNbins[n]
            rMAX = RUNbins[n+1]

            #print ('run range [%i, %i]'%(rMIN,rMAX))

            POT = np.sum( ((self._potON).query('run >= %i and run < %i'%(rMIN,rMAX)))['POT'].values ) / (1e18)
            E1D = np.sum( ((self._potON).query('run >= %i and run < %i'%(rMIN,rMAX)))['E1D'].values )
            EXT = np.sum( ((self._potOF).query('run >= %i and run < %i'%(rMIN,rMAX)))['EXT'].values )

            #print ('POT is ',POT)
            #print ('E1D is ',E1D)
            #print ('EXT is ',EXT)
            
            scaling = 0.
            if (EXT != 0):
                scaling = float(E1D)/float(EXT)


            nON = float((dfON.query('run >= %i and run < %i'%(rMIN,rMAX))).shape[0])
            nOF = float((dfOF.query('run >= %i and run < %i'%(rMIN,rMAX))).shape[0])

            #print ('selectd %.0f on-beam and %.0f off-beam events'%(nON,nOF))
            #print ('with POT of %g and scaling of %.03f'%(POT,scaling))

            errON = math.sqrt(nON)
            errOF = math.sqrt(nOF)

            # number of (on-off) after scaling
            NU = (nON - nOF * scaling)
            if ( (nON == 0) and (nOF == 0) ):
                err = NU
            elif (nON == 0):
                err = np.sqrt( (1./nOF)**2 ) * NU
            elif (nOF == 0):
                err = np.sqrt( (1./nON)**2 ) * NU
            else:
                a = 1./errON
                b = 1./errOF
                #print ('a : %.03f b : %.03f'%(a,b))
                err = np.sqrt( a**2 + b**2 )# * NU
                #print ('-> err frac : %.02f'%err)
                err *= NU

            if (nON == 0):
                continue
                
            if not (np.isfinite(nON)):
                continue
            if ( (np.isfinite(POT) == False) or (POT == 0) ):
                continue
                
            RUN_v.append(int(float((rMIN+rMAX)/2.)))
            POT_v.append(POT)
            E1D_v.append(E1D)
            EXT_v.append(EXT)

            selON_v.append(nON)
            selOF_v.append(nOF)

            selON_e.append(errON)
            selOF_e.append(errOF)
            
            selNU_v.append(NU)
            selNU_e.append(err)

            #print ('nON : %i OF : %i NU : %.02f err : %.02f'%(nON,nOF,NU,err))

            #break

        POT_v = np.array(POT_v)
        E1D_v = np.array(E1D_v)
        EXT_v = np.array(EXT_v)
        RUN_v = np.array(RUN_v)
        selON_v = np.array(selON_v)
        selOF_v = np.array(selOF_v)
        selNU_v = np.array(selNU_v)
        selNU_e = np.array(selNU_e)

        '''
        POT_v = POT_v[np.isfinite(POT_v)]
        E1D_v = E1D_v[np.isfinite(E1D_v)]
        EXT_v = EXT_v[np.isfinite(EXT_v)]
        RUN_v = RUN_v[np.isfinite(RUN_v)]
        selON_v = selON_v[np.isfinite(selON_v)]
        selOF_v = selOF_v[np.isfinite(selOF_v)]
        selON_e = selON_e[np.isfinite(selON_e)]
        selOF_e = selOF_e[np.isfinite(selOF_e)]
        selNU_v = selNU_v[np.isfinite(selNU_v)]
        selNU_e = selNU_e[np.isfinite(selNU_e)]
        '''

        if (ONBEAMONLY == True):
            return RUN_v,selON_v/POT_v, selON_e/POT_v
        
        return RUN_v, selNU_v/POT_v, selNU_e/POT_v
