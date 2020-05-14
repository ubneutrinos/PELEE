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

    def GetPOT(self,rMIN,rMAX):

        POT = np.sum( ((self._potON).query('run >= %i and run < %i'%(rMIN,rMAX)))['POT'].values ) / (1e18)
        E1D = np.sum( ((self._potON).query('run >= %i and run < %i'%(rMIN,rMAX)))['E1D'].values )
        EXT = np.sum( ((self._potOF).query('run >= %i and run < %i'%(rMIN,rMAX)))['EXT'].values )

        return POT,E1D,EXT
        
        
    def GetEvents(self, rMIN, rMAX, dfON, dfOF):

        #print ('run range [%i, %i]'%(rMIN,rMAX))
        POT,E1D,EXT = self.GetPOT(rMIN,rMAX)
        
        #print ('Run range : [%i, %i]'%(rMIN,rMAX))
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
            return None
        
        if not (np.isfinite(nON)):
            return None
        if ( (np.isfinite(POT) == False) or (POT == 0) ):
            return None

        infodict = {
            'run': int(float((rMIN+rMAX)/2.)),
            'pot': POT,
            'e1d': E1D,
            'ext': EXT,
            'nON': nON,
            'nOF': nOF,
            'eON': errON,
            'eOF': errOF,
            'nu' : NU,
            'err': err
        }

        print ('aaaa')
        
        return infodict
        

        
    def QueryByRun(self,NRUN_V,QUERY,RMIN=None,RMAX=None,ONBEAMONLY=False,POTmin=0.0):

        # get subset of data passing the query
        dfON = (self._dfON).query(QUERY)
        dfOF = (self._dfOF).query(QUERY)

        rMIN = self._runMIN
        rMAX = self._runMAX
        if (RMIN != None):
            rMIN = RMIN
        if (RMAX != None):
            rMAX = RMAX

        #print (NRUN_V)
            
        if (len(NRUN_V) == 1):
            RUNbins = np.arange(rMIN,rMAX,NRUN_V[0])
        else:
            RUNbins = NRUN_V

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

            POT = self.GetPOT(rMIN,rMAX)[0]
            ctr = 0
            
            while ( (POTmin > 0) and (POT < POTmin) and (ctr < 5) and ( (n+2) < len(RUNbins))):
                print ('POT is %g with ctr at %i'%(POT,ctr))
                ctr += 1
                n += 1
                #print ('n is now : ',n)
                rMAX = RUNbins[n+1]
                POT = self.GetPOT(rMIN,rMAX)[0]
                
            
            eventinfo = self.GetEvents(rMIN,rMAX,dfON,dfOF)

            if (eventinfo == None):
                continue

            RUN_v.append(eventinfo['run'])
            POT_v.append(eventinfo['pot'])
            E1D_v.append(eventinfo['e1d'])
            EXT_v.append(eventinfo['ext'])

            selON_v.append(eventinfo['nON'])
            selOF_v.append(eventinfo['nOF'])

            selON_e.append(eventinfo['eON'])
            selOF_e.append(eventinfo['eOF'])
            
            selNU_v.append(eventinfo['nu'])
            selNU_e.append(eventinfo['err'])

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
