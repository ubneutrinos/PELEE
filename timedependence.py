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
        
        print ('Run range : [%i, %i]'%(rMIN,rMAX))
        print ('POT is ',POT)
        print ('E1D is ',E1D)
        print ('EXT is ',EXT)
        
        scaling = 0.
        if (EXT != 0):
            scaling = float(E1D)/float(EXT)
            
            
        nON = float((dfON.query('run >= %i and run < %i'%(rMIN,rMAX))).shape[0])
        nOF = float((dfOF.query('run >= %i and run < %i'%(rMIN,rMAX))).shape[0])

        #if (nON == 0):
        #    return None
        
        #print ('selectd %.0f on-beam and %.0f off-beam events'%(nON,nOF))
        #print ('with POT of %g and scaling of %.03f'%(POT,scaling))
        
        errON = 1.
        if (nON > 0):
            errON = math.sqrt(nON)
        errOF = 1.
        if (nOF > 0):
            errOF = math.sqrt(nOF)

        errONfrac = 0.
        if (errON > 0):
            errONfrac = 1./errON
        errOFrelfrac = 0.
        if (errOF > 0):
            errOFrelfrac = scaling * 1./(errOF)

        print ('run %i has a total of %i nON and %i nOF events. Scaling is %.02f and POT %g'%(rMIN,nON,nOF,scaling,POT))
        
        # number of (on-off) after scaling
        NU = (nON - nOF * scaling)
        err = 0.
        print ('nON-nOF : %.02f'%(NU))
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
            err = np.sqrt( (errONfrac)**2 + (errOFrelfrac)**2 )
            #print ('-> err frac : %.02f'%err)
            err *= NU
        #print ('with error : %.02f. [frac : %.02f]'%(err,err/NU))
        #print ("\n")

        #if (nON == 0):
        #    return None
        
        #if not (np.isfinite(nON)):
        #    return None
        #if ( (np.isfinite(POT) == False) or (POT == 0) ):
        #    return None

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

        #print ('aaaa')
        
        return infodict

    def GetRunBins(self,NRUN_V,RMIN=None,RMAX=None):

        rMIN = self._runMIN
        rMAX = self._runMAX
        if (RMIN != None):
            rMIN = RMIN
        if (RMAX != None):
            rMAX = RMAX

        if (len(NRUN_V) == 1):
            RUNbins = np.arange(rMIN,rMAX,NRUN_V[0])
        else:
            RUNbins = NRUN_V

        return RUNbins
        
    # QUERYREL -> the query relative to QUERY for which the fractional passing rate is to be measured (i.e. # of events passing QUERYREL / # of events passing QUERY)
    def QueryRelByRun(self,NRUN_V,QUERY,QUERYREL,RMIN=None,RMAX=None,ONBEAMONLY=False,POTmin=0.0):

        # get subset of data passing the query
        dfON = (self._dfON).query(QUERY)
        dfOF = (self._dfOF).query(QUERY)

        RUNbins = self.GetRunBins(NRUN_V,RMIN,RMAX)

        fON_v = []
        fON_e = []
        fOF_v = []
        fOF_e = []
        fNU_v = []
        fNU_e = []
        RUN_v = []

        n = 0
        
        for n in range(len(RUNbins)-1):

            rMIN = RUNbins[n]
            rMAX = RUNbins[n+1]

            POT,E1D,EXT = self.GetPOT(rMIN,rMAX)

            scaling = 0.
            if (EXT != 0):
                scaling = float(E1D)/float(EXT)

            nON_d = float((dfON.query('run >= %i and run < %i'%(rMIN,rMAX))).shape[0])
            nOF_d = float((dfOF.query('run >= %i and run < %i'%(rMIN,rMAX))).shape[0])

            nON_n = float((dfON.query('run >= %i and run < %i and %s'%(rMIN,rMAX,QUERYREL))).shape[0])
            nOF_n = float((dfOF.query('run >= %i and run < %i and %s'%(rMIN,rMAX,QUERYREL))).shape[0])            

            nNU_d = (nON_d - nOF_d * scaling)
            nNU_n = (nON_n - nOF_n * scaling)

            fON  = -1
            fONe = 0.
            fOF  = -1
            fOFe = 0.
            fNU  = -1
            fNUe = 0.

            if (nON_d > 0):
                fON = nON_n / nON_d
                fONe = np.sqrt( fON * (1-fON) / nON_d )
            if (nOF_d > 0):
                fOF = nOF_n / nOF_d
                fOFe = np.sqrt( fOF * (1-fOF) / nOF_d )

            if (nNU_d > 0):
                fNU = nNU_n / nNU_d
                #eNU_d = np.sqrt( (1./nON_d)**2 + (1./ (nOF_d * scaling))**2 ) # fractional error on denominator of neutrino events
                fNUe = np.sqrt( fON * (1-fON) / nON_d )
            
            RUN_v.append( (rMIN+rMAX)/2. )
            fON_v.append(fON)
            fOF_v.append(fOF)
            fNU_v.append(fNU)
            fON_e.append(fONe)
            fOF_e.append(fOFe)
            fNU_e.append(fNUe)

        RUN_v = np.array(RUN_v)
        fON_v = np.array(fON_v)
        fOF_v = np.array(fOF_v)
        fNU_v = np.array(fNU_v)
        fON_e = np.array(fON_e)
        fOF_e = np.array(fOF_e)
        fNU_e = np.array(fNU_e)
            
        return RUN_v, fON_v, fOF_v, fNU_v, fON_e, fOF_e, fNU_e

        
    def QueryByRun(self,NRUN_V,QUERY,RMIN=None,RMAX=None,ONBEAMONLY=False,POTmin=0.0):

        # get subset of data passing the query
        dfON = (self._dfON).query(QUERY)
        dfOF = (self._dfOF).query(QUERY)

        RUNbins = self.GetRunBins(NRUN_V,RMIN,RMAX)

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

        n = -1
        
        # for each run-interval, compute passing events, POT, and return binned results
        while ( (n+2) < len(RUNbins) ):

        #for n in range(len(RUNbins)-1):

            n += 1

            rMIN = RUNbins[n]
            rMAX = RUNbins[n+1]

            POT = self.GetPOT(rMIN,rMAX)[0]
            ctr = 0

            #nabs = n
            
            while ( (POTmin > 0) and (POT < POTmin) and (ctr < 5) and ( (n+2) < len(RUNbins))):
                ctr += 1
                n += 1
                #nabs += 1
                rMAX = RUNbins[n+1]
                POT = self.GetPOT(rMIN,rMAX)[0]

            #n = nabs

            if (POT < POTmin):
                continue
            
            eventinfo = self.GetEvents(rMIN,rMAX,dfON,dfOF)#,QUERYREL=None)

            if (eventinfo == None):
                continue

            #print('filling info for run range [%i, %i] at %i try. n is now %i'%(rMIN,rMAX,ctr,n))

            RUN_v.append(eventinfo['run'])
            POT_v.append(eventinfo['pot'])
            E1D_v.append(eventinfo['e1d'])
            if (eventinfo['ext'] == 0):
                EXT_v.append(1e10)
            else:
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

        #if (ONBEAMONLY == True):
        #    return RUN_v,selON_v/POT_v, selON_e/POT_v
        
        return RUN_v, selNU_v/POT_v, selNU_e/POT_v, selON_v/POT_v, selON_e/POT_v, selOF_v/(EXT_v/1e5), selOF_e/(EXT_v/1e5), selON_v
