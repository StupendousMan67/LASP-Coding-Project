import numpy as np
from astropy.io import ascii    
from astropy.table import Table
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# define location of input files...
indir = '../laspCodingTestDataFiles/'

def countRate( counts, integrationTimeMs, darkCorrection=False,
               detectorTemp=None): 
    '''
    Given observed counts (counts) and integrationTimeMs (ms), return
    count rate in photons/sec/cm^2/nm.  If optional flag
    darkCorrection is set, then apply a correction for the dark
    current based on the tabulated detector temperature, an array
    which must be matched in time to counts and integratimeTimeMs.
    '''
    
    integrationTime = 1000. * integrationTimeMs # convert to seconds from ms
    cr = counts / integrationTime # [counts / sec / nm] (the per nm is
    # from the sampling at the specific grating position)

    # optional correction for dark current...
    if darkCorrection:
        cr = applyDarkCorrection( cr, detectorTemp)
    
    apArea = .01 # [cm^2] (aperature area)
    photonsPerSecondPerCm2 = cr / apArea # [photons/sec/cm^2/nm] 
    return photonsPerSecondPerCm2

def wavelength( gratingPosition):
    '''
    Given gratingPosition, use the grating equation to compute and
    return wavelength [nm]
    '''
    
    offset = 239532.38
    stepSize = 2.4237772022101214E-6 # rad
    d = 277.77777777777777 # nm
    phiGInRads = 0.08503244115716374 # rad
    ang1 = (offset - gratingPosition) * stepSize

    wavelength = 2 * d * np.sin(ang1) * np.cos(phiGInRads / 2.0) # nm
    
    return wavelength

def arrayMatch( values, times, newtimes):
    '''
    Given an array of values (values) with values the change at
    specific times (times), plus an array of new time values
    (newtimes) when we desire to now the value of "values" return the
    array (newvalues) representing the values at times "btime".  This
    assumes that the values of a are constant until they change.
    '''
    
    # allocate space to store results... 
    numNew = len(newtimes)
    newvalues = np.zeros(numNew)
    
    # loop over new times...
    for i in range(numNew):
    
        nt = newtimes[i]
        
        # locate the previous time...
        buf = np.where( times <= nt)
        good = buf[0]
        try:
            goodtimes = times[good]
        except:
            print 'times=', times
            print 'good=', good
            raise ValueError("no good times")
        
        # locate the index of the max time...
        imax = np.argmax( goodtimes)
        j = good[imax]
        
        # store the value...
        newvalues[i] = values[j]
        
    return newvalues

def irradiance( countRate, wavelength):
    '''
    Given count rate (photons/sec/cm^2/nm) and wavelength (nm),
    return irradiance (watts/m^2/nm)
    '''
    wavelengthInMeters = wavelength / 1.e9 # convert to meters from nm
    h = 6.62606957E-34 # [J*s]
    c = 299792458.0 # [m/s]
    energyPerPhoton = h * c / wavelengthInMeters # [J]
    wattsPerM2 = countRate * 1e2 * 1e2 * energyPerPhoton # [watts/m^2/nm]
    fudgeFactor = 1.e6
    wattsPerM2 = wattsPerM2 * fudgeFactor
    return wattsPerM2

def cleanTelemetry(d):
    '''
    Given a dataset object, remove from the telemetry table any rows
    for which the grating position (gratPos) is zero.  The
    dataset object is edited in place.
    '''

    # create shortcut to gratPos...
    g = d['instrumentTelemetry']['gratPos']

    # locate good data points...
    buf = np.where( g > 0.)
    good = buf[0]
    
    if len(good) == 0:
        raise ValueError( 'no good data points')

    # retain good rows only...
    d['instrumentTelemetry'] = d['instrumentTelemetry'][good]

def resampleTable( d, tableName1, timeCol1, valueCol1,
                  tableName2, timeCol2):
    '''
    Given a dataset object (d) with a table (tableName1)
    having a named time column named (timeCol1) 
    and a corresponding value column (valueCol1), 
    and a second table (tableName2), 
    having a named time column named (timeCol2) 
    and a corresponding value column (valueCol2),
    replace table1 with a new version that is synced to table2.
    '''
    
    v = arrayMatch( d[tableName1][valueCol1],
                    d[tableName1][timeCol1],
                    d[tableName2][timeCol2])

    # construct a new table with the resampled data...
    newTable = Table()
    newTable[timeCol1]  = d[tableName2][timeCol2]
    newTable[valueCol1] = v
    
    # replace original contents of this table 
    # with the resampled version...
    d[tableName1] = newTable

def readDatasetFull():
    '''
    Return a dict object with the following tables:
    - detectorTemp
    - distanceAndDoppler
    - instrumentTelemetry
    - integrationTime
    '''

    dataset = dict()
    
    # loop over data types...
    types = ['detectorTemp',
             'distanceAndDoppler',
             'instrumentTelemetry',
             'integrationTime']

    for t in types:
        inFile = indir + t + '.txt'
        dataset[t] = ascii.read( inFile)

    return dataset

def sliceDatasetByTime( d, t1, t2):
    '''
    Given a dataset object (d) plus starting and ending times 
    (t1 and t2 in ms), return a dataset object
    with data tables corresponding to that time period:
    - detectorTemp
    - distanceAndDoppler
    - instrumentTelemetry
    - integrationTime
    '''
    
    # create a new dict to hold output...
    dataset = dict()

    # loop over data types...
    types = d.keys()
    for t in types:
    
        # integration time is a special case, because values are sparse.
        # construct an array of integration times for each time we
        # obtained counts...
        if t == 'integrationTime':
            resampleTable( d, t, 'microsecondsSinceGpsEpoch', 'intTime (ms)', 
                  'instrumentTelemetry', 'microsecondsSinceGpsEpoch')            

        # identify good rows...
        buf = np.where( (d[t]['microsecondsSinceGpsEpoch'] >= t1) &
                        (d[t]['microsecondsSinceGpsEpoch'] <= t2)) 
        good = buf[0]
        
        # save the good rows to a table in the output dataset...
        dataset[t] = d[t][good]

    return dataset

def readDatasetByTime( t1, t2):
    '''
    Given starting and ending times (t1 and t2 in ms), return a dict object
    with data tables corresponding to that time period:
    - detectorTemp
    - distanceAndDoppler
    - instrumentTelemetry
    - integrationTime
    '''

    # read full dataset...
    d = readDatasetFull()
    return sliceDatasetByTime( d, t1, t2)

def readDatasetByName( datasetName):
    '''
    Given a dataset name (datasetName), read and return data.  Valid
    names are: QuickScan, ConstantWavelength, DownScan, Dark, UpScan
    '''

    # initialize d...
    d = None
    
    # read plan file...
    inFile = indir + 'plans.txt'
    data = ascii.read( inFile)

    # match on planName...
    for row in data:

        # extract elements from row...
        planName = row['planName']
        t1 = row['startTime']
        t2 = row['endTime']
    
        if datasetName == planName:
            d = readDatasetByTime( t1, t2)
            break

    return d

def applyDarkCorrection( countRate, detectorTemp):
    '''
    Given arrays of signal (countRate, units=counts/sec) and detector
    temperature (detectorTemp, unit=deg C), compute and subtract the
    estimated dark current and return corrected signal.
    '''

    # sanity check: arrays must be same length...
    if len(countRate) != len(detectorTemp):
        msg = "countRate and detectorTemp array must be same length (%d vs. %d)" % (len(countRate), len(detectorTemp))
        raise ValueError(msg)

    # construct polynomial fit based on coefficients we derived from
    # Dark data set... 
    z = np.array([1.34849318e-02, -9.67799807e-01, 2.35761388e+01,
                  -1.90484234e+02]) 
    p = np.poly1d(z)

    # use the polynomial fit to derive dark current for given temp...
    darkCurrent = p(detectorTemp)

    # apply dark correction...
    correctedCountRate = countRate - darkCurrent
    return correctedCountRate

def applyDopplerCorrection( observedWavelength, wavelengthTimeStamp,
                            rvcor, rvcorTimeStamp): 
    '''
    Given an array of observed wavelengths (observedWavelength) 
    at time wavelengthTimeStamp, 
    plus an array of radial velocity correction factors (rvcor) 
    tabulated at time (rvcorTimeStamp),
    update the wavelength scale to correct for radial velocity.
    '''

    # determine radial velocity corrections for each point in the spectrum...
    rvcorInterp = np.interp( wavelengthTimeStamp, rvcorTimeStamp, rvcor)
    
    # apply the correction to determine the wavelengths in the rest frame...
    restWavelength = observedWavelength * rvcorInterp
    
    return restWavelength

def applyDistanceCorrection( counts,
                             countTimeStamp,
                             distanceCorrection,
                             distanceCorrectionTimeStamp): 
    '''
    Given an array of observed counts (counts) at correponding times
    countTimeStamp, plus an array of distance correction factors
    (distanceCorrection) tabulated at time
    (distanceCorrectionTimeStamp), update the counts to correct for
    distance to source.
    '''
    
    # determine distance corrections appropriate for each measurement...
    distanceCorrectionInterp = np.interp( countTimeStamp,
                                          distanceCorrectionTimeStamp,
                                          distanceCorrection)
    
    # apply the correction to the counts...
    correctedCounts = counts * distanceCorrectionInterp
    
    return correctedCounts

def addSpectrum( d, sort=False,
                 darkCorrection=False,
                 dopplerCorrection=False,
                 distanceCorrection=False,
                 verbose=False):
    '''
    Given structure d, add a new 'spectrum' table to the dataset object
    with these columns:
    - wavelength(nm):
    - countrate
    - irradiance (watts/m^2)

    Optional flags:
    - If "sort" flag is set, return spectrum sorted on wavelength.  
    - If "darkCorrection" flag is set, apply correction to countRate for dark
    current.
    - If "dopplerCorrection" flag is set, apply radial velocity correction to
                 wavelength.
    - If "distanceCorrection" flag is set, apply distance correction to
                 countrate.
    '''
                         
    # transfer variables to local arrays for ease of use...
    g = d['instrumentTelemetry']['gratPos']
    counts = d['instrumentTelemetry']['counts']
    countTimeStamp = d['instrumentTelemetry']['microsecondsSinceGpsEpoch']
    
    # generate wavelengths from grating position...
    w = wavelength( g)

    # optionally apply Doppler correction...
    if dopplerCorrection:
        observedWavelength  = w
        rvcor               = d['distanceAndDoppler']['sunObserverDopplerFactor']
        rvcorTimeStamp      = d['distanceAndDoppler']['microsecondsSinceGpsEpoch']
        restWavelength = applyDopplerCorrection( observedWavelength,
                                                 countTimeStamp,
                                                 rvcor,
                                                 rvcorTimeStamp)
        w = restWavelength
        
    # create shortcuts to arrays...
    iTime = d['integrationTime']['intTime (ms)']
    iTimeTimeStamp = d['integrationTime']['microsecondsSinceGpsEpoch']
    
    # construct an array of integration times for each time we
    # obtained counts (although this is now done in readDatasetByTime, we do it again here
    # in case any points were dropped by cleaning the telemetry of points with gratPos==0)...
    iTimeInterp = arrayMatch( iTime, iTimeTimeStamp, countTimeStamp)
    
    # confirm that the timestamps match...
    if len(counts) != len(iTimeInterp):
        print len(counts), len(iTimeInterp)
        raise ValueError('counts and iTimeInterp arrays are unequal in length')

    # if applying dark correction, construct an interpolated version
    # of the detectorTemp array...
    if darkCorrection:
        detectorTemp = d['detectorTemp']['temp (C)']
        detectorTempTimeStamp = d['detectorTemp']['microsecondsSinceGpsEpoch']
        detectorTempInterp = np.interp( countTimeStamp,
                                        detectorTempTimeStamp,
                                        detectorTemp)
    else:
        detectorTempInterp = None

    # get count rate...
    cr = countRate( counts, iTimeInterp,
                    darkCorrection=darkCorrection,
                    detectorTemp=detectorTempInterp)

    # optionally apply distance correction to countrate...
    if distanceCorrection:
        distanceCorrection = d['distanceAndDoppler']['sunObserverDistanceCorrection']
        distanceCorrectionTimeStamp = d['distanceAndDoppler']['microsecondsSinceGpsEpoch']
        cr = applyDistanceCorrection( cr,
                                      countTimeStamp,
                                      distanceCorrection,
                                      distanceCorrectionTimeStamp)
        
    # convert counts to irradiance...
    irr = irradiance( cr, w)
    
    if verbose:
        print "g_min=", np.min(g), "g_max=", np.max(g)
        print "wavelength_min=", np.min(w), "Wavelength_max=", np.max(w)

    # optional wavelength sort...
    if sort:
        order = np.argsort(w)
    else:
        order = range(len(w))
        
    # store results in new table...
    t = Table()
    t['wavelength(nm)'] = w[order]
    t['countrate'] = cr[order]
    t['irradiance (watts/m^2)'] = irr[order]
    t['microsecondsSinceGpsEpoch'] = countTimeStamp[order]
    
    # append spectrum table to structure...
    d['spectrum'] = t

def gaussian( x, x0, a, fwhm, background):
    '''
    Given array of ordinal values (x), center (x0), amplitude (a), 
    and width (sigma), and constant background (b), 
    return array of y values representing the Gaussian curve.
    '''
    sigma = fwhm/2.355
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + background

def getCenter( x, y, background=None, xc=None, amplitude=None, fwhm=None, plot=False, title=None):
    '''
    Given a curve of data (x,y), fit a Gaussian and return best-fit parameters (center and amplitude)
    '''
    
    # if background is not defined, use the minimum value...
    if background is None:
        background = np.min(y)
    
    # if center (xc) is not defined, use peak position...
    if xc is None:
        
        # i is the index of the max value in y...
        i = np.argmax(y)
        
        # xc is the x location of the peak...
        xc = x[i]
        
    # if peak value above background is not defined, use max value...
    if amplitude is None:
        amplitude = np.max(y) - background
        
    # use a default value for FWHM...
    if fwhm is None:
        fwhm = 1
    
    # optimize the fit...
    popt,pcov = curve_fit( gaussian, x, y, p0=[xc, amplitude, fwhm, background])
    
    # unpack best-fit parameters...
    xc_best, amplitude_best, fwhm_best, background_best = popt
    
    # optional plot...
    if plot:
        
        # generate initial curve...
        xMin = np.min(x)
        xMax = np.max(x)
        x_initial = np.linspace(xMin, xMax)
        y_initial = gaussian( x_initial, xc, amplitude, fwhm, background)
        
        # save best-fit...
        x_final = x_initial
        y_final = gaussian( x_final, xc_best, amplitude_best, fwhm_best, background_best)
    
        # generate plot...
        plt.figure(figsize=(10,5))
        plt.plot( x, y, 'ro',
                 x_initial, y_initial, 'y--',
                 x_final, y_final, 'b-')
        plt.ylabel( "x")
        plt.xlabel( "y")
        plt.axvline(x=xc_best, color='r')
        if title is not None:
            plt.title( title)
        plt.show
        
    return xc_best, amplitude_best

def getRegionOfInterest( x, y, xMin, xMax):
    '''
    Given spectrum (x,y) and wavelength range (xMax,yMax),
    return the portions of the spectrum lying within the 
    wavelength range.
    '''
    
    buf = np.where((x >= xMin) & (x <= xMax))
    good = buf[0]
    return x[good], y[good]

def getWavelengthShift( x1, y1, x2, y2, xMin, xMax, plot=True, verbose=True):
    '''
    Given spectrum 1 (x1,y1), spectrum 2 (x2,y2), and a wavelength range 
    of interest (xMin,xMax), fit Guassians to the spectra and determine 
    shift required to align spectrum 2 with spectrum 1.
    '''
    
    # fit curve to spectrum 1...
    x1_good, y1_good = getRegionOfInterest( x1, y1, xMin, xMax)
    xc1,amp1 = getCenter( x1_good, y1_good, fwhm=0.2, plot=plot, title='Spectrum 1')
    
    # fit curve to spectrum 2...
    x2_good, y2_good = getRegionOfInterest( x2, y2, xMin, xMax)
    xc2,amp2 = getCenter( x2_good, y2_good, fwhm=0.2, plot=plot, title='Spectrum 2')
    
    dx = xc1-xc2

    if verbose:
        print 'xc1=', xc1
        print 'xc2=', xc2
        print 'dx=', dx

    return dx  
