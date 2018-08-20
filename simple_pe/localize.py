from numpy import *
import copy
import detectors
import fstat
import likelihood
import lal
import numpy as np
import random as rnd
from scipy import special
from astropy.time import Time
from astropy import units
from scipy.optimize import brentq
from scipy.misc import logsumexp

import cosmology

# a list of ifos that we can consider
ifos = ("H1", "H2", "I1", "K1", "L1", "V1", "ETdet1", "ETdet1", "ETdet1")

##################################################################
# Helper functions
##################################################################
def snr_projection(f_sig, method):
    """
    Function to calculate the SNR projection matrix P for a given set
    of detector responses, f_sig
    :param f_sig: an Nx2 array of detector responses
    :param method: the way we project (one of "time", "coh", "left", "right")
    """
    if method == "time":
        P = identity(len(f_sig))
    elif method == "coh":
        M = zeros((2, 2))
        for f in f_sig:
            M += outer(f, f)
        P = inner(inner(f_sig, linalg.inv(M)), f_sig)
    elif method == "right":
        cf = array([complex(f[0], f[1]) for f in f_sig])
        P = outer(cf.conjugate(), cf) / inner(cf.conjugate(), cf)
    elif method == "left":
        cf = array([complex(f[0], f[1]) for f in f_sig])
        P = outer(cf, cf.conjugate()) / inner(cf, cf.conjugate())
    else:
        raise NameError("Invalid projection method: %s" % method)
    return P


def evec_sigma(M):
    """
    Calculate the eigenvalues and vectors of M.
    sigma is defined as the reciprocal of the eigenvalue
    :param M: square matrix for which we calculate the eigen-vectors
    and sigmas
    """
    ev, evec = linalg.eig(M)
    epsilon = 1e-10
    sigma = 1 / sqrt(ev + epsilon)
    evec = evec[:, sigma.argsort()]
    sigma.sort()
    return evec, sigma

##################################################################
# Class to store detector information
##################################################################
class det(object):
    """
    class to hold the details of a detector
    """
    

    def __init__(self, location, response, found_thresh=5.0,
                 loc_thresh=4.0, duty_cycle=1.0, det_range = 200, f_mean = 100, f_band = 100, ifo=None):
        """
        Initialize
        :param location: array with detector location
        :param response: matrix with detector response
        :param det_range: the BNS range of the detector
        :param f_mean: float, mean frequency
        :param f_band: float, frequency bandwidth
        :param found_thresh: threshold for declaring an event found
        :param loc_thresh: threshold for declaring an event localized
        :param duty_cycle: fraction of time the detector is operational
        """
        self.location = location
        self.response = response
        self.det_range = det_range
        self.sigma = 2.26 * self.det_range * 8  # this gives the SNR at 1 Mpc
        self.f_mean = f_mean
        self.f_band = f_band
        self.found_thresh = found_thresh
        self.loc_thresh = loc_thresh
        if ifo == "ET_L_Eu_2" or ifo == "ET_L_Eu": #or ifo=="ET_L_US_2" or ifo=="ET_L_Aus_US_2" or ifo== "ET_L_Central_Africa_2"or ifo=="ET_L_US" or ifo=="ET_L_Aus_US" or ifo== "ET_L_Central_Africa":
            self.duty_cycle = 1.0 # set duty cycle of triangular ET to one due to redundancy.
        else:
            self.duty_cycle = duty_cycle

    def calculate_sensitivity(self, event):
        """
        Calculate the sensitivity of the detector to an event
        :param event: object, containing ra, dec, psi, gmst
        """
        self.f_plus, self.f_cross = lal.ComputeDetAMResponse(self.response,
                                                             event.ra, event.dec,
                                                             event.psi, event.gmst)
    
    def calculate_mirror_sensitivity(self, event):
        """
        Calculate the sensitivity of the detector to an event, in its mirror sky location
        :param event: object, containing mirror_ra, mirror_dec, psi, gmst
        """
        self.mirror_f_plus, self.mirror_f_cross = \
            lal.ComputeDetAMResponse(self.response,
                                     event.mirror_ra, event.mirror_dec,
                                     event.psi, event.gmst)

    def calculate_snr(self, event):
        """
        Calculate the expected SNR of the event in the detector
        :param event: object, containing ra, dec, psi, gmst, phi, cosi
        :return the complex SNR for the signal
        """
        self.calculate_sensitivity(event)
        self.snr = self.sigma / event.D * \
                   complex(cos(2 * event.phi), -sin(2 * event.phi)) * \
                   complex(self.f_plus * (1 + event.cosi ** 2) / 2, self.f_cross * event.cosi)

    def get_fsig(self, mirror=False):
        """
        Method to return the sensitivity of the detector
        :param mirror: boolean, is this the mirror position
        :return length 2 array: sigma * (F_plus, F_cross)
        """
        if mirror:
            return self.sigma * array([self.mirror_f_plus, self.mirror_f_cross])
        else:
            return self.sigma * array([self.f_plus, self.f_cross])


##################################################################
# Class to store network information 
##################################################################
class network(object):
    """
    class to hold the details of the network.
    """

    def __init__(self, threshold=12.0):
        """

        :param threshold: detection threshold for the network
        """
        self.threshold = threshold
        self.ifos = []

    def add_ifo(self, ifo, location, response, found_thresh=5.0, loc_thresh=4.0, duty_cycle=1.0, det_range=200, f_mean=100, f_band=100):
        """
        :param ifo: name of ifo
        :param location: ifo location
        :param response: matrix with detector response
        :param det_range: the BNS range of the detector
        :param f_mean: float, mean frequency
        :param f_band: float, frequency bandwidth
        :param found_thresh: threshold for declaring an event found
        :param loc_thresh: threshold for declaring an event localized
        :param duty_cycle: fraction of time the detector is operational
        """
        d = det(location, response, found_thresh=found_thresh, loc_thresh=loc_thresh, duty_cycle=duty_cycle, ifo=ifo, det_range=det_range, f_mean=f_mean, f_band=f_band)
        setattr(self, ifo, d)
        self.ifos.append(ifo)

    def set_configuration(self, configuration, found_thresh=5.0, loc_thresh=4.0,
                          duty_cycle=1.0, orientation_angle_1 = 0, orientation_angle_2 = 0,
                          ranges = None, fmeans = {'V1': 130.0, 'H1': 100.0, 'L1': 100.0},
                          fbands = {'V1': 148.9, 'H1': 117.4, 'L1': 117.4}, seed = None):

        """
        set the details of the detectors based on the given configuration.
        data is stored in the detectors module
        :param configuration: name of configuration
        :param found_thresh: threshold for single ifo detection
        :param loc_thresh: threshold for single ifo localization
        :param duty_cycle: fraction of time detectors are operational
        :param orientation_angle_1/2: the orientation of detectors in ET network in detectors.py
        :param ranges: dictionary of ranges (Mpc) for detectors in network
        :param fband: dictionary of frequency bandwidths (Hz) for detectors in network
        :param fmean: dictionary of mean frequency values (Hz) for detectors in network
        :param seed: if not None will set random so that each network will use the same random params
        """
        random.seed(seed)
        ifos = detectors.all_net(configuration)
        location, response = detectors.detectors(ifos, orientation_angle_1=orientation_angle_1, orientation_angle_2=orientation_angle_2)
        # the following three parameters are overwritten when the set_event_config function is called but we set them here in case we wish to ignore cosmology and have a more approximate result.
        if ranges==None: # if detector ranges aren't specified, read them from detectors.py
            try:
                ranges = detectors.range_8(configuration)
                fmeans = detectors.fmean(configuration)
                fbands = detectors.bandwidth(configuration)
            except KeyError: # if there aren't BNS ranges/bands just create zeros arrays
                ranges, fmeans, fbands = {}, {}, {}
                for ifo in ifos:
                    ranges[ifo] = zeros(len(ifos))
                    fmeans[ifo] = zeros(len(ifos))
                    fbands[ifo] = zeros(len(ifos))
        for ifo in ifos:
            if ifo in ["ET_L_Eu","ET_L_Eu_2"]:
                self.add_ifo(ifo, location[ifo], response[ifo], found_thresh, loc_thresh,
                             1.0,
                             ranges[ifo],
                             fmeans[ifo],
                             fbands[ifo])
            else:
                self.add_ifo(ifo, location[ifo], response[ifo], found_thresh, loc_thresh, 
                             duty_cycle, ranges[ifo], fmeans[ifo], fbands[ifo])

    def set_event_config(self, event, ignore_cosmology = False):
        """
        if ignore_cosmology is False, this overwrites the range, fband and fmeans that were set in the set_configuration as a default for the nework, with the estimated range fband and fmean for a given event and cosmology.
        """
        Dmax = 0
        
        if ignore_cosmology:
            for ifo in self.ifos:
                dhr = getattr(self,ifo).det_range * 2.26
                if dhr > Dmax:
                    Dmax = dhr
            return Dmax
        else:
            for ifo in self.ifos:
                dhr = detectors.read_dhr(ifo, event.mchirp, event.eta)
                getattr(self,ifo).det_range = dhr[0][0]/2.26
                getattr(self,ifo).sigma = dhr[0][0]*8
                getattr(self,ifo).f_mean = detectors.read_fmean(ifo, event.mchirp, event.eta)[0][0]
                getattr(self,ifo).f_band = detectors.read_fband(ifo, event.mchirp, event.eta)[0][0] # I changed BAND to band so that the default bandwidth is overwritten by this function.
                
                
                if dhr > Dmax:
                    Dmax = dhr

            return Dmax[0][0]

    def get_data(self, data):
        """
        get the relevant data for each detector and return it as an array
        :param data: name of data to return from a detector
        :return array containing requested data
        """
        return array([getattr(getattr(self, i), data) for i in self.ifos])


##################################################################
# Class to store localization information
##################################################################
class localization(object):
    """
    class to hold the details of a localization method
    """

    def __init__(self, method, M, snr, dt, z, event, mirror=False,
                 p=0.9, Dmax=1000, area = 0):
        """
        Initialization

        :param method: how we do localization, one of "time", "coh", "left, "right", "marg"
        :param M: localization matrix
        :param snr: snr of event
        :param dt: time offset
        :param z: complex snr
        :param event: details of event
        :param mirror: are we looking in the mirror location
        :param p: probability
        :param Dmax: maximum distance to consider
        """
        self.method = method
        self.M = M
        self.snr = snr
        self.dt = dt
        self.z = z
        self.mirror = mirror
        self.p = p
        self.like = 0.
        A  = fstat.snr_f_to_a(self.z, event.get_fsig(mirror))
        self.D, self.cosi, _, _ = fstat.a_to_params(A)
        if method is not "time" and method is not "marg": 
            self.approx_like(event, Dmax)
        
        if area != 0:
            self.area = area
        else:
            if M is not None:
                self.sky_project(event)
                self.calc_area()
            else:
                self.area = 1e6

    def sky_project(self, event):
        """
        Project localization matrix to zero out components in direction of source
        This is implementing equations 10 and 11 from the advanced localization paper
        :param event: structure with the event details
        """
        if self.mirror == True:
            source = event.mirror_xyz
        else:
            source = event.xyz
        P = identity(3) - outer(source, source)
        self.PMP = inner(inner(P, self.M), P)
        self.evec, self.sigma = evec_sigma(self.PMP)

    def calc_area(self):
        """
        Calculate the localization area
        :param p: probability for event to be contained in the given area
        """
        # calculate the area of the ellipse
        ellipse = - log(1. - self.p) * 2 * math.pi * (180 / math.pi) ** 2 * \
                self.sigma[0] * self.sigma[1]
        # calculate the area of the band 
        band = 4 * math.pi * (180 / math.pi) ** 2 * sqrt(2) * special.erfinv(self.p) * self.sigma[0]
        # use the minimum (that's not nan)
        self.area = nanmin((ellipse, band))

    def approx_like(self, event, Dmax=1000):
        """
        Calculate the approximate likelihood, based on equations XXX
        :param event: structure giving details of the event
        :param mirror: boolean indicating whether we are looking at the mirror position
        :param Dmax: maximum distance, used for normalization
        """
        if self.snr == 0:
            self.like = 0
            return
        Fp, Fc = event.get_f(self.mirror)
        self.like = self.snr ** 2 / 2
        if (self.method == "left") or (self.method == "right"):
            cos_fac = sqrt((Fp ** 2 + Fc ** 2) / (Fp * Fc))
            cosf = min(cos_fac / sqrt(self.snr), 0.5)
            self.like += log((self.D / Dmax) ** 3 / self.snr ** 2 * cosf)
        else:
            self.like += log(32. * (self.D / Dmax) ** 3 * self.D ** 4 / (Fp ** 2 * Fc ** 2)
                             / (1 - self.cosi ** 2) ** 3)

##################################################################
# Class to store event information
##################################################################
class event(object):
    """
    class to hold the events that we want to localize
    """

    def __init__(self, net, min_mch = 1.21877, max_mch = 1.21878, min_eta = .2499, max_eta = 0.25, 
        gps=1000000000, comoving_distn = None, params = None, 
        inject_uniform_in_componant_masses = False):
        """
        Initialize event
        :param Dmax: maximum distance to consider
        :param gmst: greenwich mean sidereal time
        :param params: parameters in form used by first 2 years paper
        """
     
     
        self.gps = lal.LIGOTimeGPS(gps,0)
        self.gmst = lal.GreenwichMeanSiderealTime(self.gps)
        self.phi = random.uniform(0, 2 * math.pi)
        
        if params==None: # choose random params unless we pass params argument
            self.ra = random.uniform(0, 2 * math.pi)
            self.dec = arcsin(random.uniform(-1, 1))
            self.psi = random.uniform(0, 2 * math.pi)
            self.cosi = random.uniform(-1, 1)
            if inject_uniform_in_componant_masses:
                m1 = random.uniform(min_mch/(2**(-1./5)), max_mch/(2**(-1./5)))
                m2 = random.uniform(min_mch/(2**(-1./5)), max_mch/(2**(-1./5)))

                self.eta = (m1*m2)/(m1+m2)**2
                mch = (m1*m2)**(3/5.)/(m1+m2)**(1/5.)
            else:
                mch = random.uniform(min_mch, max_mch)
                self.eta = random.uniform(min_eta, max_eta)
                    
            # Below choose between (1) injecting out to a specified redshift or (2) out to ET estimated range:
            
            # Option (1):
            z_max_for_mass_range=16
            Dco_max_for_mass_range=float(cosmology.cosmo.comoving_distance(z_max_for_mass_range)/ cosmology.unt.Mpc)

            # Option (2):
#            Dco_max_for_mass_range=cosmology.max_comoving_distance(max_mch) # interpolates detector response to given mass (hardcoded to ET detector)
#            z_max_for_mass_range=cosmology.redshift(Dco_max_for_mass_range) # unused parameter

            # New injection taking into account redshift:
            if comoving_distn:
                Dco = comoving_distn(random.uniform(0,1)) 
            else: Dco = random.uniform(0, 1) ** (1. / 3) * Dco_max_for_mass_range
            
        else:
            self.psi = params[0]
            self.cosi = params[1]
            Dco = params[2]
            mch = params[3]
            self.ra = params[4]
            self.dec = params[5]
            try: self.eta = params[6]
            except IndexError: self.eta = random.uniform(min_eta, max_eta)

        
        self.z = cosmology.redshift(Dco)
        self.D = Dco * (1 + self.z)
        self.mchirp = mch * (1 + self.z)

        # general content:
        self.xyz = detectors.xyz(self.ra - self.gmst, self.dec)
        self.ifos = []
        self.mirror = False
        self.detected = False
        self.localization = {}
        self.area = {}
        self.patches = {}

    def add_network(self, network):
        """
        calculate the sensitivities and SNRs for the various detectors in network
        :param network: structure containing details of the network
        """
        self.threshold = network.threshold
        self.found = 0
        self.localized = 0
        self.snrsq = 0
        for ifo in network.ifos:
            i = getattr(network, ifo)
            if rnd.random() < i.duty_cycle: #don't use numpy's RNG as messes up seeding for networks
                det = copy.deepcopy(i)
                det.calculate_snr(self)
                # calculate SNR and see if the signal was found/useful for loc
                s = abs(det.snr)
                setattr(self, ifo, det)
                if s > det.found_thresh:
                    self.found += 1
                # below used for exploring detectors below threshold.
#                else:
#                    det_ignored = True
#                    print "SNR in %s is only %.1f so we ignore it" % (ifo, s)
#                    ignored_det_fsig = det.get_fsig()
#                    print ignored_det_fsig
                if s > det.loc_thresh:
                    if ifo not in ["H2", "ETdet2", "ETdet3", "ET_L_Eu_2","ET_L_US_2", \
                                   "ET_L_Aus_US_2", "ET_L_Central_Africa_2", \
                                   "ET_Tri_Eu_2", "ET_Tri_Eu_3"]:
                        self.localized += 1
                    self.snrsq += s ** 2
                    # add the details to the event
                    self.ifos.append(ifo)                
        if self.found >= 2 and self.snrsq > self.threshold ** 2: self.detected = True
            
            # below for testing what difference detectors below threshold make to the PE
#        if det_ignored:
#            M = zeros((2, 2))
#            fsigs = list(self.get_fsig())
#            fsigs.append(ignored_det_fsig)
#            for f in fsigs:
#                M += outer(f, f)
#            F = sqrt(linalg.eig(M)[0])
#            F.sort()
#            print "f+, fx sensitivity is improved if including F[::-1] is %s" % F[::-1]

    def get_data(self, data):
        """
        get the relevant data for each detector and return it as an array
        :param data: string describing required data
        :return array with the data (for all ifos)
        """
        return array([getattr(getattr(self, i), data) for i in self.ifos])

    def get_fsig(self, mirror=False):
        """
        get the F_plus/cross times sigma for each detector
        :param mirror: boolean indicating whether we are considering the mirror location
        :return array with the sensitivities of the detectors
        """
        return array([getattr(self, i).get_fsig(mirror) for i in self.ifos])

    def get_f(self, mirror=False):
        """
        get the network sensitivity to plus and cross in the dominant polarization
        :param mirror: boolean indicating whether we are considering the mirror location
        :return length 2 array containing F_+, F_x response
        """
        M = zeros((2, 2))
        for f in self.get_fsig(mirror):
            M += outer(f, f)
        F = sqrt(linalg.eig(M)[0])
        F.sort()
        return F[::-1]

    def get_snr(self):
        """
        get the relevant data for each detector and return it as an array
        """
        return array([getattr(self, i).snr for i in self.ifos])

    def calculate_mirror(self):
        """
        calculate the mirror location and detector sensitivity there
        """
        if len(self.ifos) == 3:
            self.mirror_loc = {}
            l = self.get_data("location")
            x = l[1] - l[0]
            y = l[2] - l[0]
            normal = cross(x, y)
            normal /= linalg.norm(normal)
            self.mirror_xyz = self.xyz - 2 * inner(self.xyz, normal) * normal
            mra, mdec = detectors.phitheta(self.mirror_xyz)
            mra += self.gmst
            self.mirror_ra = mra
            self.mirror_dec = mdec
            self.mirror = True
            for i in self.ifos:
                getattr(self, i).calculate_mirror_sensitivity(self)

    def calculate_sensitivity(self):
        """
        calculate the network sensitivity
        """
        self.sensitivity = linalg.norm(self.get_fsig())
        if self.mirror:
            self.mirror_sensitivity = linalg.norm(self.get_fsig(mirror=True))


    def projected_snr(self, method, mirror=False):
        """
        Calculate the projected SNR for a given method at either original or mirror
        sky location
        """
        f_sig = self.get_fsig(mirror)
        P = snr_projection(f_sig, method)
        zp = inner(self.get_snr(), P)
        return (zp)
    
    def orientate(self, levels=[0.9,0.5,0.1]):
        """
        Calculate the alpha for the network, and the probablity the source would be recovered face-on.
        :param level: the confidence level with which the source is face-on
        """
        self.f_plus, self.f_cross = self.get_f()
        self.alpha = self.f_cross / self.f_plus

        a = fstat.params_to_a(d = self.D, cosi = self.cosi, psi = self.psi, phi = self.phi)
        a, D = fstat.set_snr(a, self.f_plus, self.f_cross, sqrt(self.snrsq))
        like_approx, like = likelihood.like_approx(a, self.f_plus, self.f_cross)
        prob = {}
        for key in like:
            prob[key] = exp(-like_approx + like[key])
        self.P_fo = prob['left'] + prob['right'] # probability source is face on
        
    def localize(self, method, mirror=False, p=0.9):
        """
        Localization of a source by a network of detectors, given the
        complex snr, sensitivity, bandwidth, mean frequency, location
        of the detectors.
        Here, we keep all the projection operators -- required if Z is not
        compatible with being a signal from the given sky location
        """
        f_mean = self.get_data("f_mean")
        f_band = self.get_data("f_band")
        # Calculate bar(f_sq)
        f_sq = (f_mean ** 2 + f_band ** 2)

        z = self.get_snr()
        locations = self.get_data("location")

        # calculate projection:
        f_sig = self.get_fsig(mirror)
        P = snr_projection(f_sig, method)

        # work out the localization factors
        B_i = 4 * pi ** 2 * real(sum(outer(f_sq * z.conjugate(), z) * P, axis=1))
        c_ij = 4 * pi ** 2 * real(outer(f_mean * z.conjugate(), f_mean * z) * P)
        C_ij = B_i * eye(len(B_i)) - c_ij
        c_i = sum(C_ij, axis=1)
        c = sum(c_i)

        CC = 1. / 2 * (outer(c_i, c_i) / c - C_ij)
        M = zeros([3, 3])

        for i1 in xrange(len(self.ifos)):
            for i2 in xrange(len(self.ifos)):
                M += outer(locations[i1] - locations[i2],
                           locations[i1] - locations[i2]) / (3e8) ** 2 \
                     * CC[i1, i2]

        # work out the offset factors
        A_i = 4 * pi * imag(sum(outer(f_mean * z.conjugate(), z) * P, axis=1))
        try:
            dt_i = 1. / 2 * inner(linalg.inv(C_ij), A_i)
        except:
            print("for method %s: Unable to invert C, setting dt=0" % method)
            dt_i = zeros_like(A_i)
        zp = self.projected_snr(method, mirror)
        snrsq = linalg.norm(zp) ** 2

        # work out the extra SNR from moving to the peak
        extra_snr = inner(dt_i, inner(C_ij, dt_i))
        # check it's reasonable:
        if (max(abs(dt_i * (2 * pi * f_band))) < 1. / sqrt(2)) and \
           ((snrsq + extra_snr) <= (self.snrsq + 1e-6)):
            if extra_snr > 0:
                # location of second peak is reasonable -- use it
                snrsq += extra_snr
                z_dt = z * (1 - 2 * pi ** 2 * (f_mean ** 2 + f_band ** 2) * dt_i ** 2
                            + 2.j * pi * dt_i * f_mean)
                zp = inner(z_dt, P)
        else:
            # failed to find a maximum within the linear regime
            M = None 
            snrsq = 0 
            dt_i = 0
            zp = zeros_like(zp)


        if mirror:
            self.mirror_loc[method] = \
                localization(method, M, sqrt(snrsq), dt_i, zp, self, mirror, p)
        else:
            self.localization[method] = \
                localization(method, M, sqrt(snrsq), dt_i, zp, self, mirror, p)

    def combined_loc(self, method):
        """
        Calculate the area from original and mirror locations for the given method
        p = the confidence region (assume it's what was used to get a1, a2)
        """
        patches = 1
        p = self.localization[method].p
        a0 = - self.localization[method].area / log(1. - p)
        if self.mirror:
            if method == "marg":
                drho2 = 2 * (self.localization[method].like -
                             self.mirror_loc[method].like)
            else:
                drho2 = (self.localization[method].snr ** 2 -
                         self.mirror_loc[method].snr ** 2)
            prob_ratio = self.mirror_loc[method].p / p
            a_ratio = self.mirror_loc[method].area / self.localization[method].area
            if drho2 > 2 * (log(prob_ratio) + log(1 + p * a_ratio) - log(1 - p)):
                a = - log(1 - p * (1 + a_ratio * prob_ratio * exp(-drho2 / 2))) * a0
            else:
                patches = 2
                a = a0 * ((1 + a_ratio) * (-log(1 - p) + log(1 + a_ratio)
                                           - log(1 + a_ratio * prob_ratio * exp(-drho2 / 2)))
                          - a_ratio * (drho2 / 2 - log(prob_ratio)))
            if isnan(a): 
                print("for method %s: we got a nan for the area" % method) 
        if not self.mirror or isnan(a):
            a = - log(1. - p) * a0
            patches = 1

        self.patches[method] = patches
        self.area[method] = a

        
    def marg_loc(self, mirror=False, p=0.9):
        """
        Calculate the marginalized localization.
        """
        if mirror:
            loc = "mirror_loc"
        else:
            loc = "localization"
        l = getattr(self, loc)
        # set coherent to zero if we don't trust it:
        if (l["coh"].snr ** 2 - l["right"].snr ** 2 < 2) or \
                (l["coh"].snr ** 2 - l["left"].snr ** 2 < 2):
            l["coh"].like = 0
        keys = ["left", "right", "coh"]
        r_max = 1.1 * sqrt(max([l[k].area for k in keys])/pi)
        r_min = 0.9 * sqrt(min([l[k].area for k in keys])/pi)
        r = brentq(f, r_min, r_max, (keys, l, p))
        l["marg"] = localization("marg", M=None, snr=l["coh"].snr, dt=l["coh"].dt, 
                                 z=l["coh"].z, event=self, mirror=mirror, 
                                 p=p, area = pi*r**2)
        l["marg"].like = logsumexp([l[k].like + log(l[k].area) - log(-2*pi*log(1-l[k].p)) 
                                    for k in keys])
        l["marg"].like -= log(l["marg"].area) - log(-2*pi*log(1-p))

                
    def localize_all(self, p=0.9):
        """
        Calculate all localizations
        """
        self.calculate_mirror()
        self.calculate_sensitivity()
        for method in ["time", "coh"]: #, "left", "right"]:
            self.localize(method, mirror=False, p=p)
            if self.mirror: self.localize(method, mirror=True, p=p)
        # self.marg_loc(p=p)
        # if self.mirror: self.marg_loc(mirror=True, p=p)
        for method in ["time", "coh"]: #, "marg"]:
            self.combined_loc(method)


def f(r, keys, l, p):
    f = 0
    lmax = max([l[k].like for k in keys])
    for k in keys:
        s2 = l[k].area/(-2*pi*log(1-l[k].p))
        f += exp(l[k].like - lmax) * s2 * (1 - p - exp(-r**2/(2*s2)))
    return f
