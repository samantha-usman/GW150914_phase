from numpy import *
import lal
import h5py
from scipy import interpolate

################################
# define the detectors
################################
def detectors(ifos, india="bangalore", south_africa="sutherland"):
  """
  Set up a dictionary of detector locations and responses. 
  Either put indigo in bangalore or a different site
  """
  location = {}
  response = {}
  # Use cached detectors for known sites:
  lho = lal.lal.CachedDetectors[lal.LALDetectorIndexLHODIFF]

  if "H1" in ifos:
    location["H1"] = lho.location
    response["H1"] = lho.response
  if "H2" in ifos:
    location["H2"] = lho.location
    response["H2"] = lho.response
  if "L1" in ifos:
    llo = lal.CachedDetectors[lal.LALDetectorIndexLLODIFF]
    location["L1"] = llo.location
    response["L1"] = llo.response
  if "V1" in ifos:
    virgo = lal.CachedDetectors[lal.LALDetectorIndexVIRGODIFF]
    location["V1"] = virgo.location
    response["V1"] = virgo.response
  if "XX" in ifos:
    response["XX"] = array([[0.5,0,0],[0,-0.5,0],[0,0,0]],  dtype=float32)
  if "K1" in ifos:
    # KAGRA location:
    # Here is the coordinates
    # 36.25 degree N, 136.718 degree E
    # and 19 degrees from North to West.
    location["K1"], response["K1"] = calc_location_response(136.718, 36.25, -19)
  if "I1" in ifos:
    if india == "bangalore":
      # Could you pl run us the Localisation Plot once more with
      # a location close to Bangalore that is seismically quiet
      # 14 deg 14' N
      # 76 deg 26' E?
      location["I1"], response["I1"] = \
          calc_location_response(76 + 26./60, 14 + 14./60, 270)
    elif india == "gmrt":
      # Here is the coordinates
      # location: 74deg  02' 59" E 19deg 05' 47" N 270.0 deg (W)
      location["I1"], response["I1"] = \
        calc_location_response(74 + 3./60, 19 + 6./60, 270)
  if "S1" in ifos:
    # SOUTH AFRICA
    # from Jeandrew
    # Soetdoring
    # -28.83926,26.098595
    # Sutherland
    # -32.370683,20.691833
    if south_africa == "sutherland":
      location["S1"], response["S1"] = \
        calc_location_response(20.691833, -32.370683, 270)
    elif south_africa == "soetdoring":
      location["S1"], response["S1"] = \
        calc_location_response(26.098595, -28.83926, 270)
  if "ETdet1" in ifos:
    location["ETdet1"], response["ETdet1"] = \
          calc_location_response(76 + 26./60, 14 + 14./60, 270)
  if "ETdet2" in ifos:
    location["ETdet2"], response["ETdet2"] = \
          calc_location_response(76 + 26./60, 14. + 14./60, 270-45)
  if "ETdet3" in ifos:
    location["ETdet3"], response["ETdet3"] = \
          calc_location_response(16 + 26./60, 84. + 14./60, 270)
  if "ETdet4" in ifos:
    location["ETdet4"], response["ETdet4"] = \
          calc_location_response(54 + 26./60, 34. + 14./60, 180)
   
  return( location, response )

def calc_location_response(longitude, latitude, arms):
  """
  Calculate the location and response for a detector with longitude, latitude in degrees
  The angle gives the orientation of the arms and is in degrees from North to East
  """
  phi = radians(longitude)
  theta = radians(latitude)
  angle = radians(arms)
  r = 6.4e6
  location = r * xyz(phi, theta)
  r_hat = location / linalg.norm(location)
  # Take North, project onto earth's surface...
  e_n = array([0,0,1])
  e_n = e_n - r_hat * inner(e_n, r_hat)
  # normalize
  e_n = e_n / linalg.norm(e_n)
  # and calculate east
  e_e = cross(e_n, r_hat)
  # Calculate arm vectors
  u_y = e_e * sin(angle) + e_n * cos(angle)
  u_x = e_e * sin(angle + pi/2) + e_n * cos(angle + pi/2)
  response = array(1./2 * (outer(u_x, u_x) - outer(u_y, u_y)), dtype=float32)
  return location, response
  

################################
# co-ordinate transformations 
################################
def xyz(phi, theta):
  """
  phi, theta -> x,y,z
  """
  x = cos(theta) * cos(phi)
  y = cos(theta) * sin(phi)
  z = sin(theta)
  loc = asarray([x,y,z])
  return(loc)

def phitheta(loc):
  """
  x,y,z -> phi, theta
  """
  x = loc[0]
  y = loc[1]
  z = loc[2]
  r = sqrt(x**2 + y**2 + z**2)
  theta = arcsin(z/r)
  phi = arctan2(y,x)
  return(phi, theta)

def range_8(configuration):
  """
  return the detector ranges for a given configuration
  """
  range_dict_all = {
    "design" : {'H1' : 197.5, 'L1' : 197.5, 'V1': 128.3 }, 
    "ET1" : {'H1' : 197.5, 'L1' : 197.5, 'V1': 128.3, 'ETdet1': 1500., 'ETdet2': 1500. }, # Triangular ET
    "ET2" : {'H1' : 197.5, 'L1' : 197.5, 'V1': 128.3, 'ETdet1': 1500., 'ETdet3': 1500. }, # L-shaped at 2 places
    "ET3" : {'ETdet1': 1500., 'ETdet3': 1500., 'ETdet4': 1500. }, # 3 L-shaped ET at three different places
  }
  return(range_dict_all[configuration])

def bandwidth(configuration):
  """
  return the detector bandwidths for a given configuration
  """
  bandwidth_dict_all = {
    "design" : {'H1' : 117.4, 'L1' : 117.4, 'V1': 148.9 }, 
    "ET1" : {'H1' : 117.4, 'L1' : 117.4, 'V1': 148.9, 'ETdet1': 117.4, 'ETdet2': 117.4 },
    "ET2" : {'H1' : 117.4, 'L1' : 117.4, 'V1': 148.9, 'ETdet1': 117.4, 'ETdet3': 117.4 },
    "ET3" : {'ETdet1': 117.4, 'ETdet3': 117.4, 'ETdet4': 117.4 }, # 3 L-shaped ET at three different places
  }
  return(bandwidth_dict_all[configuration])

def fmean(configuration):
  """
  return the detector mean frequencies for a given configuration
  """
  fmean_dict_all = {
    "design" : {'H1' : 100.0, 'L1' : 100.0, 'V1': 100.0, "I1" : 100.0 }, 
    "ET1" : {'H1' : 100., 'L1' : 100., 'V1': 130., 'ETdet1':100., 'ETdet2':100 },
    "ET2" : {'H1' : 100., 'L1' : 100., 'V1': 130., 'ETdet1':100., 'ETdet3':100 },
    "ET3" : {'ETdet1' : 100., 'ETdet3':100., 'ETdet4':100 }, # 3 L-shaped ET at three different places
  }
  return(fmean_dict_all[configuration])

def sigma_t(configuration):
  """
  return the timing accuracy.  We use SNR of 10 in LIGO, but scale the expected
  SNR in other detectors based on the range.
  It's just 1/(20 pi sigma_f for LIGO.  
  But 1/(20 pi sigma_f)(r_ligo/r_virgo) for Virgo; 
  5 seconds => no localization from det.
  """
  b = bandwidth(configuration)
  r = range_8(configuration)
  s = {}
  for ifo in r.keys():
    s[ifo] = 1./20/math.pi/b[ifo]*r["H1"]/r[ifo]
  return(s)

##################################################################
# Read SNRs, fmean, fband
##################################################################

data = {}
fname = '/home/spxvt/scripts/simple_pe/localizeET/simple_pe/aligo_mchmin-1.0_mchmax-2.0_etamin-0.05_etamax-0.25.h5py'
infile = h5py.File(fname, 'r')
for key in infile.keys():
    data[key] = infile[key][:]

mchmin, mchmax = infile['mchirp'].attrs['mchmin'], infile['mchirp'].attrs['mchmax']
etamin, etamax = infile['eta'].attrs['etamin'], infile['eta'].attrs['etamax']
deat, dmch = infile['eta'].attrs['deta'], infile['mchirp'].attrs['dmch']

infile.close()

nx = len(unique(data['mchirp']))
ny = len(data['mchirp'])/nx
ax = (data['mchirp'].reshape((nx, ny))).transpose()[0]
ay = data['eta'].reshape((nx, ny))[0]
adh = data['d_horizon'].reshape((nx, ny))
amean = data['fmean'].reshape((nx, ny))
aband = data['fband'].reshape((nx, ny))
dh_interp = interpolate.RectBivariateSpline(ax, ay, adh)
fm_interp = interpolate.RectBivariateSpline(ax, ay, amean)
fb_interp = interpolate.RectBivariateSpline(ax, ay, aband)

def read_dhr(mchirp, eta):
    ''' return horizon distance for a chirp mass and eta value '''
    return dh_interp(mchirp, eta)

def read_fmean(mchirp, eta):
    ''' return mean frequency value for a chirp mass and eta value '''
    snr = 8*read_dhr(mchirp, eta)
    return fm_interp(mchirp, eta)*(4./snr**2)

def read_band(mchirp, eta):
    ''' return frequency bandwidth for a chirp mass and eta value '''
    snr = 8*read_dhr(mchirp, eta)
    return fb_interp(mchirp, eta)*(2./snr)
