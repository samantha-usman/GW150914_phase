from numpy import *
import lal
import h5py
import subprocess
import os
from scipy import interpolate

################################
# define the detectors
################################
def detectors(ifos, india="bangalore", south_africa="sutherland", orientation_angle_1=0, orientation_angle_2=0):
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
        calc_location_response(76 + 26./60, 14 + 14./60, -13.2)
    elif india == "gmrt":
      # Here is the coordinates
      # location: 74deg  02' 59" E 19deg 05' 47" N 270.0 deg (W)
      location["I1"], response["I1"] = \
        calc_location_response(74 + 3./60, 19 + 6./60, -13.2)
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
          calc_location_response(76 + 26./60, 14 + 14./60, 270, 60)
  if "ETdet1" in ifos:
    location["ETdet1"], response["ETdet1"] = \
          calc_location_response(76 + 26./60, 14 + 14./60, 270, 60)
  if "ETdet2" in ifos:
    location["ETdet2"], response["ETdet2"] = \
          calc_location_response(76 + 26./60, 14. + 14./60, 270-45)
  if "ETdet3" in ifos:
    location["ETdet3"], response["ETdet3"] = \
          calc_location_response(16 + 26./60, 84. + 14./60, 270)
  if "ETdet4" in ifos:
    location["ETdet4"], response["ETdet4"] = \
          calc_location_response(54 + 26./60, 34. + 14./60, 180)

  # ET detectors below are placed in the European and American configs presented in
  # Raffai et al (2013) the orientation angle was chosen by using the same FOM in paper
  # - the alignment factor (measuring if network is sensitive to both polarizations)

  # ET Europe :
  if "ET_L_Eu" in ifos:
      location["ET_L_Eu"], response["ET_L_Eu"] = \
          calc_location_response(18.7, 48.5, 0 )#+ orientation_angle)
  if "ET_L_Eu_2" in ifos:
      location["ET_L_Eu_2"], response["ET_L_Eu_2"] = calc_location_response(18.7, 48.5, 45)

  # ET Europe Triangle:
  if "ET_Tri_Eu_1" in ifos:
      location["ET_Tri_Eu_1"], response["ET_Tri_Eu_1"] = \
          calc_location_response(18.7, 48.5, 0., 60. )
  if "ET_Tri_Eu_2" in ifos:
      location["ET_Tri_Eu_2"], response["ET_Tri_Eu_2"] = \
          calc_location_response(18.7, 48.5, 120., 60. )
  if "ET_Tri_Eu_3" in ifos:
      location["ET_Tri_Eu_3"], response["ET_Tri_Eu_3"] = \
          calc_location_response(18.7, 48.5, 240., 60. )

  # 3 ET Europe config
  if "ET_L_Aus_Eu" in ifos:
      location["ET_L_Aus_Eu"], response["ET_L_Aus_Eu"] = \
          calc_location_response(139.2, -26.5, 0 )#+ orientation_angle)
  if "ET_L_Argentina" in ifos:
      location["ET_L_Argentina"], response["ET_L_Argentina"] = \
          calc_location_response(68.9, 43.7, 0 +85)#+ orientation_angle)


  # 3 ET US Config (with 2 detectors to mimic triangle)
  if "ET_L_US" in ifos:
      location["ET_L_US"], response["ET_L_US"] = \
          calc_location_response(-98.4, 38.9, 0) #198)#175 + 23)
  if "ET_L_US_2" in ifos:
      location["ET_L_US_2"], response["ET_L_US_2"] = \
          calc_location_response(-98.4, 38.9, 175 + 23 - 45)
  if "ET_L_Aus_US" in ifos:
      location["ET_L_Aus_US"], response["ET_L_Aus_US"] = \
          calc_location_response(146.9, -35.8, 84.4) #+orientation_angle_1) #) #70.0)# 0 + orientation_angle_1 )#65)
  if "ET_L_Aus_US_2" in ifos:
      location["ET_L_Aus_US_2"], response["ET_L_Aus_US_2"] = \
          calc_location_response(146.9, -35.8, 65 -45)
  if "ET_L_Central_Africa" in ifos:
      location["ET_L_Central_Africa"], response["ET_L_Central_Africa"] = \
          calc_location_response(17.2, -9.9, 82.4) #+orientation_angle_2) #) # 169.4)# 0+ orientation_angle_2)# 135 + 36)
  if "ET_L_Central_Africa_2" in ifos:
      location["ET_L_Central_Africa_2"], response["ET_L_Central_Africa_2"] = \
          calc_location_response(17.2, -9.9, 135 + 36 -45)

  # 2xCE US Config:
  if "CE_US" in ifos:
      location["CE_US"], response["CE_US"] = \
          calc_location_response(-98.4, 38.9, 0)
  if "CE_Aus_US" in ifos:
      location["CE_Aus_US"], response["CE_Aus_US"] = \
          calc_location_response(146.9, -35.8, 84.4)

  if "CE_Central_Africa" in ifos:
      location["CE_Central_Africa"], response["CE_Central_Africa"] = \
          calc_location_response(17.2, -9.9, 82.4)


  #LIGO BLUE BIRD CONFIG
  if "LBB_H1" in ifos:
      location["LBB_H1"] = lho.location
      response["LBB_H1"] = lho.response
  if "LBB_L1" in ifos:
      llo = lal.CachedDetectors[lal.LALDetectorIndexLLODIFF]
      location["LBB_L1"] = llo.location
      response["LBB_L1"] = llo.response
  if "LBB_I1" in ifos:
      location["LBB_I1"], response["LBB_I1"] = calc_location_response(76 + 26./60, 14 + 14./60, -13.2)
  # for comparing alignment plots to klimenko 2011:
  if "LBB_V1" in ifos:
      virgo = lal.CachedDetectors[lal.LALDetectorIndexVIRGODIFF]
      location["LBB_V1"] = virgo.location
      response["LBB_V1"] = virgo.response
  if "LBB_K1" in ifos:
        # KAGRA location:
        # Here is the coordinates
        # 36.25 degree N, 136.718 degree E
        # and 19 degrees from North to West.
      location["LBB_K1"], response["LBB_K1"] = calc_location_response(136.718, 36.25, -19)

  if "LBB_A" in ifos:
      # from klimenko et al 2011 table 1
      location["LBB_A"], response["LBB_A"] = calc_location_response(115.66, 31.33, -45)
  if "LBB_A-" in ifos:
      # from klimenko et al 2011 table 1
      location["LBB_A-"], response["LBB_A-"] = calc_location_response(115.66, 31.33, 0)



            



  return( location, response )

def calc_location_response(longitude, latitude, arms, opening=90.):
  """
  Calculate the location and response for a detector with longitude, latitude in degrees
  The angle gives the orientation of the arms and is in degrees from North to East
  """
  phi = radians(longitude)
  theta = radians(latitude)
  angle = radians(arms)
  op = radians(opening)
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
  u_x = e_e * sin(angle + op) + e_n * cos(angle + op)
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

################################
# Networks
################################

def all_net(configuration):
  """
  return the detector ranges for a given configuration
  """
  net_dict_all = {
    "design" : ['H1', 'L1', 'V1' ],
    "GW170817" : ['H1', 'L1', 'V1' ],
    "GW170814" : ['H1', 'L1', 'V1' ],
    "GW170817_without_Virgo" : ['H1', 'L1' ],
    "ET" : ["ET_L_Eu", "ET_L_Eu_2"], # Triangular ET
    "ET1" : ['H1', 'L1', 'V1', 'ETdet1', 'ETdet2' ], # Triangular ET +LVC
    "ET2" : ['H1', 'L1', 'V1', 'ETdet1', 'ETdet3' ], # L-shaped at 2 places +LVC
    "ET3" : ['ETdet1', 'ETdet3', 'ETdet4'], # 3 L-shaped ET at three different places
    "ET3L_EU" : ["ET_L_Eu", "ET_L_Aus_Eu", "ET_L_Argentina"],
    "3ET" : ["ET_L_US", "ET_L_Aus_US", "ET_L_Central_Africa"],
    "3CE" : ["CE_US", "CE_Aus_US", "CE_Central_Africa"],
    "1CE-ET" : ["CE_US", "ET_L_Eu", "ET_L_Eu_2"],
    "2CE-ET" : ["CE_US", "CE_Aus_US", "ET_L_Eu", "ET_L_Eu_2"], #named 1 and 2 to distinguish from CE-ET (below) in Mills et al 2018.
    "CE-ET" : ["CE_US", "CE_Aus_US", "ET_L_Eu", "ET_L_Eu_2"],
    "Voyager-ET" : ["LBB_H1", "LBB_L1", "LBB_I1", "ET_L_Eu", "ET_L_Eu_2"],
    # next three networks are for calculating the impact of duty cycle on the Voyager-ET network
    "VoyagerLI-ET" : ["LBB_L1", "LBB_I1", "ET_L_Eu", "ET_L_Eu_2"],
    "VoyagerHI-ET" : ["LBB_H1", "LBB_I1", "ET_L_Eu", "ET_L_Eu_2"],
    "VoyagerHL-ET" : ["LBB_H1", "LBB_L1", "ET_L_Eu", "ET_L_Eu_2"],
    
    "VoyagerETtri" : ["LBB_H1", "LBB_L1", "LBB_I1", "ET_Tri_Eu_1", "ET_Tri_Eu_2",  "ET_Tri_Eu_3"],
    "Voyager" : ["LBB_H1", "LBB_L1", "LBB_I1"],
    "VoyagerWithAL" : ["LBB_H1", "LBB_L1", "LBB_I1", "ALV1", "ALK1"],
    "3_TriangularET" : ["ET_L_US", "ET_L_Aus_US", "ET_L_Central_Africa","ET_L_US_2", "ET_L_Aus_US_2", "ET_L_Central_Africa_2"],
    # for comparing to klimenko et al 2011:
    'LHVA2' : ["LBB_L1","LBB_H1","LBB_V1","LBB_A-"],
    'LHVA' : ["LBB_L1","LBB_H1","LBB_V1","LBB_A"],
    'LHVJ' : ["LBB_L1","LBB_H1","LBB_V1","LBB_K1"],
    'LHVAJ' : ["LBB_L1","LBB_H1","LBB_V1","LBB_A","LBB_K1"],
    # for calculating alignment factor distributions in inclincation paper
    "HL" : ["H1", "L1"],
    "HLV" : ["H1", "L1", "V1" ],
    "HLVK" : ["L1","H1","V1","K1"],
    "HLVKI" : ["L1","H1","V1","K1", "I1"],
    

    #for optimizing the orientations of ET3L_EU w.r.t. polarization metric (see optimizing polarization notebook)
    #first optimize for the two detector network:
    "ET2L_EU" : ["ET_L_Eu", "ET_L_Aus_Eu"],
    "2ET" : ["ET_L_US", "ET_L_Aus_US"],
    #ranges
    }
  return(net_dict_all[configuration])


def range_8(configuration):
  """
  return the detector ranges for a given configuration
  """
  range_dict_all = {
      # updated aLIGO design sensitivity range from 197.5 to 181.5 Mpc on 9 Apr 2018 to reflect T1800044-v4
    "HL" : {'H1' : 181.5, 'L1' : 181.5},
    "HLV" : {'H1' : 181.5, 'L1' : 181.5, 'V1': 128.3 },
    "HLVK" : {'H1' : 181.5, 'L1' : 181.5, 'V1': 128.3, 'K1' : 160.0},
    "HLVKI" : {'H1' : 181.5, 'L1' : 181.5, 'V1': 128.3, 'K1' : 160.0, 'I1' : 181.5},
    "GW170817" : {'H1': 107/2.26 *1.26 , 'L1': 218/2.26, 'V1': 58/2.26}, # 1.26 is the improvement factor for H1's range due to data processing.
    "GW170817_without_Virgo" : {'H1': 107/2.26 *1.26 , 'L1': 218/2.26},
    "GW170814" : {'H1': 53, 'L1': 98, 'V1': 26}, # 1.26 is the improvement factor for H1's range due to data processing.
    "design" : {'H1' : 181.5, 'L1' : 181.5, 'V1': 128.3 },
    "early" : {'H1' : 60., 'L1': 60.},
    "half_ligo" : {'H1' : 99, 'L1' : 99, 'V1': 128.3 },
    "half_virgo" : {'H1' : 181.5, 'L1' : 181.5, 'V1': 64 },
    "nosrm" : {'H1' : 159, 'L1' : 159, 'V1': 109 },
    "india" : {'H1' : 181.5, 'L1' : 181.5, 'V1': 128.3, "I1" : 181.5 },
    "kagra" : {'H1' : 181.5, 'L1' : 181.5, 'V1': 128.3, "I1" : 181.5 , \
        "K1" : 160.0},
    "bala" : {'H1' : 181.5, 'H2' : 181.5, 'L1' : 181.5, 'V1': 128.3, \
        "I1" : 181.5 , "K1" : 160.0},
    "sa" : {'H1' : 181.5, 'L1' : 181.5, 'V1': 128.3, "I1" : 181.5 , \
        "K1" : 160.0, "S1":181.5},
    "sa2" : {'H1' : 181.5, 'L1' : 181.5, 'V1': 128.3, "I1" : 181.5 , \
        "K1" : 160.0, "S1":181.5},
    "steve" : {'H1' : 160.0, 'L1' : 160.0, 'V1': 160.0, "I1" : 160.0 },
    "s6vsr2" : {'H1' : 20., 'L1' : 20., 'V1': 8. }
  }
  return(range_dict_all[configuration])

def bandwidth(configuration):
  """
  return the detector bandwidths for a given configuration
  """
  bandwidth_dict_all = {
    "HL" : {'H1' : 117.4, 'L1' : 117.4},
    "HLV" : {'H1' : 117.4, 'L1' : 117.4, 'V1': 148.9},
    "HLVK" : {'H1' : 117.4, 'L1' : 117.4, 'V1': 148.9, 'K1' : 148.9},
    "HLVKI" : {'H1' : 117.4, 'L1' : 117.4, 'V1': 148.9, 'K1' : 148.9, 'I1' : 117.4},
    "GW170817" : {'H1' : 117.4, 'L1' : 117.4, 'V1': 148.9},
    "GW170817_without_Virgo" : {'H1' : 117.4, 'L1' : 117.4},
    "GW170814" : {'H1' : 117.4, 'L1' : 117.4, 'V1': 148.9},
    "design" : {'H1' : 117.4, 'L1' : 117.4, 'V1': 148.9 },
    "early"  : {'H1' : 123.7, 'L1' : 123.7 },
    "half_virgo" : {'H1' : 117.4, 'L1' : 117.4, 'V1': 148.9 },
    "half_ligo" : {'H1' : 117.4, 'L1' : 117.4, 'V1': 148.9 },
    "nosrm" : {'H1' : 43, 'L1' : 43, 'V1': 58 },
    "india" : {'H1' : 117.4, 'L1' : 117.4, 'V1': 148.9, "I1" : 117.4 },
    "kagra" : {'H1' : 117.4, 'L1' : 117.4, 'V1': 148.9, "I1" : 117.4, \
        "K1" : 89.0 },
    "bala" : {'H1' : 117.4, 'H2' : 117.4, 'L1' : 117.4, 'V1': 148.9, \
        "I1" : 117.4, "K1" : 89.0 },
    "sa" : {'H1' : 117.4, 'L1' : 117.4, 'V1': 148.9, "I1" : 117.4, \
        "K1" : 89.0 , "S1": 117.4},
    "sa2" : {'H1' : 117.4, 'L1' : 117.4, 'V1': 148.9, "I1" : 117.4, \
        "K1" : 89.0 , "S1": 117.4},
    "steve" : {'H1' : 100.0, 'L1' : 100.0, 'V1': 100.0, "I1" : 100.0 },
    "s6vsr2" : {'H1' : 100., 'L1' : 100., 'V1': 120. }
  }
  return(bandwidth_dict_all[configuration])

def fmean(configuration):
  """
  return the detector mean frequencies for a given configuration
  """
  fmean_dict_all = {
    "HL" : {'H1' : 100., 'L1' : 100.},
    "HLV" : {'H1' : 100., 'L1' : 100., 'V1': 130.},
    "HLVK" : {'H1' : 100., 'L1' : 100., 'V1': 130., 'K1' : 130.},
    "HLVKI" : {'H1' : 100., 'L1' : 100., 'V1': 130., 'K1' : 130., 'I1' : 100.},
    "GW170817" : {'H1' : 100., 'L1' : 100., 'V1': 130.},
    "GW170814" : {'H1' : 117.4, 'L1' : 117.4, 'V1': 148.9},
    "GW170817_without_Virgo" : {'H1' : 100., 'L1' : 100.},
    "steve" : {'H1' : 100.0, 'L1' : 100.0, 'V1': 100.0, "I1" : 100.0 },
    "design" : {'H1' : 100., 'L1' : 100., 'V1': 130. },
    "india" : {'H1' : 100., 'I1' : 100., 'L1' : 100., 'V1': 130. },
    "s6vsr2" : {'H1' : 180., 'L1' : 180., 'V1': 150. }
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

data, fname, dh_interp, fm_interp, fb_interp = {}, {}, {}, {}, {}
current_dir = os.getcwd()
data_dir = current_dir+'/localizeET/Data/'
fname = {'AL': data_dir + 'aligo_mchmin-1.0_mchmax-300.0_etamin-0.05_etamax-0.25.h5py',
    'ET':  data_dir + 'ETD_mchmin-1.0_mchmax-650.0_etamin-0.05_etamax-0.25.h5py',
        'LB': data_dir + 'LIGOBlueBird_mchmin-1.0_mchmax-500.0_etamin-0.05_etamax-0.25.h5py',
            'CE' : data_dir + 'CE_mchmin-1.0_mchmax-650.0_etamin-0.05_etamax-0.25.h5py'}
# AL = Advanced LIGO, ET = Einstein Telescope, CE = Cosmic Explorer, LB = LIGO Blue Bird

for key in fname.keys():
    infile = h5py.File(name = fname[key], mode = 'r')
    for dkey in infile.keys():
        data[dkey] = infile[dkey][:]

#mchmin, mchmax = infile['mchirp'].attrs['mchmin'], infile['mchirp'].attrs['mchmax']
#etamin, etamax = infile['eta'].attrs['etamin'], infile['eta'].attrs['etamax']
#deat, dmch = infile['eta'].attrs['deta'], infile['mchirp'].attrs['dmch']

    infile.close()

    nx = len(unique(data['mchirp']))
    ny = len(data['mchirp'])/nx
    ax = (data['mchirp'].reshape((nx, ny))).transpose()[0]
    ay = data['eta'].reshape((nx, ny))[0]
    adh = data['d_horizon'].reshape((nx, ny))
    amean = data['fmean'].reshape((nx, ny))
    aband = data['fband'].reshape((nx, ny))
    dh_interp[key] = interpolate.RectBivariateSpline(ax, ay, adh)
    fm_interp[key] = interpolate.RectBivariateSpline(ax, ay, amean)
    fb_interp[key] = interpolate.RectBivariateSpline(ax, ay, aband)

def read_dhr(key, mchirp, eta):
    ''' return horizon distance for a chirp mass and eta value '''
    key = key[0:2]   # so make sure det name starts with ET, LB, AL, or CE!
    return dh_interp[key](mchirp, eta)

def read_fmean(key, mchirp, eta):
    ''' return mean frequency value for a chirp mass and eta value '''
    key = key[0:2]
    snr = 8*dh_interp[key](mchirp, eta)
    return fm_interp[key](mchirp, eta)

def read_fband(key, mchirp, eta):
    ''' return frequency bandwidth for a chirp mass and eta value '''
    key = key[0:2]
    snr = 8*dh_interp[key](mchirp, eta)
    return fb_interp[key](mchirp, eta)
