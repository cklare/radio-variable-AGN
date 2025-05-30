import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky
import os
import pickle
import functions as fns

# path to VaST data
#f = input("input path to VAST data:")
# currently, hardcoded
print('are we using the correct version')
f = '../../all_siblings/straight_from_pipe/'

# first, read in the files with the agn and vast source crossmatches and merge
# there are three separate agn catalogs - AGN identified in SDSS only, AGN identified in WISE only, and SDSS identified in both WISE and SDSS
print('creating sdss-vast sample...')
sdss_pilot, sdss_extragalactic, sdss_racs, sdss_ids = fns.combine_agn_vast(f'{f}sdss')
print('creating sdss-wise-vast sample...')
sdss_wise_pilot, sdss_wise_extragalactic, sdss_wise_racs, sdss_wise_ids = fns.combine_agn_vast(f'{f}sdss_wise')
print('creating wise-vast sample...')
wise_pilot,  wise_extragalactic, wise_racs, wise_ids = fns.combine_agn_vast(f'{f}wise')

# now, account for cross-agn catalog contamination
print('checking for cross-agn catalog contamination...')

sdss_ids,sdss_wise_ids,wise_ids = fns.cross_agn_survey_checks(sdss_ids,sdss_wise_ids,wise_ids)
print(f'total number of unique vast-agn sources: {len(sdss_ids)+len(sdss_wise_ids)+len(wise_ids)}')

# update the sdss-wise and wise catalogs
mask = sdss_wise_pilot['vast_id'].isin(sdss_wise_ids['vast_id_pilot'])
sdss_wise_pilot = sdss_wise_pilot[mask]
mask = sdss_wise_extragalactic['vast_id'].isin(sdss_wise_ids['vast_id_extragalactic'])
sdss_wise_extragalactic = sdss_wise_extragalactic[mask]
mask = sdss_wise_racs['vast_id'].isin(sdss_wise_ids['vast_id_racs'])
sdss_wise_racs = sdss_wise_racs[mask]

# update wise
mask = wise_pilot['vast_id'].isin(wise_ids['vast_id_pilot'])
wise_pilot = wise_pilot[mask]
mask = wise_extragalactic['vast_id'].isin(wise_ids['vast_id_extragalactic'])
wise_extragalactic = wise_extragalactic[mask]
mask = wise_racs['vast_id'].isin(wise_ids['vast_id_racs'])
wise_racs = wise_racs[mask]




# now, get the measurements
print('loading measurement files...')
pilot_measurements= fns.load_measurements(f'{f}pilot_measurements',sdss_ids,sdss_wise_ids, wise_ids, 'pilot')
extragalactic_measurements= fns.load_measurements(f'{f}extragalactic_measurements',sdss_ids,sdss_wise_ids, wise_ids, 'extragalactic')
racs_measurements= fns.load_measurements(f'{f}racs_measurements',sdss_ids,sdss_wise_ids, wise_ids, 'racs')

# first, I will calculate the scintillation modulation, max two-epoch variability metric, and max two-epoch modulation

sdss_ids = fns.calculate_variability_metrics(sdss_ids,pilot_measurements, extragalactic_measurements, racs_measurements)
sdss_wise_ids = fns.calculate_variability_metrics(sdss_wise_ids,pilot_measurements, extragalactic_measurements, racs_measurements)
wise_ids = fns.calculate_variability_metrics(wise_ids,pilot_measurements, extragalactic_measurements, racs_measurements)

var_metrics = pd.concat([sdss_ids['var_max'],sdss_wise_ids['var_max'],wise_ids['var_max']])
mods = pd.concat([sdss_ids['mod_max'],sdss_wise_ids['mod_max'],wise_ids['mod_max']])


if not os.path.isdir('../../distributions/'):
    os.mkdir('../../distributions/')

plt.figure()
plt.hist(var_metrics, density=True, color = 'm')
plt.axvline(x=4.3, linestyle= '--',color = 'k',label='variability threshold')
#plt.yscale('log')
plt.xlabel('Variability Metric')
plt.ylabel('Number of Sources in Sample')
plt.title('Distribution of max flux epoch vs min flux epoch variability metrics')
plt.legend()
plt.savefig('../../distributions/var_metrics.png')
plt.close()

plt.figure()
plt.hist(mods, density=True, color = 'purple')
plt.axvline(x=0.26, linestyle='--',color='k', label='variability threshold')
max_mod_riss = (888/8000)**(17/30)
plt.axvline(x=max_mod_riss, linestyle='--',color='r',label='max RISS modulation')
plt.legend()
plt.title('Distribution of maximum modulation from median flux')
plt.xlabel('Modulation')
plt.ylabel('Number of sources in sample')
plt.savefig('../../distributions/mod_metrics.png')
plt.close()

# now, plot the modulations and modulations due to riss


# now, perform the light curve morphology cuts 
print('beginning light curve cuts on sdss sample')
sdss_ids, sdss_pilot, sdss_extragalactic, sdss_racs = fns.light_curve_cuts(sdss_pilot, sdss_extragalactic, sdss_racs, pilot_measurements,\
                                                                           extragalactic_measurements,racs_measurements,sdss_ids)
print('beginning light curve cuts on sdss-wise sample')
sdss_wise_ids, sdss_wise_pilot, sdss_wise_extragalactic, sdss_wise_racs = fns.light_curve_cuts(sdss_wise_pilot, sdss_wise_extragalactic, sdss_wise_racs,\
                                                                    pilot_measurements,extragalactic_measurements,racs_measurements,sdss_wise_ids)
print('beginning light curve cuts on wise sample')
wise_ids, wise_pilot, wise_extragalactic,wise_racs = fns.light_curve_cuts(wise_pilot, wise_extragalactic, wise_racs, pilot_measurements,\
                                                                          extragalactic_measurements,racs_measurements,wise_ids)

# now, determining expected effects due to scintillation
nu_vast = 888 # vast frequency, in MHz
delta_nu_vast = 288 # vast bandwidth, in MHz

print('beginning ISS calculations')

# determine modulation index and timescale due to ISS
# we assume all sources are point sources (as suggested by our compactness cut), as this gives an upper bound to the allowed magnitude of ISS modulations
sdss_pilot,sdss_extragalactic,sdss_racs = fns.scintillation(sdss_pilot,sdss_extragalactic,sdss_racs,nu_vast,delta_nu_vast)
sdss_wise_pilot,sdss_wise_extragalactic,sdss_wise_racs = fns.scintillation(sdss_wise_pilot,sdss_wise_extragalactic,sdss_wise_racs,nu_vast,delta_nu_vast)
wise_pilot,wise_extragalactic,wise_racs = fns.scintillation(wise_pilot,wise_extragalactic,wise_racs,nu_vast,delta_nu_vast)

# finally, plot the lightcurves
print('plotting lightcurves')
plots_f = '../../candidate_plots_with_ISS/'
if not os.path.isdir(plots_f):
    os.mkdir(plots_f)
fns.plot_light_curves(sdss_pilot, sdss_extragalactic,sdss_racs,pilot_measurements,extragalactic_measurements, racs_measurements,sdss_ids,plots_f)
fns.plot_light_curves(sdss_wise_pilot, sdss_wise_extragalactic,sdss_wise_racs,pilot_measurements,extragalactic_measurements, racs_measurements, sdss_wise_ids,plots_f)
fns.plot_light_curves(wise_pilot, wise_extragalactic,wise_racs,pilot_measurements,extragalactic_measurements, racs_measurements,wise_ids,plots_f)

