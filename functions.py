# mega file with all my user defined functions

# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky
import os
import pickle

################## function #0 ##################

# in the case that multiple vast sources are mapped to the same agn, choose which vast source to keep
# in these cases, many are due to 'siblings' ie multiple beams fit to the same source
# adf = df with the agn catalog data
# vdf = df with the vast catalog data
# agn_id_str = what the column of agn source identifiers is called (for SDSS sources, ObjID, and for WISE sources, WISEA)
# returns the dfs with unique vast to agn mappings
def choose_vast_to_keep(adf,vdf,agn_id_str):
    # first, look at the percent of epochs in each source with siblings, since these epochs will have to get thrown out
    # we want to drop any sources with all epochs contaminated by siblings
    mask = vdf['n_siblings']==vdf['n_selavy']
    print(f'number of sources in vast survey with no sibling-free selavy measurements: {sum(mask)}')
    vdf = vdf[~mask]
    ratio = vdf['n_siblings']/vdf['n_selavy']
    # insert this column into the df
    vdf = vdf.assign(percent_siblings=ratio)
    
    # sort by the percentage of contaminated epochs, since we'll keep the least contaminated
    vdf = vdf.sort_values(by='percent_siblings',ascending=True)
    # drop the extra vast sources from the vast df
    vdf = vdf.drop_duplicates(subset=agn_id_str,keep='first')
    # update the agn df accordingly
    mask = adf['vast_id'].isin(vdf['vast_id'])
    adf = adf[mask]
    # output the results
    print(f'{len(ratio)-len(vdf)} superfluous vast sources dropped')
 
    return adf,vdf

##################### function #1 ########################

# in the case that multiple agn are mapped to the same vast source (not common), choose one to keep, and merge the vast and agn df
def choose_agn_to_keep(adf,vdf,agn_id_str):
    # first, make sure all agn ids are unique, since we should always be dropping the vast duplicates first
    ids = adf[agn_id_str]
    if len(np.unique(ids.values))!=len(ids.values):
        print('superfluous vast sources still present')
    ids = vdf[agn_id_str]
    if len(np.unique(ids.values))!=len(ids.values):
        print('superfluous vast sources still present in vast df')
    # get the vast ids
    vast_ids = adf['vast_id']
    # make sure the same vast sources are in both dataframes still
    test = vdf['vast_id']
    if len(np.unique(vast_ids.values))!=len(np.unique(test.values)):
        print('frick')
    # if there are duplicate matches, merge the dfs and return
    if len(np.unique(vast_ids.values))==len(vast_ids.values):
        print('yay! no superfluous agn matches')
        df = adf.merge(vdf,how='outer',on=agn_id_str,suffixes=('','_vast'))
        return df
    
    # keep the closest on-sky match
    vdf = vdf.sort_values(by='sep',ascending=True)
    vdf = vdf.drop_duplicates(subset='vast_id',keep='first')
    adf = adf.sort_values(by='sep',ascending=True)
    adf = adf.drop_duplicates(subset='vast_id',keep='first')

    # output the results
    print(f'{len(vast_ids)-len(vdf)} superfluous agn dropped')

    # check that the dataframes are the same size
    if len(vdf)!=len(adf):
        print('problem')
    
    # merge the vast and agn dataframes and return
    df = adf.merge(vdf,how='outer',on=agn_id_str,suffixes=('','_vast'))
    return df

######################### function #3 ###########################

# since a pandas merge will drop sources not found in all vast surveys, we need to add null values to the id dataframe
def force_square_dfs(all_ids,survey_ids,agn_id_str, survey_str):
    # adding null values (0) for the sources not in extragalactic
    print(f'number of sources: {len(all_ids)}')
    mask = all_ids[agn_id_str].isin(survey_ids[agn_id_str])
    # if all the agn are in the vast survey, no need to do anything
    if sum(mask)==len(mask):
        print(f'all sources found in {survey_str} survey')
        all_ids = all_ids.merge(survey_ids,how='outer',on=agn_id_str)
        all_ids = all_ids.rename(columns={'vast_id':f'vast_id_{survey_str}'})
        return all_ids
    # otherwise, insert zeros for the 'vast_id_survey' values for all the agn ids in ids[mask]
    if sum(mask)<len(mask):
        print(f'{sum(mask)} sources out of {len(mask)} total sources found in {survey_str} survey')
        # make a two column dataframe to add onto our id frame for non-survey sources
        missing_ids = all_ids[~mask]
        missing_ids = missing_ids[[agn_id_str]]
        null_ids_to_insert = np.zeros((len(missing_ids),),dtype=int)
        # insert the null vast ids
        missing_ids = missing_ids.assign(vast_id=null_ids_to_insert)
        # concatenate with the original survey id df
        survey_ids = pd.concat([survey_ids,missing_ids])
        all_ids = all_ids.merge(survey_ids,how='outer',on=agn_id_str)
        all_ids = all_ids.rename(columns={'vast_id':f'vast_id_{survey_str}'})
        return all_ids

################# function #4 ######################

# unionize the crossmatched agn and vast datasets
# f is the location of the data
def combine_agn_vast(f):
    # read in the dataframes which were crossmatched
    # _agn indicates the df has the sdss or wise data, while _vast indicates the df has the vast data.
    # Both have the corresponding id of the source they crossmatched with
    # pilot survey
    pilot_agn = pd.read_pickle(f'{f}_pilot_pairs_agn')
    pilot_vast = pd.read_pickle(f'{f}_pilot_pairs_vast')
    # full survey (extragalactic pipeline run) - there were no crossmatches with the galactic fields
    extragalactic_agn = pd.read_pickle(f'{f}_eg_pairs_agn')
    extragalactic_vast = pd.read_pickle(f'{f}_eg_pairs_vast')
    # the racs low band data
    racs_agn = pd.read_pickle(f'{f}_racs_low_pairs_agn')
    racs_vast = pd.read_pickle(f'{f}_racs_low_pairs_vast')

    print(f'number of agn-pilot matches (may not be 1-to-1) {len(pilot_agn)}' )
    print(f'number of agn-extragalactic matches (may not be 1-to-1) {len(extragalactic_agn)}' )

    # figure out which source name to use
    if 'ObjID' in extragalactic_agn.columns:
        agn_id_str = 'ObjID'
    if not 'ObjID' in extragalactic_agn.columns:
        agn_id_str = 'WISEA'

    # first, remove sources from racs not in at least one vast catalog, since we don't care about those
    ids = np.concatenate([pilot_agn[agn_id_str].values,extragalactic_agn[agn_id_str].values])
    test = np.concatenate([pilot_vast[agn_id_str].values,extragalactic_vast[agn_id_str].values])
    # if these are different, we have an issue
    if not ids.all()==test.all():
        print('error')

    # correct for agn which are mapped to multiple distinct vast sources
    pilot_agn,pilot_vast = choose_vast_to_keep(pilot_agn,pilot_vast,agn_id_str)
    extragalactic_agn,extragalactic_vast = choose_vast_to_keep(extragalactic_agn,extragalactic_vast,agn_id_str)

    # repeat for vast sources mapped to multiple distinct agn, and merge the dataframes
    pilot = choose_agn_to_keep(pilot_agn,pilot_vast,agn_id_str)
    extragalactic = choose_agn_to_keep(extragalactic_agn,extragalactic_vast,agn_id_str)

    # now, address racs 
    # first, we want to drop any racs sources not in the pilot or extragalactic
    agn_ids = np.concatenate([pilot[agn_id_str].values,extragalactic[agn_id_str].values])
    agn_ids = np.unique(agn_ids)
    mask = racs_agn[agn_id_str].isin(agn_ids)
    racs_agn = racs_agn[mask]
    mask = racs_vast[agn_id_str].isin(agn_ids)
    racs_vast = racs_vast[mask]

    # now, merge the racs dfs
    racs_agn,racs_vast = choose_vast_to_keep(racs_agn,racs_vast,agn_id_str)
    racs = choose_agn_to_keep(racs_agn,racs_vast,agn_id_str)

    # print updates
    print('number of unique sources for current agn catalog [note, there may be cross-agn catalog contamination, where a vast source maps to a source in more than one agn catalog]')
    print(f'total number of unique sources in agn catalog: {len(agn_ids)}')
    print(f'number of sources detected in pilot survey: {len(pilot)}/{len(agn_ids)}')
    print(f'number of sources detected in extragalactic survey: {len(extragalactic)}/{len(agn_ids)}')
    print(f'number of sources detected in racs: {len(racs)}/{len(agn_ids)}')


    # now, we shall get the ids of all sources into one id dataframe, so we can track sources in multiple vast surveys
    pilot_ids = pilot[[agn_id_str,'vast_id']]
    extragalactic_ids = extragalactic[[agn_id_str,'vast_id']]
    racs_ids = racs[[agn_id_str,'vast_id']]
    
    # make a df with all ids
   
    # turn the id array into a dataframe
    df_dict = {agn_id_str:agn_ids}
    ids_df = pd.DataFrame(df_dict)

    # insert the null values for sources not in all vast surveys
    ids_df = force_square_dfs(ids_df,pilot_ids,agn_id_str,'pilot')
    ids_df = force_square_dfs(ids_df,extragalactic_ids,agn_id_str,'extragalactic')
    ids_df = force_square_dfs(ids_df,racs_ids,agn_id_str,'racs')

    
    # output the number of sources in the final df
    print(f'number of total unique sources: {len(ids_df)} (should be equal to {len(agn_ids)})')
    print(ids_df.shape)

    # return the merged dataframes, as well as the dataframe containing the id information
    return pilot,extragalactic,racs,ids_df

###################### function #5 ######################

# this function is called by function 6, and it needs to be run 9 times total, which is tedious to keep typing out (while function 6 only runs thrice)
# we compare two of the three dataframes for a given vast source in one vast survey
def compare_surveys(survey_str,ids1,ids2,df1,df2):
    # we want to exclude the null values so we don't accidentally remove them
    mask = ids1[f'vast_id_{survey_str}']>0
    ids1 = ids1[mask]
    mask = ids2[f'vast_id_{survey_str}']>0
    ids2 = ids2[mask]

    # check for any overlapping sources
    mask = ids1[f'vast_id_{survey_str}'].isin(ids2[f'vast_id_{survey_str}'])
    # if no overlapping sources, we're good to return
    if sum(mask)==0:
        print('yay! no cross agn catalog contamination')
        return df1,df2, ids1, ids2
    
    # if not, proceed
    print(f'{sum(mask)} vast sources found in 2 agn catalogs')
    # isolate the problem sources
    bad_ids = ids1[mask]
    # ids we will drop from df1
    bad_ids1 = np.array([])
    # ids we will drop from df2
    bad_ids2 = np.array([])
    # for each problem source, choose the closest on-sky agn as the true agn
    for i in range(len(bad_ids)):
        mask = df1['vast_id']==bad_ids.iloc[i]
        sep1 = df1[mask].sep
        mask = df2['vast_id']==bad_ids.iloc[i]
        sep2 = df2[mask].sep

        # compare the separations between the vast source and agn in the original crossmatch
        # if the source in df1 is further away, we will drop this source from df1
        if sep1>sep2:
            bad_ids1 = np.concatenate([bad_ids1,bad_ids.iloc[i]])
        # else, drop it from df2
        if sep2>=sep1:
            bad_ids2 = np.concatenate([bad_ids2,bad_ids.iloc[i]])

    # drop the problem sources from the main dfs
    mask = ~df1['vast_id'].isin(bad_ids1)
    df1 = df1[mask]
    mask = ~df2['vast_id'].isin(bad_ids2)
    df2 = df2[mask]

    # output the results
    print(f'removed {len(bad_ids1)+len(bad_ids2)} cross-agn-catalog contaminated sources')
    # update the id dataframe
    mask = ~ids1[f'vast_id_{survey_str}'].isin(bad_ids1)
    ids1 = ids1[mask]
    mask = ~ids2[f'vast_id_{survey_str}'].isin(bad_ids2)
    ids2 = ids2[mask]
    
    # return the updated dfs and id dfs
    return df1,df2, ids1,ids2

# singles - there are duplicates only in one survey
def check_for_duplicates(sdss_ids,sdss_wise_ids,wise_ids):
    sdss_ids = sdss_ids[['vast_id_pilot','vast_id_extragalactic','vast_id_racs']]
    sdss_wise_ids = sdss_wise_ids[['vast_id_pilot','vast_id_extragalactic','vast_id_racs']]
    wise_ids = wise_ids[['vast_id_pilot','vast_id_extragalactic','vast_id_racs']]
    all_ids = pd.concat([sdss_ids,sdss_wise_ids,wise_ids],axis=0)
    mask_pilot = (all_ids['vast_id_pilot']>0) & (all_ids.duplicated(subset='vast_id_pilot'))
    mask_extragalactic = (all_ids['vast_id_extragalactic']>0) & (all_ids.duplicated(subset='vast_id_extragalactic'))
    mask_racs = (all_ids['vast_id_racs']>0) & (all_ids.duplicated(subset='vast_id_racs'))
    print(f'duplicated ids pilot: {sum(mask_pilot)}')
    print(f'duplicated ids extragalactic: {sum(mask_extragalactic)}')
    print(f'duplicated ids racs: {sum(mask_racs)}')
    return all_ids, mask_pilot, mask_extragalactic, mask_racs



################ function #7 ##################

# possibly, agn from the three agn dataframes (sdss-only sources, sdss-and-wise sources, and wise sources) will be matched to the same vast source
# however, since we crossmatched the agn datasets FIRST (and sdss and wise have better resolution), this isn't physically possible, so we have to assign any
# polygamous vast sources to one agn across the three dfs

def cross_agn_survey_checks(sdss_ids,sdss_wise_ids,wise_ids):
    # there shouldn't be any vast sources found in more than one agn catalog, since we know these catalogs are mutually exclusive
    all_ids, mask_pilot, mask_extragalactic, mask_racs = check_for_duplicates(sdss_ids,sdss_wise_ids,wise_ids)

    if sum(mask_pilot)+sum(mask_extragalactic)+sum(mask_racs)==0:
        print('yay! no cross-survey contamination!')
        return sdss_ids, sdss_wise_ids, wise_ids


    # first, address pilot
    if sum(mask_pilot)>0:
        problem_pilot = all_ids[mask_pilot]
        mask = wise_ids['vast_id_pilot'].isin(problem_pilot['vast_id_pilot'])
        if sum(mask)>0:
            wise_ids = wise_ids[~mask]
        if sum(mask)==0:
            mask = sdss_wise_ids['vast_id_pilot'].isin(problem_pilot['vast_id_pilot'])
            sdss_wise_ids = sdss_wise_ids[~mask]

    # check if the issue is resolved
    all_ids, mask_pilot, mask_extragalactic, mask_racs = check_for_duplicates(sdss_ids,sdss_wise_ids,wise_ids)

    if sum(mask_pilot)+sum(mask_extragalactic)+sum(mask_racs)==0:
        print('yay! no cross-survey contamination after pilot duplicates removed!')
        return sdss_ids, sdss_wise_ids, wise_ids

    # second, address extragalactic
    if sum(mask_extragalactic)>0:
        problem_extragalactic = all_ids[mask_extragalactic]
        mask = wise_ids['vast_id_extragalactic'].isin(problem_extragalactic['vast_id_extragalactic'])
        if sum(mask)>0:
            wise_ids = wise_ids[~mask]
        if sum(mask)==0:
            mask = sdss_wise_ids['vast_id_extragalactic'].isin(problem_extragalactic['vast_id_extragalactic'])
            sdss_wise_ids = sdss_wise_ids[~mask]

      # check if the issue is resolved
    all_ids, mask_pilot, mask_extragalactic, mask_racs = check_for_duplicates(sdss_ids,sdss_wise_ids,wise_ids)

    if sum(mask_pilot)+sum(mask_extragalactic)+sum(mask_racs)==0:
        print('yay! no cross-survey contamination after pilot and extragalactic duplicates removed!')
        return sdss_ids, sdss_wise_ids, wise_ids

    # lastly, address racs
    if sum(mask_racs)>0:
        problem_racs = all_ids[mask_racs]
        mask = wise_ids['vast_id_racs'].isin(problem_racs['vast_id_racs'])
        if sum(mask)>0:
            wise_ids = wise_ids[~mask]
        if sum(mask)==0:
            mask = sdss_wise_ids['vast_id_racs'].isin(problem_racs['vast_id_racs'])
            sdss_wise_ids = sdss_wise_ids[~mask]

    # check if the issue is resolved
    all_ids, mask_pilot, mask_extragalactic, mask_racs = check_for_duplicates(sdss_ids,sdss_wise_ids,wise_ids)

    if sum(mask_pilot)+sum(mask_extragalactic)+sum(mask_racs)==0:
        print('yay! no cross-survey contamination after pilot and extragalactic and racs duplicates removed!')
        return sdss_ids, sdss_wise_ids, wise_ids

    if sum(mask_pilot)>0:
        problem_pilot = all_ids[mask_pilot]
        mask = sdss_wise_ids['vast_id_pilot'].isin(problem_pilot['vast_id_pilot'])
        sdss_wise_ids = sdss_wise_ids[mask]

     # check if the issue is resolved
    all_ids, mask_pilot, mask_extragalactic, mask_racs = check_for_duplicates(sdss_ids,sdss_wise_ids,wise_ids)
    if sum(mask_pilot)+sum(mask_extragalactic)+sum(mask_racs)==0:
        print('yay! no cross-survey contamination after fu pilot and extragalactic and racs duplicates removed!')
        return sdss_ids, sdss_wise_ids, wise_ids

    if sum(mask_extragalactic)>0:
        problem_extragalactic = all_ids[mask_extragalactic]
        mask = sdss_wise_ids['vast_id_extragalactic'].isin(problem_extragalactic['vast_id_extragalactic'])
        sdss_wise_ids = sdss_wise_ids[mask]

    # check if the issue is resolved
    all_ids, mask_pilot, mask_extragalactic, mask_racs = check_for_duplicates(sdss_ids,sdss_wise_ids,wise_ids)
    if sum(mask_pilot)+sum(mask_extragalactic)+sum(mask_racs)==0:
        print('yay! no cross-survey contamination after fu pilot and fu extragalactic and racs duplicates removed!')
        return sdss_ids, sdss_wise_ids, wise_ids

    if sum(mask_racs)>0:
        problem_racs = all_ids[mask_racs]
        mask = sdss_wise_ids['vast_id_racs'].isin(problem_racs['vast_id_racs'])
        sdss_wise_ids = sdss_wise_ids[mask]

    # check if the issue is resolved
    all_ids, mask_pilot, mask_extragalactic, mask_racs = check_for_duplicates(sdss_ids,sdss_wise_ids,wise_ids)
    if sum(mask_pilot)+sum(mask_extragalactic)+sum(mask_racs)==0:
        print('yay! cross-survey contamination finally is resolved!')
        return sdss_ids, sdss_wise_ids, wise_ids

    print('wtf is the issue here')
    return sdss_ids, sdss_wise_ids,wise_ids
    

#################### function #8 ######################

# read in the vast measurements and remove any bad or irrelevant data
# I have been removing the epochs affected by holography and epochs with siblings
def load_measurements(f, sdss_ids,sdss_wise_ids,wise_ids,survey_str):
    # combine all the vast ids from the three agn catalogs
    vast_ids = np.concatenate([sdss_ids[f'vast_id_{survey_str}'].values, sdss_wise_ids[f'vast_id_{survey_str}'].values, wise_ids[f'vast_id_{survey_str}'].values])
    
    # read in the measurement file
    measurements = pd.read_pickle(f)

    # keep only measurements corresponding to sources in catalog
    mask = measurements['source'].isin(vast_ids)
    measurements = measurements[mask]

    # only keep measurements with no siblings
    mask = measurements['has_siblings']==False
    measurements = measurements[mask]

    # output an update
    print(f'dropping {sum(~mask)}/{len(mask)} {survey_str} measurements contaminated with siblings')

    # now, drop the epochs affected by the holography issue
    mask = (measurements['time']>np.datetime64('2023-08-12')) & (measurements['time']<np.datetime64('2024-01-05'))
    measurements = measurements[~mask]
    
    return measurements

################################### function #9 ############################
# select sources based on their light curve morphology. I love to change the criteria obsessively 
# this function is for a single light curve

def morphology_cuts(measurements, cuts):
    
    # first, drop measurements with low-quality measurements - either negative or >15% error
    mask = (measurements['flux_peak']>0) & (measurements['flux_peak_err']/measurements['flux_peak']<0.15)
    measurements = measurements[mask]

    # sort the measurements by time in case they come out of the pickle file all wonky and out of order
    measurements = measurements.sort_values(by='time',ascending=True)

    # get the flux and flux error measurements
    fluxes = measurements['flux_peak'].values
    flux_errs = measurements['flux_peak_err'].values

    # first criteria - we must have three quality measurements
    if len(fluxes)<3:
        cuts[0] = cuts[0]+1
        return False, cuts    
    

    # second criteria - require one epochs to be above 2.4mJy (10x vast detection threshold)
    mask = measurements['flux_peak']>2.4
    if sum(mask)==0:
        cuts[1] = cuts[1]+1
        return False, cuts
    
    # third, we need the source to be actually variable - I am using this metric from Dykaar (2024)
    var_metric = abs(fluxes[::-1]-fluxes[0::])/np.sqrt(flux_errs[::-1]**2+flux_errs[0::]**2)
    if np.max(var_metric)<5:
        cuts[2] = cuts[2]+1
        return False, cuts
    
    # fourth, we want the maximum flux value to occur before the minimum, since we don't want decreasing light curves
    if np.argmax(fluxes)<np.argmin(fluxes):
        cuts[3] = cuts[3]+1
        return False, cuts
    
    # fifth, we want the flux to actually go up; the last flux value should be >1.2x the first AND the max>1.2x the min

    if np.max(fluxes)<1.2*np.min(fluxes) or fluxes[-1]<1.2*fluxes[0]:
        cuts[4] = cuts[4]+1
        return False, cuts
    
    # sixth, we want to eliminate zig zags
    max_flux_diff = np.max(fluxes)-np.min(fluxes)
    flux_changes = fluxes[::-1]-fluxes[0::]
    if np.max(abs(flux_changes))>0.8*max_flux_diff:
        cuts[5] = cuts[5]+1
        return False, cuts
    
    # if the source has made it this far, it passes! (for now)
    return True, cuts

##################### function 9 ###############################

# perform the light curve cuts on the sample
def light_curve_cuts(pdf, edf, rdf, pmdf,emdf,rmdf,ids):
    # let us know what we're doing
    print(f'beginning light curve cuts on {len(ids)} sources')

    # does the source pass the light curve cuts? Default is no
    good_sources = False*np.ones((len(ids),),dtype=bool)
    # this is to keep track of where the sources get cut
    cuts = np.zeros((6,),dtype=int)

    # loop through all the sources
    for i in range(len(ids)):
        # get the ids of the source in each survey
        pilot_id = ids['vast_id_pilot'].iloc[i]
        extragalactic_id = ids['vast_id_extragalactic'].iloc[i]
        racs_id = ids['vast_id_racs'].iloc[i]

        # get this source's pilot measurements, if they exist
        if pilot_id>0:
            mask = pmdf['source']==pilot_id
            measurements = pmdf[mask]

        # get the racs measurements, if they exist
        if racs_id>0:
            mask = rmdf['source']==racs_id
            try:
                measurements = pd.concat([measurements,rmdf[mask]])
            except:
                measurements = rmdf[mask]

        # finally, add the extragalactic measurements
        if extragalactic_id>0:
            mask = emdf['source']==extragalactic_id
            try:
                measurements = pd.concat([measurements,emdf[mask]])
            except:
                measurements = emdf[mask]

        # execute the morphology cuts
        
        good_sources[i], cuts = morphology_cuts(measurements,cuts)

    # output the results
    n_left = len(good_sources)
    print(f'number of sources in survey: {n_left}')
    n_left = n_left-cuts[0]
    print(f'sources with >2 selavy measurements: {n_left}')
    n_left = n_left-cuts[1]
    print(f'sources with >0 epochs above 2.4mJy: {n_left}')
    n_left = n_left-cuts[2]
    print(f'sources passing variability metric: {n_left}')
    n_left = n_left-cuts[3]
    print(f'sources with peak flux occuring after min flux: {n_left}')
    n_left = n_left-cuts[4]
    print(f'sources with flux that rises over time: {n_left}')
    n_left = n_left-cuts[5]
    print(f'sources with no sharp flux zig-zags: {n_left}')
    print(f'sources meeting all light curve morphology requirements: {sum(good_sources)}')
       
    ids = ids[good_sources]

    # update the dfs
    
    mask = pdf['vast_id'].isin(ids['vast_id_pilot'])
    pdf = pdf[mask]
    mask = edf['vast_id'].isin(ids['vast_id_extragalactic'])
    edf = edf[mask]
    mask = rdf['vast_id'].isin(ids['vast_id_racs'])
    rdf = rdf[mask]
    
    return ids, pdf, edf, rdf

################################ function 10 #############################
# insert information we need to calculate scintillation effects
def get_nu0(lat,long):
    if abs(lat)>50:
        nu0 = 8000 # MHz (8 GHZ)
        return nu0
    if abs(lat)>30:
        nu0 = 10000 # MHz
        return nu0
    if abs(lat)>15:
        nu0 = 15000 # MHz
        return nu0
    if abs(lat)>10:
        nu0 = 20000 # MHz
        return nu0
    if abs(lat)>5:
        if long<-25 or long >35:
            nu0 = 20000 # Mhz
            return nu0
    nu0 = 40000 # Mhz
    return nu0
    


######################################### function #11 ####################################
# nu is the observing frequency, theta_s is the size of the source, and df is the source df
def calculate_scintillation_parameters(df, nu,delta_nu):

    # determine the transition frequencies to use based on galactic coordinates
    # note, this needs to be updated with more recent electron density distributions
    ras = df['ra'].values
    decs = df['dec'].values
    coords = SkyCoord(ras,decs,unit=u.deg)
    coords = coords.galactic
    lats = coords.l
    longs = coords.b
    lats = lats.value
    longs = longs.value
    df = df.assign(lat_g=lats)
    df = df.assign(long_g=longs)
    
    nu_0s = np.zeros((len(coords),),dtype=float)
    
    for i in range(len(df)):
        nu_0s[i] = get_nu0(lats[i],longs[i])

    df = df.assign(nu0 = nu_0s)

    # now, calculate xi and determine the scattering regime
    # if xi <<1, we have weak scattering, and if xi >>1, we have strong scattering
    xis = (nu_0s/nu)**(17/10)
    df = df.assign(xi=xis)
    # for each source, determine regime and calculate m and t
    ms_weak = np.zeros((len(xis),),dtype=float)
    ts_weak = np.zeros((len(xis),),dtype=float)
    ms_riss = np.zeros((len(xis),),dtype=float)
    ts_riss = np.zeros((len(xis),),dtype=float)
    ms_diss = np.zeros((len(xis),),dtype=float)
    ts_diss = np.zeros((len(xis),),dtype=float)


    for i in range(len(xis)):
        if xis[i] > 0.3 and xis[i] < 1.7:
            print('warning! asymptotic limit of ISS may not be valid')
        # weak scattering regime
        if xis[i] <1:
           
            # modulation index for a point source
            ms_weak = xis[i]**(5/6)
            # timescale for a point source
            ts_weak = 2*np.sqrt(nu_0s[i]/nu) # in hours
    
        # if we are at the transition frequency
        if xis[i] == 1:
            print('observing frequency is equal to the transition frequency')
            ms_weak[i] = 1 # 100% modulation in this case
            ts_weak[i] = 2 # hours

        # strong scattering regime
        if xis[i]>=1:
            
            # first, check if we have the frequency resolution to detect diffractive ISS
            max_bandwidth = nu*xis[i]**(-2)
            if delta_nu<max_bandwidth:
                
                # assume as an upper limit, that the scale of the wavefront corrugations is not smaller than the source angular size
                ms_diss[i] = 1
                ts_diss[i] = 2*(nu/nu_0s[i])**(6/5) #in hours
             

            # now, for refractive ISS
            
            ms_riss[i] = (nu/nu_0s[i])**(17/30)
            ts_riss[i] = 2*(nu_0s[i]/nu)**(11/5) # in hours
           

    # now, insert these modulation indices and timescales into the df
    df = df.assign(m_weak=ms_weak)
    df = df.assign(t_weak=ts_weak)
    df = df.assign(m_riss=ms_riss)
    df = df.assign(t_riss=ts_riss)
    df = df.assign(m_diss=ms_diss)
    df = df.assign(t_diss=ts_diss)

    return df

################################# function 12 ##############################
# get ISS scintillation effects, for each vast survey

def scintillation(pdf,edf,rdf,nu,delta_nu):
    # first, we insert the galactic coordinates into each df
    pdf = calculate_scintillation_parameters(pdf,nu,delta_nu)
    edf = calculate_scintillation_parameters(edf,nu,delta_nu)
    rdf = calculate_scintillation_parameters(rdf,nu,delta_nu)

    return pdf,edf,rdf

def make_plots(df,measurements,out):
    measurements = measurements.sort_values(by='time',ascending=True)

    fluxes = measurements['flux_peak']
    time = measurements['time']
    flux_errs = measurements['flux_peak_err']
    median_flux = np.median(fluxes.values)

    # get the optical or infrared magnitude
    if 'PSFMag_g' in df.columns:
        g_mag = df['PSFMag_g'].iloc[0]
        # convert to flux
        ref_flux = 1000*3730*10**(-g_mag/2.5) # in mJy
        ref_label = 'g-flux'
    if not 'PSFMag_g' in df.columns:
        w1mag = df['W1mag'].iloc[0]
        # convert to flux
        ref_flux = 1000*309.54*10**(-w1mag/2.5) # in mJy
        ref_label = 'w1-flux'

    # get the scintillation info - should be the same for all three surveys, since it is dependent on coordinates and vast specs
    m_weak = df['m_weak'].iloc[0]
    m_riss = df['m_riss'].iloc[0]
    m_diss = df['m_diss'].iloc[0]
    t_weak = df['t_weak'].iloc[0]
    t_riss = df['t_riss'].iloc[0]
    t_diss = df['t_diss'].iloc[0]


    # create subplots and axes labels
    f,axs = plt.subplots(1,2)
    f.set_size_inches(16,10)
    axs[0].set_box_aspect(1)
    axs[1].set_box_aspect(1)
    axs[0].set_title('VAST Flux')
    axs[1].set_title('Radio-Loudness')
    axs[0].set_ylabel('Radio Flux [mJy]')
    axs[0].set_xlabel('Date')
    axs[1].set_ylabel(f'radio-flux/{ref_label}')
    axs[1].set_xlabel('Date')

    # calculate the % increase in flux
    increase = (np.max(fluxes.values)-np.min(fluxes.values))/np.min(fluxes.values)
    increase = int(100*increase)

    # plot the light curve
    axs[0].plot(time,fluxes,color='purple',label='radio flux')
    axs[0].errorbar(time,fluxes,yerr=flux_errs,fmt='o',color='k')
    axs[0].axhline(y=median_flux,color = 'g', linestyle = '--', label = 'median flux')
    axs[0].axhline(y=np.max(fluxes),color='r',linestyle='--',label=f'{increase}% in flux' )
    axs[0].axvline(x=np.datetime64('2023-08-12'),color = 'k',linestyle= ':', label = 'removed epochs')
    axs[0].axvline(x=np.datetime64('2024-01-05'),color= 'k',linestyle=':')
    
    # plot the radio-loudness
    radio_loudness = fluxes/ref_flux
    rl_error = radio_loudness*(flux_errs/fluxes)
    axs[1].plot(time,radio_loudness,color = 'm')
    axs[1].errorbar(time,radio_loudness, yerr = rl_error, fmt = 'o', color='k')

    # now, plot the magnitude of modulations due to scintillation
    if m_weak>0:
        max_flux = median_flux*(1+m_weak)
        min_flux = median_flux*(1-m_weak)
        axs[0].axhline(y=max_flux, color='m',linestyle='--',label='WISS')
        axs[0].axhline(y=min_flux, color = 'm',linestyle='--')
       
    if m_riss>0:
        max_flux = median_flux*(1+m_riss)
        min_flux = median_flux*(1-m_riss)
        axs[0].axhline(y=max_flux, color='m',linestyle='--',label='RISS')
        axs[0].axhline(y=min_flux, color = 'm',linestyle='--')
  
    if m_diss>0:
        max_flux = median_flux*(1+m_diss)
        min_flux = median_flux*(1-m_diss)
        axs[0].axhline(y=max_flux, color='b',linestyle='--',label='DISS')
        axs[0].axhline(y=min_flux, color = 'b',linestyle='--')

    source = df['vast_id'].iloc[0]
    axs[0].legend(loc=2)
    f.suptitle(f'VAST Light Curve \n {source}')
    plt.tight_layout()
    plt.savefig(f'{out}{str(source)}.png')
    plt.close()
    return

def in_three_surveys(edf,pmdf,emdf,rmdf,pid,eid,rid,out):
    # get the subsets of each df corresponding to the source
    mask = pmdf['source']==pid
    pmdf = pmdf[mask]
    mask = edf['vast_id']==eid
    edf = edf[mask]
    mask = emdf['source']==eid
    emdf = emdf[mask]
    mask = rmdf['source']==rid
    rmdf = rmdf[mask]

    # now, combine measurements
    measurements = pd.concat([pmdf,emdf,rmdf])
    make_plots(edf,measurements,out)
    return
 

def in_two_surveys(sdf,mdf1,mdf2,id1,id2,out):
    # get the subsets of each df corresponding to the source
    mask = mdf1['source']==id1
    mdf1 = mdf1[mask]
    mask = sdf['vast_id']==id1
    sdf = sdf[mask]
    mask = mdf2['source']==id2
    mdf2 = mdf2[mask]

    # now, combine measurements
    measurements = pd.concat([mdf1,mdf2])
    make_plots(sdf,measurements,out)
 
    return

def in_one_survey(df,mdf,id,out):
    mask = mdf['source']==id
    mdf = mdf[mask]
    mask = df['vast_id']==id
    df = df[mask]
    make_plots(df,mdf,out)
    return

##################################### function #13 #####################################
# plotting the light curves

def plot_light_curves(pdf,edf,rdf,pmdf,emdf,rmdf,ids,out):
    # first, gather all the measurements for a source
    for i in range(len(ids)):
        pilot_id = ids['vast_id_pilot'].iloc[i]
        extragalactic_id = ids['vast_id_extragalactic'].iloc[i]
        racs_id = ids['vast_id_racs'].iloc[i]

        # make plots if the source is in all three surveys
        if pilot_id>0 and extragalactic_id>0 and racs_id>0:
            in_three_surveys(edf,pmdf,emdf,rmdf,pilot_id,extragalactic_id,racs_id,out)

        # make plots if the source is in two surveys
        if pilot_id==0 and extragalactic_id>0 and racs_id>0:
            in_two_surveys(edf,emdf,rmdf,extragalactic_id,racs_id,out)

        if pilot_id>0 and extragalactic_id>0 and racs_id==0:
            in_two_surveys(edf,emdf,pmdf,extragalactic_id,pilot_id,out)

        if pilot_id>0 and extragalactic_id==0 and racs_id>0:
            in_two_surveys(pdf,pmdf,rmdf,pilot_id,racs_id,out)

        # make plots if the source is in one survey
        if pilot_id>0 and extragalactic_id==0 and racs_id==0:
            in_one_survey(pdf,pmdf,pilot_id,out)

        if pilot_id==0 and extragalactic_id>0 and racs_id==0:
            in_one_survey(edf,emdf,extragalactic_id,out)

    return

       
        

