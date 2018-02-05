# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import requests
import networkx as nx
from sklearn import cluster, covariance
import numpy as np
import scipy.sparse as spr
import scipy
import statsmodels.api as sm
# Importing library for GPR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
# Cross validation in time
from sklearn.model_selection import TimeSeriesSplit
#metrics
from sklearn.metrics import mean_squared_error, r2_score

###############
# Clear crude #
###############

# Remove all years for which i have not google trend observations (starts from 2004)
# However for 2004 we had 50 states, so we decided to drop also 2004
def clear_crude(c_dataframe):
    for year in range(1995, 2005):
        crude = c_dataframe[c_dataframe['Year'] != year]
        
    # Remove from crude all states for which i have not a google trend (not in the official 51 states)
    crude = crude[crude['LocationAbbr'] != 'GU']
    crude = crude[crude['LocationAbbr'] != 'PR']
    crude = crude[crude['LocationAbbr'] != 'VI']
    crude = crude[crude['LocationAbbr'] != 'US']
    crude = crude[crude['LocationAbbr'] != 'UW']
    
    # Remove all columns with no observations or useless content
    columns = ['Year', 'LocationAbbr', 'Data_Value', 'Low_Confidence_Limit', 'High_Confidence_Limit', 'Sample_Size']
    crude = crude[columns]
    
    # Re-set index
    crude = crude.set_index([[i for i in range(len(crude))]])
    
    return crude



######################
# Clear age_adjusted #
######################

def  clear_age_adjusted(aa_dataframe):
    # Remove from crude all states for which i have not a google trend (not in the official 51 states)
    age_adjusted = aa_dataframe[aa_dataframe['LocationAbbr'] != 'GU']
    age_adjusted = age_adjusted[age_adjusted['LocationAbbr'] != 'PR']
    age_adjusted = age_adjusted[age_adjusted['LocationAbbr'] != 'VI']
    
    # Remove all columns with no observations or useless content
    columns = ['Year', 'LocationAbbr','Data_Value', 'Low_Confidence_Limit', 'High_Confidence_Limit', 'Sample_Size']
    age_adjusted = age_adjusted[columns]
    
    # Re-set index
    age_adjusted = age_adjusted.set_index([[i for i in range(len(age_adjusted))]])
    
    return age_adjusted






def build_google_trend_dataframe(list_of_trends, keywords, starting_year, ending_year):        
    time_interval = (ending_year + 1 - starting_year)

    final = pd.DataFrame(index = [i for i in range(time_interval)]*51 ,columns = keywords)
    for data in range(len(list_of_trends)):
        trend = []
        for year in range(starting_year, ending_year + 1):
            for obs in list_of_trends[data][str(year)]:
                trend.append(obs)
        final[keywords[data]] = trend

    year_col = []
    for year in range(starting_year, ending_year + 1):
        for i in range(51):
            year_col.append(year)
    final['Year'] = year_col
    
    #Multi-index dataframe
    with open(os.path.abspath('BIN/data/states.txt'), 'r', encoding = 'UTF') as r:
        st = r.read()
    states = eval(st)

    iterat = [[int(i) for i in range(starting_year,ending_year+1)], states]
    index = pd.MultiIndex.from_product(iterat, names=['Year', 'states'])
    final = pd.DataFrame(final.values, index=index, columns=final.columns)
    return final



def get_google_trend_correlation_in_space(google_data, ground_truth, keywords):

    correlations = []    
    # We retrieve the correlation coefficent for every keyword, across every year we explored.
    for word in keywords:
        for year in range(2005, 2017):
            # Retrieve list of information for every state regarding the given keyword from the dataframe 
            # that we computed in the google trend step and store as list
            a = list(google_data[word][google_data['Year'] == year])
            # Retrieve list of information for every state regarding for the current year
            b = list(ground_truth['Data_Value'][ground_truth['Year'] == year])
            b = [float(i) for i in b]
            
            # Compute correlation coefficient
            corr = np.corrcoef(a, b)[1, 0]
            correlations.append([corr, word, year])
    
    correlations = pd.DataFrame(correlations, columns = ['Correlation', 'Keyword', 'Year'])
   
    #pivoting to print data
    correlations = correlations.pivot(index='Year',columns='Keyword')['Correlation']

    return correlations


def get_google_trend_correlation_in_time(crude,dic,all_keys):
    correlations = []   
    diff = pd.DataFrame(crude.set_index('Year').groupby('LocationAbbr')\
                        .apply(lambda x: x['Data_Value'].diff()))

    d = diff.stack().unstack(0)
    if crude.shape[0]>500:
        d.index = d.index.droplevel(level=1)

    # We retrieve the correlation coefficent for every keyword, across every state and keyword we explored.
    for word in all_keys:
        for state in list(dic[word].columns):
            from_date = max(dic[word].index[0],d.index[0])
            to_date = min(dic[word].index[-1],d.index[-1])

            a = list(dic[word].apply(lambda x: x.diff(),axis=0)[[state]]\
                     .loc[from_date+1:to_date].values.flatten())

            b = list(d[[state]].loc[from_date+1:to_date].values.flatten())
            b = [float(i) for i in b]

            # Compute correlation coefficient
            corr = np.corrcoef(a, b)[1, 0]
            correlations.append([corr, word, state])

    correlations = pd.DataFrame(correlations, columns = ['Correlation', 'Keyword', 'State'])

    #pivoting to print data
    correlations = correlations.pivot(index='Keyword',columns='State')['Correlation']
    
    return(correlations)




#######################################
# Download and clear US Census Bureau #
#######################################

def download_US_Census_Bureau(google_trend):
    # Download the dataset from US Census Bureau and save it as .xls
    #US_Census_Bureau = 'https://www2.census.gov/programs-surveys/demo/tables/p60/259/statepov.xls'
    #census_xls = requests.get(US_Census_Bureau)
    #with open('BIN/data/Census.xls', 'wb') as census_bureau:
     #   census_bureau.write(census_xls.content)
    
    # this will help to clear the downloaded excel file...
    states = list(google_trend.index)
    
    # Read excel file and save it as a dataframe
    census = pd.read_excel('BIN/data/Census.xls')
    census.columns = [str(i) for i in range(len(census.columns))]
    
    # Clear the dataset and save only informations about the 51 states for given time intervals(2014-2016, 2013-2014, 2015, 2016)
    census_data = []
    for state in census['0']:
        try:
            str_state = state.strip('.')
            str_state = str_state.strip('…')
            if str_state in states:
                y14_16 = list(census['1'][census['0'] == state])
                y13_14 = list(census['3'][census['0'] == state])
                y15_16 = list(census['5'][census['0'] == state])
                census_data.append([str_state, y14_16[0], y13_14[0], y15_16[0]])
        except:
            AttributeError
    
    # Save the final data for US Census Bureau in a dataframe
    census_data = pd.DataFrame(census_data, columns = ['state', 'y14_16', 'y13_14', 'y15_16'])
    return census_data


#################################
# Ground Truth Data Preparation #
#################################

def ground_truth_data_preparation(age_adjusted, crude):

    aa14_16 = []
    aa13_14 = []
    aa15_16 = []
    c14_16 = []
    c13_14 = []
    c15_16 = []
    
    # Extract from age_adj and crude the information about the year of interest and put them in the proper list
    # We have two list with 2-year-interval and one list with 3-year-interval
    for year in range(2013, 2017):
        aa = list(age_adjusted['Data_Value'][age_adjusted['Year'] == year])
        c = list(crude['Data_Value'][crude['Year'] == year])
        aa = [float(i) for i in aa]
        c = [float(i) for i in c]
        if year == 2013:
            aa13_14.append(aa)
            c13_14.append(c)
        elif year == 2014:
            aa13_14.append(aa)
            c13_14.append(c)
            aa14_16.append(aa)
            c14_16.append(c)
        elif year == 2015:
            aa14_16.append(aa)
            c14_16.append(c)
            aa15_16.append(aa)
            c15_16.append(c)
        elif year == 2016:
            aa14_16.append(aa)
            c14_16.append(c)
            aa15_16.append(aa)
            c15_16.append(c)
    
    # Compute the mean of the lists        
    aa13_14 = np.mean(aa13_14, axis = 0)
    aa14_16 = np.mean(aa14_16, axis = 0)
    aa15_16 = np.mean(aa15_16, axis = 0)
    c13_14 = np.mean(c13_14, axis = 0)
    c14_16 = np.mean(c14_16, axis = 0)
    c15_16 = np.mean(c15_16, axis = 0)
    
    
    all_prevalence = [aa13_14, aa14_16, aa15_16, c13_14, c14_16, c15_16]
    prevalence_df = pd.DataFrame(all_prevalence, index = ['aa13_14', 'aa14_16', 'aa15_16', 'c13_14', 'c14_16', 'c15_16'])
    prevalence_df = prevalence_df.transpose()
    return prevalence_df

################################
# US Census Bureau correlation #
################################

def census_correlation(ground_truth_data, census_data):
    final_corr = []
    for col_p in ground_truth_data.columns:
        for col_c in census_data.columns:
            # Compute the correlation between two columns only if they have the same time interval
            cc = col_c.strip('y')
            cp = col_p.strip('aa')
            cp = cp.strip('c')
            if cp == cc:
                corr = np.corrcoef(ground_truth_data[col_p], census_data[col_c])[0][1]
                final_corr.append([col_c, col_p, corr])
    
    final = pd.DataFrame(final_corr, columns = ['US Census Bureau year', 'Ground Truth year', 'Correlation'])
    
    #pivoting to print data
    final = final.pivot(index='US Census Bureau year',columns='Ground Truth year')['Correlation']
    return final




################################
# correlation between states   #
################################


def correlation_between_states(crude, states, threshold):

    correlations = []    
    # We retrieve the correlation coefficent for every keyword, across every year we explored.
    diff = pd.DataFrame(crude.set_index('Year').groupby('LocationAbbr')\
                        .apply(lambda x: x['Data_Value'].diff()))

    d = diff.stack().unstack(0)
    if crude.shape[0]>500:
        d.index = d.index.droplevel(level=1)

        # Calculate the correlation between individuals. We have to transpose first, because the corr function calculate the pairwise correlations between columns.
    corr = d.corr()
    corr = pd.DataFrame(corr.values,columns=states, index=states)

    # Transform it in a links data frame (3 columns only):
    links = corr.stack().reset_index()
    links.columns = ['var1', 'var2','value']

    # Keep only correlation over a threshold and remove self correlation (cor(A,A)=1)
    links_filtered=links.loc[ (links['value'] > threshold) & (links['var1'] != links['var2']) ]
    
    
    # Build your graph
    G=nx.from_pandas_dataframe(links_filtered, 'var1', 'var2')
    #pivoting to print data
    links = links.pivot(index='var1',columns='var2')['value']
    
    #computing the graph partition
    l = nx.laplacian_matrix(G=G)
    L = spr.csr_matrix(l).toarray()
    eig , vec = np.linalg.eig(L)
    sort = np.argsort(np.diff(np.abs(eig)))
    i = 3
    v =vec[:,sort[-i]].real/sum(vec[:,sort[-i]].real)
    label = v.astype(int)

    return (links,G, label)


def correlation_between_states_2(crude, states, threshold):

    correlations = []    
    # We retrieve the correlation coefficent for every keyword, across every year we explored.
    diff = pd.DataFrame(crude.set_index('Year').groupby('LocationAbbr')\
                        .apply(lambda x: x['Data_Value'].diff()))

    d = diff.stack().unstack(0)
    if crude.shape[0]>500:
        d.index = d.index.droplevel(level=1)

    # Learn a graphical structure from the correlations
    edge_model = covariance.GraphLassoCV()

    # standardize the time series: using correlations rather than covariance
    # is more efficient for structure recovery
    d = d.loc[1998:]
    corr = d.values
    X = corr.copy()
    X /= X.std(axis=0)
    edge_model.fit(X)
    
    # Transform it in a links data frame (3 columns only):
    corr = pd.DataFrame(edge_model.covariance_, index=states, columns=states)
    links = corr.stack().reset_index()
    links.columns = ['var1', 'var2','value']

    # Keep only correlation over a threshold and remove self correlation (cor(A,A)=1)
    links_filtered=links.loc[ (links['value'] > threshold) & (links['var1'] != links['var2']) ]


    # Build your graph
    G=nx.from_pandas_dataframe(links_filtered, 'var1', 'var2')
    #pivoting to print data
    links = links.pivot(index='var1',columns='var2')['value']
    
        
    #computing the graph partition
    l = nx.laplacian_matrix(G=G)
    L = spr.csr_matrix(l).toarray()
    eig , vec = np.linalg.eig(L)
    sort = np.argsort(np.diff(np.abs(eig)))
    i = 1
    v =vec[:,sort[-i]].real/sum(vec[:,sort[-i]].real)
    label = v.astype(int)

    return (links,G, label)




################################
# regression functions         #
################################


def prepare_data_simple_lasso(crd,google,dic, keywords, all_keys,
                              census = [],states=[], from_ = 2004, to_=2016, ins =[]):
    #insurance cleaning
    ins.State = states
    ins.drop('Insurance', axis=1, inplace=True)
    ins = ins.transpose()
    ins.columns = states
    ins.drop('State', inplace=True)
    ins = ins.stack()
    #census cleanin
    census.set_index('state', inplace=True)
    census.index = states
    cen = census.transpose()
    cen.drop('y14_16', inplace=True)
    cen.index = [2013,2016]
    cen = cen.stack()
    # data
    Y = crd.set_index(['Year','LocationAbbr'])[['Data_Value']]
    X_1 = google.drop(['Year'], axis=1)
    X_1.columns = [s+'_region' for s in keywords]
    X_2 = pd.concat([dic[s].stack() for s in all_keys],axis=1)
    X_2.columns = [s+'_time' for s in all_keys]
    
    if census.shape[0]>0:
        all_data = pd.concat([Y.loc[from_:], X_1.loc[from_:to_], X_2.loc[from_:to_], cen],axis=1,
                             ignore_index = True)
        if ins.shape[0]>0:
            all_data = pd.concat([all_data,ins],axis = 1, ignore_index= True)
    all_data = pd.concat([Y.loc[from_:], X_1.loc[from_:to_], X_2.loc[from_:to_]],axis=1)

    # clean and fill Nan
    index = all_data.Data_Value.index[all_data.Data_Value.apply(np.isnan)]
    all_data.drop(index, inplace=True)
    all_data = all_data.fillna(all_data.mean(),axis=0,inplace=True)

    # data
    X = all_data.drop('Data_Value', axis=1).values
    y = all_data.Data_Value.values
    return(X,y)

def prepare_data_for_state(crd,dic,all_keys,st):
    # X
    X_2 = pd.concat([dic[s].stack() for s in all_keys],axis=1)
    X_2.columns = [s+'_time' for s in all_keys]
    state = X_2.xs(st, level=1)
    ind_col_null = X_2.columns[(state.sum()==0)]
    state = state.drop(ind_col_null, axis = 1)
    state = state.interpolate(method='spline',order = 3, s = 0)
    state_X = state.fillna(state.mean()).loc[2005:2016]

    # y
    Y = crd.set_index(['Year','LocationAbbr'])[['Data_Value']]
    state_y = Y.xs(st,level='LocationAbbr').loc[2005:2016]
    return(state_X.values, state_y.values)


def regression_for_states(X_s,y_s):
    X_s = sm.add_constant(X_s)
    from_ = 5
    prd = [np.mean(y_s) for _ in range(from_)]
    #tscv = TimeSeriesSplit(n_splits= len(y_s)-from_)
    #for train_index, test_index in tscv.split(X_s):
     #   X_train, X_test = X_s[train_index], X_s[test_index]
      #  y_train, y_test = y_s[train_index], y_s[test_index]
    mod = sm.RecursiveLS(y_s,X_s)
    res = mod.fit()
    #prd.append(res.predict(X_test))
    prd = [res.predicted_state[i,i] for i in range(y_s.shape[0])]

    resid = y_s.flatten() -  prd

    prd = np.array(prd)
    resid = y_s.flatten() -  prd
    
    X = np.arange(len(y_s))[:, np.newaxis]

    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))

    gp = GaussianProcessRegressor(kernel=kernel,
                                  alpha=0.0).fit(X, resid)

    X_ = np.linspace(0,12,12)
    y_mean, y_cov = gp.predict(X_[:, np.newaxis], return_cov=True)
    return({'pred': prd + y_mean, 'var': y_cov})




################################
# clear insurance              #
################################


def clear_insurance(type_ins, value):
    
    insurance = pd.read_excel('BIN/data/hic04_acs.xls')
    insurance.columns = [i for i in range(len(insurance.columns))]
    
    if value == 'p':
        columns = [0, 1] + [i for i in range(4, 38, 4)]
    else:   
        columns = [0, 1] + [i for i in range(2, 38, 4)]
        
    insurance = insurance[columns]
    
    column_names = ['State', 'Insurance'] + sorted([i for i in range(2008, 2017)], reverse = True)
    insurance.columns = column_names
    
    insurance = insurance.drop(insurance.index[[i for i in range(576, 580)]])
    insurance = insurance.drop(insurance.index[[i for i in range(15)]])
    insurance = insurance.set_index([[i for i in range(len(insurance))]])
    
    for i in range(len(insurance['State'])):
        if insurance.loc[i, 'State'] is np.nan:
            insurance.loc[i, 'State'] = insurance.loc[i - 1, 'State']
    
    type_of_insurances = []
    for i in range(11):
        type_of_insurances.append(insurance.loc[i, 'Insurance'])
    
    type_of_insurances.remove(type_ins)
    
    for i in type_of_insurances:
        insurance = insurance[insurance['Insurance'] != i]
        
    return insurance