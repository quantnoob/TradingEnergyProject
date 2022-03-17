import pandas as pd
import numpy as np
import datetime
import pandas_market_calendars as mcal
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from Sparse_jump import *
from sklearn.preprocessing import StandardScaler
US_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
from sklearn.neighbors import KNeighborsClassifier
from TradingTools.TradingFunc import *

def process_data():
    ## quaterly data, already the gdp growth forecast
    #gdpNow = pd.read_csv('./data/GDPNOW_2011_2022.csv').rename(
    #        columns = {'DATE':'date','GDPNOW':'gdp now'}
    #        )
    #gdpNow['date'] = pd.to_datetime(gdpNow['date'])
    #gdpNow.sort_values('date',inplace=True)

    gdp = pd.read_excel('./data/GDPC1.xls',skiprows=10)
    gdp.columns = ['date','gdp']
    gdp['date'] = pd.to_datetime(gdp['date'])
    gdp.sort_values('date',inplace=True)
    gdp['gdp growth'] = gdp['gdp'].pct_change(4)
    del gdp['gdp']

    ## monthly spotprice
    InflationRate = pd.read_csv('./data/1yrExpectedInflationRate.csv').rename(
    columns={'DATE':'date','EXPINF1YR':'IR'})
    InflationRate['date'] = pd.to_datetime(InflationRate['date'])
    InflationRate.sort_values('date',inplace=True)
    InflationRate['IR_y2y_ratio'] = InflationRate['IR'].pct_change(12)

    CPI = pd.read_csv('./data/cpi.csv').rename(columns={'DATE':'date','CPIAUCSL':'CPI'})
    CPI['date'] = pd.to_datetime(CPI['date'])
    CPI.sort_values('date',inplace=True)
    CPI['CPI_y2y_ratio'] = CPI['CPI'].pct_change(12)

    pmi = pd.read_csv('./data/ISM-MAN_PMI.csv').sort_values('Date').rename(
            columns = {'Date':'date','PMI':'PMI'}
        )
    pmi['date'] = pd.to_datetime(pmi['date'])
    pmi.sort_values('date',inplace=True)
    pmi['pmi_y2y_ratio'] = pmi['PMI'].pct_change(12)

    ## weekly data
    nfci = pd.read_csv('./data/NFCI_1971_2022.csv').rename(
            columns = {'DATE':'date','NFCI':'NFCI'}
        )
    nfci['date'] = pd.to_datetime(nfci['date'])
    nfci.sort_values('date',inplace=True)

    Inventory = pd.read_csv('./data/Inventory.csv',skiprows=4).rename(
    columns = {
        'Week of':'date','Weekly U.S. Ending Stocks of Crude Oil and Petroleum Products Thousand Barrels':'Inventory'})
    Inventory['date'] = pd.to_datetime(Inventory['date'])
    Inventory.sort_values('date',inplace=True)
    Inventory['Inventory_y2y_ratio'] = Inventory['Inventory'].pct_change(12)
    Inventory['Inventory_mom_3_12'] = Inventory['Inventory'].rolling(3).mean() - Inventory['Inventory'].rolling(12).mean() 
    ## daily data
    USDIndex = pd.read_csv('./data/usd_index.csv')[['Date','Adj Close']].rename(
        columns = {'Date':'date','Adj Close':'USD'}
    )
    USDIndex['date'] = pd.to_datetime(USDIndex['date'])
    USDIndex.sort_values('date',inplace=True)
    USDIndex.dropna(inplace=True)
    USDIndex['usd_mom_20_125'] = USDIndex['USD'].rolling(20).mean() - USDIndex['USD'].rolling(125).mean()

    VIX = pd.read_csv('./data/VIX.csv').rename(
                columns = {'DATE':'date','VIXCLS':'VIX'}
            )
    VIX.drop(VIX[VIX['VIX'] == '.'].index, inplace=True)
    VIX['date'] = pd.to_datetime(VIX['date'])
    VIX['VIX'] = pd.to_numeric(VIX['VIX'])
    VIX.sort_values('date',inplace=True)

    wti = pd.read_excel('./data/PET_PRI_FUT_S1_D.xls', sheet_name=1,skiprows=2)
    wti.columns = ['date','F1','F2','F3','F4']
    wti['date'] = pd.to_datetime(wti['date'])
    wti = wti[wti['date']>=pd.Timestamp('20051201')]
    wti.sort_values('date',inplace=True)
    wti['dollar_ret'] = wti['F1'].diff()
    wti['dollar_vol_monthly'] = wti['dollar_ret'].rolling(20).std() * np.sqrt(252/20)
    wti['carry'] = wti['F1'] - wti['F4']
    wti['carry_mom_1_5'] = wti['carry'] - wti['carry'].rolling(5).mean()
    wti['carry_mom_5_20'] = wti['carry'].rolling(5).mean() - wti['carry'].rolling(20).mean()
    wti['carry_mom_20_125'] = wti['carry'].rolling(20).mean() - wti['carry'].rolling(125).mean()

    spotprice = pd.read_excel('./data/WTIprice.xls',skiprows=10)
    spotprice.columns = ['date','F1']
    spotprice['date'] = pd.to_datetime(spotprice['date'])
    spotprice.sort_values('date',inplace=True)
    spotprice['mom_1_5'] = spotprice['F1'] - spotprice['F1'].rolling(5,closed = 'both').mean()
    spotprice['mom_5_20'] = spotprice['F1'].rolling(5,closed = 'both').mean() - spotprice['F1'].rolling(20,closed = 'both').mean()
    spotprice['mom_20_250'] = spotprice['F1'].rolling(20,closed = 'both').mean() - spotprice['F1'].rolling(250,closed = 'both').mean()
    del spotprice['F1']
    ## merge data
    monthy_data = InflationRate.merge(CPI, how='outer').merge(pmi, how='outer')\
                .sort_values('date')
    #monthy_data = monthy_data[monthy_data['date']>=pd.Timestamp('20051201')]
    monthy_data.set_index('date',inplace=True)

    weekly_data = nfci.merge(Inventory, how='outer')\
                .sort_values('date')
    #weekly_data = weekly_data[weekly_data['date']>=pd.Timestamp('20051201')]
    weekly_data.set_index('date',inplace=True)

    daily_data = USDIndex.merge(wti,how='outer').merge(VIX, how='outer').merge(spotprice,how='outer').sort_values('date')
    #daily_data = daily_data[daily_data['date']>=pd.Timestamp('20051201')]
    daily_data.set_index('date',inplace=True)

    quaterly_data = gdp.sort_values('date')
    #quaterly_data = quaterly_data[quaterly_data['date']>=pd.Timestamp('20051201')]
    quaterly_data.set_index('date',inplace = True)

    ## forward data filling
    data = pd.concat([quaterly_data,monthy_data,
                             weekly_data,daily_data],axis=1)
    data.dropna(how='all', inplace=True)
    data.fillna(method='ffill', inplace=True)
    data.dropna(how='all',axis=0, subset = ['F1','F2','F3','F4'], inplace = True)
    data = data[data.index>=pd.Timestamp('20051201')]
    #data = data[[i for i in data.columns if not i in ['CPI','F1','F2','F3','F4','Inventory']]]
    return data

def knn_jump(n_states = 3, jump_penalty=1,n_neighbors = 20):
    data = process_data()
    data = data.dropna(how='any')
    data_test = data[data.index>pd.Timestamp('20140101')]
    data_train = data[data.index<=pd.Timestamp('20140101')]

    features_name_macro_exluded = [i for i in data.columns if not i in ['F1','F2','F3','F4','Inventory',
                'PMI','NFCI','CPI','IR','gdp growth','USD']]

    data_selected = data_train[features_name_macro_exluded]

    scaler = StandardScaler()
    dt = scaler.fit_transform(data_selected)
    states, feature_weights = sparse_jump(dt, n_states=n_states, max_features=20,jump_penalty=jump_penalty)
    pd.DataFrame(feature_weights/sum(feature_weights), index = data_selected.columns).plot(kind='bar', figsize=(12,5))
    data_selected['states'] = states
    # we want to see the summary statistics of different states, w.r.t different states
    # states are ordered by mom_20_250 signal
    important_features = data_selected.columns[np.where(feature_weights>=0.025)]
    summary_table_macro_excluded = data_selected[important_features.to_list()+['states']].groupby('states').mean()\
                                    .sort_values('mom_20_250').T
    # we make an order for the states, bull +1, bear -1, static 0
    print('original states(ordered by mom_20_250 signal) is: {}'.format(summary_table_macro_excluded.columns.tolist()))
    print('The meaningful states should be: [-1, 0, 1]')
    states_map = dict(zip(summary_table_macro_excluded.columns.tolist(), [-1,0,1]))
    print(states_map)
    data_selected['True_state'] = data_selected['states'].apply(lambda x: states_map[x])
    plt.suptitle('Training set result, parameters: n_states={}, jump_penalty={}'.format(n_states, jump_penalty))
    fig, axes = plt.subplots(ncols=1,nrows=2, sharex=False,figsize=(20,12))
    axes[0].scatter(data_train.index.to_list(),data_train['F1'],c=data_selected['True_state'].values, s = 0.2)
    axes[1].plot(data_train.index.to_list(),data_selected['True_state'].values)
    # standardize the training data, use the previously fitted scaler to transform test data
    X_train = scaler.fit_transform(data_selected[features_name_macro_exluded])
    X_test = scaler.transform(data_test[features_name_macro_exluded])
    y_train = data_selected['True_state'].values

    # knn for state prediction
    knn = KNeighborsClassifier(n_neighbors=n_neighbors,weights='uniform')
    knn_model = knn.fit(X_train,y_train)

    pred = knn_model.predict(X_test)

    data_test['s'] = pred

    wti_data = pd.read_excel('./data/PET_PRI_FUT_S1_D.xls', sheet_name=1,skiprows=2)
    exp_date = pd.read_excel('./data/Expiry Calendar HW1.xlsx')

    wti_data = processing_data(wti_data,type = 'WTI')
    wti_PL = get_everyday_position(wti_data.copy(), exp_date, type='WTI',consider_transaction=False)

    wti_daily_PL = wti_PL[['Date','dF']]
    wti_daily_PL.columns = ['Date','daily_PL']
    wti_daily_PL.set_index('Date',inplace=True)

    signal = data_test['s'].reset_index()
    signal.columns = ['Date','signal']
    signal.set_index('Date',inplace=True)

    PL_signal = pd.concat([signal, wti_daily_PL],axis=1)
    PL_signal.dropna(how='any',inplace=True)

    # if market state is 0, -1, signal is -1, short the futures
    # if market state is 1, signal is 1, long the future
    PL_signal['signal'] = PL_signal['signal'].apply(lambda x: x if x>0 else -1)

    return PL_signal