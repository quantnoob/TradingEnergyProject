# %%
import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
import matplotlib.pyplot as plt

# %% [markdown]
# ##### Load Data

# %%


# %% [markdown]
# ##### Functions Needed

# %%
def get_rolling_date_before_expiration(dt, data):
    '''
    return five business days before given date, considering US FederalHoliday
    '''
    dt_curr = dt-pd.offsets.CustomBusinessDay(n=5, calendar=USFederalHolidayCalendar())

    return  data[data['Date']<=dt_curr]['Date'].iloc[-1]
###
#print(rolling_date_before_expiration(datetime.datetime.today()))

def get_nearest_next_expiration_date(dt, exp_date,type=None):
    '''
    return the nearest future expiration date after given date
    '''
    exp_dt = exp_date[type]
    exp_dt = exp_dt[exp_dt>=dt]
    return exp_dt.iloc[0]


def processing_data(dt,type = None):
    '''
    processing rbob data, considering date range, nan value, and column names
    '''
    dt['Date'] = pd.to_datetime(dt['Date'])
    dt = dt[np.logical_and(dt['Date']>=pd.Timestamp('20100101'),
                            dt['Date']<=pd.Timestamp('20211231'))]
    dt = dt.sort_values('Date')
    if type == 'RBOB':
        dt.fillna(2.686)
    dt.columns = ['Date','F1','F2','F3','F4']
    dt.reset_index(drop=True, inplace=True)
    return dt

def get_everyday_position(data_frame, exp_date,type = None, consider_transaction=True):
    '''
    find everyday position of data (WTI, RBOB),
    based on roll adjusted long only strategy.
    if the transaction cost is 0, the cummulative Pt is equity line
    (implied by the sample answer given in this course)

    input:
        data_frame: data
        exp_date:   expiration calendar
        type: 'RBOB' or 'WTI'
    return: new dataframe containing P&L and cumulative P&L
    '''
    data_frame['transaction_occurs'] = False
    # all ft is f1, expept the time period between T-5 and T, closed boundary
    # we will adjust ft latter in this program
    data_frame['Ft'] = data_frame['F1']
    # chosse transaction cost
    if consider_transaction:
        if type == 'WTI':
            transaction_cost = 0.01
        else:
            transaction_cost = 0.0005
    else:
        transaction_cost = 0
    

    dt = data_frame['Date'].iloc[0]
    rolling_date_idx_lst = []
    while(dt<=data_frame['Date'].iloc[-1] and dt<exp_date[type].iloc[-1]):
        ## get next expiration date
        next_expiration_date = get_nearest_next_expiration_date(dt,
                                    exp_date, type = type)
        ## calculate rolling date from expiration date, -5 business date
        rolling_date = get_rolling_date_before_expiration(next_expiration_date, data_frame)
        #print(rolling_date)
        rolling_date_idx = data_frame[data_frame['Date']==rolling_date].index.to_list()
        rolling_date_idx_lst.append(rolling_date_idx)
        ## set ft = f2 for period between T-5 and T, closed boundary, then we further fix T-5
        ## on T-5 df = f1(T-5) - f1(T-6)
        ## on T-4 df = f2(T-4) - f2(T-5)
        ## on T+1 df = f1(T+1) - f2(T)
        ## in order to calculate P&L
        data_frame.loc[rolling_date_idx,'transaction_occurs'] = True
        change_idx = data_frame[np.logical_and(data_frame['Date']>=rolling_date,
                                    data_frame['Date']<=next_expiration_date)].index.to_list()
        ## adjust ft using f2
        data_frame.loc[change_idx,'Ft'] = data_frame.loc[change_idx,'F2']
        ## get next date
        if data_frame['Date'].iloc[-1]<=next_expiration_date:
            break
        dt = data_frame[data_frame['Date']>next_expiration_date]['Date'].iloc[0]
    ## in this way on T-5 the daily P&L is calculated as: f2(T-5) - f1(T-6) which need to be fixed
    ## in order to satisfy on T-5 df = f1(T-5) - f1(T-6)
    data_frame['dF'] = data_frame['Ft'].diff(1)
    data_frame['dF'].fillna(0, inplace = True)
    
    # considering transaction cost and rolling yield
    # calculate P&L
    transaction_idx = data_frame[data_frame['transaction_occurs']==True].index.to_list()
     # fix the P&L on T-5, also when calculating P&L
    # we take the transaction cost into consideration
    data_frame.loc[transaction_idx,'dF'] = data_frame.loc[transaction_idx,'dF']\
                                            - data_frame.loc[transaction_idx, 'F2']\
                                            + data_frame.loc[transaction_idx, 'F1']\
                                        -2 * transaction_cost
    # Equity Line roll-adjusted
    data_frame['Pt'] = data_frame['dF']
    data_frame['Pt'].loc[0] = data_frame['F1'].iloc[0]
    data_frame['Pt'] = data_frame['Pt'].cumsum()
    return data_frame

# %%
def calcualte_momentum_signals(equity_line, n):
    '''
        As mentioned in lecture, to avoid jump, we calcualte momentum using equity line.
        The trading signal's date is the date the signal is computed.
        params:
            equity_line: dataframe with index as date
            n          : consider n-day momentum
        output:
            signal     : trading signal with index as date. (int: 0: close position, 1: buy, -1: short sell)
    '''
    def func(x):
        # no need to consider equals to zero condition, since in practice there is no perfect equal
        return 1 if (x[-1] - (1/len(x))*x.sum()) > 0 else -1
    
    signal = equity_line.rolling(window=n).apply(func).fillna(0)
    signal.columns = ['signal']
    return signal

def calcualte_carry_signals(data,epsilon):
    '''
        Calculate carry signal based on the closing price of F1(t)-F4(t) everyday.
        Although we don't hold the F1 position on T-5 to T,  where T is expiration day,
        we can still use F1(t) - F4(t) as the signal indicator.

        params: 
            data: dataframe for wti_data
            epsilon : threshold
        output:
            signal  : trading signal with index as date. (int: 0: close position, 1: buy, -1: short sell)
    '''
    def func(x, epsilon):
        if x > epsilon:
            return 1
        elif x< - epsilon:
            return -1
        else:
            return 0
    data['F1-F4'] = data['F1'] - data['F4']
    signal = data['F1-F4'].apply(lambda x: func(x, epsilon))
    signal.index = data['Date']
    signal = signal.to_frame()
    signal.columns = ['signal']
    return signal

# %%
def trading_strategy_backtest(daily_PL, signal):
    '''
        This is a general backtest algorithm for calculating PL of trading signals:

            Short selling logic is: since the short selling is just the opposite position of long position,
                                    the daily p&l need to multiplied by -1.
                                    As for the rolling actions, we only need to close the short position on F1,
                                    and open another short position on F1 on T-5, in the same way as long strategy does.

            Transaction cost logic is: from +1/-1 to 0, single-side transaction, from -1 to +1, double-side transaction

            P&L calculation logic is: the trading signal is generated on day t and the transaction is completed at the
                                    closing price on day t, so that the P&L at t+1 is the long daily P&L times the position
                                    signal on day t, which is: dF_long_short(t+1) = dF_long(t+1) * signal(t)

        params:
            daily_PL: DataFrame of The roll adjusted daily P&L for long position, without considering any transaction cost
            signal  : trading signals. (int: 0: close position, 1: buy, -1: short sell)
        output:
            cum_PL  : cummulated p&l
    '''
    def calculate_transaction(x, transaction_cost):
        if x[0] == x[1]:
            return 0
        elif abs(x[0]-x[1]) == 1:
            return transaction_cost
        else:
            return transaction_cost * 2

    transaction_cost = 0.01
    data = pd.concat([daily_PL, signal], axis=1)
    data['effective_position'] = data['signal'].shift(1)
    data['transaction_cost'] = data['signal'].rolling(window=2).apply(lambda x: calculate_transaction(x, transaction_cost))
    ## because we don't know the signal indicated position at the beginning
    data['trading_daily_PL'] = data['daily_PL'] * data['effective_position']
    data['trading_PL-t_cost'] = data['trading_daily_PL'] - data['transaction_cost']
    data['Cummulated P&L'] = data['trading_PL-t_cost'].cumsum()
    data['Cummulated P&L'].fillna(0.0, inplace=True)
    data['trading_PL-t_cost'].fillna(0.0, inplace=True)
    return data

def calculate_summary_stats(data):
    '''
    calculate summary statistics such as P&L, SR, MDD

    params:
        data: data frame containing daily PL, etc.
    Output:
        result : (Average Annual P&L, Sharpe Ratio, Maximum Drawdown)
    '''
    Annualized_PL = ((data['Cummulated P&L'][-1])*1000*100/(len(data)))*250
    Annualized_std = (data['trading_PL-t_cost']*1000*100).std() * np.sqrt(250)
    SR = Annualized_PL/Annualized_std

    def calculate_mdd(arr):
        '''
        calcluate maximum drawdown
        '''
        _max = arr[0]
        _dd = []
        for i in range(len(arr)):
            if arr[i]>=_max:
                _dd.append(0)
                _max = arr[i]
            else:
                _dd.append(_max - arr[i])
        return max(_dd)
    MDD = calculate_mdd((1000*100*data['Cummulated P&L']).to_list())
    return (Annualized_PL, SR, MDD)



def find_optimal_parameter(equity_line, wti_data, wti_daily_PL, params, signal = None):
    signal_lst = []
    result_lst = []
    if signal=='Momentum':
        for param in params:
            _signal = calcualte_momentum_signals(equity_line,param)
            _result = trading_strategy_backtest(wti_daily_PL, _signal)
            signal_lst.append(_signal)
            result_lst.append(_result)
    elif signal == 'Carry':
        for param in params:
            _signal = calcualte_carry_signals(wti_data, param)
            _result = trading_strategy_backtest(wti_daily_PL, _signal)
            signal_lst.append(_signal)
            result_lst.append(_result)
    else:
        return
    SR_lst = []

    for res in result_lst:
        _, SR, _ = calculate_summary_stats(res)
        SR_lst.append(SR)
    
    SR_max = max(SR_lst)
    id_SR_max = SR_lst.index(SR_max)
    optimal_param = params[id_SR_max]

    if signal == 'Momentum':
        print('The optimal parameter is: n = {}'.format(optimal_param))
        print('The optimal Sharpe Ratio is: {}'.format(SR_max))

    if signal == 'Carry':
        print('The optimal parameter is: e = {}'.format(optimal_param))
        print('The optimal Sharpe Ratio is: {}'.format(SR_max))
        

def find_optimal_weight(result_momentum, result_carry):
    '''
    find the optimal weights of combination of momenteum and carry strategy
    '''
    new_result = pd.DataFrame()
    new_result.index = result_momentum.index

    w = np.arange(0,1.01,0.01)
    SR_lst = []
    for _w in w:
        new_result['Cummulated P&L'] = (1-_w) * result_momentum['Cummulated P&L'] + (_w) * result_carry['Cummulated P&L']
        new_result['trading_PL-t_cost'] = (1-_w) * result_momentum['trading_PL-t_cost'] + (_w) * result_carry['trading_PL-t_cost']
        _, SR, _ = calculate_summary_stats(new_result)
        SR_lst.append(SR)
    SR_max = max(SR_lst)
    id_SR_max = SR_lst.index(SR_max)
    optimal_weight = w[id_SR_max]

    print('The optimal weight is: w = {}'.format(optimal_weight))
    print('The optimal Sharpe Ratio is: {}'.format(SR_max))

if __name__ == '__main__':
# %% [markdown]
# ##### Process Data

# %%
    wti_data = pd.read_excel('./PET_PRI_FUT_S1_D.xls', sheet_name=1,skiprows=2)
    exp_date = pd.read_excel('./Expiry Calendar HW1.xlsx')

    wti_data = processing_data(wti_data,type = 'WTI')
    ## we want to first calculate the equity line, and then calculate signals
    ## in order to calculate equity line, we do not consider transaction costs
    wti_PL = get_everyday_position(wti_data.copy(), exp_date, type='WTI',consider_transaction=False)

    # %%
    wti_data.head(5)

    # %%
    wti_PL.head(5)

    # %%
    equity_line = wti_PL[['Date','Pt']].set_index('Date')
    equity_line.plot(figsize=(12,5), title = 'equity line')

    # %% [markdown]
    # ##### Signals and Strategies

    # %%
    n = 20
    epsilon = 0
    momentum_signal = calcualte_momentum_signals(equity_line,n)
    carry_signal = calcualte_carry_signals(wti_data, epsilon)

    # %%
    wti_daily_PL = wti_PL[['Date','dF']]
    wti_daily_PL.columns = ['Date','daily_PL']
    wti_daily_PL.set_index('Date',inplace=True)

    # %%
    result_momentum_n_20 = trading_strategy_backtest(wti_daily_PL, momentum_signal)

    # %%
    result_momentum_n_20['Cummulated P&L'].plot(figsize=(12,5),title='Cummulated P&L of Momentum n=20 strategy')

    # %%
    result_carry_e_0 = trading_strategy_backtest(wti_daily_PL, carry_signal)

    # %%
    result_carry_e_0['Cummulated P&L'].plot(figsize=(12,5),title='Cummulated P&L of Carry e=0 strategy')

    # %% [markdown]
    # ##### Summary Statistics

    # %%
    from tabulate import tabulate as tb

    stat_carry_e_0 = (calculate_summary_stats(result_carry_e_0))
    stat_carry_e_0 = list(zip(('Annualized P&L','Sharpe Ratio','Maximum Drawdowm'),stat_carry_e_0))
    stat_carry_e_0 = [list(i) for i in stat_carry_e_0]

    # %%
    print('Summary Statistics of Carry epsilon=0 strategy')
    print(tb(stat_carry_e_0, tablefmt='fancy_grid'))

    # %%
    stat_momentum_n_20 = (calculate_summary_stats(result_momentum_n_20))
    stat_momentum_n_20 = list(zip(('Annualized P&L','Sharpe Ratio','Maximum Drawdowm'),stat_momentum_n_20))
    stat_momentum_n_20 = [list(i) for i in stat_momentum_n_20]
    print('Summary Statistics of Momentum n=20 strategy')
    print(tb(stat_momentum_n_20, tablefmt='fancy_grid'))

    # %% [markdown]
    # ##### Calculate Optimal Parameters

    # %%
    n_lst = np.arange(5,41)
    epsilon_lst = np.arange(0,0.51,0.01)

    # %%
    momentum_signal = calcualte_momentum_signals(equity_line,n)
    carry_signal = calcualte_carry_signals(wti_data, epsilon)

    # %%
    find_optimal_parameter(equity_line,wti_data, wti_daily_PL,n_lst,signal='Momentum')

    # %%
    find_optimal_parameter(equity_line,wti_data, wti_daily_PL,epsilon_lst,signal='Carry')

    # %% [markdown]
    # ##### Mini-Portfolio based on Optimal Parameters for Momentum and Carry

    # %%
    result_momentum_n_15 = trading_strategy_backtest(wti_daily_PL, calcualte_momentum_signals(equity_line, 15))
    result_momentum_n_15.head(5)

    # %%
    result_carry_e_0_43 = trading_strategy_backtest(wti_daily_PL, calcualte_carry_signals(wti_data, 0.43))
    result_carry_e_0_43.head(5)

    # %%
    find_optimal_weight(result_momentum_n_15, result_carry_e_0_43)

    # %% [markdown]
    # ##### Write result in the excel sheet

    # %%
    res_carry_e_0 = result_carry_e_0[['trading_daily_PL','transaction_cost','Cummulated P&L']]
    res_carry_e_0.columns = ['daily_PL(no transaction_cost)','transaction_cost','Cummulated P&L(t_cost included)']

    # %%
    res_mom_n_20 = result_momentum_n_20[['trading_daily_PL','transaction_cost','Cummulated P&L']]
    res_mom_n_20.columns = ['daily_PL(no transaction_cost)','transaction_cost','Cummulated P&L(t_cost included)']

    # %%
    res_carry_e_0_43 = result_carry_e_0_43[['trading_daily_PL','transaction_cost','Cummulated P&L']]
    res_carry_e_0_43.columns = ['daily_PL(no transaction_cost)','transaction_cost','Cummulated P&L(t_cost included)']

    res_mom_n_15 = result_momentum_n_15[['trading_daily_PL','transaction_cost','Cummulated P&L']]
    res_mom_n_15.columns = ['daily_PL(no transaction_cost)','transaction_cost','Cummulated P&L(t_cost included)']

    # %%
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter('./Strategy_P&L.xlsx', engine='xlsxwriter')

    # Write each dataframe to a different worksheet.
    res_carry_e_0.to_excel(writer, sheet_name='carry_e=0 P&L')
    res_mom_n_20.to_excel(writer, sheet_name='mom_n=20 P&L')
    res_carry_e_0_43.to_excel(writer, sheet_name='carry_e=0.43_optimal P&L')
    res_mom_n_15.to_excel(writer, sheet_name='mom_n=15_optimal P&L')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()

    # %%
    w = 0.55
    #mini_optimal_port = pd.DataFrame()
    mini_optimal_port = res_mom_n_15 * (1-w) + res_carry_e_0_43 * w

    # %%
    plt.figure(figsize=(20,8))
    (100*1000*mini_optimal_port['Cummulated P&L(t_cost included)']).plot(
                                                        title = 'optimal mini-portfolio P&L')
    (100*1000*res_mom_n_15['Cummulated P&L(t_cost included)']).plot()
    (100*1000*res_carry_e_0_43['Cummulated P&L(t_cost included)']).plot()
    plt.legend(['Carry-Momenteum', 'Momentum n=15','Carry e=0.43'])

    # %%
    plt.figure(figsize=(20,8))
    (100*1000*res_mom_n_20['Cummulated P&L(t_cost included)']).plot()
    (100*1000*res_carry_e_0['Cummulated P&L(t_cost included)']).plot()
    plt.title('Equity Lines for Momentum and Carry strategy')
    plt.legend(['Momentum n=20','Carry e=0'])

# %%



