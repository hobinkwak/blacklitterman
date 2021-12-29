import os
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
import datetime
import warnings
warnings.simplefilter('ignore')
from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting
from pypfopt import EfficientFrontier, objective_functions
from itertools import product
from tqdm import tqdm
import time
import cvxpy as cp

def L1_norm(w, b = 1):
    return b * cp.norm(w, 1)

def lr_momentum(df):
    df = np.log(df)
    x = np.arange(len(df))
    res = sp.stats.linregress(x, df)
    return ((1 + res.slope) ** 105) if res.slope > 0 else ((-1 - res.slope) ** 105)
    #return ((1 + res.slope) ** 105) * (res.rvalue ** 2) if res.slope > 0 else (((1 + res.slope) ** 105) * (res.rvalue ** 2)) * (-1)
    if res.rvalue **2 >= 0.4:
        return ((1 + res.slope) ** 105) if res.slope > 0 else ((-1 - res.slope) ** 105)
    else:
        return ((1 + res.slope) ** 105) * (res.rvalue ** 2) if res.slope > 0 else (((1 + res.slope) ** 105) * (res.rvalue ** 2)) * (-1)
    

def lr_std(df):
    df = np.log(df)
    x = np.arange(len(df))
    res = sp.stats.linregress(x, df)
    return res.stderr


def sharpe_ratio(df):
    returns = df.pct_change().dropna()
    vol = returns.std() * np.sqrt(252)    
    annualized = (returns + 1).prod() ** (252/len(returns)) - 1
    #excess = annualized - 0.03
    excess = annualized
    return excess / vol


def momentum(df, method='basic', days=100):
    
    if method == 'basic':
        return_df = (np.log(df) - np.log(df.shift(days))).dropna()
        momentum_df = return_df.iloc[-1, :]
        momentum_df= pd.DataFrame(momentum_df)
        momentum_df.columns = ['momentum']
        momentum_df['rank'] = momentum_df.momentum.rank(ascending=False)
        momentum_df=momentum_df.sort_values(by='rank')
        return momentum_df
    
    if method == 'sharpe':
        df2 = df.iloc[-days-1:].copy()
        data = df2.copy()
        for col in df.columns:
            data[col] = df2[col].rolling(days).apply(sharpe_ratio)
            data[col+'_stderr'] = df2[col].rolling(days).std()
        lr_df = data[-1:].reset_index(drop=True).T.iloc[:len(df.columns), :]
        lr_stderr_df = data[-1:].reset_index(drop=True).T.iloc[len(df.columns):, :]
        lr_df.columns =['lr']
        lr_stderr_df.columns =  ['lr_stderr']
        lr_df['rank'] = lr_df.lr.rank(ascending=False)
        lr_df = lr_df.sort_values(by='rank')
        return lr_df, lr_stderr_df
    
    if method == 'momentum':
        df2 = df.iloc[-days:].copy()
        data = df2.copy()
        for col in df.columns:
            data[col] = df2[col].rolling(days).apply(lr_momentum)
            data[col+'_stderr'] = df2[col].rolling(days).apply(lr_std)
        lr_df = data[-1:].reset_index(drop=True).T.iloc[:len(df.columns), :]
        lr_stderr_df = data[-1:].reset_index(drop=True).T.iloc[len(df.columns):, :]
        lr_df.columns =['lr']
        lr_stderr_df.columns =  ['lr_stderr']
        lr_df['rank'] = lr_df.lr.rank(ascending=False)
        lr_df = lr_df.sort_values(by='rank')
        return lr_df, lr_stderr_df
    
    
name = ['KODEX 200','TIGER 코스닥150','TIGER 미국S&P500선물(H)','TIGER 유로스탁스50(합성,H)','KINDEX 일본 Nikkei225(H)','TIGER 차이나CSI300','KOSEF 국고채10년', 'KBSTAR 중기우량회사채','TIGER 단기선진하이일드(합성,H)','KODEX 골드선물(H)','TIGER 원유선물Enhanced(H)','KODEX 인버스','KOSEF 미국달러선물','KOSEF 미국달러인버스선물', 'KOSEF 단기자금']

code_cap = ['69500', '233160', '143850','195930','238720', '192090', '148070','136340','182490','132030','130680','114800','138230','139660', '130730']
code = ['069500', '233160', '143850','195930','238720', '192090', '148070','136340','182490','132030','130680','114800','138230','139660', '130730']


def load_info(date):
    try:
        marcap = pd.read_csv('./ETF/{}.csv'.format(date), encoding='ISO-8859-1')
    except:
        print("Probably Weekend, Not a trading day")
        return
    marcap[marcap.columns[0]] = marcap[marcap.columns[0]].astype('str')
    marcap = marcap.loc[marcap.iloc[:,0].isin(code_cap)][[marcap.columns[0], marcap.columns[11]]]
    marcap.columns = ['code','marcap']
    marcap.reset_index(drop=True, inplace=True)
    marcap.code.values[2] = '069500'
    info = pd.DataFrame(zip(name, code), columns=['name', 'code'])
    info = pd.merge(info, marcap, on='code')
    return info
    
    
def load_data(start, end, kind='data'):
    if kind=='market':
        df = fdr.DataReader('069500', start=start, end=end)['Close']
        return df    
    if kind=='data':
        df = pd.DataFrame([fdr.DataReader(c, start=start, end=end)['Close'] for c in code]).T
        df.columns = name
        return df    
    
    
def bl_prior(df, market, marcap, plot_cov=False, plot_prior=False):
    """Return: sigma, d, marcap_dict, market_prior"""
    sigma = risk_models.CovarianceShrinkage(df).ledoit_wolf()
    #sigma = risk_models.risk_matrix(df)
    d = black_litterman.market_implied_risk_aversion(market)
    d = 1 if d <= 0 else d
    if plot_cov:
        plt.rcParams["font.family"] = 'NanumGothic'
        plotting.plot_covariance(sigma, plot_correlation=True)
        plt.show()
    marcap_dict = {}
    for name in marcap.name.values:
        marcap_dict[name] = marcap.loc[marcap.name==name].marcap.values[0]
    marcap_dict = pd.Series(marcap_dict)
    market_prior = black_litterman.market_implied_prior_returns(marcap_dict, d, sigma)
    if plot_prior:
        # matplotlib 한글폰트
        plt.rcParams["font.family"] = 'NanumGothic'
        market_prior.plot(kind='barh')
        plt.show()
    return sigma, d, marcap_dict, market_prior
    
    
def bl_investors_view(df, method, days,sigma, display_rank=True):
    m, stderr = momentum(df, method=method, days=days)
    if display_rank:
        display(m)
    view = {}
    for c in m.index.values:
        view[c] = m.loc[m.index == c].lr.values[0]
    normalized_stderr = stderr.values.squeeze() / stderr.values.squeeze().sum()
    confidence = (1 - normalized_stderr) * (1 - np.diag(sigma))
    return view, confidence
    
    
def bl_posterior(sigma, market_prior, view, confidence, marcap_dict, d, plot_return=True):

    model = BlackLittermanModel(sigma, pi=market_prior, absolute_views = view, omega="idzorek",view_confidences = confidence, market_caps=marcap_dict, risk_aversion=d)
    returns_bl = model.bl_returns()
    returns_df = pd.DataFrame([market_prior, returns_bl, pd.Series(view)], index=["Prior", "Posterior", "Views"]).T
    sigma_bl = model.bl_cov()
    if plot_return:
        display(returns_df)
        plt.rcParams["font.family"] = 'NanumGothic'
        returns_df.plot.bar(figsize=(20,10))
        plt.show()
        
    return returns_bl, sigma_bl

def max_qu_allocate(returns_bl, sigma_bl,d, plot=True):
    ef = EfficientFrontier(returns_bl, sigma_bl)
    #ef.add_objective(objective_functions.L2_reg)
    ef.add_objective(L1_norm)
    
    
    k, ki, u, ui = returns_bl[0],  returns_bl[11], returns_bl[12],  returns_bl[13]
    
    constraints = [lambda x : x[0]+x[1]>=0.1 , lambda x : x[0]+x[1]<=0.4, lambda x : x[0]<=0.4,
                  lambda x : x[1]<=0.2, lambda x : x[2]<=0.2,lambda x : x[3]<=0.2,
                  lambda x : x[4]<=0.2, lambda x : x[5]<=0.2,lambda x : x[2]+x[3]+x[4]+x[5] <= 0.4,
                  lambda x : x[2]+x[3]+x[4]+x[5] >= 0.1,lambda x : x[6]<=0.5,
                   lambda x : x[7]<=0.4, lambda x : x[8]<=0.4,lambda x : x[8]>=0.05,
                   lambda x : x[6] + x[7] + x[8] >=0.2,lambda x : x[6] + x[7] + x[8] <=0.6,
                  lambda x : x[9]<=0.15,lambda x : x[10]<=0.15,
                   lambda x : x[9] + x[10]>=0.05,lambda x : x[9] + x[10]<=0.2,
                   lambda x : x[11]<=0.1,lambda x : x[12]<=0.2,
                   lambda x : x[13]<=0.2,lambda x : x[12] + x[13]<=0.2, lambda x: x[14]>=0.01]
    for c in constraints:
        ef.add_constraint(c)
    if k > ki:
        ef.add_constraint(lambda x : x[11] == 0)
    if k < ki:
        ef.add_constraint(lambda x : x[0] == 0)
    if u < ui:
        ef.add_constraint(lambda x : x[12] == 0)
    if u > ui:
        ef.add_constraint(lambda x : x[13] == 0)
        
    ef.max_quadratic_utility(risk_aversion=d)
    
    final_qu = ef.clean_weights(rounding=4)
    if plot:
        plt.rcParams["font.family"] = 'NanumGothic'
        pd.Series(final_qu).plot.pie(figsize=(10,10));
        
    return final_qu

def max_sharpe_allocate(returns_bl, sigma_bl, plot=True):
    ef = EfficientFrontier(returns_bl, sigma_bl)
    #ef.add_objective(objective_functions.L2_reg)
    ef.add_objective(L1_norm)
    
    
    k, ki, u, ui = returns_bl[0],  returns_bl[11], returns_bl[12],  returns_bl[13]
    
    constraints = [lambda x : x[0]+x[1]>=0.1 , lambda x : x[0]+x[1]<=0.4, lambda x : x[0]<=0.4,
                  lambda x : x[1]<=0.2, lambda x : x[2]<=0.2,lambda x : x[3]<=0.2,
                  lambda x : x[4]<=0.2, lambda x : x[5]<=0.2,lambda x : x[2]+x[3]+x[4]+x[5] <= 0.4,
                  lambda x : x[2]+x[3]+x[4]+x[5] >= 0.1,lambda x : x[6]<=0.5,
                   lambda x : x[7]<=0.4, lambda x : x[8]<=0.4,lambda x : x[8]>=0.05,
                   lambda x : x[6] + x[7] + x[8] >=0.2,lambda x : x[6] + x[7] + x[8] <=0.6,
                  lambda x : x[9]<=0.15,lambda x : x[10]<=0.15,
                   lambda x : x[9] + x[10]>=0.05,lambda x : x[9] + x[10]<=0.2,
                   lambda x : x[11]<=0.1,lambda x : x[12]<=0.2,
                   lambda x : x[13]<=0.2,lambda x : x[12] + x[13]<=0.2, lambda x: x[14]>=0.01]
    for c in constraints:
        ef.add_constraint(c)
    if k > ki:
        ef.add_constraint(lambda x : x[11] == 0)
    if k < ki:
        ef.add_constraint(lambda x : x[0] == 0)
    if u < ui:
        ef.add_constraint(lambda x : x[12] == 0)
    if u > ui:
        ef.add_constraint(lambda x : x[13] == 0)
    
    ef.max_sharpe()
    final_sharpe = ef.clean_weights(rounding=4)
    
    if plot:
        plt.rcParams["font.family"] = 'NanumGothic'
        pd.Series(final_sharpe).plot.pie(figsize=(10,10))

    return final_sharpe

    
def make_graph(df, year, month, day):
    try:
        os.mkdir('{}'.format(str(datetime.datetime(year,month,day))[:10]))
    except:
        print("FAIL")
        pass
    for name in df.columns:
        df[name].plot(title=name)
        plt.savefig('./{}/{}.png'.format(str(datetime.datetime(year,month,day))[:10], name), dpi=200)
        plt.show()
    for name in df.columns:
        df[name][-30:].plot(title=name)
        res = sp.stats.linregress(np.arange(len(df[name].values[-10:])), df[name].values[-10:])
        plt.plot(df[name].index.values[-10:], res.slope * np.arange(len(df[name].values[-10:])) + res.intercept, lw=5)
        plt.savefig('./{}/{}.png'.format(str(datetime.datetime(year,month,day))[:10], name + '_momentum'), dpi=200)
        plt.show()

        
def lr_slope(df):
    t = df
    x = np.arange(len(t))
    res = sp.stats.linregress(x, t)
    return res.slope


def lr_intercept(df):
    t = df
    x = np.arange(len(t))
    res = sp.stats.linregress(x, t)
    return res.intercept


def lr_stderr(df):
    t = df
    x = np.arange(len(t))
    res = sp.stats.linregress(x, t)
    return res.stderr


def sharpe_strategy(data, date, window, weight, count):
    df = data.rolling(window).apply(sharpe_ratio).dropna()
    cond_lower = df.iloc[window -1:] <= (df.rolling(window).mean().dropna() - df.rolling(window).std().dropna()) * weight
    #cond_upper = df.iloc[window-1:] >= (df.rolling(window).mean().dropna() + df.rolling(window).std().dropna()) * weight
    cond = (cond_lower).sum(axis=1)
    return data.loc[date:][cond >= count]
    

def momentum_strategy(data, date, window, weight, count):
    pred = (data.rolling(window).apply(lr_slope)*(window-1)) + data.rolling(window).apply(lr_intercept)
    cond_lower = data.iloc[window-1:] <= pred.dropna() - data.rolling(window).apply(lr_stderr).dropna() * weight
    #cond_upper = data.iloc[window-1:] >= pred.dropna() + data.rolling(window).apply(lr_stderr).dropna() * weight
    cond = (cond_lower).sum(axis=1)
    # date는 실제 운용일 다음날부터
    return data.iloc[window-1:][cond >= count].loc[date:]


def sigma_strategy(data, date, window, weight, count):
    """ 하이 """
    cond_lower = data.iloc[window-1:] <= data.rolling(window=window).mean().dropna() - \
                           weight*data.rolling(window=window).std().dropna()
    #cond_upper = data.iloc[window-1:] >= data.rolling(window=window).mean().dropna() + \
    #                        weight*data.rolling(window=window).std().dropna()
    cond =  (cond_lower).sum(axis=1)
    #return data.iloc[window-1:][cond >= count].loc['2021-06-01':]
    # date는 실제 운용일 다음날부터
    return data.iloc[window-1:][cond >= count].loc[date:]

    
def market_return(start, end):
    data = fdr.DataReader('069500', start=start, end=end)['Close']
    cum_profit = (1 + data.pct_change().dropna()).cumprod()
    return cum_profit


def asset_allocate(days, method, timedelta, end_y, end_m, end_d, marcap_fix=False):
    
    end_date = datetime.datetime(end_y, end_m, end_d)
    start_date = end_date - datetime.timedelta(timedelta)
    
    if len(str(end_m)) == 1 :
        date_m=  '0' + str(end_m)
    else:
        date_m = str(end_m)
    if len(str(end_d)) == 1:
        date_d = '0' + str(end_d)
    else:
        date_d = str(end_d)
    if marcap_fix:
        info = load_info('0531')
    else:
        info = load_info(date_m + date_d)
    
    marcap = info[['name','marcap']]
    df = load_data(start_date, end_date)
    market = load_data(start_date, end_date, 'market')
    sigma, d, marcap_dict, market_prior = bl_prior(df, market, marcap)
    view, confidence = bl_investors_view(df,method, days, sigma, display_rank=False)
    returns_bl, sigma_bl = bl_posterior(sigma, market_prior,view, confidence, marcap_dict, d, plot_return=False)
    first_final_sharpe = max_sharpe_allocate(returns_bl, sigma_bl, plot=False)
    
    return first_final_sharpe



def main_backtest(kind, method, first_weights, momentum_days, freq, ub,lb, date, windows, weights, counts, start_year, start_month, start_day, end_year,end_month,end_day,marcap_fix=False):
    
    """kind : 리밸런싱 기준 sigma/momentum/sharpe
       date : 첫 운용일 다음 날
       freq : 4나 5권장
       method : investor's view를 모델링할 방법
       momentum_days : investor's view를 모델링할 때 사용할 window size
       marcap_fix : marcap 데이터가 없을 때.
    """
    cases = list(product(windows, weights, counts))
    
    final_result  = {}
    for window, weight, count in tqdm(cases):
        
        if kind=='sigma':
            dff = load_data(datetime.datetime(start_year, start_month, start_day)-datetime.timedelta(window + (window//5)*2), datetime.datetime(end_year, end_month, end_day))
            result = sigma_strategy(dff, date, window, weight, count) 
        if kind=='momentum':
            dff = load_data(datetime.datetime(start_year, start_month, start_day)-datetime.timedelta(window + (window//5)*2), datetime.datetime(end_year, end_month, end_day))
            result = momentum_strategy(dff, date, window, weight, count) 
        if kind == 'sharpe':
            dff = load_data(datetime.datetime(start_year, start_month, start_day)-datetime.timedelta((window + (window//5)*2)*2), datetime.datetime(end_year, end_month, end_day))
            result = sharpe_strategy(dff, date, window, weight, count) 
        ls = result.index.tolist()
        for dt in ls:
            for i in range(1, freq):
                nextday = dt + datetime.timedelta(days=i)
                if nextday in ls:
                    del ls[ls.index(nextday)]  
        
        if len(ls) > ub or len(ls) <=lb : 
            print("too many or little reblancing at {}-{}-{}, num:{}".format(window, weight, count, len(ls)))
            continue
        if len(ls) == 0 :
            print("No Rebalancing Signal at {}-{}-{}".format(window, weight, count))
            continue
        final_result['{},{},{}'.format(window, weight, count)] = []

        for i in range(len(ls)):
            
            try: 
                year = ls[i].year
                month = ls[i].month
                day = ls[i].day
                year2 = ls[i+1].year
                month2 = ls[i+1].month
                day2 = ls[i+1].day
            except:
                year = ls[i].year
                month = ls[i].month
                day = ls[i].day
                year2 = end_year
                month2 = end_month
                day2 = end_day
                
            if i == 0:
                #수익률 계산 부분 (5.31 ~ 6.11)
                start_date = datetime.datetime(start_year, start_month, start_day)
                if datetime.datetime(year, month, day).weekday() == 4:
                    end_date = datetime.datetime(year, month, day) + datetime.timedelta(3)
                else:
                    end_date = datetime.datetime(year, month, day) + datetime.timedelta(1)
                    
                test_df = load_data(start=start_date, end=end_date)
                df_cum_profit = (1 + test_df.pct_change().dropna()).cumprod()

                start = 10000000000 * np.array(list(dict(first_weights).values()))
                cashflow = (df_cum_profit * start).sum(axis=1)
                cash = (df_cum_profit.iloc[-1] * start).sum()
                final_result['{},{},{}'.format(window, weight, count)].append(cash)
            
            
            # Weight 계산 부분 (6.10 종가 기준) (6.16 종가 기준) (6.25 종가 기준) ... (7.15)
            end_date = datetime.datetime(year, month, day)
            final_sharpe = asset_allocate(momentum_days, method, 180, end_date.year, end_date.month, end_date.day, marcap_fix)
            
            
            # 수익률 계산 부분
            test_df = load_data(start=datetime.datetime(start_year,start_month,start_day),end=datetime.datetime.today())
            # 6월 3일부터 6월 10일 + 1일까지
            # 6월 12일부터 6.16+1 (17)일까지 리밸런싱한 놈으로 수익률 반영
            # 6월 18일부터 6.25+1 (26)일까지 리밸런싱한 놈으로 수익률 반영
            # 6월 27일부터 7.7+1 (8)일까지 리밸런싱한 놈으로 수익률 반영
            #...
            # 7월 17일부터 7.26+1 (27)일까지 리밸런싱한 놈으로 수익률 반영
            # 7월 26+2 (28)일부터 ~
            
            start_date = datetime.datetime(year, month, day) #6.10
            end_date = datetime.datetime(year2, month2, day2) #6.16
            
            if start_date.weekday() in [3, 4]:
                start_date = start_date + datetime.timedelta(4)
            else:
                start_date = start_date + datetime.timedelta(2)
            if end_date.weekday() == 4:
                end_date = end_date + datetime.timedelta(3)
            else:
                end_date = end_date + datetime.timedelta(1)
            if i == len(ls)-1:
                end_date = datetime.datetime(end_year, end_month, end_day)
            
            test_df = test_df.pct_change().dropna().loc[start_date:end_date]
            df_cum_profit = (1 + test_df).cumprod() 
            if len(df_cum_profit) ==0: 
                print(year, month, day, window, weight, count)
                continue
            cash_allocated = cash * np.array(list(dict(final_sharpe).values()))
            cash = (df_cum_profit.iloc[-1] * cash_allocated).sum()
            cashflow = cashflow.append((df_cum_profit * cash_allocated).sum(axis=1))
            
            final_result['{},{},{}'.format(window, weight, count)].append([year, month, day, cash])
            final_result['{},{},{}'.format(window, weight, count)].append(final_sharpe)
        final_result['{},{},{}'.format(window, weight, count)].append(cashflow)
        df = {key : final_result[key][-1] for key in final_result.keys()}
        df = pd.DataFrame(df)
        df.to_csv(f'{datetime.datetime.today().date()}.csv')
    
    return final_result, df
