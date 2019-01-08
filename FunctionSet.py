import numpy as np 
import pandas as pd
from collections import deque
import calendar
import datetime

class FS:
    def __init__(self, data):
        self.data = data

    def Ma(self, days, target = None, fillmethod = 0):
        if target == None:
            if type(self.data) != list:
                print('Format Error')
                return None
            else:
                return pd.Series(self.data).rolling(int(days)).mean().fillna(fillmethod)
        else:
            '''
            data = pandas.dataframe
            target = str
            days = int
            '''
            return pd.Series(self.data[target]).rolling(int(days)).mean().fillna(fillmethod)

    def Roll_std(self, target, days, fillmethod = 0):
        '''
        data = pandas.dataframe
        target = str
        days = int
        '''
        return pd.Series(self.data[target]).rolling(int(days)).std().fillna(fillmethod)

    def BBand(self, target, days, std_range = 2, fillmethod = 0):
        ma = self.Ma(target = target, days = days)
        std = self.Roll_std(target, days)
        bband = np.vstack([[q - std_range * b for q, b in zip(ma, std)],
                        [q + std_range * b for q, b in zip(ma, std)]])
        bband = pd.DataFrame(bband.T)  
        bband.columns = ['Lower', 'Upper']
        return bband

    def TR(self):
        a = [x - y for x, y in zip(self.data['High'].iloc[1:], self.data['Low'].iloc[1:])]
        b = [abs(x - y) for x, y in zip(self.data['High'].iloc[:-1], self.data['Close'].iloc[1:])]
        c = [abs(x - y) for x, y in zip(self.data['Low'].iloc[:-1], self.data['Close'].iloc[1:])]
        return [max(x, y, z) for x, y, z in zip(a, b, c)]

    def KD(self, days, columns = None, weight = (2/3, 1/3)):
        data = pd.DataFrame(self.data)
        if columns != None:
            data = data[columns]
        mx, mn = [], []
        for i in range(data.shape[0]):
            mx.append(max(data.iloc[i,:]))
            mn.append(min(data.iloc[i,:]))
        rs_h = pd.Series(mx).rolling(days).max() 
        rs_l = pd.Series(mn).rolling(days).min()
        RSV = [(z-y)/(x-y) for x, y, z in zip(rs_h.iloc[days-1:], rs_l.iloc[days-1:], data['Close'].iloc[days-1:])]
        del mx, mn, rs_h, rs_l
        K_val, D_val = [weight[1]*RSV[0]], [weight[1]*weight[1]*RSV[0]]
        for i in range(1, len(RSV)):
            K_val.append(weight[0]*K_val[i-1] + weight[1]*RSV[i])
            D_val.append(weight[0]*D_val[i-1] + weight[1]*K_val[i])
        return K_val, D_val
        
    def ATR(self, days):
        atr = self.Ma(target = 'Close', days = days)
        return atr

    def DI(self):
        return [(x + y + 2 * z)/4 for x, y, z in zip(self.data['High'], self.data['Low'], self.data['Close'])]


    def MACD(self, days, short = 12, long = 26):
        if short >= long:
            print('short should be less than long')
            return None
        di = self.DI()
        ema_s, ema_l = [sum(di[:short])/short], [sum(di[:long])/long]
        for i in range(len(di)):
            if short + 1 + i <= len(di) - 1:
                ema_s.append((ema_s[-1] * (short - 1) + di[short + 1 + i] * 2) / (short + 1))
                if long + 1 + i <= len(di) - 1:
                    ema_l.append((ema_l[-1] * (long - 1) + di[long + 1 + i] * 2) / (long + 1))
        ema_s = ema_s[long - short : ]
        dif = [x - y for x, y in zip(ema_s , ema_l)]
        macd = [sum(dif[:days])/days]
        for i in range(len(dif[days:])):
            macd.append((macd[-1] * (days - 1) + dif[days + i] * 2) / (days + 1))
        return macd

    def BIAS(self, target, days):
        ma = self.Ma(self.data, target, days)[days:]
        bias = [(x - y) / y for x, y in zip(self.data[target][days:], ma)]
        return bias

    def RSI(self, days):
        sub = [x - y for x, y in zip(self.data['Open'], self.data['Close'])]
        up = sum([x for x in sub if x >= 0])
        dn = sum([abs(x) for x in sub if x < 0])
        return up/dn

class Action:
    def __init__(self, data, initial, account = [], coin_hold = [], coin_owed = [], status = []):
        self.data = data
        self.initial = initial
        self.account = account
        self.coin_hold = coin_hold
        self.coin_owed = coin_owed
        self.status = status
    
    def get_coin_hold(self):
        return self.coin_hold

    def get_coin_owed(self):
        return self.coin_owed

    def get_account(self):
        return self.account
    
    def get_status(self):
        return self.status

    def start(self):
        self.coin_hold = [100]
        self.coin_owed = [0]
        self.account = [self.initial]
        self.status = ['begin']

    def hold(self):
        self.coin_hold.append(self.coin_hold[-1])
        self.coin_owed.append(self.coin_owed[-1])
        self.account.append(self.initial)
        self.status.append('hold')

    def buy(self, price, unit):
        if price * unit >= self.initial :
            unit = self.initial // price
        self.coin_hold.append(self.coin_hold[-1]+unit)
        self.coin_owed.append(self.coin_owed[-1])
        self.initial -= price * unit
        self.account.append(self.initial)
        self.status.append('buy')
    
    def sell(self, price, unit):
        if unit >= self.coin_hold[-1]:
            unit = self.coin_hold[-1] 
        self.coin_hold.append(self.coin_hold[-1]-unit)
        self.coin_owed.append(self.coin_owed[-1])
        self.initial += price * unit
        self.account.append(self.initial)
        self.status.append('sell')
    
    def short(self, price, unit, method = False):
        if method == True:
            unit += self.coin_hold[-1]
            self.coin_hold.append(0)
        else:
            self.coin_hold.append(self.coin_hold[-1])
        self.coin_owed.append(unit)
        self.initial += price * unit
        self.account.append(self.initial)
        self.status.append('short')
    
    def stop_loss(self, price, unit, method = True):
        if method == True:
            unit = self.coin_owed[-1]
        self.coin_hold.append(self.coin_hold[-1])
        self.coin_owed.append(self.coin_owed[-1] - unit)
        self.initial -= price * unit
        self.account.append(self.initial)
        self.status.append('stop_loss')
    
    def end(self, price):
        self.initial += self.coin_hold[-1] * price
        self.initial -= self.coin_owed[-1] * price
        self.coin_hold.append(0)
        self.coin_owed.append(0)
        self.account.append(self.initial)
        self.status.append('Ending')
