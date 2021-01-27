import numpy as np
import pandas as pd
import os
import datetime

class Calc:
    def __init__(self, data, pred, start_date, enddate, timegap = 1):
        self.data = data
        self.pred = pred
        self.start_date = start_date
        self.enddate = enddate
        self.timegap = timegap
    
    def _auto_set(self):
        if self.start_date == None:
            self.start_date = self.data[0].keys()
        if self.enddate == None:
            self.enddate = self.data[-1].keys()
        return self.start_date, self.enddate
    
    def get_table(self, object):
        self.start_date, self.enddate = self._auto_set()
        __obj = pd.DataFrame(object).transpose()
        for i in range(__obj.shape[1]):
            __obj.iloc[:,i] = pd.Series(__obj.iloc[:,i]).interpolate()
        __obj['time'] = pd.to_datetime(__obj.index, format = '%Y%m%d', errors = 'ignore')
        __obj = (__obj['time'] >= self.start_date) & (__obj['time'] <= self.enddate)
        return __obj
    
    # Tax Rate Only Charge when Sell Out;
    # Procedure Rate Charget at Each Transaction;
    # ==========================================================
    # Unmaintained：Trigger to buy/sell and if it succeed
    # Current：Must able to buy or sell fully
    # ==========================================================
    def cal(self, tax = 0.003, procedure = 0.001425):
        per = 1
        buy_price = []
        for i in range(1, len(self.data)):
            if self.pred[i] / self.data[i-1] > procedure + 1:
                buy_price.append(self.data[i-1])
            elif self.data[i-1] / self.pred[i] > procedure + 1: 
                per += sum([(item - self.data[i-1]) / self.data[i-1] for item in buy_price]) * per
                buy_price = []
        return per

    def exec(self, method):
        print('The calculation based on : tax rate = 0.3% ; procedure = 0.1425%')
        if method == 'auto':
            if max(self.data['time']) - min(self.data['time']) > 250:
                yr = datetime.datetime.now().year
                m_yr = min(self.data['time']).year
                y_list = list(range(m_yr, yr))
                per = []
                for y in y_list:
                    per.append([y, self.cal()])
        elif method == 'manual':
            self.data = self.get_table(self.data)
            self.pred = self.get_table(self.pred)
            per = self.cal()
        return per
