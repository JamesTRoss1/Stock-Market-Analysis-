#James Ross; student at ASU  
#main file for GUI application

import tkinter as tk
import yfinance as yf 
import re
import inspect
import pandas as pd
import csv 
import plotly as ply 
import cufflinks as cf 
import datetime
import calendar
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib 
from matplotlib import pyplot as plt
import statsmodels.api as sm 
import numpy as np
import statistics
import sklearn.neural_network as sknn
import sklearn.preprocessing as sk
import sklearn.cross_decomposition as skcd 
import sklearn.ensemble as ske
from scipy.stats import norm
from scipy.stats import laplace
from scipy.stats import linregress
from sklearn.metrics import r2_score
class Ticker(): 
    duration = ""
    ticker = ""
    time = ""
    interval = ""
    intervalTime = ""
    def setDuration(self, newVar):
        self.duration = newVar
    def setTicker(self, newVar):
        self.ticker = newVar
    def setTime(self, newVar): 
        self.time = newVar
    def setInterval(self, newVar): 
        self.interval = newVar
    def setAll(self, newVar2, newVar3, newVar4):
        self.ticker = str(newVar2).strip() 
        self.time = str(newVar3).strip() 
        self.intervalTime = str(newVar4).strip()

class GUI():
    root = tk.Tk() 
    elements = []
    ticker = Ticker()
    history = None 
    def onButton1Click(self, elements):
        for element in elements: 
            GUI.clearElement(self, element)
        elements = []
        GUI.createCSV(self)
        GUI.createGUI2(self)
        print("Calculating...")
        GUI.News(self)
        #GUI.AllRegressorPriceChangeVolume(self)
        #GUI.SVRPriceChangeVolume(self)
        #GUI.LinearPriceChangeVolume(self)
        #GUI.PolyPriceChangeVolume(self)
        #GUI.PriceChangeVolume(self)
        #GUI.AbsNetChange(self)
        #GUI.Volume(self)
        #GUI.PercentChangeAnalysis(self)
        #GUI.createPlot(self)
    def createCSV(self):
        tick = yf.Ticker(str(GUI.ticker.ticker))   
        GUI.history = tick.history(period = str(GUI.ticker.time) + str(GUI.ticker.duration), interval = str(GUI.ticker.intervalTime) + str(GUI.ticker.interval))
        GUI.history = GUI.history.dropna()
        time = pd.to_datetime(GUI.history.Close.index.values)
        self.start_time = pd.to_datetime(str(GUI.history.Close.index.values[0]).split(" ")[0])
        self.start_time = self.start_time.date()
        self.end_time = pd.to_datetime(str(GUI.history.Close.index.values[len(GUI.history.Close.index.values) - 1]).split(" ")[0])
        self.end_time = self.end_time.date()
        print(self.start_time)
        print(self.end_time)
        stockClose = GUI.history.Close.values
        with open('ticker.csv', 'w+', newline = '') as file:
            writer = csv.writer(file, delimiter = ',') 
            writer.writerow(["Time", "Close"]) 
            for indice in range(0, len(time)):
                writer.writerow([time[indice], stockClose[indice]])
    def createGUI2(self):
        GUI.root.title("Stock Analysis Chart")
    def createPlot(self):
        data = pd.read_csv("ticker.csv", index_col = 0)
        data.index = pd.to_datetime(data.index)
        #SARIMA model based upon price 
        #period is based upon the time range perspective of stock market, not calendar year 
        stockPeriod = GUI.StockPeriod()
        print(type(GUI.history))
        print()
        #SARIMA
        multiplicity_number = 1
        forecast_values = GUI.SARIMA(data=data.values, stockPeriod=stockPeriod, multiplicity_number=multiplicity_number)
        print("SARIMA Predicted Direction For " + (str(GUI.ticker.ticker)) + " The Next " + str(stockPeriod * multiplicity_number) + str(GUI.ticker.interval))
        forecast_list = forecast_values.summary_frame().values.tolist()
        print(forecast_list)
        start = GUI.history.values[len(GUI.history) - 1][3]
        end = forecast_list[len(forecast_list) - 1][0]
        #percent change formula 
        percent_change = GUI.PercentChange(start=start, end=end)
        #Determines if Bullish, Bearish, or sideways 
        forecast = GUI.forecast(percent_change=percent_change, negligble_value=.75)
        print(forecast)
        print()
        print(len(GUI.history.values))
    def PercentChangeAnalysis(self):
        #Data Starts As Dataframe 
        data = pd.read_csv("ticker.csv", index_col = 0)
        data.index = pd.to_datetime(data.index)
        stockPeriod = GUI.StockPeriod(self)
        starting_value = GUI.history.Close.values[len(GUI.history.Close.values) - 1]
        starting_index = -1
        adjusted_data = []
        for indice in range(0, len(data.values)):
            adjusted_data.append(GUI.PercentChange(self, start=starting_value, end=data.values[indice]))
            starting_index = starting_index + 1
            starting_value = data.values[starting_index]
        print(adjusted_data)
        df = pd.DataFrame(data=adjusted_data)
        #Stationary Set; Ready for SARIMA 
        diff_df = df.diff()
        df = df.dropna()
        count = df.sum()
        length = len(df.values)
        #Mean Used For Laplace Distribution In Order To Highlight Volitility and Possible Corrections As Return Will Correct Towards Mean
        loc = float(count / length)
        scale = 1
        #Range is set to remove extreme outliers
        print(df.values)
        n, bins, p = plt.hist(df.values, density=True)
        min_value = round(float(min(df.values)))
        max_value = round(float(max(df.values)))
        x = np.arange(min_value, max_value, 1)
        mu, std = norm.fit(df.values)
        x = np.linspace(min_value, max_value, 100)
        p = norm.pdf(x,mu,std)
        plt.plot(x,p,'k', linewidth=2, label="Gaussian Distribution")
        mu, std = laplace.fit(df.values)
        x = np.linspace(min_value, max_value, 100)
        p = laplace.pdf(x,mu,std)
        plt.plot(x,p, linewidth=2,color='red', label="Laplace Distribution")
        plt.xlabel("Return Percent Change")
        plt.ylabel("Probability")
        plt.legend()
        #Scale 
        plt.ylim(auto=True)
        #Post Chart and Info On GUI
        LM = tk.Label(GUI.root, text = "Mean: " + str(mu))
        LM.pack()
        LS = tk.Label(GUI.root, text= "Standard Deviation: " + str(std))
        LS.pack()
        plt.show()
    def Volume(self):
        data = GUI.history.Volume.values
        n, bins, p = plt.hist(data, density=True)
        min_value = round(float(min(data)))
        max_value = round(float(max(data)))
        x = np.arange(min_value, max_value, 1)
        mu, std = norm.fit(data)
        x = np.linspace(min_value, max_value, 100)
        p = norm.pdf(x,mu,std)
        plt.plot(x,p,'k', linewidth=2, label="Gaussian Distribution")
        mu, std = laplace.fit(data)
        x = np.linspace(min_value, max_value, 100)
        p = laplace.pdf(x,mu,std)
        plt.plot(x,p, linewidth=2,color='red', label="Laplace Distribution")
        plt.title(str(GUI.ticker.ticker) + " Volume")
        plt.xlabel("Volume")
        plt.ylabel("Probability")
        plt.legend()
        #Scale 
        plt.ylim(auto=True)
        plt.xlim(auto=True)
        #Post Chart and Info On GUI
        LM = tk.Label(GUI.root, text = "Mean: " + str(mu))
        LM.pack()
        LS = tk.Label(GUI.root, text= "Standard Deviation: " + str(std))
        LS.pack()
        #Calculate Current Volume and distance away from mean
        data_point = GUI.history.Volume.values[len(GUI.history.Volume.values) - 1]
        zscore = (data_point - mu) / (std)
        LZ = tk.Label(GUI.root, text= "Z-Score: " + str(round(float(zscore), 2)))
        LZ.pack()
        plt.show()
    def AbsNetChange(self): 
        data = []     
        for indice in range(0, len(GUI.history.Close.values)):
            data.append(abs((GUI.history.Close.values[indice] - GUI.history.Open.values[indice])) * GUI.history.Volume.values[indice])
        n, bins, p = plt.hist(data, density=True)
        min_value = round(float(min(data)))
        max_value = round(float(max(data)))
        x = np.arange(min_value, max_value, 1)
        mu, std = norm.fit(data)
        x = np.linspace(min_value, max_value, 100)
        p = norm.pdf(x,mu,std)
        plt.plot(x,p,'k', linewidth=2, label="Gaussian Distribution")
        mu, std = laplace.fit(data)
        x = np.linspace(min_value, max_value, 100)
        p = laplace.pdf(x,mu,std)
        plt.plot(x,p, linewidth=2,color='red', label="Laplace Distribution")
        plt.title(str(GUI.ticker.ticker) + " Volume")
        plt.xlabel("Volume")
        plt.ylabel("Probability")
        plt.legend()
        #Scale 
        plt.ylim(auto=True)
        plt.xlim(auto=True)
        #Post Chart and Info On GUI
    #    LM = tk.Label(GUI.root, text = "Mean: " + str(mu))
    #    LM.pack()
    #    LS = tk.Label(GUI.root, text= "Standard Deviation: " + str(std))
    #    LS.pack()
    #    #Calculate Current Volume and distance away from mean
    #    data_point = ((GUI.history.Close.values[indice] - GUI.history.Open.values[indice])) * GUI.history.Volume.values[indice]
    #    zscore = (data_point - mu) / (std)
    #    LZ = tk.Label(GUI.root, text= "Z-Score: " + str(round(float(zscore), 2)))
    #    LZ.pack()
    #    plt.show()
    def PriceChangeVolume(self):   
        data = []     
        for indice in range(0, len(GUI.history.Close.values)):
            data.append(abs((GUI.history.Close.values[indice] - GUI.history.Open.values[indice]) / (GUI.history.Open.values[indice])) * GUI.history.Volume.values[indice])
        n, bins, p = plt.hist(data, density=True)
        GUI.DistributionPlot(self, data, 100, 100, 1, 1, str(GUI.ticker.ticker), "Price-Change * Volume", "Probability")
        #Post Chart and Info On GUI
        mu, std = laplace.fit(data)
        LM = tk.Label(GUI.root, text = "Mean: " + str(mu))
        LM.pack()
        LS = tk.Label(GUI.root, text= "Standard Deviation: " + str(std))
        LS.pack()
        #Calculate Current Volume and distance away from mean
        data_point = ((GUI.history.Close.values[indice] - GUI.history.Open.values[indice]) / (GUI.history.Open.values[indice])) * GUI.history.Volume.values[indice]
        zscore = (data_point - mu) / (std)
        LZ = tk.Label(GUI.root, text= "Z-Score: " + str(round(float(zscore), 2)))
        LZ.pack()
        plt.show()
    def LinearPriceChangeVolume(self): 
        y = []     
        x = []
        for indice in range(0, len(GUI.history.Close.values)):
            y.append((((GUI.history.Close.values[indice] - GUI.history.Open.values[indice]) / (GUI.history.Open.values[indice])) * 100))
            x.append(GUI.history.Volume.values[indice])
        regression = linregress(x,y)
        print(regression)
    def PolyPriceChangeVolume(self):
        y = []     
        x = []
        degree = 2
        degreemax = 20
        maxr2 = 0
        for indice in range(0, len(GUI.history.Close.values)):
            y.append((((GUI.history.Close.values[indice] - GUI.history.Open.values[indice]) / (GUI.history.Open.values[indice])) * 100))
            x.append(GUI.history.Volume.values[indice])
        for deg in range(degree, degreemax):
            poly = np.polyfit(x,y,deg)
            p = np.poly1d(poly)
            r2 = r2_score(y, p(x))
            if(r2 > maxr2):
                maxr2 = r2 
                Poly = poly 
                Degree = deg
                R2 = r2 
        print("Coeff: " + str(Poly))
        print("Degree: " + str(Degree))
        print("R^2 " + str(R2))
    def SVRPriceChangeVolume(self):
        y = []     
        x = []
        for indice in range(0, len(GUI.history.Close.values)):
            y.append((((GUI.history.Close.values[indice] - GUI.history.Open.values[indice]) / (GUI.history.Open.values[indice])) * 100))
            x.append(GUI.history.Volume.values[indice])
        prediction, score = GUI.SupportVectorRegression(self, x=x, y=y)
    def AllRegressorPriceChangeVolume(self):
        y = []     
        x = []
        for indice in range(0, len(GUI.history.Close.values)):
            y.append((((GUI.history.Close.values[indice] - GUI.history.Open.values[indice]) / (GUI.history.Open.values[indice])) * 100) * GUI.history.Volume.values[indice])
            x.append(indice)
        Regr = GUI.Regressors()
        x = np.reshape(x, (-1, 1))
        y = np.reshape(y, (-1,1))
        print(y)
        pred = GUI.runAllRegressors(self, Regr, x, y, y)
        print(pred.items())
        row = 0
        column = 0
        index = 1
        for predict in pred.values():
            GUI.RegressionPlot(self, predict, x, y, 100, 100, row, column, "Regression", "Volume", "Price", index)
            row = row + 1
            column = column + 1
            index = index + 1
    def News(self):
        #Uses Finviz 
        import stock_news
        dates, indices = self.getDates()
        pcv = []
        pc = self.getPriceChange()
        for ind in indices:
            pcv = self.getPriceChangeVolume()
            pcv.pop(ind)
        stock_news.Init(str(GUI.ticker.ticker), str(self.start_time), str(self.end_time))
        fin = stock_news.FinnHub(str(self.start_time), str(self.end_time), self.start_time, self.end_time, str(GUI.ticker.ticker), "stock_news.txt")
        fin.setDate(dates)
        fin.aggregate_sentiment()
        counts = fin.getCounts()
        aggScores = fin.getTotalScore()
        countScores = {}
        for key, value in counts.items():
            if key in aggScores:
                countScores[key] = float(counts[key]) * float(aggScores[key])
        print("Average Value " + str(self.average(counts.values())))
        print("Average Value " + str(self.average(aggScores.values())))
        #Equal x, y coords for plot 
        while len(pcv) > len(countScores.values()):
            pcv.pop()
        while len(pc) > len(countScores.values()):
            pc.pop()
        plt.scatter(pc, countScores.values())
        plt.show()
    def getDates(self):
        y = []
        indices = []
        data = pd.read_csv("ticker.csv", index_col = 0)
        print(data)
        for b,c in enumerate(str(data.index.values).split(" ")):
            if "-" in str(c):
                print(c)
                d = re.sub("[|]|'", "", str(c))
                if str(d) not in y and not str(d).__contains__("["):
                    y.append(str(d))
                else:
                    print(b)
                    indices.append(b)
        print("INDICES TO REMOVE")
        print(indices)
        return y, indices 
    def getPriceChange(self):
        """Returns a dictionary of dates and associated price change"""
        y = []     
        for indice in range(0, len(GUI.history.Close.values)):
            y.append((((GUI.history.Close.values[indice] - GUI.history.Open.values[indice]) / (GUI.history.Open.values[indice])) * 100))
        return y
    def getVolume(self): 
        """Returns a dictionary of dates and associated volume"""
        x = []     
        for indice in range(0, len(GUI.history.Close.values)):
            x.append(GUI.history.Volume.values[indice])
        return x 
    def getPriceChangeVolume(self):
        y = []     
        for indice in range(0, len(GUI.history.Close.values)):
            y.append((((GUI.history.Close.values[indice] - GUI.history.Open.values[indice]) / (GUI.history.Open.values[indice])) * 100) * GUI.history.Volume.values[indice])
        return y
    def average(self, values):
        return sum(values) / len(values) 
    def verify(self, l1a, l2a):
    #    while(len(l1a) != len(l2a)):
     #       try:
      #          if(len(l1a) == len(l2a)):
       #             return 
        #        elif len(l1a) > len(l2a):
        #
        #            for indice in range(0, len(l1a)):
         #               if(str(l1a[indice]) in l2a):
          #                  continue
           #             else:
            #                l1a.tolist()
             #               l1a = list(l1a).pop(indice)
              #              print(type(l1a))
#                else:
 #                       for indice in range(0, len(l2a)):
  #                             continue
 #                           else:
 #                               l2a = list(l2a).pop(indice)
 #                               print(l2a)
 #                               print(type(l2a))
 #           except IndexError:
 #               return indice
 #       return l1a, l2a 
        pass
    def SpecificRegressorPriceChangeVolume(self, reg):
        #Reg is an index integer to represent the method to run 
        #Work in progress
        y = []     
        x = []
        for indice in range(0, len(GUI.history.Close.values)):
            y.append((((GUI.history.Close.values[indice] - GUI.history.Open.values[indice]) / (GUI.history.Open.values[indice])) * 100))
            x.append(GUI.history.Volume.values[indice])
        Regr = GUI.Regressors() 
    def SupportVectorRegression(self,x,y): 
        from sklearn.svm import SVR
        svr = SVR()
        x = np.reshape(x, (-1, 1))
        fit = svr.fit(x,y)
        #Prediction is the y-values
        prediction = svr.predict(x)
        score = svr.score(x,y)
        return prediction, score
    def SARIMA(self, data, stockPeriod, multiplicity_number):
        sarima = sm.tsa.statespace.SARIMAX(data, seasonal_order=(1,1,1,stockPeriod), trend='ct')
        print("Fitting Data")
        results = sarima.fit(maxiter=2000, method = 'nm')
        forecast_values = results.get_forecast(steps=stockPeriod * multiplicity_number)
        return forecast_values
    def SARIMAPlotPreProsessing(self, stockPeriod, multiplicy_number, forecast_list):
        X = np.arange(0, len(GUI.history.values), 1)
        X1 = np.arange(len(GUI.history.values), (len(GUI.history.values) - 1) + (stockPeriod * multiplicy_number), 1)
        forecast_values = []
        forecast_lower = []
        forecast_upper = []
        for indice in range(0, len(forecast_list) - 1):
            forecast_values.append(forecast_list[indice][0])
            forecast_lower.append(forecast_list[indice][2])
            forecast_upper.append(forecast_list[indice][3])
    def StockPeriod(self): 
        if GUI.ticker.interval == 'm':
            stockPeriod = 60
        elif GUI.ticker.interval == 'h':
            stockPeriod = 6
        elif GUI.ticker.interval == 'd':
            if GUI.ticker.duration == "wk": 
                stockPeriod = 5
            else: 
                stockPeriod = 21
        elif GUI.ticker.interval == 'wk':
           stockPeriod = 4
        elif GUI.ticker.interval == 'mo':
            stockPeriod = 12
        else:
            stockPeriod = 1
        return stockPeriod 
        #Create Price Plot SARIMAX And Post It To Screen 
    #Regressors 
    class Regressors:
        def MLPRegressor(self, x, y, pred):
            mlp = sknn.MLPRegressor().fit(x,y)
            prediction = mlp.predict(pred)
            return prediction
     #   def RandomForestRegressor(self, x, y, pred): 
     #       rfc = ske.RandomForestRegressor().fit(x,y)
     #       prediction = rfc.predict(pred)
     #       return prediction
        def AdaBoostRegressor(self, x, y, pred):
            abr = ske.AdaBoostRegressor().fit(x,y)
            prediction = abr.predict(pred)
            return prediction
        def BayesianRidgeRegressor(self, x, y, pred): 
            from sklearn.linear_model import BayesianRidge
            brr = BayesianRidge().fit(x,y)
            return brr.predict(pred) 
     #   def LogisticRegression(self, x, y, pred): 
      #      from sklearn.linear_model import LogisticRegression
       #     lr = LogisticRegression().fit(x,y)
        #    return lr.predict(pred)
        def RANSACRegressor(self, x, y, pred):
            from sklearn.linear_model import RANSACRegressor
            rr = RANSACRegressor().fit(x,y)
            return rr.predict(pred)
   #     def PoissonRegressor(self, x, y, pred):
    #        from sklearn.linear_model import PoissonRegressor
     #       x = np.reshape(x, (-1,1))
      ##      y = np.reshape(y, (-1,1))
      #      pr = PoissonRegressor().fit(x,y)
      #      return pr.predict(pred)
        def KNeighborsRegressor(self, x, y, pred): 
            from sklearn.neighbors import KNeighborsRegressor
            knr = KNeighborsRegressor().fit(x,y)
            prediction = knr.predict(pred)
            return prediction
     #   def IsotonicRegressor(self, x, y, pred):
     #       from sklearn.isotonic import IsotonicRegression
     #       x = np.ravel(x)
     #       y = np.ravel(y)
     #       ir = IsotonicRegression().fit(x,y)
     #       prediction = ir.predict(pred)
     #       return prediction
        def PLSRegressor(self, x, y, pred):
            pls = skcd.PLSRegression().fit(x,y)
            prediction = pls.predict(pred)
            return prediction
        def VotingRegressor(self,x,y,pred):
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            r1 = LinearRegression()
            r2 = RandomForestRegressor(n_estimators=10, random_state=1)
            abr = ske.VotingRegressor([('lr', r1), ('rf', r2)]).fit(x,y)
            prediction = abr.predict(pred)
            return prediction
    def runAllRegressors(self, r, x, y, pred):
        #r is an instance of the regressor class
        predictions = {} 
        attrs = (getattr(r, name) for name in dir(r))
        methods = filter(inspect.ismethod, attrs)
        for method in methods:
            try:
                predictions.update({str(method): method(x, y, pred)})
            except TypeError:
                pass
        return predictions
    def SARIMAPlot(self, X, X1, data, forecast_values, forecast_lower, forecast_upper, figsizex, figsizey, row, column):
        matplotlib.use("TkAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        figure = Figure(figsize=(figsizex, figsizey), dpi=100)
        plot = figure.add_subplot(1,1,1)
        plot.plot(X, data.values)
        plot.plot(X1, forecast_values, color='r')
        plot.fill_between(X1, forecast_lower, forecast_upper, color = 'r', alpha = .3)
        #Places Plot In GUI
        canvas = FigureCanvasTkAgg(figure, GUI.root)
        canvas.get_tk_widget().grid(row=row, column=column) 
      #  LF = tk.Label(GUI.root, text = "SARIMA Forecast: " + str(forecast) + " at $" + str(round(float(end), 2)))
      #  LF.grid(row=1, column = 0)
      #  LR = tk.Label(GUI.root, text = "Estimated Return: " + str(round(float(percent_change), 2)) + "%")
      #  LR.grid(row=2, column = 0)
    def DistributionPlot(self, data, figsizex, figsizey, row, column, title, xlabel, ylabel): 
        matplotlib.use("TkAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        figure = Figure(figsize=(int(figsizex), int(figsizey)), dpi=100)
        plot = figure.add_subplot(1,1,1)
        min_value = round(float(min(data)))
        max_value = round(float(max(data)))
        x = np.arange(min_value, max_value, 1)
        mu, std = norm.fit(data)
        x = np.linspace(min_value, max_value, 100)
        p = norm.pdf(x,mu,std)
        plot.plot(x,p,'k', linewidth=2, label="Gaussian Distribution")
        mu, std = laplace.fit(data)
        x = np.linspace(min_value, max_value, 100)
        p = laplace.pdf(x,mu,std)
        plot.plot(x,p, linewidth=2,color='red', label="Laplace Distribution")
        plt.title(str(GUI.ticker.ticker) + str(title))
        plt.xlabel(str(xlabel))
        plt.ylabel(str(ylabel))
        plt.legend()
        #Scale 
        plt.ylim(auto=True)
        plt.xlim(auto=True)
        canvas = FigureCanvasTkAgg(figure, GUI.root)
        canvas.get_tk_widget().grid(row=int(row), column=int(column)) 
    def RegressionPlot(self, predictiony, x, y, figsizex, figsizey, row, column, title, xlabel, ylabel, index):
        matplotlib.use("TkAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        figure = Figure(figsize=(int(figsizex), int(figsizey)), dpi=100)
        plot = figure.add_subplot(3, 3, index)
        x = x.flatten()
        y = y.flatten()
        predictiony = predictiony.flatten()
        print(len(y))
        print(len(predictiony))
        plt.plot(x, y, 'k', linewidth = 2, label="Actual")
        plt.plot(x, predictiony, color='r', label="Prediction")
        plt.title(str(GUI.ticker.ticker) + str(title))
        plt.title(str(GUI.ticker.ticker) + str(title))
        plt.xlabel(str(xlabel))
        plt.ylabel(str(ylabel))
        plt.legend()
        #Scale 
        plt.ylim(auto=True)
        plt.xlim(auto=True)
        plt.show()
        #canvas = FigureCanvasTkAgg(figure, GUI.root)
        #canvas.get_tk_widget().grid(row=int(row), column=int(column)) 
    def Forecast(self, negligble_value, percent_change):
        if abs(percent_change) < negligble_value: 
            forecast = "Sideway Trend"
        elif percent_change > 0: 
            forecast = "Bullish Trend"
        else: 
            forecast = "Bearish Trend"
    def PercentChange(self, start, end): 
        print("Start: " + (str(start)))
        print("End: " + str(end))
        percent_change = ((end - start) / start) * 100 
        return percent_change
    def clearElement(self, element): 
        element.destroy()
    def CreateGUI1(self):
        GUI.root.title("Stock Analysis Program")
        GUI.root.geometry('500x500')
        L1 = tk.Label(GUI.root, text = "Stock Ticker")
        E1 = tk.Entry(GUI.root, bd = 5) 
        E4 = tk.Entry(GUI.root, bd = 5)
        L4 = tk.Label(GUI.root, text = "Interval")
        L5 = tk.Label(GUI.root, text = "Units of Interval")
        svar = tk.StringVar()
        E5A = tk.Radiobutton(GUI.root, text = "Minutes", variable = svar, value = 'A', command = lambda: GUI.ticker.setInterval("m"))
        E5B = tk.Radiobutton(GUI.root, text = "Hours", variable = svar, value = 'B', command = lambda: GUI.ticker.setInterval("h"))
        E5C = tk.Radiobutton(GUI.root, text = "Days", variable = svar, value = 'C', command = lambda: GUI.ticker.setInterval("d"))
        E5D = tk.Radiobutton(GUI.root, text = "Weeks", variable = svar, value = 'D', command = lambda: GUI.ticker.setInterval("wk"))
        E5E = tk.Radiobutton(GUI.root, text = "Months", variable = svar, value = 'E', command = lambda: GUI.ticker.setInterval("mo"))
        L2 = tk.Label(GUI.root, text = "Duration")
        E2 = tk.Entry(GUI.root, bd = 5) 
        L3 = tk.Label(GUI.root, text = "Units of Duration") 
        var = tk.StringVar()
        E3B = tk.Radiobutton(GUI.root, text = "Hours", variable = var, value = 'B', command = lambda: GUI.ticker.setDuration("h"))
        E3C = tk.Radiobutton(GUI.root, text = "Days", variable = var, value = 'C', command = lambda: GUI.ticker.setDuration("d"))
        E3D = tk.Radiobutton(GUI.root, text = "Weeks", variable = var, value = 'D', command = lambda: GUI.ticker.setDuration("wk"))
        E3E = tk.Radiobutton(GUI.root, text = "Months", variable = var, value = 'E', command = lambda: GUI.ticker.setDuration("mo"))
        E3F = tk.Radiobutton(GUI.root, text = "Years", variable = var, value = 'F', command = lambda: GUI.ticker.setDuration("y"))
        B1 = tk.Button(GUI.root, text = "Submit", command = lambda: [GUI.ticker.setAll(E1.get(), E2.get(), E4.get()), GUI.onButton1Click(self, elements)])
        L1.pack()
        E1.pack()
        L4.pack()
        E4.pack()
        L5.pack()
        E5A.pack()
        E5B.pack()
        E5C.pack()
        E5D.pack()
        E5E.pack()
        L2.pack()
        E2.pack()
        L3.pack()
        E3B.pack()
        E3C.pack()
        E3D.pack()
        E3E.pack()
        E3F.pack()
        B1.pack()
        elements = [L1, E1, L2, E2, L3, E3B, E3C, E3D, E3E, E3F, B1, L4, E4, L5, E5A, E5B, E5C, E5D, E5E]
        GUI.root.mainloop()

def main(): 
    window = GUI()
    window.CreateGUI1() 

if(__name__ == "__main__"):
        main()

