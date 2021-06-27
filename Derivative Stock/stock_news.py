import os
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
import json
from langdetect import detect
import sqlite3
import time
import time
import bs4 as bs
FINHUB_KEY='c2siv8aad3ic1qis15mg'
class Init():
    """Class that initializes global value for the module. It also use general method to initialize value.
     """

    def __init__(self, stock, start, end):
        """Built-in method to inialize the global values for the module
        Attributes
        -------------------
        `self.start.date` : str
            start date of the training period. Must be within the last year for the free version of FinHub. Format
            must be "YYYY-mm-dd"
        `self.end_date` : str
            end date of the training period. Format must be "YYYY-mm-dd"
        `self.ticker` : list
            tickers on which we want to perform the test. Can be one ticker in form of a list as well as a list
            of tickers like the s&p 500.
        `self.db_name` : str
            name of the sqlite3 database
        `self.dir_path` : str
            directory where the data are saved. It takes into account the `self.start_date` and `self.end_date`
        `self.start_date_` : datetime object
            same thing as `start_date` but as a datetime object
        `self.end_date_` : datetime object
            same thing as `start_date` but as a datetime object
        """

        #initialize value here
        self.start_date = start
        self.end_date = end
        self.ticker = stock
    
        self.db_name = 'financial_data'
        self.dir_path = os.path.dirname(os.path.realpath(__file__)) + '/output/' + self.start_date + '_' + \
                        self.end_date + '/'
        Path(self.dir_path).mkdir(parents=True, exist_ok=True) #create new path if it doesn't exist
        self.start_date_ = datetime.strptime(self.start_date, "%Y-%m-%d")  #datetime object
        self.end_date_ = datetime.strptime(self.end_date, "%Y-%m-%d")    #datetime object
        self.delta_date = abs((self.end_date_ - self.start_date_).days) #number of days between 2 dates

        try:
            self.start_date_ > self.end_date_
        except:
            print("'start_date' is after 'end_date'")

        try :
            datetime.strptime(self.start_date, "%Y-%m-%d") <= (datetime.now()- relativedelta(years=1))
        except:
            print("'start_date' is older than 1 year. It doesn't work with the free version of FinHub")
class FinnHub():
    """Class to make API calls to FinnHub"""

    def __init__(self,start_date,end_date,start_date_,end_date_,ticker,dir_path):
        """ Class constructor
        Parameters
        ----------
        `start_date` : str
            Start date of the request. Must be within 1 year from now for must request
            with the free version of FinHub
        `end_date` : str
            End date of the request.
        `start_date_` : datetime object
            Same thing as `start_date` but as a datetime object
        `end_date_` : datetime object
             Same thing as `start_date` but as a datetime object
        `ticker` : str
            Ticker symbol
        `db_name` : str
            Name of the sqlite database
        `dir_path` : str
            Directory  where our data will be stored
        Attributes
        ----------
        `self.max_call` : int
            maximum api calls per minute for the finhub API
        `self.time_sleep` : int
            seconds to sleep before making a new API call. Default is 60 seconds as the maximum number of API calls is
            per minute
        `self.nb_request` : int
            nb of request made so far. Set to 0 in constructor `__init__` as we may loop through ticker
            and want to avoid the variable to reset to 0 when exiting the wrapper `iterate_day()` (which could generate
            an error)
        `self.finhub_key` : str
            finhub unique API key. Get yours here : https://finnhub.io/
        `self.db_name : str
            default file name for the sql database
        """

        #Initialize attributes values here
        self.max_call = 60
        self.time_sleep = 60
        self.nb_request = 0
        self.dates = []
        self.finhub_key = FINHUB_KEY
        self.news_header = ['category', 'datetime','headline','id','image','related','source','summary','url']
        self.start_date = start_date
        self.end_date = end_date
        self.ticker = ticker
        self.ticker_request = ticker #different value because ticker like 'ALL' (All State) can generate error in SQLite
                                    #database
        self.dir_path = dir_path
        self.js_data = []
        self.summary = ""
        self.scores = []
        self.time = []
        self.start_date_ = start_date_ #datetime object
        self.end_date_ = end_date_ #datetime object

        #call the methods to access historical financial headlines
        tickers = self.ticker
        ticker_ = self.ticker
        self.ticker = ticker_ + '_'
        self.ticker_request = ticker_
        self.req_new(start_date, end_date)
    def setDate(self, dates):
        self.dates = dates
    def req_new(self, startdate_, enddate_):
        """ Method that makes news request(s) to the Finnhub API"""
        request_ = requests.get('https://finnhub.io/api/v1/company-news?symbol=' + self.ticker_request + '&from=' +
                                startdate_ + '&to=' + enddate_ + '&token=' + self.finhub_key)
        self.js_data += request_.json()
        for indice in range(0, len(self.js_data)):
            self.summary = self.summary + self.js_data[indice]['summary']
            self.scores.append(self.weighted_sentiment_score(self.js_data[indice]['summary'], 1))
            epoch_time = self.js_data[indice]['datetime']
            time_formatted = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch_time))
            self.time.append(time_formatted)
    def getTotalScore(self):
        """ Returns the sentiment scores dict object """
        return self.total_scores
    def getCounts(self):
        """ Returns the number of articles recorded each day. Returns a dict """
        return self.counts
    def aggregate_sentiment(self):
        """ Method that sorts the individual sentiment data based off of similar dates. Returns a dict """
        self.total_scores = {}
        self.counts = {}
        for indice in range(0, len(self.time)):
            if self.time[indice].split(" ")[0] in self.dates: 
                if not self.total_scores.__contains__(str(str(self.time[indice]).split(" ")[0])):
                    self.total_scores.update({str(str(self.time[indice]).split(" ")[0]) : int(0)})
                    self.counts.update({str(str(self.time[indice]).split(" ")[0]) : int(1)})
                else:
                    self.counts[str(str(self.time[indice]).split(" ")[0])] = self.counts[str(str(self.time[indice]).split(" ")[0])] + 1
                    self.total_scores[str(str(self.time[indice]).split(" ")[0])] = self.total_scores[str(str(self.time[indice]).split(" ")[0])] + float(self.scores[indice])
    def weighted_sentiment_score(self, sentence, weighting):
        """Method that determines the sentiment value associated with each summary news article
        -----
        Parameters
        -----
        'Weighting' : list of floats or a single float value
            weighting that will be applied to each article 

        """
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(str(sentence))
        return float(sentiment['compound']) * weighting

