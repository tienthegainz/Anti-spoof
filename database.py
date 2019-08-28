# -*- coding: utf-8 -*-
"""
    FCam.database
    -------

    This module implement database of FCam.

    :copyright 2019 by FTECH team.
"""

import pickle
import numpy as np
import datetime
from random import *
import os
import json

import config

class DBManagement():
    __instance = None
    global data

    @staticmethod
    def getInstance():
        """ Static access method. """
        if DBManagement.__instance == None:
            DBManagement()
        return DBManagement.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if DBManagement.__instance != None:
            print ('DB class is a singleton!')
        else:
            DBManagement.__instance = self
            self.get_database(config.db_file)

    def get_database(self, db_file):
        try:
            self.data = pickle.loads(open(db_file, "rb").read())
        except:
            return

    def save(self):
        f = open(config.db_file, "wb")
        f.write(pickle.dumps(self.data))
        f.close()

    def save_data(self,features, ids):
        f = open(config.db_file, "wb")
        self.data = {"features": features, "ids":ids}
        f.write(pickle.dumps(self.data))
        f.close()

    def get_features(self):
        return np.array(self.data['features'])

    def set_features(self,features):
        self.data = {"features": features, "ids": self.get_ids()}
        self.save()

    def get_ids(self):
        return np.array(self.data['ids'])

    def set_ids(self,ids):
        self.data = {"features": self.get_features(), "ids": ids}
        self.save()


    def write_logs_test(self):
        days_ago = 3
        day_start = datetime.datetime.now() - datetime.timedelta(days=days_ago)
        for day_step in range(days_ago+1):
            day_log = day_start + datetime.timedelta(days=day_step)
            logs_path = "data/logs/" + str(day_log.date()) + ".txt"
            file = open(logs_path, "a")
            for minutes_step in range(24*60):
                time_log = day_log + datetime.timedelta(minutes=minutes_step)
                if time_log.date() != day_log.date():
                    break
                faceId = "faceId:" + str(randint(0, 9))
                file.write(str(time_log) + " " + faceId + os.linesep)

            file.close()
        return

    def write_logs(self, faceId):
        logs_path = "data/logs/" + str(datetime.datetime.now().date()) + ".txt"
        file = open(logs_path, "a")
        time_log = datetime.datetime.now()
        file.write(str(time_log) + " faceId:" + str(faceId) + os.linesep)
        file.close()
        return

    def get_logs(self, faceId = None, fromDate = None, toDate = None):
        logs = {}
        for filename in os.listdir(config.logs_path):
            logs_daily = []
            path = os.path.join(config.logs_path, filename)
            print(path)
            file = open(path, "r")
            for line in file:
                logArray = line.split(" faceId:")
                if (logArray[1][:1] == str(faceId)):
                    logs_daily.append(logArray[0])
            file.close()
            date_str = filename.split(".")[0]
            logs[date_str] = logs_daily

        print(json.dumps(logs))
        return logs
