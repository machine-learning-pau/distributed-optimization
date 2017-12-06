#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 6:55:25 2017

@author: Yann Vernaz
"""
# utilitaires pour le Meetup ML Pau

import os
import numpy as np
from operator import add
from pyspark.mllib.regression import LabeledPoint

def data_read(sc, fileName = "retail.csv"):
    '''Reads the dataset (CSV format) and forms a labePoint RDD'''
    
    if os.path.isfile(fileName) != True:
        print("Le fichier de donnés "+ fileName + " n'existe pas.")
        return
    else:
        retailRDD = sc.textFile(fileName).map(lambda line: line.split(','))
        header = retailRDD.first()
        retailRDD = retailRDD.filter(lambda line: line != header)
        labelPointRDD = retailRDD.map(lambda line: LabeledPoint(line[0],[line[2:]]))
    return labelPointRDD

def data_scaled(labelPointRDD):
    '''Normalizes features with mean 0 and std 1.'''
    
    n = labelPointRDD.count()
    
    # mean vector : .mean()
    mean = labelPointRDD.map(lambda row: row.features.toArray()).reduce(add)/n

    # std vector : stdev()
    std = np.sqrt(labelPointRDD.map(lambda row: np.power(row.features.toArray()-mean, 2)).reduce(add))

    # scaled features
    data_scaled = labelPointRDD.map(lambda row: LabeledPoint(row.label, np.append(1.0, (row.features.toArray()-mean)/std)))

    return data_scaled, n, mean, std
