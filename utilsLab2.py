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

def modifLabel(x):
    '''(Re)encoding the label (target variable)'''
    y = -1.
    if (x<0.5):
        y = 1.
    return y

def data_read(sc, fileName = "retail.csv"):
    '''Reads the dataset (CSV format) and forms a labePoint RDD'''
    
    if os.path.isfile(fileName) != True:
        print("Le fichier de donnés "+ fileName + " n'existe pas.")
        return
    else:
        retailRDD = sc.textFile(fileName).map(lambda line: line.split(','))
        header = retailRDD.first()
        retailRDD = retailRDD.filter(lambda line: line != header)
        labelPointRDD = retailRDD.map(lambda line: LabeledPoint(modifLabel(float(line[0])),[line[2:]]))
    return labelPointRDD

def data_read_undersampling(sc, fileName = "retail.csv"):
    '''Reads the dataset (CSV format) and forms a labePoint RDD
        
        Stratified Sampling : we're keeping all instances of the Churn=True class,
        but downsampling the Churn=False class to 0.1 fraction.
        '''
    
    if os.path.isfile(fileName) != True:
        print("Le fichier de donnés "+ fileName + " n'existe pas.")
        return
    else:
        retailRDD = sc.textFile(fileName).map(lambda line: line.split(','))
        header = retailRDD.first()
        retailRDD = retailRDD.filter(lambda line: line != header)
        underSamplingRDD = retailRDD.map(lambda line: (modifLabel(float(line[0])),line[2:]))
        # specify the exact fraction desired from each key as a dictionary
        fractions = {1: 0.1, -1: 1.0}
        underSamplingRDD = underSamplingRDD.sampleByKey(False, fractions)
        underSamplingRDD = underSamplingRDD.map(lambda line: LabeledPoint(line[0],line[1]))
    return underSamplingRDD

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
