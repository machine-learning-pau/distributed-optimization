#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 7:01:25 2017

@author: Yann Vernaz
"""
# utilitaires pour le Meetup ML Pau

import os
import csv
import numpy as np
from random import sample
import matplotlib.pyplot as plt

def data_generate(fileName='data.csv', w=[1., 2., -1], n=100):
    '''Generates 2-D features and binary labels (-1,+1) from a linear model. Save the dataset into file.
            
        Parameters
        ----------
        fileName : (string) output file name
        w : (ndarray) weights (d,) 
        n: (int) number of generate exemples
            
        Returns
        -------
        (bool)
        
        Examples
        --------
        >>> w = np.array([2., 3., -1.])
        >>> utilsLab1.data_generate('data_train.csv', w, n=1000)
        True
    '''
    #np.random.seed(50)
    x1 = np.random.normal(loc=1, scale=2, size=n)
    x2 = np.random.normal(loc=4, scale=4, size=n)
    noise = np.random.normal(loc=0, scale=1, size=n)
    v = w[0] + w[1]*x1 + w[2]*x2 + noise    # linear 2-D model
    y = 2*(sigmoid(v) > 0.5)-1              # labels : +1 or -1
    #y = 2*(v>0) - 1                        # labels : +1 or -1
    data = zip(y, x1, x2)
    with open(fileName,'w') as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)
    return True

def data_read(fileName='data_training.csv'):
    '''Reads the generate dataset (CSV format)

        Parameters
        ----------
        fileName : (string) input file name
            
        Returns
        -------
        x : (ndarray) features matrix (n by d)
        y : (ndarray) label vector (n,)
        
        Examples
        --------   
        >>> x,y = utilsLab1.data_read('data_training.csv')
    '''
    features = []
    labels = []
    if os.path.isfile(fileName) != True:
        print("Le fichier de données "+ fileName + " n'existe pas.")
        return np.array(features), np.array(labels)
    else:
        with open(fileName,'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # add intercept and parsing
                features.append([1.0, float(row[1]),float(row[2])])
                labels.append(float(row[0]))
    return np.array(features), np.array(labels)

def data_scaled(x):
    '''Normalizes features with mean 0 and std 1.
        
        Parameters
        ----------
        x : (ndarray) features matrix (n,d)
        
        Returns
        -------
        x_scaled : (ndarray) scaled features matrix (n,d), with mean 0 and std 1.
        
        Examples
        --------
        >>> x_scaled = utilsLab1.data_scaled(x)
    '''
    n,d = x.shape
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    x_scaled = np.ones((n,d))
    for i in range(d-1):
        x_scaled[:,i+1] = (x[:,i+1] - mean[i+1])/std[i+1]
    return x_scaled

def weight_rescaled(w, mean, std):
    '''Rescales the weights'''
    v = np.zeros(3)
    v[0] = w[0] - w[1]*mean[1]/std[1] - w[2]*mean[2]/std[2]
    v[1] = w[1]/std[1]
    v[2] = w[2]/std[2]
    return v

def sampling(x,y,size=1):
    '''Sampling method'''
    n,d = x.shape
    z = np.concatenate((x, y.reshape(n,1)), axis=1)
    i = sample(range(n), size)
    sampleData = z[i,:]
    sample_x = sampleData[:,0:d]
    sample_y = sampleData[:,d]
    return sample_x, sample_y

def sigmoid(z):
    '''Sigmoid function'''
    return 1 / (1 + np.exp(-z))

def prediction(x,w,threshold):
    '''Prediction function'''
    if (sigmoid(-np.dot(x,w))>threshold):
        pred = -1.
    else:
        pred = 1.
    return pred

def huber_loss(w, clip_delta=1):
    '''Huber loss function
        
        Examples
        --------
        >>> f=np.vectorize(huber_loss)
        >>> w = np.linspace(-5,5,100)
        >>> plt.plot(w, f(w))
    '''
    error = np.abs(w)
    quadratic_part = np.minimum(error, clip_delta)
    return 0.5 * np.square(quadratic_part) + clip_delta * (error - quadratic_part)

def pseudo_huber_loss(w, clip_delta=1):
    '''Pseudo-Huber loss function'''
    delta = clip_delta*clip_delta
    z = 1. + np.power(w,2)/delta
    return delta*np.sum(np.sqrt(z) - 1.)

def pseudo_huber_grad(w, clip_delta=1):
    '''Pseudo-Huber gradient function'''
    delta = clip_delta*clip_delta
    z = 1. + np.power(w,2)/delta
    return np.sum(np.divide(w, np.sqrt(z)))

def pseudo_huber_hessian(w, clip_delta=1):
    '''Pseudo-Huber hessian function'''
    delta = clip_delta*clip_delta
    z = 1. + np.power(w,2)/delta
    return np.sum( np.divide( 1. + np.power(w,2)*(1. + np.sqrt(z))/delta, np.power(z,1.5) ) )

def show_progress(k,w,f,gradf):
    '''Shows algorithm progress. 
    
        Parameters
        ----------
        k : (int) interation/epoch number
        w : (ndarray) weights (d,)
        f : (float) loss function value at w
        gradf : (ndarray) gradient value at w (d,)
            
        Returns
        -------
        (bool)
            
        Examples
        --------
        >>> utilsLab1.show_progress(k, w, fval, gradient)
        
    '''
    msg = "epoch %-5s w=%-23s f(w)=%-8.4f Grad_f(w)=%.6f" % (k+1, np.round(w,2), f, np.linalg.norm(gradf,2))
    print(msg)
    return True

def show_progress2(k,f,gradf):
    '''Shows algorithm progress.
        
        Parameters
        ----------
        k : (int) interation/epoch number
        f : (float) loss function value at w
        gradf : (ndarray) gradient value at w (d,)
        
        Returns
        -------
        (bool)
        
        Examples
        --------
        >>> utilsLab1.show_progress(k, w, fval, gradient)
        
        '''
    msg = "epoch %-5s f(w)=%-8.4f Grad_f(w)=%.6f" % (k+1, f, np.linalg.norm(gradf,2))
    print(msg)
    return True

# Plotting functions

def data_plot(x, y, w):
    '''Plots the dataset and true frontier line.
    
        Parameters
        ----------
        x : (ndarray) features matrix (n,d)
        y : (ndarray) labels vector (n,)
        w : (ndarray) weights vector (n,)
        
        Examples
        --------
        >>> utilsLab1.data_plot(x, y, w)
    '''
    n,d = x.shape
    plt.figure(figsize=(10,5))
    cols = {1: 'g', -1: 'r'}
    for i in range(n):
        plt.plot(x[i,1], x[i,2], cols[y[i]]+'o')
    plt.xlabel("x1")
    plt.ylabel("x2")
    x1 = [np.min(x[:,1]),np.max(x[:,1])]
    # x2 = - w0/w2 - (w1/w2)*x1
    x2 = [-(w[0] + w[1]*i)/w[2] for i in x1] 
    plt.plot(x1, x2, linewidth=2.0)
    plt.title("Classe +1 (verte), Classe -1 (rouge)")
    plt.grid()
    plt.show()

def iteration_plot(nbGraphs=1, f=0, label=["line1"]):
    '''Plots function(s) objective vs. iterations.
    
        Parameters
        ----------
        nbGraphs : (int) number of curves to plot
        f : (ndarray) objective fonction values (f_1,f_2,...,f_nbGraphs)
        label : ()
        
        Examples
        --------
        >>> utilsLab1.iter_plot(f_GD, f_SGD)
    '''

    plt.figure(figsize=(15,5))
    nbIter = len(f[:,1])
    colori = ["black", "blue", "red", "green", "pink", "grey", "cyan"]
    
    for i in range(nbGraphs):
        plt.plot(range(nbIter), f[:,i], color=colori[i], label=label[i],
                 linewidth=1.0, linestyle="-")
    plt.xlim(0, nbIter)
    plt.xlabel('Iterations')
    plt.ylabel('Perte logistique')
    plt.legend()
    plt.show()

def data_plot_solution(x, y, truew, algo, lambda1=0.0, lambda2=0.0):
    '''
        Parameters
        ----------
        x : (ndarray) features matrix (n,d)
        y : (ndarray) labels vector (n,1)
            
        Returns
        -------
        (bool)
        
        Examples
        --------
        >>> utilsLab1.data_plot_solution(x_train, y_train, w, miniBatchGD)
    '''
    #np.random.seed(800)
    n,d = x.shape
    x1 = [np.min(x[:,1]),np.max(x[:,1])]
    plt.figure(figsize=(15,8))
    w = truew
    x2 = [-(w[0] + w[1]*i) / w[2] for i in x1]
    plt.plot(x1, x2, 'b', label="Vraie ligne", linewidth=4.0)
    
    cols = {1: 'r', -1: 'b'}
    for i in range(n):
        plt.plot(x[i,1], x[i,2], cols[y[i]]+'o')
    
    w = np.random.uniform(size=3)
    x2 = [-(w[0] + w[1]*i) / w[2] for i in x1]
    plt.plot(x1, x2, 'r--', label="0 iteration", linewidth=2.0)

    x_scaled = data_scaled(x)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    wopt, f_tab = algo(x_scaled, y, w0=np.array(w),
                       iterations=20, sampleSize=10,
                       learningRate=0.05, momentum=0.9, nesterov=True,
                       lambda1=lambda1, lambda2=lambda2)
    v = weight_rescaled(wopt,mean,std)
    x2 = [-(v[0] + v[1]*i) / v[2] for i in x1]
    #x2 = [-(wopt[0] + wopt[1]*i) / wopt[2] for i in x1]
    plt.plot(x1, x2, 'g--', label="20 iterations", linewidth=2.0)

    wopt, f_tab = algo(x_scaled, y, w0=np.array(wopt),
                       iterations=30, sampleSize=10,
                       learningRate=0.05, momentum=0.9, nesterov=True,
                       lambda1=lambda1, lambda2=lambda2)
    v = weight_rescaled(wopt,mean,std)
    x2 = [-(v[0] + v[1]*i) / v[2] for i in x1]
    #x2 = [-(wopt[0] + wopt[1]*i) / wopt[2] for i in x1]
    plt.plot(x1, x2, 'c--', label="50 iterations", linewidth=2.0)
    
    wopt, f_tab = algo(x_scaled, y, w0=np.array(wopt),
                       iterations=50, sampleSize=10,
                       learningRate=0.05, momentum=0.9, nesterov=True,
                       lambda1=lambda1, lambda2=lambda2)
    v = weight_rescaled(wopt,mean,std)
    x2 = [-(v[0] + v[1]*i) / v[2] for i in x1]
    #x2 = [-(wopt[0] + wopt[1]*i) / wopt[2] for i in x1]
    plt.plot(x1, x2, 'y--', label="100 iterations", linewidth=2.0)
    
    wopt, f_tab = algo(x_scaled, y, w0=np.array(wopt),
                       iterations=100, sampleSize=10,
                       learningRate=0.05, momentum=0.9, nesterov=True,
                       lambda1=lambda1, lambda2=lambda2)
    v = weight_rescaled(wopt,mean,std)
    x2 = [-(v[0] + v[1]*i) / v[2] for i in x1]
    #x2 = [-(wopt[0] + wopt[1]*i) / wopt[2] for i in x1]
    plt.plot(x1, x2, 'm--', label="200 Iterations", linewidth=2.0)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, fontsize=20, borderaxespad=0.)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid()
    plt.show()

