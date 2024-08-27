import os
import string
import numpy as np
from itertools import islice
import random
import csv
from time import time
import json
import pandas as pd
from tqdm import trange


'''
    readcsv: Read feature tensors from csv data packet
    args:
        target: the directory that stores the csv files
        fealen: the length of feature tensor, related to to discarded DCT coefficients
    returns: (1) numpy array of feature tensors with shape: N x H x W x C
             (2) numpy array of labels with shape: N x 1 
'''
def readcsv(target, fealen=32):
    #read label
    path  = os.path.join(target, 'label.csv')
    label = np.genfromtxt(path, delimiter=',')
    if os.path.isfile(os.path.join(target, 'dc.npy')):
        trange_desc = "reading npy files"
    else:
        trange_desc = "reading csv files"
    #read feature
    feature = []
    assert fealen == 32
    for i in trange(fealen, desc=trange_desc):
        if i==0:
            file = 'dc.csv'
            path = os.path.join(target, file)
            file_npy = 'dc.npy'
            path_npy = os.path.join(target, file_npy)
            if os.path.isfile(path_npy):
                featemp = np.load(path_npy)
            else:
                featemp = pd.read_csv(path, header=None).to_numpy()
                np.save(path_npy, featemp)
            feature.append(featemp)
        else:
            file = 'ac'+str(i)+'.csv'
            path = os.path.join(target, file)
            file_npy = 'ac'+str(i)+'.npy'
            path_npy = os.path.join(target, file_npy)
            if os.path.isfile(path_npy):
                featemp = np.load(path_npy)
            else:
                featemp = pd.read_csv(path, header=None).to_numpy()
                np.save(path_npy, featemp)
            feature.append(featemp)          
    feature = np.rollaxis(np.asarray(feature), 0, 3)[:, :, 0:fealen]
    return feature, label


'''
    processlabel: adjust ground truth for biased learning
    args:
        label: numpy array contains labels
        cato : number of classes in the task
        delta1: bias for class 1
        delta2: bias for class 2
    return: softmax label with bias
'''
def processlabel(label, cato=2, delta1 = 0, delta2=0):
    label_int = [int(_lb) for _lb in label]
    softmaxlabel = np.eye(cato)[label_int]
    '''
    softmaxlabel = np.zeros((len(label), cato), dtype=np.float32)
    for i in range(0, len(label)):
        if int(label[i])==0:
            softmaxlabel[i,0]=1-delta1
            softmaxlabel[i,1]=delta1
        if int(label[i])==1:
            softmaxlabel[i,0]=delta2
            softmaxlabel[i,1]=1-delta2
    return softmaxlabel
    '''
    return softmaxlabel


'''
    data: a class to handle the training and testing data, implement minibatch fetch
    args: 
        fea: feature tensor of whole data set
        lab: labels of whole data set
        ptr: a pointer for the current location of minibatch
        maxlen: length of entire dataset
        preload: in current version, to reduce the indexing overhead of SGD, we load all the data into memeory at initialization.
    methods:
        nextinstance():  returns a single instance and its label from the training set, used for SGD
        nextbatch(): returns a batch of instances and their labels from the training set, used for MGD
            args: 
                batch: minibatch number
                channel: the channel length of feature tersor, lenth > channel will be discarded
                delta1, delta2: see process_label
        sgd_batch(): returns a batch of instances and their labels from the trainin set randomly, number of hs and nhs are equal.
            args:
                batch: minibatch number
                channel: the channel length of feature tersor, lenth > channel will be discarded
                delta1, delta2: see process_label

'''
class data:
    def __init__(self, fea, lab, preload=False):
        self.ptr_n=0
        self.ptr_h=0
        self.ptr=0
        self.dat=fea
        self.label=lab
        self.epoch_cnt=0
        with open(lab) as f:
            self.maxlen=sum(1 for _ in f)
        if preload:
            print("loading data into the main memory...")
            self.ft_buffer, self.label_buffer=readcsv(self.dat)
    

    def nextinstance(self):
        temp_fea=[]
        label=None
        idx=random.randint(0,self.maxlen)
        for dirname, dirnames, filenames in os.walk(self.dat):
            for i in range(0, len(filenames)-1):
                    if i==0:
                        file='/dc.csv'
                        path=self.dat+file
                        with open(path) as f:
                            r=csv.reader(f)
                            fea=[[int(s) for s in row] for j,row in enumerate(r) if j==idx]
                            temp_fea.append(np.asarray(fea))
                    else:
                        file='/ac'+str(i)+'.csv'
                        path=self.dat+file
                        with open(path) as f:
                            r=csv.reader(f)
                            fea=[[int(s) for s in row] for j,row in enumerate(r) if j==idx]
                            temp_fea.append(np.asarray(fea))        
        with open(self.label) as l:
            temp_label=np.asarray(list(l)[idx]).astype(int)
            if temp_label==0:
                label=[1,0]
            else:
                label=[0,1]
        return np.rollaxis(np.array(temp_fea),0,3),np.array([label])


    def sgd(self, channel=None, delta1=0, delta2=0):
        with open(self.label) as l:
            labelist=np.asarray(list(l)).astype(int)
        length=labelist.size
        idx=random.randint(0, length-1)
        temp_label=labelist[idx]
        if temp_label==0:
            label=[1,0]
        else:
            label=[0,1]
        ft= self.ft_buffer[idx]

        return ft, np.array(label)
    def sgd_batch_2(self, batch, channel=None, delta1=0, delta2=0):
        with open(self.label) as l:
            labelist=np.asarray(list(l)).astype(int)
            labexn = np.where(labelist==0)[0]
            labexh = np.where(labelist==1)[0]
        n_length = labexn.size
        h_length = labexh.size
        if not batch % 2 == 0:
            print('ERROR:Batch size must be even')
            print('Abort.')
            quit()
        else:
            num = batch // 2
        idxn = labexn[(np.random.rand(num)*n_length).astype(int)]
        idxh = labexh[(np.random.rand(num)*h_length).astype(int)]
        label = np.concatenate((np.zeros(num), np.ones(num)))
        label = processlabel(label,2, 0,0 )
        ft_batch = np.concatenate((self.ft_buffer[idxn], self.ft_buffer[idxh]))
        ft_batch_nhs = self.ft_buffer[idxn]
        label_nhs = np.zeros(num)
        return ft_batch, label


    def sgd_batch(self, batch, channel=None, delta1=0, delta2=0):
#        with open(self.label) as l:
#            labelist=np.asarray(list(l)).astype(int)
#            labexn = np.where(labelist==0)[0]
#            labexh = np.where(labelist==1)[0]
        labexn = np.where(self.label_buffer == 0)[0]
        labexh = np.where(self.label_buffer == 1)[0]
        n_length = labexn.size
        h_length = labexh.size
        if not batch % 2 == 0:
            print('ERROR:Batch size must be even')
            print('Abort.')
            quit()
        else:
            num = batch // 2
        idxn = labexn[(np.random.rand(num)*n_length).astype(int)]
        idxh = labexh[(np.random.rand(num)*h_length).astype(int)]
        label = np.concatenate((np.zeros(num), np.ones(num)))
        #label = processlabel(label,2, delta1, delta2)
        ft_batch = np.concatenate((self.ft_buffer[idxn], self.ft_buffer[idxh]))
        ft_batch_nhs = self.ft_buffer[idxn]
        label_nhs = np.zeros(num)
        return ft_batch, label, ft_batch_nhs, label_nhs


    '''
    nextbatch_beta: returns the balalced batch, used for training only
    return:
        ft_batch: balanced batch
    '''
    def nextbatch_beta(self, batch, channel=None, delta1=0, delta2=0):
        def update_ptr(ptr, batch, length):
            if ptr+batch<length:
                ptr+=batch
            if ptr+batch>=length:
                ptr=ptr+batch-length
            return ptr
        labexn = np.where(self.label_buffer == 0)[0]
        labexh = np.where(self.label_buffer == 1)[0]
        n_length = labexn.size
        h_length = labexh.size

        if not batch % 2 == 0:
            print('ERROR:Batch size must be even')
            print('Abort.')
            quit()
        else:
            num = batch//2
            if num>=n_length or num>=h_length:
                print('ERROR:Batch size exceeds data size')
                print('Abort.')
                quit()
            else:
                if self.ptr_n+num <n_length:
                    idxn = labexn[self.ptr_n:self.ptr_n+num]
                elif self.ptr_n+num >=n_length:
                    idxn = np.concatenate((labexn[self.ptr_n:n_length], labexn[0:self.ptr_n+num-n_length]))
                self.ptr_n = update_ptr(self.ptr_n, num, n_length)
                if self.ptr_h+num <h_length:
                    idxh = labexh[self.ptr_h:self.ptr_h+num]
                elif self.ptr_h+num >=h_length:
                    print("Epoch {} finished.".format(self.epoch_cnt))
                    self.epoch_cnt += 1
                    idxh = np.concatenate((labexh[self.ptr_h:h_length], labexh[0:self.ptr_h+num-h_length]))
                self.ptr_h = update_ptr(self.ptr_h, num, h_length)
                #print self.ptr_n, self.ptr_h
                label = np.concatenate((np.zeros(num), np.ones(num)))
                #label = processlabel(label,2, delta1, delta2)
                ft_batch = np.concatenate((self.ft_buffer[idxn], self.ft_buffer[idxh]))
                ft_batch_nhs = self.ft_buffer[idxn]
                label_nhs = np.zeros(num)
        return ft_batch, label, ft_batch_nhs, label_nhs


    '''
    nextbatch_without_balance: returns the normal batch. Suggest to use for training and validation
    '''
    def nextbatch_without_balance_alpha(self, batch, channel=None, delta1=0, delta2=0):
        def update_ptr(ptr, batch, length):
            if ptr+batch<length:
                ptr+=batch
            if ptr+batch>=length:
                ptr=ptr+batch-length
            return ptr
        if self.ptr + batch < self.maxlen:
            label = self.label_buffer[self.ptr:self.ptr+batch]
            ft_batch = self.ft_buffer[self.ptr:self.ptr+batch]
        else:
            print("Epoch {} finished.".format(self.epoch_cnt))
            self.epoch_cnt += 1
            label = np.concatenate((self.label_buffer[self.ptr:self.maxlen], self.label_buffer[0:self.ptr+batch-self.maxlen]))
            ft_batch = np.concatenate((self.ft_buffer[self.ptr:self.maxlen], self.ft_buffer[0:self.ptr+batch-self.maxlen]))
        self.ptr = update_ptr(self.ptr, batch, self.maxlen)
        return ft_batch, label
    
    
    def nextbatch(self, batch, channel=None, delta1=0, delta2=0):
        #print('recommed to use nextbatch_beta() instead')
        databat=None
        temp_fea=[]
        label=None
        if batch>self.maxlen:
            print('ERROR:Batch size exceeds data size')
            print('Abort.')
            quit()
        if self.ptr+batch < self.maxlen:
            temp_label = self.label_buffer[self.ptr:self.ptr+batch]
            label=processlabel(temp_label, 2, delta1, delta2)
            for dirname, dirnames, filenames in os.walk(self.dat):
                for i in range(0, len(filenames)-1):
                    # if i % 2 == 0:
                    #     continue
                    if i==0:
                        file='/dc.csv'
                        path=self.dat+file
                        with open(path) as f:
                            temp_fea.append(np.genfromtxt(islice(f, self.ptr, self.ptr+batch),delimiter=','))
                    else:
                        file='/ac'+str(i)+'.csv'
                        path=self.dat+file
                        with open(path) as f:
                            temp_fea.append(np.genfromtxt(islice(f, self.ptr, self.ptr+batch),delimiter=','))
            self.ptr=self.ptr+batch
        elif (self.ptr+batch) >= self.maxlen:
            
            #processing labels
            with open(self.label) as l:
                a=np.genfromtxt(islice(l, self.ptr, self.maxlen),delimiter=',')
            with open(self.label) as l:
                b=np.genfromtxt(islice(l, 0, self.ptr+batch-self.maxlen),delimiter=',')
            #processing data
            if self.ptr==self.maxlen-1 or self.ptr==self.maxlen:
                temp_label=b
            elif self.ptr+batch-self.maxlen==1 or self.ptr+batch-self.maxlen==0:
                temp_label=a
            else:
                temp_label=np.concatenate((a,b))
            label=processlabel(temp_label,2, delta1, delta2)
            #print label.shape
            for dirname, dirnames, filenames in os.walk(self.dat):
                for i in range(0, len(filenames)-1):
                    # if i % 2 == 0:
                    #     continue
                    if i == 0:
                        file='/dc.csv'
                        path=self.dat+file
                        with open(path) as f:
                            a=np.genfromtxt(islice(f, self.ptr, self.maxlen),delimiter=',')
                        with open(path) as f:
                            b=np.genfromtxt(islice(f, None, self.ptr+batch-self.maxlen),delimiter=',')
                        if self.ptr==self.maxlen-1 or self.ptr==self.maxlen:
                            temp_fea.append(b)
                        elif self.ptr+batch-self.maxlen==1 or self.ptr+batch-self.maxlen==0:
                            temp_fea.append(a)
                        else:
                            try:
                                temp_fea.append(np.concatenate((a,b)))
                            except:
                                print (a.shape, b.shape, self.ptr)
                    else:
                        file='/ac'+str(i)+'.csv'
                        path=self.dat+file
                        with open(path) as f:
                            a=np.genfromtxt(islice(f, self.ptr, self.maxlen),delimiter=',')
                        with open(path) as f:
                            b=np.genfromtxt(islice(f, 0, self.ptr+batch-self.maxlen),delimiter=',')
                        if self.ptr==self.maxlen-1 or self.ptr==self.maxlen:
                            temp_fea.append(b)
                        elif self.ptr+batch-self.maxlen==1 or self.ptr+batch-self.maxlen==0:
                            temp_fea.append(a)
                        else:
                            try:
                                temp_fea.append(np.concatenate((a,b)))
                            except:
                                print (a.shape, b.shape, self.ptr)
            self.ptr=self.ptr+batch-self.maxlen
        return np.rollaxis(np.asarray(temp_fea), 0, 3)[:,:,0:channel], label
