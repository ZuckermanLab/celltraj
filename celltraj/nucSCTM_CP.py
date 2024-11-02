from __future__ import division, print_function; __metaclass__ = type
import numpy as np
import os, sys
import subprocess
import h5py
import matplotlib
import matplotlib.pyplot as plt
from skimage import transform as tf
from scipy.optimize import minimize
from scipy import ndimage
import scipy
import pickle
from pystackreg import StackReg
import numpy.matlib

protocol_Pickle = pickle.HIGHEST_PROTOCOL

class nucCellMasksCCRimgs():
    
    def __init__(self):
        """
        Work-in-progress init function. For now, just start adding attribute definitions in here.
        Todo
        ----
        - Most logic from initialize() should be moved in here.
        - Also, comment all of these here. Right now most of them have comments throughout the code.
        - Reorganize these attributes into some meaningful structure
        """
        
    def initialize(self,fileSpecifier,modelName):
        self.modelName=modelName
        pCommand='ls '+fileSpecifier
        p = subprocess.Popen(pCommand, stdout=subprocess.PIPE, shell=True)
        (output, err) = p.communicate()
        output=output.decode()
        fileList=output.split('\n')
        fileList=fileList[0:-1]
        self.fileList=fileList
        nF=len(fileList)
        self.nF=nF
        self.visual=False
        self.imgdim=2
        self.imgchannel=None #None for single-channel images, or chosen channel for multi-channel images
        self.mskchannel=None #None for single-channel masks, or chosen channel for multi-channel masks
        self.maximum_cell_size=250 #biggest linear edge of square holding a single cell image
        self.ntrans=30
        self.maxtrans=60.0 #sqrt number of points and max value for brute force registration
        try:
            self.get_image_data(1)
            self.imagesExist=True
        except:
            sys.stdout.write('problem getting images \n')
            self.imagesExist=False

    def get_image_data(self,n_frame):
        """
        Example function with PEP 484 type annotations.
        The return type must be duplicated in the docstring to comply
        with the NumPy docstring style.
        Parameters
        ----------
        param1
            The first parameter.
        param2
            The second parameter.
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        self.n_frame=n_frame
        nF=self.nF
        timeList=np.array([])
        imgfileList=np.array([])
        imgs=[None]*nF
        nucImgs=[None]*nF
        nmsks=[None]*nF
        e_images=np.zeros(nF)
        for iF in range(self.nF):
            fileName=self.fileList[iF]
            try:
                dataIn=h5py.File(fileName,'r')
                dsetName = "/images/img_%d/image" % int(n_frame)
                e = dsetName in dataIn
                if e:
                    e_images[iF]=1
                    dset=dataIn[dsetName]
                    imgs[iF]=dset[:]
                    time=dset.attrs['time']
                    dset_nuc_msk = "/images/img_%d/nmsk" % int(n_frame) # Retrieve nuclear masks info
                    dset_nmsk=dataIn[dset_nuc_msk]
                    nmsks[iF]=dset_nmsk[:]
                    
                    dset_nucImg = "/images/img_%d/nucImg" % int(n_frame) # Retrieve nuclear image info
                    dset_nuc = dataIn[dset_nucImg]
                    nucImgs[iF] = dset_nuc[:]
                    timeList = np.append(timeList, time)
                    imgfileList = np.append(imgfileList, iF)
                dataIn.close()
            except:
                sys.stdout.write('error in '+fileName+str(sys.exc_info()[0])+'\n')
        indimages = np.where(e_images > 0)
        imgs = np.array(imgs)
        nucImgs = np.array(nucImgs)
        nmsks = np.array(nmsks)
        imgs = imgs[indimages]
        nucImgs = nucImgs[indimages]
        nmsks = nmsks[indimages]
        if imgs.ndim < 3:
            imgs = imgs[0]
            imgs = np.expand_dims(imgs, axis = 0)
            nucImgs = nucImgs[0]
            nucImgs = np.expand_dims(nucImgs, axis = 0)
            nmsks = nmsks[0]
            nmsks = np.expand_dims(nmsks, axis = 0)
        if self.imgchannel is None:
            pass
        else:
            imgs = imgs[:, :, :, self.imgchannel]
        if self.mskchannel is None:
            pass
        else:
            nucImgs = nucImgs[:, :, :, self.mskchannel]
            nmsks = nmsks[:, :, :, self.mskchannel]
        self.imgs = imgs
        self.nucImgs = nucImgs
        self.nmsks = nmsks
        self.timeList = timeList
        self.imgfileList = imgfileList

    def get_frames(self):
        numFiles=np.array([])
        numImages=np.array([])
        frameList=np.array([])
        nImage=1
        n_frame=0
        while nImage>0:
            nImage=0
            for iF in range(self.nF):
                fileName=self.fileList[iF]
                try:
                    dataIn=h5py.File(fileName,'r')
                    dsetName = "/images/img_%d/image" % int(n_frame)
                    e = dsetName in dataIn
                    if e:
                        nImage += 1
                    dataIn.close()
                except:
                    sys.stdout.write('no images in '+fileName+str(sys.exc_info()[0])+'\n')
            if nImage>0:
                numImages = np.append(numImages, nImage)
                sys.stdout.write('Frame '+str(n_frame)+' has '+str(nImage)+' images...\n')
            n_frame += 1    
        self.numImages = numImages
        self.maxFrame = numImages.size - 1

    def get_imageSet(self,start_frame,end_frame):
        sys.stdout.write('getting images frame: '+str(start_frame)+'...\n')
        self.get_image_data(start_frame)
        self.imgSet = self.imgs.copy()
        self.nucImgSet = self.nucImgs.copy()
        self.nmskSet = self.nmsks.copy()
        self.imgfileSet = self.imgfileList.copy()
        self.frameSet = start_frame*np.ones_like(self.imgfileSet)
        self.timeSet = self.timeList.copy()
        self.start_frame = start_frame
        self.end_frame = end_frame
        for iS in range(start_frame+1, end_frame+1):
            sys.stdout.write('getting images frame: '+str(iS)+'...\n')
            self.get_image_data(iS)  
            self.imgSet = np.append(self.imgSet, self.imgs, axis=0)
            self.nucImgSet = np.append(self.nucImgSet, self.nucImgs, axis=0)
            self.nmskSet = np.append(self.nmskSet, self.nmsks, axis=0)
            self.imgfileSet = np.append(self.imgfileSet, self.imgfileList)
            self.frameSet = np.append(self.frameSet, iS*np.ones_like(self.imgfileList))
            self.timeSet = np.append(self.timeSet, self.timeList)
        self.imgfileSet = self.imgfileSet.astype(int)
        self.frameSet = self.frameSet.astype(int)

    def get_imageSet_trans_turboreg(self):
        nimg=self.imgfileSet.size
        tSet=np.zeros((nimg,3))
        stack_inds=np.unique(self.imgfileSet).astype(int)
        for istack in stack_inds:
            sys.stdout.write('registering phase images '+self.fileList[istack]+'\n')
            inds=np.where(self.imgfileSet==istack)
            inds=inds[0]
            img0=self.imgSet[inds,:,:]
            img0=np.abs(img0)>0
            sr = StackReg(StackReg.TRANSLATION)
            tmats = sr.register_stack(img0, reference='previous')
            nframes=tmats.shape[0]
            for iframe in range(nframes):
                tmatrix=tmats[iframe,:,:]
                tSet[inds[iframe],1]=tmatrix[0,2] 
                tSet[inds[iframe],2]=tmatrix[1,2]
                sys.stdout.write('    stack '+str(istack)+' frame '+str(iframe)+' transx: '+str(tSet[inds[iframe],1])+' transy: '+str(tSet[inds[iframe],2])+'\n')
        self.imgSet_t=tSet

    def get_nucImgSet_trans_turboreg(self):
        nimg=self.imgfileSet.size
        tSet=np.zeros((nimg,3))
        stack_inds=np.unique(self.imgfileSet).astype(int)
        for istack in stack_inds:
            sys.stdout.write('registering nuclear images '+self.fileList[istack]+'\n')
            inds=np.where(self.imgfileSet==istack)
            inds=inds[0]
            img0=self.nucImgSet[inds,:,:]
            img0=np.abs(img0)>0
            sr = StackReg(StackReg.TRANSLATION)
            tmats = sr.register_stack(img0, reference='previous')
            nframes=tmats.shape[0]
            for iframe in range(nframes):
                tmatrix=tmats[iframe,:,:]
                tSet[inds[iframe],1]=tmatrix[0,2] 
                tSet[inds[iframe],2]=tmatrix[1,2]
                sys.stdout.write('    stack '+str(istack)+' frame '+str(iframe)+' transx: '+str(tSet[inds[iframe],1])+' transy: '+str(tSet[inds[iframe],2])+'\n')
        self.imgSet_t=tSet

