#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: DevAJ

#####FORWARD Layers are definrd in this block
"""

import numpy as np
def convolution_layer(image,filt,bias,stride=1):
    (no_filter, no_channel_filter, f, _) = filt.shape 
    no_channel, in_dim, _ = image.shape
    #print("no of channnels======",no_channel)
    #print("no of channnels in filter=  ",no_channel_filter)
    out_dim = int((in_dim - f)/stride)+1 
    
    assert no_channel == no_channel_filter, "Dimensions of filter or input image are not equal"
   #print("output dimm in convolution for===",out_dim)
    output = np.zeros((no_filter,out_dim,out_dim))
    for curr_f in range(no_filter):
        curr_y = out_y = 0
        while curr_y + f <= in_dim:
            curr_x = out_x = 0
            while curr_x + f <= in_dim:
                output[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:,curr_y:curr_y+f, curr_x:curr_x+f]) + bias[curr_f]
                curr_x += stride
                #print("current x in while after stride add= ",curr_x)
                out_x += 1
            curr_y += stride
            #print("current y== ",curr_y)
            out_y += 1
           
    return output

def maxpool_layer(image, f=2, stride=2):
    no_channel, h_prev, w_prev = image.shape
    h = int((h_prev - f)/stride)+1
    w = int((w_prev - f)/stride)+1
    # print("height=",h)
    # print("width= ",w)
    output = np.zeros((no_channel, h, w))
    for i in range(no_channel):
        curr_y = out_y = 0
        while curr_y + f <= h_prev:
            curr_x = out_x = 0
            while curr_x + f <= w_prev:
                output[i, out_y, out_x] = np.max(image[i, curr_y:curr_y+f, curr_x:curr_x+f])
                curr_x += stride
                #print("current x in while after stride add= ",curr_x)
                out_x += 1
            curr_y += stride
            out_y += 1
    return output

def nanargmax(arr):
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs    

def softmax(X):
    out = np.exp(X)
    return out/np.sum(out)

def relu(X):
    X[X<=0] = 0
    return X

def categoricalCrossEntropy(probs, label):
    return -np.sum(label * np.log(probs))

def finllayer(X,w3,b3):
    z = w3.dot(X) + b3
    return z

def initializeFilter(size, scale = 1.0):
    stddev = scale/np.sqrt(np.prod(size))
    return np.random.normal(loc = 0, scale = stddev, size = size)

def initializeWeight(size):
    return np.random.standard_normal(size=size) * 0.01

def flatten_layer(res):
    (nf2, dim2, _) = res.shape
    fnl = res.reshape((nf2 * dim2 * dim2, 1))    
    return fnl

import numpy as np

def convolution_backward(dconv_prev,conv_in,filt,stride):
    (no_filter, no_channel, f, _) = filt.shape
    (_, orig_dim, _) = conv_in.shape
    dout = np.zeros(conv_in.shape) 
    
    dfilt = np.zeros(filt.shape)
    
    dbias = np.zeros((no_filter,1))

    # print("no of filter ",no_filter)
    # print("original diamension ",orig_dim)

    for curr_f in range(no_filter):
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
    
                dfilt[curr_f] += dconv_prev[curr_f, out_y, out_x] * conv_in[:, curr_y:curr_y+f, curr_x:curr_x+f]
                #print("derivative filter",dfilt)
                dout[:, curr_y:curr_y+f, curr_x:curr_x+f] += dconv_prev[curr_f, out_y, out_x] * filt[curr_f] 
               #print(dout)
                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1

        dbias[curr_f] = np.sum(dconv_prev[curr_f])
    print(dout)
    return dout, dfilt, dbias

def maxpoolBackward(dpool, orig, f, stride):
    (no_channel, orig_dim, _) = orig.shape
    dout = np.zeros(orig.shape)
   # print("no of channels in maxpool back= ",no_channel)
    for channel in range(no_channel):
        
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            
            while curr_x + f <= orig_dim:
                
                (a, b) = nanargmax(orig[channel, curr_y:curr_y+f, curr_x:curr_x+f])
                
                dout[channel, curr_y+a, curr_x+b] = dpool[channel, out_y, out_x]
                #print("*****************dout=",dout)
                curr_x += stride
                
                out_x += 1
                
            curr_y += stride
            out_y += 1
        #print("##################################################",out_y)
    return dout


#%%

####### Create weight bias filters randomlyand test on image

import glob
import cv2
import pickle
import matplotlib.pyplot as plt
# =============================================================================
# save_path = '/home/dev/Documents/IMGPRO 2-11-2019/my model/models/paramsfnl.pkl'
# 
# #save_path = "/home/dev/Documents/IMGPRO 2-11-2019/my model/model/paramsfnl.pkl"
# #cost = train(save_path = save_path)
# params, cost = pickle.load(open(save_path, 'rb'))
# [f1, f2, w3, w4, b1, b2, b3, b4] = params
# 
# =============================================================================
#img=cv2.imread("/home/dev/Pictures/7.jpg")
#plt.imshow(img)
nooflabels=3
num_filt1=8
num_filt2=8
filt_size=5
img_depth=1
f1=(num_filt1 ,img_depth,filt_size,filt_size)
f2=(num_filt1 ,num_filt2,filt_size,filt_size)
w3=(128,800)
w4=(nooflabels,128)
f1=initializeFilter(f1)
f2=initializeFilter(f2)
w3=initializeWeight(w3)
w4=initializeWeight(w4)
b1 = np.zeros((f1.shape[0],1))
b2 = np.zeros((f2.shape[0],1))
b3 = np.zeros((w3.shape[0],1))
b4 = np.zeros((w4.shape[0],1))

# =============================================================================
# print("f1 shape==  ",f1.shape)
# print("f2 shape==  ",f2.shape)
# 
# print("w3 shape==  ",w3.shape)
# print("w4 shape==  ",w4.shape)
# print("b1 shape==  ",b1.shape)
# print("b2 shape==  ",b2.shape)
# print("b3 shape==  ",b3.shape)
# print("b4 shape==  ",b4.shape)
# =============================================================================
label=['2','5','7']
label = np.eye(nooflabels,nooflabels)
print(label)
i=0
#labels = np.eye(nooflabels).[int(nooflabels[1])].reshape(nooflabels, 1) 
#labels=[int(labels[0])].reshape(nooflabels, 1) 
#print("labes shape",labels.shape)



for img in glob.glob("/home/dev/Pictures/pic/*.jpg"):
    #print(img)
    if(i<nooflabels):
    #print("labels=====================",labels[i])
        labels=label[i].reshape(nooflabels,1) 
        print(i)
        i=i+1
    
    img= cv2.imread(img)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(28,28))
    #plt.imshow(img)
    img=img.reshape(1,28,28)    
    
    res = convolution_layer(img, f1, b1, stride=1)
    res=relu(res)
    conv1=res
    res = convolution_layer(res, f2, b2, stride=1) 
    res=relu(res)
    conv2=res
    res = maxpool_layer(res, f=2, stride=2)
    pooledfitures=res
    #print("before ====== ",res.shape)
    fnl=flatten_layer(res)
# =============================================================================
    z=finllayer(fnl,w3,b3)
    z=relu(z)
    #z = w3.dot(fnl) + b3 #dense
    #z[z<=0] = 0 #relu
    
   #out = w4.dot(z) + b4 # dense2
    
# =============================================================================
# =============================================================================
#     print("Z= ",z.shape)
#     print("B= ",b4.shape)
#     print("W= ",w4.shape)
# =============================================================================
    out=finllayer(z,w4,b4)
    probs = softmax(out)
      
    loss=categoricalCrossEntropy(probs,labels)
    dout = probs - labels  
        
    dw4 = dout.dot(z.T)
    #print("dw4= ",dw4)
    db4=np.sum(dout,axis=1).reshape(b4.shape)
    
    dz = w4.T.dot(dout) 
    dz[z<=0] = 0 
    dw3 = dz.dot(fnl.T)
    db3 = np.sum(dz, axis = 1).reshape(b3.shape)
    
    dfc = w3.T.dot(dz)
    dpool=dfc.reshape(pooledfitures.shape)
    #print(dpool) 
    dconv2 = maxpoolBackward(dpool, conv2, f=2, stride=2) 
    dconv2=relu(conv2)
     
    dconv1, df2, db2 = convolution_backward(dconv2, conv1, f2,stride=1)
    dconv1=relu(conv1)
    
    dimage, df1, db1 = convolution_backward(dconv1, img, f1, stride=1) 
    
    grads = [df1, df2, dw3, dw4, db1, db2, db3, db4] 

#%%
# =============================================================================
# import cv2
# import glob
# num_filt1=8
# num_filt2=8
# filt_size=28
# img_depth=1
# f1=(num_filt1 ,img_depth,filt_size,filt_size)
# f2=(num_filt2 ,img_depth,filt_size,filt_size)
# w3=(128,800)
# w4=(10,128)
# f1=initializeFilter(f1)
# f2=initializeFilter(f2)
# w3=initializeWeight(w3)
# w4=initializeWeight(w4)
# #print(f1)
# imgdata = []
# for img in glob.glob("/home/dev/Pictures/pic/*.jpg"):
#     n= cv2.imread(img)
#     n=cv2.cvtColor(n,cv2.COLOR_RGB2GRAY)
#     n=cv2.resize(n,(28,28))
#     n=n.reshape(1,28,28)
#     #print(n.shape)
#     imgdata.append(n)
# #print(imgdata)
# totalres=[]
# for i in imgdata:  
#     res=convolution_layer(i,f1,b3,stride=1)
#     totalres.append(res)
#     #print(res)
# print(totalres)
# =============================================================================

#%%

loss = categoricalCrossEntropy(probs, label)      
dout = probs - label 
dw4 = dout.dot(z.T) 
db4 = np.sum(dout, axis = 1).reshape(b4.shape)     
    dz = w4.T.dot(dout) 
    dz[z<=0] = 0 
    dw3 = dz.dot(fc.T)
    db3 = np.sum(dz, axis = 1).reshape(b3.shape)    
    dfc = w3.T.dot(dz) 
    dpool = dfc.reshape(pooled.shape) 
    
    dconv2 = maxpoolBackward(dpool, conv2, maxpool_filter, maxpool_stride) 
    dconv2[conv2<=0] = 0 
    
    dconv1, df2, db2 = convolutionBackward(dconv2, conv1, f2, conv_Stride) 
    dconv1[conv1<=0] = 0 
    
    dimage, df1, db1 = convolutionBackward(dconv1, image, f1, conv_Stride) 
    
    grads = [df1, df2, dw3, dw4, db1, db2, db3, db4] 
    
    return grads, loss

