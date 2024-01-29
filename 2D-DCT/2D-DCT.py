import cv2
import numpy as np
import time
import math

def dct2(img):
    M, N = img.shape
    dct_img = np.zeros((M, N), np.float32)
    Cu = np.ones(M)
    Cu[0] = 1/np.sqrt(2)
    Cv = np.ones(N)
    Cv[0] = 1/np.sqrt(2)
    cos_x = np.cos(np.pi*(2*np.arange(M)[:, np.newaxis] + 1)*np.arange(M)/(2*M))
    cos_y = np.cos(np.pi*(2*np.arange(N)[:, np.newaxis] + 1)*np.arange(N)/(2*N))
    dct_img = (2/np.sqrt(M*N))*np.outer(Cu, Cv)*np.dot(cos_x.T, img).dot(cos_y)
    return dct_img

def idct2(img):
    M, N = img.shape
    idct_img = np.zeros((M, N), np.float32)
    Cu = np.ones(M)
    Cu[0] = 1/np.sqrt(2)
    Cv = np.ones(N)
    Cv[0] = 1/np.sqrt(2)
    cos_u = np.cos(np.pi*(2*np.arange(M)[:, np.newaxis] + 1)*np.arange(M)/(2*M))
    cos_v = np.cos(np.pi*(2*np.arange(N)[:, np.newaxis] + 1)*np.arange(N)/(2*N))
    idct_img = (2/np.sqrt(M*N))*np.dot(cos_u, np.outer(Cu, Cv)*img).dot(cos_v.T)
    return idct_img.astype(np.uint8)

def dct1(img):
    N = img.shape[0]
    X = np.zeros(N)
    Cu = np.ones(N)
    Cu[0] = 1/np.sqrt(2)
    cos_x = np.cos(np.pi*(2*np.arange(N)[:,np.newaxis] + 1)*np.arange(N)/(2*N))
    X = np.sqrt(2/N)*Cu*np.dot(cos_x.T, img)
    return X

def two_dct1(img):
    M, N = img.shape
    dct_img = np.zeros((M, N))
    rows = dct1(img)
    cols = dct1(rows.T)
    dct_img = cols.T
    #for i in range(M):
        #dct_img[i,:] = dct1(img[i,:])
    #for j in range(N):
        #dct_img[:,j] = dct1(dct_img[:,j])
    return dct_img

if __name__ == "__main__":
    img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
    #img = cv2.resize(img, (128,128))
    #2D-DCT
    st = time.time()
    dct = dct2(img) 
    dct2_time = time.time()-st
    #2D-DCT visualize in log domain
    dct_log = np.log10(abs(dct))
    dct_log_scaled = (dct_log - np.min(dct_log)) / (np.max(dct_log) - np.min(dct_log))
    image_dct2 = np.clip(dct_log_scaled, 0, 1) 
  
    #2D-IDCT
    image_idct2 = idct2(dct).astype(np.uint8) 
   
    #Two 1D-DCT
    st = time.time()
    image_two_dct1 = two_dct1(img) 
    two_dct1_time = time.time()-st
    
    #Calculate PSNR
    mse = np.mean((img - image_idct2)**2)
    psnr = 20 * np.log10(255 / np.sqrt(mse))
    
    #Output
    cv2.imshow('2D-DCT',image_dct2)
    cv2.imshow('2D-IDCT',image_idct2)
    cv2.imshow('Two 1D-DCT',image_two_dct1)
    print('2D-DCT TIME:', dct2_time, 'sec')
    print('Two 1D-DCT TIME:', two_dct1_time, 'sec')
    print('PSNR:', psnr)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()