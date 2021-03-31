# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 16:29:51 2021

@author: Ekaterina.tolstaya
"""

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

def MNinv(A,Wm,d,m0,mapr,mu,ml,eps,rel,Nit):
# % Input parameters:
# %           A - forward model matrix
# %           Wm - weighting vector (size Wm = size m0)
# %           d - data
# %           m0 - initial approximation, here it is not used. m=0 is used.
# %           mapr - a priori model
# %           mu, ml - upper and lower boundary for density
# %           eps - desired preciseness
# %           rel - relaxation parameter for alpha
# %           e - focusing parameter 
# %           Nit - maximum number of iterations
# %           Nz - number of cells in vertical direction (used in MGS inv, and useless here)
# %Output parameters:
# %           m - final model (uor solution)
# %           x - number of performed iterations
# %           al - vector of alphas in every itration
# %           P1 - values of a parametric functional
# %           S - values of a stabilizer functional
# %           I - values of a misfit
# %                      size P1 = size I = size al = size S = vector with length x
    Nd, Nm=A.shape 
    maprw=Wm*mapr
    print('Starting smooth inversion...')
    Aw=np.dot(A,np.diag(1/Wm))
    m=m0
    
    der0=(-ml)*(mu)/(mu-ml)
    der0=np.ones(ml.shape)
    mlog=np.log(np.abs((m-ml)/(mu-m)))
    mwlog=mlog*Wm
    maprw=mapr*Wm
    
    dm=((m-ml)*(mu-m)/(mu-ml))
    F=Aw
    l = -np.dot(F.T,d)
    l1=l
    sold = np.dot(l.T,l) 
    mwlog=-l*(np.dot(l1.T,l)/ np.dot( np.dot(F,l).T, np.dot(F,l)  ))*der0
    mlog = mwlog/Wm 
    m= mu -  (mu -ml )/(1+np.exp(mlog))
    mw=m*Wm

    r=np.dot(Aw,mw)-d
    mfit=np.dot(r.T,r)  
    alp=np.dot(r.T,r)/np.dot(mw.T,mw)
    alp=np.linalg.norm(d)
    x=2
    P=mfit
    print('alpha=',alp)
    print('iteration 1 , misfit=',mfit,', functional=',P)
    al = []
    P1 = []
    S = []
    I = []
    al.append(alp)
    P1.append(P)
    S.append(np.dot((mw-maprw).T,mw-maprw))
    I.append(mfit)
    # Conjugate gradient iterations
    while (mfit/np.linalg.norm(d)>eps and x<Nit):

        for x2 in range(15):
            
            F=Aw    # % Frechet matrix (derivative of A)
            dm=((m-ml)*(mu-m)/(mu-ml))/der0
            
            l = np.dot(F.T,r)*dm  +  alp*(mw-maprw)
            s = np.dot(l.T,l)
            if (s==0):
                print('s=0!')
                break
            l1 = l+l1*s/sold
            sold=s  # % cjg direction
            
            f=np.dot(F,l1*dm)
            k = np.dot(l1.T,l)/(np.dot(f.T,f)+alp*np.dot(l1.T,l1))
            
            mwlog=mwlog-k*l1
            mlog=mwlog/Wm 
            m=mu - (mu-ml)/(1+np.exp(mlog/der0))
            mw=m*Wm
              
            r=np.dot(A,m)-d
            mfit=np.dot(r.T,r)
            stab=np.dot((mw-maprw).T,(mw-maprw))
            P=mfit+alp*stab
            al.append(alp)
            P1.append(P)
            S.append(stab)
            I.append(mfit)
            x=x+1
    
            
            print('iteration ',x,', misfit=',mfit/np.linalg.norm(d),', functional=',P)
            if (x>Nit):
                break
        
        alp=alp*rel
        print('alpha=',alp)
        if (alp<1e-15):
            break
            
        print('I made ',x-1,' iterations to find min norm solution, misfit=',mfit)    
    
    return m, x, al, P1, S, I




