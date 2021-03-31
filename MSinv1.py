# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 15:17:20 2021

@author: Ekaterina.tolstaya
"""

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

def logtr(m,uc,lc,C):
    #%out=                C.*log(abs((m-lc)./(uc-m)));
    out= C*(0.5*(uc+lc) + np.log(abs((m-lc)/(uc-m))))
    #%out=C.*tan((m-lc)*pi./(uc-lc)-pi/2);
    return out

def mvtr(m,e,C):
    out=m/np.sqrt(e*e+m**2);
    #%out=sqrt(abs(m)./(e*e+abs(m)));
    #%out=sqrt(log(1+(m/e).^2));
    return out

def imvtr(mv,e,C):
    out=e*mv/np.sqrt(1-(mv)**2)
    #%out=e^2*mv.^2./(1-mv.^2);
    #%out=e*sqrt(exp(mv.^2)-1);
    return out

def ilogtr(ml,uc,lc,C):
    #%out=uc - (uc-lc)./(1+exp(ml./C));
    out= uc - (uc-lc)/(1+np.exp((ml - 0.5*(uc+lc))/C))
    #%out=lc+(uc-lc)/2.*(2/pi*atan(ml./C)+1);
    return out

def dlogtr(m,uc,lc,C):
    out=(m-lc)*(uc-m)/(uc-lc)/C
    return out
def darctr(ml,uc,lc,C):
    out=(uc-lc)/np.pi*C/(C*C+ml**2)
    return out

def dmvtr(mv,e,C):
    out=e*(1-mv**2)**(-1.5)
    #%out=2*e^2*mv.*((1-mv).^(-2));
    #%if (mv) 
    #%    out=e*(mv.*sqrt(exp(mv.^2)-1) + mv./sqrt(exp(mv.^2)-1));
    #%else out=e;
    #%end;
    return out

def MSinv(A,Wm,d,m0,mapr,mu,ml,eps,rel,e,Nit,Nz):
# % Input parameters:
# %           A - forward model matrix
# %           Wm - weighting vector (size Wm = size m0)
# %           d - data
# %           m0 - initial approximation
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
    m=m0

    print('Starting MS inversion...');
    Aw=np.dot(A,np.diag(1/Wm))
    Pold=1e35
    normd=np.linalg.norm(d)
    
    m[m<ml]=ml[m<ml]
    m[m>mu]=mu[m>mu]
 
    mvu=mvtr(mu,e,1)
    mvl=mvtr(ml,e,1)
    mv=mvtr(m,e,1)
    
    der0=np.ones(mvu.shape)
    #%der0=((-mvl).*(mvu)./(mvu-mvl));
    #max(der0)
    
    mvlog=logtr(mv,mvu,mvl,der0)
    mvwlog=mvlog*Wm
    maprw=mapr*Wm
    
    F=np.dot(Aw,np.diag((dlogtr(mv,mvu,mvl,der0))*(dmvtr(mv,e,1))))
       
    l = np.dot(F.T,(  np.dot(A,m0) -d))
    l1=l
    sold = np.dot(l.T,l) 
    mvwlog=mvwlog-l*( np.dot(l1.T,l))/( np.dot(  np.dot(F,l).T,   np.dot(F,l)))
    mvlog=mvwlog/Wm 
    mv=ilogtr(mvlog,mvu,mvl,der0)
    m=imvtr(mv,e,1)
    mw=m*Wm
    r=np.dot(A,m)-d 
    mfit=np.dot(r.T,r)
    stab=sum(((mw-maprw)**2)/((mw-maprw)**2 + e*e));
    r0 = np.dot(A,m0)-d
    alp=np.dot(r0.T,r0)/stab
    x=0
    P0=mfit
    print('iteration ',x,', misfit=',mfit/normd,', functional=%f',P0)
    al = []
    P1 = []
    S = []
    I = []
            
    while (mfit/normd>eps and x<Nit):
        
        for x2 in range(Nd):
            
            F=Aw
            dm=(dlogtr(mv,mvu,mvl,der0))*(dmvtr(mv,e,1))
            l = np.dot(F.T,r)*dm+alp*mv
            s = np.dot(l.T,l)
            if (s==0):
                break
            l1 = l+l1*s/sold  # % cjg direction
            sold=s
            f=np.dot(F, l1*dm )
            
            k = np.dot(l1.T,l)/(   np.dot(f.T,f)+alp*(  np.dot(l1.T,l1)   ))

            mvwlog=mvwlog-k*l1
            mvlog=mvwlog/Wm 
            mv=ilogtr(mvlog,mvu,mvl,der0)
            m=imvtr(mv,e,1)
            mw=m*Wm
            r=np.dot(A,m)-d
            mfit=np.dot(r.T,r)
            stab=np.sum(((mw-maprw)**2)/((mw-maprw)**2 + e*e))
            P=mfit+alp*stab
            
            print('iteration ',x,', misfit=',mfit/normd,', functional=',P/P0)
            
            al.append(alp)
            P1.append(P)
            S.append(stab)
            I.append(mfit)
            x=x+1
                 
            if (abs(Pold-P)<0.0001*P):
                    break
            if (P>=Pold): 
                    break
            Pold=P
            if (mfit/normd<eps or x>Nit):
                break


        alp=alp*rel;
        print('alpha=',alp)
        print('I made ',x-1,' iterations to find focused solution, misfit/norm(d)=',mfit/normd)
        
    return m, x, al, P1, S, I        




