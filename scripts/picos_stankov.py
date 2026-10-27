# SPDX-License-Identifier: GPL-3.0-or-later
"""Symmetry check of the argon case of Stankov et al. (Section 7.2 of the article): peak
current of each half period and mean power of each period, from superficie.dat, with
I = dQ/dt and Q = eps0 eps_d (V - phi_left) A / Delta, A = 7.26 cm2.

Usage: python3 scripts/picos_stankov.py DIR [DIR ...]
"""
import json, sys, numpy as np
EPS0=8.8541878128e-12; A=7.26e-4; T=4.1666666666666665e-05
for d in sys.argv[1:]:
    a=json.load(open(d+"/parametros.json"))["argumentos"]
    s=np.loadtxt(d+"/superficie.dat",skiprows=1,ndmin=2)
    t,V,phi=s[:,0],s[:,1],s[:,2]
    q=EPS0*a["eps_dieletrico"]*(V-phi)/a["espessura"]*A
    I=np.gradient(q,t)*1e3
    n=int(round(t[-1]/T))
    print("==", d, "v_peak", a["v_pico"], "periods", n)
    for k in range(n):
        m1=(t>=k*T)&(t<(k+0.5)*T); m2=(t>=(k+0.5)*T)&(t<(k+1)*T)
        i1=I[m1]; i2=I[m2]
        j1=np.argmax(abs(i1)); j2=np.argmax(abs(i2))
        P=np.trapezoid(V[(t>=k*T)&(t<=(k+1)*T)], q[(t>=k*T)&(t<=(k+1)*T)])/T
        print(f"{k+1:3d}  first half {i1[j1]:+7.2f} mA at {(t[m1][j1]-k*T)*1e6:5.2f} us | second half {i2[j2]:+7.2f} mA at {(t[m2][j2]-k*T)*1e6:5.2f} us | P {P:.4f} W")
