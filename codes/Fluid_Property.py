# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 14:20:37 2019

@author: Jamiu Ekundayo
"""

import numpy as np
from scipy.optimize import fsolve
import math as mt


def gas_compr_factor(Input_Pressure, Input_Temperature, EOS_method,Gas, phase=0):
    
    """
    Calculates gas compressibility factor for methane and helium using 
    six different EOS
    
    It can be expanded to accommodate more pure gases and even gas mixtures
    
    Inputs:
    
        1. Input_Pressure = List of Pressures in bar
        2. Input_Temperature = List of Temperatures in Celsius
        3. EOS method
            - PR
            - PR-Peneloux
            - SRK
            - SRK-Peneloux
            - SBWR
            - Lee-Kesler.
        4. Gas - only Methane and Helium. 
            
            
    
    """
    # constants 
    R = 8.314       # Gas constant, Jmol-1K-1
    
    #EOS parameters for methane
    if Gas == 'Methane':
        Tc = 190.40     # Critical temperature of methane, K
        Pc = 4.6        # Critical pressure of methane, Mpa
        omega = 0.0113
    
    elif Gas == 'Helium':
        Tc = 5.19     # Critical temperature of methane, K
        Pc = 0.227        # Critical pressure of methane, Mpa
        omega = 0.0        
    
    Compr_factor = []
    for j in range(len(Input_Pressure)):
        
        pre = Input_Pressure[j]*1e5
      
        Temp = 273.15 + Input_Temperature[j] 
        Tr = Temp/Tc

        if EOS_method == "PR":
            
            #omega = 0.008 #0.0113
            #Alp = mt.exp((2.0 + 0.8145*Tr)*(1-Tr**(0.134 + 0.508*omega -0.0467*omega**2)))
            Alp = (1+(0.37464 + 1.54226*omega - 0.26992*omega**2)*(1-Tr**0.5))**2
            
            a = 0.457535*Alp*R**2*Tc**2/(Pc*1e6)
            
            b = 0.077796*R*Tc/(Pc*1e6)
                   
            A = a*pre/(R**2*Temp**2)
            
            B = b*pre/(R*Temp)
            
            coeff0 = 1
            coeff1 = -1*(1-B)
            coeff2 = A - 2*B - 3*B**2
            coeff3 = -1*(A*B - B**2 - B**3)
            
            coeffs = np.array([coeff0, coeff1, coeff2, coeff3])
        
            Z = abs(max(np.roots(coeffs)))
            
        elif EOS_method == "PR-Peneloux":
            #omega = 0.008 #0.0113            
            
            #Alp = mt.exp((2.0 + 0.8145*Tr)*(1-Tr**(0.134 + 0.508*omega -0.0467*omega**2)))
            Alp = (1+(0.37464 + 1.54226*omega - 0.26992*omega**2)*(1-Tr**0.5))**2  
            cpen = 0.40768*R*Tc/(Pc*1e6)*(0.29441-0.29056+0.08775*omega)
            a = 0.457535*Alp*R**2*Tc**2/(Pc*1e6)
            b = 0.077796*R*Tc/(Pc*1e6)# - cpen
            A = a*pre/(R**2*Temp**2)
            
            B = b*pre/(R*Temp)
            C = cpen*pre/(R*Temp)
            
            coeff0 = 1
            coeff1 = B + 4*C-1
            coeff2 = A -2*B -3*B**2 -4*B*C -2*C + 2*C**2   #A-2B-〖3B〗^2-4BC-2C+〖2C〗^2
            coeff3 = -1*(A*B - B**2 - B**3 + 2*B*C**2 + 2*C**2)           #AB-B^2-B^3  +2BC^2+2C^2

            coeffs = np.array([coeff0, coeff1, coeff2, coeff3])
        
            Z = abs(max(np.roots(coeffs)))
            
        elif EOS_method == "SRK":
            #omega = 0.008
            Alp = (1+(0.48 + 1.574*omega - 0.176*omega**2)*(1-Tr**0.5))**2
            
            a = 0.42748*Alp*R**2*Tc**2/(Pc*1e6)
            
            b = 0.08664*R*Tc/(Pc*1e6)
                   
            A = a*pre/(R**2*Temp**2)
            
            B = b*pre/(R*Temp)
            
            coeff0 = 1
            coeff1 = -1
            coeff2 = A - B - B**2
            coeff3 = -1*A*B

            coeffs = np.array([coeff0, coeff1, coeff2, coeff3])
        
            Z = abs(max(np.roots(coeffs)))         
                
        elif EOS_method == "SRK-Peneloux":
            #omega = 0.008
            Alp = (1+(0.48 + 1.574*omega - 0.176*omega**2)*(1-Tr**0.5))**2
            
            a = 0.42748*Alp*R**2*Tc**2/(Pc*1e6)
            cpen = 0.40768*R*Tc/(Pc*1e6)*(0.29441-0.29056+0.08775*omega)
            b = 0.08664*R*Tc/(Pc*1e6) #- cpen
            
            A = a*pre/(R**2*Temp**2)
            
            B = b*pre/(R*Temp)
            
            C = cpen*pre/(R*Temp)
            
            coeff0 = 1
            coeff1 = 3*C-1
            coeff2 = A - B - B**2 - 2*B*C - 3*C + 2*C**2
            coeff3 = -1*(A*B + B*C + 2*C**2 + C*B**2 + 2*B*C**2)

            coeffs = np.array([coeff0, coeff1, coeff2, coeff3])
        
            Z = abs(max(np.roots(coeffs)))
        
        elif EOS_method == "SBWR":    # Soave (1999)'s modified BWR EOS
            Zc = 0.2908 - 0.099*omega + 0.04*omega**2;
            # Definition of parameters
            d1 = 0.4912 + 0.6478*omega;
            d2 = 0.3 + 0.3619*omega;
            e1 = 0.0841 + 0.1318*omega + 0.0018*omega**2;
            e2 = 0.075 + 0.2408*omega - 0.014*omega**2;
            e3 = -0.0065 + 0.1798*omega - 0.0078*omega**2;
            f = 0.77;
            e = (2 - 5*Zc)*mt.exp(f)/(1 + f + 3*f**2 - 2*f**3);
            d = (1 - 2*Zc - e*(1 + f - 2*f**2)*mt.exp(-f))/3;
            b = Zc - 1 - d - e*(1 + f)*mt.exp(-f);
            bc = b*Zc;
            dc = d*Zc**4;
            ec = e*Zc**2;
            ff = f*Zc**2;
            beta = bc + 0.422*(1 - 1/Tr**1.6) + 0.234*omega*(1 - 1/Tr**3);
            delta = dc*(1 + d1*(1/Tr - 1) + d2*(1/Tr - 1)**2);
            eta = ec + e1*(1/Tr - 1) + e2*(1/Tr - 1)**2 + e3*(1/Tr - 1)**3;
            
            if Tr>1:
                y0 = pre/(Pc*1e6)/Tr/(1+beta*pre/(Pc*1e6)/Tr);
            else:
                if phase==0:
                    y0 = pre/(Pc*1e6)/Tr/(1+beta*pre/(Pc*1e6)/Tr);
                else:
                    y0 = 1/Zc**(1+(1-Tr)**(2/7));
            
            fun = lambda y: y*(1+beta*y+delta*y**4+eta*y**2*(1+ff*y**2)*mt.exp(-ff*y**2))-pre/(Pc*1e6)/Tr;
            
            yi = fsolve(fun,y0);
            
            Z = float(pre/(Pc*1e6)/yi/Tr);

        elif EOS_method == "Lee-Kesler":

            omega_r = 0.3978
            #Pr = pre/(Pc*1e6)
            def LeeKesler(fluid):
                """
                Lee-Kesler equation of state
                Based on the repo
                https://github.com/j-jith/pythermophy/blob/master/pythermophy/lee_kesler.py
                
                """

                b1 = [0.1181193, 0.2026579]
                b2 = [0.265728, 0.331511]
                b3 = [0.154790, 0.027655]
                b4 = [0.030323, 0.203488]
            
                c1 = [0.0236744, 0.0313385]
                c2 = [0.0186984, 0.0503618]
                c3 = [0.0, 0.016901]
                c4 = [0.042724, 0.041577]
            
                d1 = [0.155488e-4, 0.48736e-4]
                d2 = [0.623689e-4, 0.0740336e-4]
            
                beta = [0.65392, 1.226]
                gamma = [0.060167, 0.03754]
 
                if fluid=='simple':
                    i = 0
                elif fluid=='reference':
                    i = 1
                else:
                    return None
            
                B = b1[i] - b2[i]/Tr - b3[i]/Tr**2 - b4[i]/Tr**3
                
                C = c1[i] - c2[i]/Tr + c3[i]/Tr**3
                
                D = d1[i] + d2[i]/Tr
                
                obj_func = lambda x: -(pre/(Pc*1e6))*x/Tr + 1 + B/x + C/x**2 + D/x**5 + c4[i]/Tr**3/x**2 * (beta[i] + gamma[i]/x**2) * mt.exp(gamma[i]/x**2)
                
                init_vol = Tr/(pre/(Pc*1e6))

                #res = root(obj_func, init_vol, method='lm')
                res = fsolve(obj_func, init_vol)
                #Vr = res.x
                Vr = res

                return(Vr)
            z0 = (pre/(Pc*1e6))/Tr * LeeKesler('simple')
            zr = (pre/(Pc*1e6))/Tr * LeeKesler('reference')
        
            # departure term
            #z1 = (zr - z0)/omega_r
            
            Z = float(z0 + omega*(zr - z0)/omega_r)

        Compr_factor.append(float(format(Z, '.4f')))
        #Compr_factor.append(Z)

    return(Compr_factor)
