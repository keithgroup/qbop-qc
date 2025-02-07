# MIT License
# 
# Copyright (c) 2025, Barbaro Zulueta
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Computes the vibrational partition function energy contributions using QBOP-1
   rho_v :: compute the thermal correlator
   lnq :: the natural logarithm of the vibrational partition function
   Uvib :: vibrational internal energy
   CVvib :: the specific heat capacity at constant volume for vibrations
   Svib :: the entropy of vibrations
   conditional_statements :: conditational statements for bonding and anti-bonding
   
   """
import os,sys
import numpy as np
from . import parameters as param

# QBOP sub-equations
def rho_v(net_be, params, size):
    """Calculate the thermal correlator 
    
       Parameters
       ----------
       net_be: :obj:`numpy.ndarray`
           the net bond energies in a 1-d array
       params: :obj:`numpy.float64`
           the parameters for the thermal correlators
       size: :obj:`numpy.int64`
           the number of bonds in molecule 
           
       Returns
       -------
       value: :obj:`numpy.ndarray`  
           the value of the thermal correlator
       """
    values = []
    for l in range(size):
        alpha = np.delete(params[:,0],l)
        beta = np.delete(params[:,1],l)
        zeta = np.delete(params[:,2],l)
        phi = np.delete(params[:,3],l)
        kappa = np.delete(params[:,4],l)
        neigh_bes = np.delete(net_be,l)
        numerator = alpha - beta * np.exp( - zeta * (neigh_bes - phi)**2)
        denominator = kappa
        values += np.sum(numerator) / np.sum(denominator),
    return np.array(values)

def lnq(rho, zpebop2, be, k, T, df_vib):
    """Calculate the natural logarithm of the 
       vibrational partition function
    
       Parameters
       ----------
       rho: :obj:`numpy.ndarray`
           the thermal correlator
       zpebop2: :obj:`numpy.float64`
           the total zpes calculated from ZPE-BOP2
       be: :obj:`numpy.ndarray`
           the net bond energies in a 1-d array
       k: :obj:`numpy.ndarray`
           the scaled-shift parameter
       T: :obj:`numpy.float64`
           the temperature in K
       df_vib: :obj:`numpy.int64`
           the vibrational degree of freedoms
           
       Returns
       -------
       lnq_bot: :obj:`numpy.float64`     
           the value of the natural log vibrational partition
           starting from the ground state (e.g., bottom-of-the-well)
       lnq_v0: :obj:`numpy.float64`
           the value of the natural log vibrational partition
           starting from the first excited state
       """
        
    k_B = 1.987204259e-3 # Boltzmann's constant in kcal/(mol K)
    fraction1 = - zpebop2 / (k_B * T)
    fraction2 = np.float128( (k * be * rho) / (k_B * T))
    lnq_v0 = (df_vib / be.shape[0]) * np.sum(np.log(1 - np.exp(-fraction2))) 
    lnq_bot = fraction1 - lnq_v0 
    return (lnq_bot, lnq_v0)

def Uvib(rho, zpebop2, be, k, T, df_vib):
    """Calculate the internal vibrational thermal effects
    
       Parameters
       ----------
       rho: :obj:`numpy.ndarray`
           the thermal correlator
       zpebop2: :obj:`numpy.float64`
           the total zpes calculated from ZPE-BOP2
       be: :obj:`numpy.ndarray`
           the net bond energies in a 1-d array
       k: :obj:`numpy.ndarray`
           the scaled-shift parameter
       T: :obj:`numpy.float64`
           the temperature in K
       df_vib: :obj:`numpy.int64`
           the vibrational degree of freedoms
           
       Returns
       -------
       U_vib: :obj:`numpy.float64`  
           internal thermal vibrational contribution (kcal/mol)
       """
        
    k_B = 1.987204259e-3 # Boltzmann's constant in kcal/(mol K)
    R =  (8.314472 * 627.5098)/ (2625.5002e3) # Eh to kcal/mol
    fraction = np.float128( (k * be * rho) / (k_B * T))
    val =  (fraction * T) / (np.exp(fraction) - 1)
    U_vib = zpebop2 + (R * np.sum(val) * df_vib / be.shape[0])
    return U_vib

def CVvib(rho, be, k, T, df_vib):
    """Calculate the heat capacity at constant volume for vibrations
    
       Parameters
       ----------
       rho: :obj:`numpy.ndarray`
           the thermal correlator
       be: :obj:`numpy.ndarray`
           the net bond energies in a 1-d array
       k: :obj:`numpy.ndarray`
           the scaled-shift parameter
       T: :obj:`numpy.float64`
           the temperature in K
       df_vib: :obj:`numpy.int64`
           the vibrational degree of freedoms
           
       Returns
       -------
       CV_vib: :obj:`numpy.float64`   
           specific heat capacity at constant volume (kcal/(mol * T))
       """
        
    k_B = 1.987204259e-3 # Boltzmann's constant in kcal/(mol K)
    R =  (8.314472 * 627.5098)/ (2625.5002e3) # Eh to kcal/mol
    fraction = np.float128( (k * be * rho) / (k_B * T))
    val =  np.exp(fraction) * (fraction / (np.exp(fraction) - 1))**2
    C_vib = R * np.sum(val) * df_vib / be.shape[0]
    return C_vib

def Svib(rho, be, k, T, df_vib):
    """Calculate the vibrational entropy effects
    
       Parameters
       ----------
       rho: :obj:`numpy.ndarray`
           the thermal correlator
       be: :obj:`numpy.ndarray`
           the net bond energies in a 1-d array
       k: :obj:`numpy.ndarray`
           the scaled-shift parameter
       T: :obj:`numpy.float64`
           the temperature in K
       df_vib: :obj:`numpy.int64`
           the vibrational degree of freedoms
           
       Returns
       -------
       S_vib: :obj:`numpy.float64` 
           vibrational entropy (kcal/(mol * T))
       """
        
    k_B = 1.987204259e-3 # Boltzmann's constant in kcal/(mol K)
    R =  (8.314472 * 627.5098)/ (2625.5002e3) # Eh to kcal/mol
    fraction = np.float128( (k * be * rho) / (k_B * T))
    first = fraction / (np.exp(fraction) - 1)
    second = np.log(1 - np.exp(-fraction))
    sum_all = np.sum(first - second)
    S_vib = R * sum_all * df_vib / be.shape[0]
    return S_vib

def conditional_statements(bo):
    """Condition protocols for qbop-1
    
       Parameters
       ----------
       bo: :obj:`numpy.float64`
           Mulliken bond order
           
       Returns
       -------
       corrections: :obj:`int` 
           the correction value: 
               True -> 0 > bo
               False -> bo > 0
       """
    # Conditional protocols for QBOP-1
    if 0 > bo:
        corrections = True # anti-bonding 
    else:
        corrections = False # bonding
    return corrections

def qbop(zpebop2, net_be, df_vib, atoms, T, parameter_path, bos):
    """Calculate vibrational thermal effects (e.g., partition functions, 
       internal energy, etc.)
    
       Parameters
       ----------
       zpebop2: :obj:`numpy.float64`
           the total zpe calculated from ZPE-BOP2
       net_be: :obj:`numpy.float64`
           NxN net vibrational bond energy matrix
       def_vib: :obj:`numpy.int64`
           the vibrational degree of freedoms
       atoms: :obj:`numpy.ndarray`
           N array showing the atoms
       T: :obj:`numpy.float64`
           the temperature in K
       parameter_path: :obj:`str`
           name of the path/folder where QBOP-1 data is located
       bos: :obj:`numpy.ndarray`
           Mulliken bond order

           
       Returns 
       -------
       U_vib: :obj:`numpy.float64`
           internal thermal vibrational contribution (kcal/mol)
       q_bot: :obj:`numpy.float64`
           the value of the vibrational partition
           starting from the ground state (e.g., bottom-of-the-well)
       q_v0: :obj:`numpy.float64` 
           the value of the vibrational partition
           starting from the first excited state
       S_vib: :obj:`numpy.float64`     
           vibrational entropy (kcal/(mol * T))
       CV_vib: :obj:`numpy.float64`     
           specific heat capacity at constant volume (kcal/(mol * T))
       """
        
    bes = []
    shifts = []
    rho_p = []
    size = atoms.shape[0]
    if size == 2:
        bes = np.array([net_be[1][0]])
        shifts = np.array([2])
        rho = np.array([1])
    elif size > 2:
        for l in range(1, size):
            for n in range(l):
                # important properties
                pair_atoms = np.array([atoms[l], atoms[n]])
                if net_be[l][n] >= (100 / 349.755):
                    bes += net_be[l][n],
                    cond_statements = conditional_statements(bos[l][n])
            
                    # fitting parameters
                    scale_shift, rho_param = param.qbop_pair(pair_atoms, parameter_path, cond_statements) # , rho_int, rho_ext
                    rho_p += rho_param, 
                    shifts += scale_shift,
                else:
                    continue
        bes = np.array(bes)
        size = bes.shape[0]
        shifts = np.array(shifts)
        rho = rho_v(bes, np.array(rho_p), size)
    
    lnq_bot, lnq_v0 = lnq(rho, zpebop2, bes, shifts, T, df_vib)
    U_vib = Uvib(rho, zpebop2, bes, shifts, T, df_vib)
    CV_vib = CVvib(rho, bes, shifts, T, df_vib)
    S_vib = Svib(rho, bes, shifts, T, df_vib)
    q_bot = np.exp(lnq_bot)
    q_v0 = np.exp(lnq_v0)
    return (U_vib, q_bot, q_v0, S_vib, CV_vib)