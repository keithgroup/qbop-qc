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

"""Parameters for the QBOP-1 code (version 1.0.0)
   zpe_pair :: ZPEBOP-2 atom-pair parameters 
   qbop_pair :: QBOP-1 atom-pair parameters
   """

import numpy as np
import json
from os import path

def zpe_pair(pair_atoms, parameter_folder_path, anti_corr, two_body = True):
    """Fitted atom-pair parameters used in the ZPE-BOP2 equation
    
       Parameters
       ----------
       pair_atoms: :obj:`numpy.ndarray`
           Name of the two-pair atoms 
       parameter_folder_path: :obj:`str`
           Path to the parameter folder for ZPE-BOP2 
       anti_corr: :obj:`bool`
           Parameters for anti-bonding (False) or bonding (True)
       two_body: :obj:`bool`, optional
           Two-body parameters (True) or three-body parameters (False)
       
       Returns 
       -------
       beta_val: :obj:`numpy.float64`
           Fixed parameter for EHT (either for bond or anti-bonding; kcal/mol) for bonding or
           anti-bonding
       pre_exp: :obj:`numpy.float64`
           Fixed negative pre-exponential factor for the anharmonic short-range interaction (kcal/mol)
       zeta: :obj:`numpy.float64`
           Fixed exponential factor for the anharmonic short-range interaction (A^-1)
       R_param: :obj:`numpy.float64`
           Fixed turning point distance parameter for the anharmonic short-range interaction (A)
       kappa: :obj:`numpy.float64`
           Fixed three-body parameters for bonding or anti-bonding (kcal/mol)^1/3
       """
    bonds1 = f'{pair_atoms[0]}~{pair_atoms[1]}'
    bonds2 = f'{pair_atoms[1]}~{pair_atoms[0]}'
    if path.exists(f'{parameter_folder_path}{bonds1}/{bonds1}_param_opt.json') == True:
        file = f'{parameter_folder_path}{bonds1}/{bonds1}_param_opt.json'
    elif path.exists(f'{parameter_folder_path}{bonds2}/{bonds2}_param_opt.json') == True:
        file = f'{parameter_folder_path}{bonds2}/{bonds2}_param_opt.json'
    with open(file, "r") as openfile:
        parameter = json.load(openfile)
    
    if two_body == True:
        if anti_corr == False:
            beta_val = parameter['beta_bond']
        else:
            beta_val = parameter['beta_anti']
        pre_exp = parameter['pre_exp']
        zeta = parameter['zeta']
        R_param = parameter['R_param']
        return (beta_val, pre_exp, zeta, R_param)
    
    else:
        if anti_corr == False:
            kappa = parameter['kappa_bond']
        else:
            kappa = parameter['kappa_anti']
        return kappa

def qbop_pair(pair_atoms, parameter_folder_path, conds):
    """Fitted atom-pair parameters used in the ZPE-BOP2 equation
    
       Parameters
       ----------
       pair_atoms: :obj:`numpy.ndarray`
           Name of the two-pair atoms 
       parameter_folder_path: :obj:`str`
           Path to the parameter folder for QBOP-1
       conds: :obj:`bool`
           Parameters for anti-bonding (False) or bonding (True)
       
       Returns 
       -------
       scale_shift: :obj:`numpy.float`
           The scale-shift value (i.e., E_N+1 - E_N = scale-shift * BE_ij) for bonding or anti-bonding
       rho_param: :obj:`numpy.ndarray`
           Parameters for thermal correlator (i.e., alpha, beta, zeta, phi, kappa) for bonding or anti-bonding
       """
    bonds1 = f'{pair_atoms[0]}~{pair_atoms[1]}'
    bonds2 = f'{pair_atoms[1]}~{pair_atoms[0]}'
    if path.exists(f'{parameter_folder_path}{bonds1}/{bonds1}_param_opt.json') == True:
        file = f'{parameter_folder_path}{bonds1}/{bonds1}_param_opt.json'
    elif path.exists(f'{parameter_folder_path}{bonds2}/{bonds2}_param_opt.json') == True:
        file = f'{parameter_folder_path}{bonds2}/{bonds2}_param_opt.json'
    with open(file, "r") as openfile:
        parameter = json.load(openfile)
        
    names = ['alpha', 'beta', 'zeta', 'phi', 'kappa']
    if conds == True:
        scale_shift = parameter['k_anti']
        rho_param = [parameter[f'{l}_anti'] for l in names]
    else:
        scale_shift = parameter['k_bond']
        rho_param = [parameter[f'{l}_bond'] for l in names]
    
    return (scale_shift, rho_param) 