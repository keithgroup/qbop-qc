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


#!/usr/bin/env python3
"""Argument parser used for computing qbop-1
   parsing:: parser for computing qbop-1
   file_properties:: class containing properties for writing the files
   write_json:: write the json file (if user wishes to)
   print_file:: print the qbop-1 output file"""

import argparse
import json
import os
from datetime import date
from datetime import datetime
import numpy as np
from .qbop1 import QBOP
from scipy.constants import physical_constants

def parsing():
    """Parsers for QBOP-1"""
    parser = argparse.ArgumentParser(description = 'compute thermal energies using QBOP-1 and the ideal gas rigid-rotor harmonic oscillator method')
    parser.add_argument('-f', action = 'store', help = 'name of the Gaussian Hartree-Fock output file or the atom symbol', required = True, type = str)
    parser.add_argument('-mult', action = 'store', help = 'multiplicity (i.e., 2S + 1)', required = True, type = str)
    parser.add_argument('-type', action = 'store', help = 'is the file a molecule or atom (type: atom) (default: molecule)?', required = False, type = str)
    parser.add_argument('-T', action = 'store', help = 'temperature in K (default:298.15 K)', required = False, type = str)
    parser.add_argument('-P', action = 'store', help = 'pressure of the system (default:1 atm)', required = False, type = str)
    parser.add_argument('-param_folder', action = 'store', help = "name of ZPEBOP-2 and QBOP-1's parameter folder (default: opt_parameters/)", required = False, type = str)
    parser.add_argument('--json',action = 'store_true', help = 'save the job output into JSON', required=False)
    return parser.parse_args()

class file_properties:
    
    def physical_properties(atoms, masses, molecular_mass, point_group, 
                            symmetry_number, rotational_constants):
        """Physical properties such as atom masses, molecular mass,
           point group, symmetry number, and rotational constants"""
        
        size = atoms.shape[0]
        print(' PHYSICAL PROPERTIES')
        print(' ' + 26 * '-')
        print(' AtomN  Element  Mass (amu)')
        print(' ' + 26 * '-')
        for l in range(size):
            atomid = l + 1
            print(f' {atomid:^5}{atoms[l]:>6}{masses[l]:>13.5f}')
        print(' ' + 26 * '-')
        print(f' Molecular Mass (amu) = {molecular_mass:0.5f}')
        print(f' Point group: {point_group}')
        print(f' Rotational symmetry number: {symmetry_number}')
        print(f' Rotational constants (GHz):',end='')
        for l in rotational_constants:
            print(f'   {l:0.7f}',end='')
        print('\n\n')
        return None
    
    def thermochemistry_table(T, P, zpe, q, U, Cv, S, H, G):
        """Print thermochemistry style similar to Gaussian"""
        
        titles = np.array(['electronic','translational','rotational','vibrational','total'])
        size = titles.shape[0]
        print(' THERMOCHEMISTRY')
        print(f' Temperature (K) = {T:0.2f}')
        print(f' Pressure (atm) = {P:0.2f}')
        print(f' Zero-point Energy (E_ZPE; kcal/mol) = {zpe:0.4f}')
        print(f' Internal Energy (U; kcal/mol) = {U[4]:0.4f}')
        print(f' Enthalpy (H; kcal/mol) = {H:0.4f}')
        print(f' Integrated Heat Capacity (H - E_ZPE; kcal/mol) = {H - zpe:0.4f}')
        print(f' Entropy (S; cal/mol*K) = {S[4] * 1e3:0.4f}')
        print(f' Gibbs Free Energy (G; kcal/mol) = {G:0.4f}')
        print(f' Specific Heat Capacity at Constant Volume (Cv; cal/mol*K) = {Cv[4] * 1e3:0.4f}')
        print(' ' + 52 * '-')
        print('                      U          Cv           S')
        print('                  kcal/mol     cal/mol*K   cal/mol*K')
        print(' ' + 52 * '-')
        for l in range(size):
            print(f' {titles[l]:^15}{U[l]:>10.4f}{Cv[l] * 1e3:>12.4f}{S[l] * 1e3:>12.4f}')
        print(' ' + 52 * '-')
        print('                         q             ln(q)')
        print(' ' + 52 * '-')
        for l in range(size):
            print(f' {titles[l]:^15}{q[l]:>15.6e}{np.log(q[l]):>15.6f}')
        print(' ' + 52 * '-' + '\n')
        return None
    
    def print_file_title(file, zpebop_parameter_folder, qbop_parameter_folder):
        """Print the title of the file"""
        
        # Initiate the string for the current month, day,  and year this code was executed 
        today = date.today()
        day_month_year = today.strftime("%d-%B-%Y")

        # Initiate the string for the current time 
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        
        print(f"\n\n\n{20 * ' '}SUMMARY OF QBOP-1 CALCULATION\n")
        print(f"{24 * ' '}QBOP-1 (Version 1.0.0)")
        print(f"{23 * ' '}{day_month_year} {current_time}\n\n")
        print(f'   HARTREE-FOCK OUTPUT:  {os.getcwd()}/{file}\n\n')
        print(f'   ZPE-BOP PARAMETER FOLDER: {zpebop_parameter_folder}\n\n')
        print(f'   Q-BOP PARAMETER FOLDER: {qbop_parameter_folder}\n\n')
        return None

def write_json(dictionary):
    """Write the json file"""
        
    with open("bopout.json", mode='w') as writer:
        json.dump(dictionary, writer, indent = 4)
            
    return None 
    
def print_file(parser_args):
    """Write the output file"""
    
    # Get information from the the ROHF output file
    if parser_args.param_folder == None:
        param_folders_zpebop = 'opt_parameters/zpebop2'
        param_folders_qbop = 'opt_parameters/qbop1'
    else:
        param_folders_zpebop = f'{parser_args.param_folder}/zpebop2'
        param_folders_qbop = f'{parser_args.param_folder}/qbop1'
    
    if (parser_args.type == None) or (parser_args.type == 'molecule'): 
        data = QBOP(parser_args.f, np.int64(parser_args.mult), atom_molecule = 'molecule', zpebop_param_folder = param_folders_zpebop, qbop_param_folder = param_folders_qbop)
    elif (parser_args.type == 'atom'):
        data = QBOP(parser_args.f, np.int64(parser_args.mult), atom_molecule = 'atom', zpebop_param_folder = param_folders_zpebop, qbop_param_folder = param_folders_qbop) 
    
    if parser_args.T == None:
        T = 298.15
    else:
        T = np.float64(parser_args.T)
    
    if parser_args.P == None:
        P = 1
    else:
        P = np.float64(parser_args.P)
        
    zpe, E_net = data.zpve_results()
    qbop_data = data.qbop_results(T = T, P = P)
    rrho_data = data.thermal_results()
    U_tot, S_tot, Cv_total = data.total_thermal_S_Cv()
    H, G = data.enthalpy_gibbs()

    # Print the title, BEBOP version, date, time, name or file, and energy
    file_properties.print_file_title(parser_args.f, param_folders_zpebop, param_folders_qbop)
    
    # Print physical properties
    h = physical_constants['Planck constant'][0]
    atoms = data.mol
    masses = data.masses
    molecular_mass = data.total_mass
    point_group = data.point_group
    symmetry_number = data.symmetry_numb
    rotational_constants = np.sort(data.rot_constants)[::-1] * 1e-9 / h # J to GHz
    file_properties.physical_properties(atoms, masses, molecular_mass, point_group, symmetry_number, rotational_constants)
              
    # Print thermochemistry
    thermal_contr, q_contr, entropy_contr, Cv_contr= rrho_data
    q = [q_contr[4], q_contr[5], q_contr[6], q_contr[2], q_contr[0]]
    U = [*thermal_contr[:4], U_tot] 
    Cv = [*Cv_contr[:4], Cv_total]
    S = [*entropy_contr[:4], S_tot]
    file_properties.thermochemistry_table(T, P, zpe, q, U, Cv, S, H, G)

    if parser_args.json == True:
        dictionary = {'method':'QBOP-1',
                      'version':'1.0.0',
                      'HF output file':os.getcwd() + '/' + parser_args.f,
                      'date':date.today().strftime("%d-%B-%Y"),
                      'time':datetime.now().strftime("%H:%M:%S"),
                      'atoms':atoms.tolist(),
                      'physical_properties':{'point_group':point_group,
                                            'symmetry_number':symmetry_number,
                                            'masses':masses.tolist(),
                                            'total_mass':molecular_mass,
                                            'rotational_constants':rotational_constants.tolist()},
                      'thermochemistry':{'T':T,
                                         'P':P,
                                         'zpe':zpe,
                                         'q':{'q_elec': float(q[0]), 'q_trans':float(q[1]), 'q_rot':float(q[2]), 'q_vib':float(q[3]), 'q_tot':float(q[4])},
                                         'lnq':{'lnq_elec':float(np.log(q[0])), 'lnq_trans':float(np.log(q[1])), 'lnq_rot':float(np.log(q[2])), 
                                                'lnq_vib':float(np.log(q[3])), 'lnq_tot':float(np.log(q[4]))},
                                         'U':{'U_elec':float(U[0]), 'U_trans':float(U[1]), 'U_rot':float(U[2]), 'U_vib':float(U[3]), 'U_tot':float(U[4])},
                                         'CV':{'CV_elec':float(Cv[0]) * 1e3, 'CV_trans':float(Cv[1]) * 1e3, 'CV_rot':float(Cv[2]) * 1e3, 
                                               'CV_vib':float(Cv[3]) * 1e3, 'CV_tot':float(Cv[4]) * 1e3},
                                         'S':{'S_elec':float(S[0]) * 1e3, 'S_trans':float(S[1]) * 1e3, 'S_rot':float(S[2]) * 1e3, 
                                              'S_vib':float(S[3]) * 1e3, 'S_tot':float(S[4]) * 1e3},
                                         'H':float(H),
                                         'G':float(G),
                                         'H_T - zpe':float(H - zpe)}
                      }
            
    
    if parser_args.json == True:
        write_json(dictionary)

    return None 

def main():
    pars_arg = parsing()
    print_file(pars_arg)
    return None

if __name__ == "__main__":
    main()
