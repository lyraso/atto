# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 02:15:24 2025

@author: k21071708
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import fsolve

EAU = 27.2114
IAU = 3.5e16
LAU = 0.052918
TAU = 2.419e-17
alpha = 1. /137
c = 1/alpha

#in SI units
wavelength = 800 #e-9 #Wavelength in nm
Int_0 = 4e14 #Intensity in W/cm2
Ip = 0.5 #* 13.5984 # Hydrogen gas target ionization potential in eV

#### conversion to atomic units
omega = 2 * np.pi * LAU * c / wavelength  # Angular frequency
TC = 2 * np.pi / omega  # Optical cycle period

t_list = np.linspace(-TC, TC, 200)  #time list

#two-colour beam, with mixing angle θ and phase shift φ
def beam_TC(theta, phi, r, s):
    Int_1 = Int_0 * (np.cos(theta))**2  #Intensity of beam 1
    Int_2 = Int_0 * (np.sin(theta))**2  #Intensity of beam 1
    omega_1 = r * omega  #Frequency of beam 1
    omega_2 = s * omega  #Frequency of beam 1

    E_01 = np.sqrt(Int_1 / IAU)  #field strength of beam 1
    E_02 = np.sqrt(Int_2 / IAU)  #field strength of beam 2

    e_field = []  #create empty lists to store electric field values for each point in time

    for i in t_list:
        beam_1 = E_01 * np.sin(omega_1 * i)
        beam_2 = E_02 * np.sin((omega_2 * i) + phi)
        total_beam = beam_1 + beam_2

        e_field.append(total_beam)

    return e_field

#vector potential A of TC beam:
def vector_potential_TC(theta, phi, r, s):
    Int_1 = Int_0 * (np.cos(theta))**2  #Intensity of beam 1
    Int_2 = Int_0 * (np.sin(theta))**2  #Intensity of beam 1
    omega_1 = r * omega  #Frequency of beam 1
    omega_2 = s * omega  #Frequency of beam 1

    E_01 = np.sqrt(Int_1 / IAU)  #field strength of beam 1
    E_02 = np.sqrt(Int_2 / IAU)  #field strength of beam 2

    vector_pot = []  
    
    for i in t_list:
        A_1 = (E_01 / omega_1) * np.cos(omega_1 * i)
        A_2 = (E_02 / omega_2) * np.cos((omega_2 * i) + phi)
        A_total = A_1 + A_2

        vector_pot.append(A_total)

    return vector_pot

plt.plot(t_list, beam_TC(0, np.pi /2, 1, 2), label='electric field')
#plt.plot(t_list, vector_potential_TC(np.pi / 4, np.pi /2, 1, 2), label='vector potential')

plt.xlabel("time")
plt.ylabel("electric field")
plt.show()

def field_strength(theta):
    Int_1 = Int_0 * (np.cos(theta))**2  #Intensity of beam 1
    Int_2 = Int_0 * (np.sin(theta))**2  #Intensity of beam 1

    E_01 = np.sqrt(Int_1 / IAU)  #field strength of beam 1
    E_02 = np.sqrt(Int_2 / IAU)  #field strength of beam 2

    return [E_01, E_02]

E_01 = field_strength(np.pi/2)[0]
E_02 = field_strength(np.pi/2)[1]

#pick values of r and s to set up frequencies:
omega_1 = omega  #Frequency of beam 1
omega_2 = 2 * omega  #Frequency of beam 2

#pick phase shift (phi):
phi = np.pi/2

def action_drv_TC(t_arr):
    #S_mono = Ip*t + 0.5*p**2*t - (p/omega)*np.cos(omega*t + np.pi/2) + 0.5*t - (1/4*omega)*np.sin(2*omega*t + np.pi)
    t = t_arr[0] + t_arr[1]*1j

    dS_dt_TC_real = np.real(Ip + 0.5* (p + (E_01/omega_1)*(np.cos(omega_1 * t)) + (E_02/omega_2)*(np.cos(omega_2*t + phi)) )**2)   #dS/dt = Ip + (p + A(t))^2 
    dS_dt_TC_imag = np.imag(Ip + 0.5* (p + (E_01/omega_1)*(np.cos(omega_1 * t)) + (E_02/omega_2)*(np.cos(omega_2*t + phi)) )**2)   #where A(t) = E_0/w * cos(wt + phi)

    return np.array([dS_dt_TC_real, dS_dt_TC_imag])

#FINDINGSADDLE POINTS
momentum = np.linspace(-2.0, 2.0, 50)

fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axes explicitly
cmap = plt.get_cmap('viridis')

for p in momentum:
    ts = []
    
    def action_drv_TC(t_arr):
        t = t_arr[0] + t_arr[1]*1j
        
        dS_dt_TC_real = np.real(Ip + 0.5* (p + (E_01/omega_1)*(np.cos(omega_1 * t)) + (E_02/omega_2)*(np.cos(omega_2*t + phi)) )**2)   #dS/dt = Ip + (p + A(t))^2 
        dS_dt_TC_imag = np.imag(Ip + 0.5* (p + (E_01/omega_1)*(np.cos(omega_1 * t)) + (E_02/omega_2)*(np.cos(omega_2*t + phi)) )**2)   #where A(t) = E_0/w * cos(wt + phi)

        return np.array([dS_dt_TC_real, dS_dt_TC_imag])

    for m in np.linspace(0, TC, 10):
        for n in np.linspace(0, TC/2, 10):
            saddles = fsolve(action_drv_TC, np.array([m,n]), xtol=1e-8)
            saddles = np.round(saddles, 3)

            if (saddles[0] > -15 and saddles[0] < TC and 
                saddles[1] > 0 and saddles[1] < TC/2 and
                np.linalg.norm(action_drv_TC(saddles)) < 0.1):
                ts.append(saddles)

    saddle_points = [complex(t[0], t[1]) for t in ts]
    saddle_points = np.unique(np.round(saddle_points, 3))

    reals = [np.real(point) for point in saddle_points]
    imags = [np.imag(point) for point in saddle_points]

    # Plot directly to the axes
    sc = ax.scatter(reals, imags, c=[p]*len(reals), cmap=cmap, 
                   vmin=min(momentum), vmax=max(momentum), alpha=0.7)

ax.set_xlabel('Real Part')
ax.set_ylabel('Imaginary Part')
ax.set_xlim(-15, 100)
ax.set_ylim(-5, 30)
ax.set_title('θ = 90')
ax.grid(True)

# Create colorbar using the scatter plot's mappable
cbar = fig.colorbar(sc, ax=ax, label='Momentum ($p$)')
cbar.set_ticks([min(momentum), max(momentum)])
cbar.set_ticklabels([f'{min(momentum):.1f}', f'{max(momentum):.1f}'])

plt.savefig("saddles theta=0.png")

plt.show()








