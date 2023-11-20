# from boom import *
# from arnett import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import mesa_reader as mr
import sys 
from sklearn.mixture import GaussianMixture
from scipy.interpolate import interp1d
import sklearn
import astropy.units as u
import astropy.constants as c 
from sklearn.preprocessing import StandardScaler
import matplotlib

#* =======================================================================================================================================+
def prepare_canvas():
    plt.rcParams['figure.figsize'] = [8, 8]
    plt.rcParams['axes.linewidth'] = 2.5
    plt.legend(prop={'size': 15})

    fontsize = 20
    ax = plt.gca()
    ax.tick_params(direction = 'in', length = 20)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        # tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        # tick.label1.set_fontweight('bold')

#! this function takes as input data (e.g. a dictionary), converts it into a dataframe and finds the correlation between its entries
def correlation(dat):
    dataframe = pd.DataFrame(dat)
    corr = dataframe.corr()
    mask = np.zeros_like(corr, dtype = bool)
    mask[np.triu_indices_from(mask)] = True
    corr[mask] = np.nan
    
    return corr

#! this function takes as input the initial mass of a star and returns the mass of the He star
def ZAMS_to_He(mass):
    mass = np.asarray(mass)
    he_mass = np.zeros_like(mass)
    he_mass[mass <= 30] = 2 * (0.0385 * mass[mass <= 30]**1.603)
    he_mass[mass > 30] = 2 * (0.5 * mass[mass > 30] - 5.87)
    return he_mass

#! this function returns the index of the entry of the array that is closest to the value
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#! this function is a power-law gen for pdf(x) proportional to x^{g-1} for a<=x<=b
def random_power_law(a, b, p, size = 1):
    g = p + 1
    """Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b"""
    r = np.random.random(size = size)
    ag, bg = a**g, b**g
    return (ag + (bg - ag) * r)**(1. / g)


def kuiper(size = 1):
    q = np.arange(0., 1., 0.0001)
    f = 2. / (1 + q)**2
    f = np.cumsum(f)
    f = f / f[-1]
    samples = np.random.random(size)
    return np.interp(samples, f, q)

#? Here is included all the data gathering from the corresponding foldes (this may need fixing for other machines accordingly) and the stellar evolution model
def model(metallicity: float, beta_val: float, alpha_outflow_val: float, eta_outflow_val: float, zeta_val: float, aturb_val: float, 
          t_15_val:float, m_max_val: float, wind_is_on: bool):
    
    parent_folder = os.path.dirname(__file__)

    subfolder = f'grid_che_helium_single_yoon_++_withN_{metallicity}_solar_CC'
    source = os.path.join(parent_folder, subfolder)
    files = sorted(os.listdir(source))
    # print(files)

    #! Initialization of arrays
    m_init_list = np.ndarray(len(files), dtype = float)
    ni56 = np.ndarray(len(files), dtype = float)
    E_exp = np.ndarray(len(files), dtype = float)
    M_ej = np.ndarray(len(files), dtype = float)
    m_grav = np.ndarray(len(files), dtype = float)
    xi25 = np.ndarray(len(files), dtype = float)
    mu4 = np.ndarray(len(files), dtype = float)
    mu4_mod = np.ndarray(len(files), dtype = float)
    mfinal = np.ndarray(len(files), dtype = float)
    rfinal = np.ndarray(len(files), dtype = float)
    mwind = np.ndarray(len(files), dtype = float)
    vkick = np.ndarray(len(files), dtype = float)
    co_mass = np.ndarray(len(files), dtype = float)
    Z = np.ndarray(len(files), dtype = float)

    #! Initialization of array index
    j = 0
    for file in files:
        minit = float(file.split("_")[0])
        if minit >= 4.5:
            try:
                # print(os.path.join(source, file))
                #* Get data from profiles and store the initial mass
                profile = mr.MesaData(os.path.join(source, file)).data
                m_init_list[j] = minit
                
                #* Run the evolution model for the specified choice of parameters
                res = criterion(profile, beta=beta_val, alpha_outflow=alpha_outflow_val, eta_outflow=eta_outflow_val,
                            zeta=zeta_val, aturb=aturb_val, t_15=t_15_val * u.s, m_max= m_max_val * u.Msun, wind=wind_is_on)
                

                #* Get the desired results
                ni56[j] = res['M_ni56'].value
                E_exp[j] = res['E_expl'].value
                M_ej[j] = res['M_ej'].value
                m_grav[j] = res['m_grav'].value
                xi25[j] = res['xi25']
                mu4[j] = res['mu4']
                mu4_mod[j] = res['mu4']
                mfinal[j] = res['M_star'].value
                rfinal[j] = res['R_star'].value
                mwind[j] = res['M_wind'].value
                vkick[j] = res['vkick'].value
                co_mass[j] = res['m_co'].value
                Z[j] = metallicity

                j += 1
            except:
                pass

    #* Store the results in a dictionary for easier access
    outputs_per_Z = {
        'm_init': m_init_list,
        'ni56': ni56,
        'E_exp': E_exp,
        'M_ej': M_ej,
        'm_grav': m_grav,
        'xi25': xi25,
        'mu4': mu4,
        'mu4_mod': mu4_mod,
        'mfinal': mfinal,
        'rfinal': rfinal,
        'mwind': mwind,
        'vkick': vkick,
        'co_mass': co_mass
    }

    return outputs_per_Z
#* ========================================================================================================================================
#TODO =====================================================================================================================================
def rhoav(r, m):
    """
    average density in g/cm**3
    """
    rhoav = (0.75 / np.pi) * m * np.maximum(r, 1.e-99)**(-3)
    if rhoav[0] == 0:
        rhoav[0] = rhoav[1]
    return rhoav


def t_infall(r, m):
    """
    infall time in sec
    """
    rho_av = rhoav(r, m)
    time = 0.5 * np.sqrt(np.pi / c.G.cgs.value) * rho_av**(-0.5)
    return time


def mdot(r, m, rho):
    """
    accretion rate in g/s
    """
    tff = t_infall(r, m)
    rho_av = rhoav(r, m)
    acc = 2 * m / tff * (rho/(rho_av-rho))
    return acc


def r_gain(mdot, m, time, offset):
    """
    gain radius in cm
    """
    rg = (mdot / m**3 * (1.2e7**3 * c.M_sun.cgs.value**2) + (1.2e6)**3)**(1/3)
    return rg


def luminosity(mdot, m, r_gain, time, zeta: float, t_15: float, **kwargs):
    """
    luminosity inoin erg/s
    """

    t_cool = t_15 * (m / (1.5 * c.M_sun.cgs.value)) ** (5.0/3.0)
    t_cool = np.maximum(t_cool, 0.1)
    l0 = ((0.3 * 0.084e0 / c.M_sun.cgs.value * c.c.cgs.value**2) * m**2 / t_cool)
    # initial luminosity \approx binding energy/cooling time scale
    #alpha = np.maximum(0, (1 - physconst.Kepler.gee * m / (5.0/7.0*r_gain * physconst.Kepler.c**2)))         # redshift factor
    lum = (zeta * c.G.cgs.value * mdot * m / r_gain + l0 *
           np.exp(-np.minimum(time / t_cool, 100)))  # * alpha ** 3
    return lum, l0* np.exp(-np.minimum(time / t_cool, 100)),t_cool,(0.084e0 / c.M_sun.cgs.value * c.c.cgs.value**2 ) * m**2


def add_wind(m_by, lum0, rg, e_expl, m_grav, i_final, t_15: float, radius=1.2e6, **kwargs):
    """
    Modifies the explosion energy and remenant mass by adding in the neutrino driven wind phase.

    """
    C=1                                                         # Proportionality constant for neutrino heating in the wind phase
    # Binding (recombination energy) ???
    e_rec_alpha = 5 * (1 * u.MeV) / c.u.cgs.value
    epsilon_v = 9.5 * m_by/c.M_sun.cgs.value            # Mean neutrino energy
    #t_cool = 5 * (m_by / (1.5 * physconst.Kepler.solmass))      # Cooling timescale
    t_cool = t_15 * (m_by / (1.5 * c.M_sun.cgs.value)) ** (5.0/3.0)
    L_0 = lum0[i_final]                                         # Neutrino luminosity at the neutron star surface
    #radius = 1.2e6                                             # Neutron star radius
    radius = rg[i_final]
    #radius = 2.2e5
    e_neutrino = L_0 * t_cool                                   # Total energy released by neutrino luminosity (assuming exp L decay)

    # Initial mass loss rate (m_dot) after accretion ceases (at the mass coordinate given by i_final)
    m_dot_0 = 1.14e-10 * C**(5/3) * (L_0/1e51)**(5/3) * epsilon_v**(10/3) * (
        radius/1e6) ** (5/3) * (1.4*c.M_sun.cgs.value/m_by)**2 * c.M_sun.cgs.value

    m_wind = m_dot_0 * t_cool                                   # Mass lost in wind (baryonic)
    e_wind = m_wind * e_rec_alpha                               # Energy contribution in wind

    e_expl =  e_expl + e_wind                               # Modifed explosion energy
    m_by   =  m_by-m_wind                                   # Modified neutron star remenant mass (baryonic)
    # m_grav =  (m_by-m_wind) - 0.084 * (m_by-m_wind)**2 / physconst.Kepler.solmass #(Convert to gravitational mass)
    m_grav = (-1.0 + np.sqrt(1.0 + 4.0 * 0.084 * m_by /
              c.M_sun.cgs.value)) / (2 * 0.084) * c.M_sun.cgs.value

    if e_expl <=0:  # <-- No modification necessary if there is no explosion (i.e. black hole)
        e_wind     =  0.
        m_wind     =  0.

    return {'e_expl'      : e_expl,
            'm_by'        : m_by,
            'm_grav'      : m_grav,
            'e_wind'      : e_wind,
            'm_wind'      : m_wind,
            }


def nuc_energy(A1, A2, mu1, mu2, X, mtot):
     # A1 -> A2 + energy
    energy_per_reaction = ((A2*mu1 - A1*mu2)*u.mu*c.c.cgs.value**2).to(u.erg)
    reactions_per_gram = (1*u.g / (A2*mu1*u.mu)).to(u.dimensionless_unscaled)/u.g
    mtot = u.Quantity(mtot,u.Msun)
    return (energy_per_reaction*reactions_per_gram *(X*mtot)).to(u.erg)


def eburn_func(p, T9, i, rho, beta: float):

    mass = p('mass')[::-1]*u.Msun
    mass   = (p('dq')*p('mass')[0])[::-1]
    m  = mass[i]
    h    = 1.007276
    he3  = 3.016029
    he4  = 4.002602
    c12  = 12.00000
    n14  = 14.00307
    o16  = 15.994915
    ne20 = 19.992440
    na23 = 22.989769
    mg24 = 23.985042
    si28 = 27.976927
    s32  = 31.972071
    ar36 = 35.967545
    ca40 = 39.962591
    ti44 = 43.582056 
    fe52 = 51.948113
    fe54 = 53.939609
    fe56 = 55.934937
    ni56 = 55.942129
    
    E = 0*u.erg
    mni56 = m*p('ni56')[::-1][i]
    if ((T9 >= 2.5) & (T9<3.5)):
            
            E += nuc_energy(1,16,h,o16,p('h1')[::-1][i],m)
            E += nuc_energy(4,16,he4,o16,p('he4')[::-1][i],m)         
            E += nuc_energy(12,16,c12,o16,p('c12')[::-1][i],m)
            E += nuc_energy(14,16,n14,o16,p('n14')[::-1][i],m)
    if ((T9 >= 3.5) & (T9<5)):
            E += nuc_energy(1,28,h,si28,p('h1')[::-1][i],m)
            E += nuc_energy(4,28,he4,si28,p('he4')[::-1][i],m)         
            E += nuc_energy(12,28,c12,si28,p('c12')[::-1][i],m)
            E += nuc_energy(14,16,n14,si28,p('n14')[::-1][i],m)
            E += nuc_energy(16,28,o16,si28,p('o16')[::-1][i],m)
            E += nuc_energy(20,28,ne20,si28,p('ne20')[::-1][i],m)
            E += nuc_energy(24,28,mg24,si28,p('mg24')[::-1][i],m)
    try:
        if (np.log10(beta*rho) > 11.62 + 1.5 * np.log10(T9)-39.17/T9) and (T9 > 5.):
        #if (T9 > 5.):
                enuc = nuc_energy(1,56,h,ni56,p('h1')[::-1][i],m)
                E += enuc
                mni56 += m*p('h1')[::-1][i] - (enuc/c.c**2).to(u.Msun).value

                enuc = nuc_energy(4,56,he4,ni56,p('he4')[::-1][i],m)
                E += enuc 
                mni56 += m*p('he4')[::-1][i] - (enuc/c.c**2).to(u.Msun).value
                
                enuc = nuc_energy(12,56,c12,ni56,p('c12')[::-1][i],m)
                E += enuc
                mni56 += m*p('c12')[::-1][i] - (enuc/c.c**2).to(u.Msun).value
                
                enuc =  nuc_energy(16,56,o16,ni56,p('o16')[::-1][i],m)
                E += enuc
     
                mni56 += m*p('o16')[::-1][i] - (enuc/c.c**2).to(u.Msun).value
                
                enuc = nuc_energy(14,16,n14,ni56,p('n14')[::-1][i],m)
                E += enuc 
                mni56 += m*p('n14')[::-1][i] - (enuc/c.c**2).to(u.Msun).value
                
                enuc = nuc_energy(20,56,ne20,ni56,p('ne20')[::-1][i],m)
                E += enuc 
                mni56 += m*p('ne20')[::-1][i] -(enuc/c.c**2).to(u.Msun).value
                
                enuc = nuc_energy(24,56,mg24,ni56,p('mg24')[::-1][i],m)
                E += enuc 
                mni56 += m*p('mg24')[::-1][i] -(enuc/c.c**2).to(u.Msun).value
                
                enuc = nuc_energy(28,56,si28,ni56,p('si28')[::-1][i],m)
                E += enuc
                mni56 += m*p('si28')[::-1][i] -(enuc/c.c**2).to(u.Msun).value
            
                enuc = nuc_energy(3,56,he3,ni56,p('he3')[::-1][i],m)
                E += enuc
                mni56 += m*p('he3')[::-1][i] -(enuc/c.c**2).to(u.Msun).value
                
                enuc = nuc_energy(32,56,s32,ni56,p('s32')[::-1][i],m)
                E += enuc
                mni56 += m*p('s32')[::-1][i] -(enuc/c.c**2).to(u.Msun).value
                
                enuc = nuc_energy(36,56,ar36,ni56,p('ar36')[::-1][i],m)
                E += enuc
                mni56 += m*p('ar36')[::-1][i] -(enuc/c.c**2).to(u.Msun).value
                
                enuc = nuc_energy(40,56,ca40,ni56,p('ca40')[::-1][i],m)
                E += enuc
                mni56 += m*p('ca40')[::-1][i] -(enuc/c.c**2).to(u.Msun).value
                
                enuc = nuc_energy(44,56,ti44,ni56,p('ti44')[::-1][i],m)
                E += enuc
                mni56 += m*p('ti44')[::-1][i] -(enuc/c.c**2).to(u.Msun).value
                
                enuc = nuc_energy(52,56,fe52,ni56,p('fe52')[::-1][i],m)
                E += enuc
                mni56 += m*p('fe52')[::-1][i] -(enuc/c.c**2).to(u.Msun).value
                
                enuc = nuc_energy(54,56,fe54,ni56,p('fe54')[::-1][i],m)
                E += enuc
                mni56 += m*p('fe54')[::-1][i] -(enuc/c.c**2).to(u.Msun).value
    except ZeroDivisionError:
        print('oh no')
        pass
    mni56 = (mni56*u.Msun).to(u.g)
    return E.value,mni56.value


def criterion_Muller(
        profile,
        r,
        m,
        rho,
        cs,
        stot,
        ybind,
        # parameters
        alpha_outflow: float,
        alpha_turb: float,
        beta: float,
        eta_outflow: float,
        t_15: float,
        zeta: float,
        m_max: float,
        addwind = True,
        ma_conv = None,
        offset = 0,
        silent = False,
        threshold = 1,
        **kwargs):

    still_accreting = 0
    late_fallback = 0
    slow_shock = 0
    dm_kick = 0.0

    con_matzner = 0.794 # 1.03
    t9_oburn = 3.5
    t9_siburn = 5 # 4.8 # incomplete silicon burning
    # t9_siburn = 4.5 # complete silicon burning

    if ma_conv is None:
        ma_conv = np.zeros_like(rho)

    macc = mdot(r, m, rho)
    # print(u.Quantity(macc, u.g/u.s).to(u.Msun/u.s))
    dot_m_out = np.zeros_like(rho)
    time = t_infall(r, m)
    # print(time)
    rg = r_gain(macc, m, time, offset)
    # print(u.Quantity(rg, u.cm).to(u.km))
    lum0, ldiff, tcool, ebnd_pns = luminosity(macc, m, rg, time, t_15 = t_15, zeta = zeta, **kwargs)
    print(lum0)
    alpha = np.maximum(0, (1 - 2 * c.G.cgs.value * m / (5.0/7.0*rg * c.c.cgs.value**2)))         #* redshift factor
    # print(alpha)
    if offset == 0:
        lume2 = lum0 * 1e-52 * (m / c.M_sun.cgs.value)**2 * alpha ** 1.5 #* 1.8 ** 0.2
    else:
        lume2 = interp1d(time, lum0)(np.maximum(time - offset, 0)) * 1e-52 * (m / c.M_sun.cgs.value)**2

    lum0=lum0*np.sqrt(alpha) #* for diagnostic purposes only
    ldiff=ldiff*np.sqrt(alpha)
    rsh = 0.55e5 * lume2**(4 / 9) * (rg * 1.e-6)**(16 / 9) / (macc**2 * m)**(1/3) * c.M_sun.cgs.value
    #* with turbulent pressure:
    rsh = alpha_turb * rsh
    # print(u.Quantity(rsh, u.cm).to(u.km))
    egain = (0.25 * c.G.cgs.value) * m / np.maximum(rsh, rg) + 8.8 * 1.602176634e-6 / c.u.cgs.value * 0.75
    egain2 = (0.25 * c.G.cgs.value) * m / rg + 8.8 * 1.602176634e-6 / c.u.cgs.value * 0.75
    # print(egain, egain2)
    tadv = 18e-3 * (np.maximum(rsh, 0.) * 1.e-7)**1.5 * np.log(np.maximum(rsh/rg, 1)) * (m / c.M_sun.cgs.value)**(-0.5)
    # print(u.Quantity(tadv, u.s).to(u.ms))
    mgain = macc * tadv
    theat = 150.e-3 * (egain / 1e19) * (rg / 1.e7)**2 / np.maximum(lume2, 1e-80)
    # print(u.Quantity(theat, u.s).to(u.ms))
    qheat = (egain * mgain) / theat
    eta_acc = qheat / macc
    psi = np.pi * ma_conv / 2.0 * (macc * m * c.M_sun.cgs.value ** 2/ rg) / np.maximum(qheat, 1.0)
    scr = 4.0 / 3.0 * 0.4649
    tadv_theat = tadv / theat * (1 + scr * (1 + psi) ** (2 / 3)) / (1 + scr)

    #* Compactness parameter
    if np.max(m / c.M_sun.cgs.value) > 2.5:
        xi25 = interp1d(m / c.M_sun.cgs.value, m/(r*1e-8 * c.M_sun.cgs.value))(2.5).tolist() #CHECK-XXX
    else:
        xi25 = 0.0
    xi17 = interp1d(m / c.M_sun.cgs.value, m/(r*1e-8 * c.M_sun.cgs.value))(1.75).tolist()
    mc4  = interp1d(stot, m / c.M_sun.cgs.value)(4.0).tolist()
    mu4  = interp1d(m / c.M_sun.cgs.value, r)(mc4+0.3).tolist() - \
        interp1d(m / c.M_sun.cgs.value, r)(mc4).tolist()
    mu4 = 0.6 / (mu4 / 1e8)
    ebn_env = interp1d(m / c.M_sun.cgs.value, ybind)(mc4).tolist()

    #* Now estimate the explosion energy:
    e_rec = 5 * u.MeV / u.u
    m_max_by  = m_max + 0.084 * m_max **2 / c.M_sun.cgs.value
    # print(m_max_by)

    try:
        jj0 = np.where(np.logical_and (
                tadv_theat > threshold,
                m < m_max_by))[0][0]
        jj = np.arange(jj0, len(m))
    except IndexError:
        jj = np.array([len(m)-1], dtype = int)

    m_init = 0.0 

    if len(jj) > 1:
        i_expl = np.min(jj)
        m_init = m[i_expl - 1]
    else:
        i_final = len(m)

    v_esc = np.sqrt(2 * c.G.cgs.value * m / r)
    # print(u.Quantity(v_esc, u.cm / u.s).to(u.km/u.s))

    e_diag = np.zeros_like(m)
    e_total = np.zeros_like(m)
    v_shock = np.zeros_like(m)
    T_9s = np.zeros_like(m)

    eburn = 0.

    mNi56 = 0.
    mO16 = 0.
    m_by_rem = m_init
    e_delayed = 0.
    m_neutrino = 0.
    i_kick = 0
    for i in jj[:-1]:
        # eta_outflow = 1.0 - alpha_outflow
        dot_m_sh = 4.0 * np.pi * v_shock[i-1] * r [i-1]**2 * rho [i-1]
        scr1 = 1.0
        if (eta_outflow * dot_m_sh > macc[i]):
            scr1 = macc[i] / dot_m_sh

        eta_1 = max( [(eta_acc[i] / np.abs(egain[i])), 0.0]) # * alpha_outflow
        eta_2 =  eta_1 * scr1

        dot_m_out [i] = macc [i] * eta_2
        m_out = (m[i] - m[i-1]) * eta_2
        e_delayed = e_delayed + (m[i] - m[i-1]) * (eta_outflow * eta_1 - eta_2) * e_rec

        e_diag[i] = e_diag[i-1] + m_out * e_rec + (ybind [i-1] - ybind[i])  * alpha_outflow

        e_total[i] = e_total[i-1] + m_out * e_rec + (ybind [i-1] - ybind[i])  * alpha_outflow + \
                    (m[i] - m[i-1]) * (eta_outflow * eta_1 - eta_2) * e_rec
        v_shock[i] = (con_matzner * np.sqrt(max(e_diag[i],0.0) / (m[i] - m_init)) * ((m[i] - m_init) / (rho[i] * r[i]**3))**0.19)

        m_by_rem = m_by_rem + (m[i] - m[i-1]) * (1 - alpha_outflow)*(1.0-eta_1) 

        T_9s[i] = ((3 / (4 * c.sigma_sb.cgs.value / c.c.cgs.value))*(beta/(beta-1))*rho[i]*v_shock[i]**2)**0.25*1e-9
        iregion = 0
        if T_9s[i] > t9_oburn:
            iregion = 1
        if T_9s[i] > t9_siburn:
            iregion = 2
        if np.log10(beta*rho[i]) < 11.62 + 1.5 * np.log10(T_9s[i])-39.17/T_9s[i] and T_9s[i] > t9_siburn:
            iregion = 3
        
        eburn_i,dmni56 = eburn_func(profile,T_9s[i],i,rho[i],beta=beta)
        mNi56 += dmni56*alpha_outflow
        e_diag[i] += eburn_i
        e_total[i] += eburn_i
        
        if iregion == 0:
            mO16 += alpha_outflow * (m[i] - m [i-1]) * profile('o16')[::-1][i]
        eburn += eburn_i
        #* determining final mass cut, v_shock ~ v_esc
        m_grav = (-1.0 + np.sqrt (1.0 + 4.0 * 0.084 * m_by_rem / c.M_sun.cgs.value)) / (2 * 0.084)  * c.M_sun.cgs.value
        v_post = (beta-1)/beta * v_shock[i]

        if (v_post > v_esc[i]):
            i_final = i
            e_diag [i_final] = e_diag [i_final]
            e_expl = e_diag[i_final]
            i_kick = i_final
            break
        if (m_by_rem > m_max_by):
            i_final = i 
            e_expl = e_diag[i_final]
            e_delayed = 0.
            dm_kick = m[i]-m_by_rem
            m_by_rem = m_by_rem + (1.0-alpha_outflow) * dm_kick
            i_kick = i_final
            print("Fallback by early accretion.",dm_kick/1.99e33)
            break
    else:
        m_by_rem = np.max(m)
        m_grav = np.max(m)
        e_expl = 0.
        mNi56 = 0.
        i_final = jj[-1]

    m_grav = (-1 + np.sqrt(1 + 4 * 0.084 * m_by_rem /
              c.M_sun.cgs.value)) / (2 * 0.084) * c.M_sun.cgs.value
    # print(u.Quantity(m_grav, u.g).to(u.Msun))
    m_wind = 0.0
    e_wind = 0.0
    # if addwind:
    #     if (m_grav < m_max):
    #         wind = add_wind(m_by_rem, lum0, rg, e_expl, m_grav, i_final, t_15 = t_15, **kwargs)
    #         e_diag[i_final] = wind['e_expl']
    #         m_by_rem = wind['m_by']
    #         m_grav = wind['m_grav']
    #         m_wind = wind['m_wind']
    #         e_wind = wind['e_wind']

    if m_by_rem > m_max_by: 
        m_grav = m_by_rem
        # print(u.Quantity(m_grav, u.g).to(u.Msun))

    e_expl_bh = e_diag[i_final]

    #* Continuing after break at i_final
    for i in range(i_final+1, jj[-1]):
        if still_accreting==1:
            m_by_rem = m_by_rem + (m[i] - m[i-1])
            dm_kick =  dm_kick + (m[i] - m[i-1])
        e_diag[i] = e_diag[i-1] + (ybind[i-1] - ybind[i]) #+ (m[i] - m_by_rem) * m_grav * c.G.cgs.value / r[i]
        e_total[i] = e_total[i-1] + (ybind[i-1] - ybind[i]) #+ (m[i] - m_by_rem) * m_grav * c.G.cgs.value / r[i]
        eexp = e_diag[i]
        v_shock[i] = (con_matzner * np.sqrt(max([eexp, 0]) / (m[i] - m_init)) * ((m[i] - m_init) / (rho[i] * r[i]**3))**0.19)
        v_post = (beta-1)/beta * v_shock[i]
        if (v_post > v_esc[i]):
            still_accreting = 0
        if (v_post < cs[i]):
            slow_shock = 1
            dm_kick = 0.0
        else:
            slow_shock = 0
        if (slow_shock):
            i_bh = i
            mO16 = 0.0
            mNi56 = 0.0
        if (slow_shock) & (late_fallback==0):
            late_fallback = 1
            e_expl_bh = e_diag [i-1]
            print("Fallback after early accretion has stopped.")
            print(e_expl_bh,ybind[i],m[i_bh]/1.99e33)

        T_9s[i] = ((3 / (4 * c.sigma_sb.cgs.value / c.c.cgs.value))
                   * (beta/(beta-1)) * rho[i] * v_shock[i]**2)**0.25*10**-9
        iregion = 0
        if T_9s[i] > t9_oburn:
            iregion = 1
        if T_9s[i] > t9_siburn:
            iregion = 2
        if T_9s[i] > t9_siburn and np.log10(beta*rho[i]) < 11.62 + 1.5 * np.log10(T_9s[i])-39.17/T_9s[i]:
            iregion = 3

        eburn_i,dmni56 = eburn_func(profile,T_9s[i],i,rho[i],beta=beta)
        e_diag[i] = eexp + eburn_i
        if (late_fallback==1):
            e_diag[i] = e_expl_bh #* energy of the sound pulse remains constant after fallback
        if (late_fallback==1) & (slow_shock):
            e_diag[i] = e_expl_bh #* energy of the sound pulse remains constant after fallback


        e_total[i] = e_total[i] + eburn_i
        e_expl = e_diag [i]
        eburn += eburn_i
        mNi56 += dmni56
        if (iregion == 0):
            mO16  += (m[i] - m [i-1]) * profile('o16')[::-1][i]

        if iregion == 0:
            mO16 += alpha_outflow * (m[i] - m [i-1]) * profile('o16')[::-1][i]
        if ((late_fallback == 1) & (e_expl_bh < ybind [i])):
            i_bh = i
            mO16 = 0.0
            mNi56 = 0.0

    if (late_fallback==1):
        print("BH formation (?) by fallback",e_expl_bh)
        e_delayed=0
        m_by_rem = m[i_bh]
        m_grav = m_by_rem
    if (m_init > m_max_by):
        m_init = 0.0

    if m_grav < m_max:
        e_expl = e_expl + e_delayed
    if e_expl < 1e45:
        m_by_rem = np.max(m)
        m_grav = np.max(m) 
        e_expl = 0.
        mNi56 = 0.
        mO16 = 0.0
        dm_kick = 0.0
        i_final = jj[-1]

    #* Calculate kick velocity

    v_kick = 0.16 * np.sqrt(max(0.0,2.0 * (m[i_kick]-m_init) * e_expl)) / m_grav
    if (late_fallback == 1):
        v_kick = 0.16 * np.sqrt(max(0.0,2.0 * dm_kick * e_expl)) / m_grav
    if (e_expl < 1e45):
        v_kick = 0.0

    #* results section

    result = {
        'crit' : tadv_theat,
        'lum' : lum0,
        'time' : time,
        'macc' : macc,
        'dot_m_out' : dot_m_out,
        'mass' : m,
        'tadv' : tadv,
        'theat' : theat,
        'rsh' : rsh,
        'rgain' : rg,
        'egain' : egain,
        'lume2' : lume2,
        'qheat' : qheat,
        'eta_acc' : eta_acc,
        'rho0' : rho,
        'r0' : r,
        'v_shock' : v_shock,
        'cs' : cs,
        'v_esc' : v_esc,
        'e_diag' : e_diag,
        'e_total' : e_total,
        'm_init' : m_init,
        'm_by' : m_by_rem,
        'm_grav' : m_grav,
        'e_expl' : e_expl,
        'e_wind' : e_wind,
        'm_wind' : m_wind,
        'xi25' : xi25,
        'xi17' : xi17,
        'mc4' : mc4,
        'mu4' : mu4,
        'm_star' : m[-1],
        'r_star' : r[-1],
        'm_ej' : m[-1]-m_by_rem,
        'eburn' : eburn,
        'Ni56' : mNi56,
        'O16' : mO16,
        'T_9s' : T_9s,
        'ebn_env': ebn_env,
        'i_final' : i_final,
        'v_kick'  : v_kick,
        'ybind'   : ybind,
        'ldiff'   : ldiff,
        'tcool'   : tcool,
        'ebnd_pns': ebnd_pns,
        'alpha'   : np.sqrt(alpha)
        }

    return result


def extra_stuff(profile, mass):
    chem_variables = ['h1', 'he3', 'he4', 'c12', 'n14', 'o16', 'ne20', 'mg24', 'si28', 's32', 'ar36', 'ca40', 'ti44', 'cr48', 'cr56', 'fe52', 'fe54', 'fe56', 'ni56']
    abundances = []
    for element in chem_variables:
        bla = profile(element)[::-1]
        abundances.append(bla)
    bla = abundances[len(abundances)-1] + abundances[len(abundances)-2] + abundances[len(abundances)-3] + abundances[len(abundances)-4]
    bla2 = abundances[len(abundances)-5] + abundances[len(abundances)-6] + abundances[len(abundances)-7] + abundances[len(abundances)-8] + abundances[len(abundances)-9] + abundances[len(abundances)-10] + abundances[len(abundances)-11] + abundances[len(abundances)-12] + abundances[len(abundances)-13] + abundances[len(abundances)-14] + abundances[len(abundances)-15] + abundances[len(abundances)-16] + abundances[len(abundances)-17] + abundances[len(abundances)-18] + abundances[len(abundances)-19] + abundances[len(abundances)-20] 
    Fe_core_mass = 0.0*u.Msun
    check = True
    for i in range(len(bla)-1,0,-1):
        if bla[i]>=bla2[i] and check:
            Fe_core_mass = mass[i]
            check=False
    CO_core_mass1 = 0.0*u.Msun
    check = True
    bla = abundances[3]+abundances[5]
    bla2 = abundances[0]+abundances[1]+abundances[2]+abundances[4]
    for i in range(len(bla)-1,0,-1):
        if bla[i]>=bla2[i] and check:
            CO_core_mass1 = mass[i]
            check=False
    bla1 = True
    bla2 = True
    menv1 = 0*u.Msun
    menv2 = 0*u.Msun
    #if abundances[4][len(bla)-1]>0.005:
    #    bla2=True
    #    val = abundances[5][len(bla)-1]/2
    val = 3e-3
    for i in range(len(bla)-1,0,-1):
        if bla1 and abundances[2][i]<1e-2:
            bla1 = False
            menv1 = max(mass) - mass[i]
            #print('menv1',i)
        if bla2 and abundances[4][i]<val:
            bla2=False
            menv2 = max(mass) - mass[i]
            #print('menv2',i)
    CO_core_mass2 = max(mass) - menv1
    surfHe = abundances[2][len(bla)-1]
    surfC = abundances[3][len(bla)-1]
    surfN = abundances[4][len(bla)-1]
    surfO = abundances[5][len(bla)-1]
    result = {
        'fecoremass':Fe_core_mass,
        'cocoremass':CO_core_mass1,
        'henenvmass':menv2,
        'surfHe':surfHe,
        'surfC':surfC,
        'surfO':surfO,
        'surfN':surfN
    }

    # print(eta_acc[i_expl] / np.abs(egain[i_expl]),np.abs(egain[i_expl])*1.66e-24/1.602e-6)


    return result


def model_Muller(metallicity: float, beta_val: float, alpha_outflow_val: float, eta_outflow_val: float, zeta_val: float, aturb_val: float, 
          t_15_val:float, m_max_val: float, wind_is_on: bool):
    
    parent_folder = os.path.dirname(__file__)

    subfolder = f'grid_che_helium_single_yoon_++_withN_{metallicity}_solar_CC'
    source = os.path.join(parent_folder, subfolder)
    files = sorted(os.listdir(source))
    # print(files)

    #! Initialization of arrays
    m_init_list = np.ndarray(len(files), dtype = float)
    ni56 = np.ndarray(len(files), dtype = float)
    E_exp = np.ndarray(len(files), dtype = float)
    M_ej = np.ndarray(len(files), dtype = float)
    m_grav = np.ndarray(len(files), dtype = float)
    xi25 = np.ndarray(len(files), dtype = float)
    mu4 = np.ndarray(len(files), dtype = float)
    mu4_mod = np.ndarray(len(files), dtype = float)
    mfinal = np.ndarray(len(files), dtype = float)
    rfinal = np.ndarray(len(files), dtype = float)
    mwind = np.ndarray(len(files), dtype = float)
    vkick = np.ndarray(len(files), dtype = float)
    co_mass = np.ndarray(len(files), dtype = float)
    Z = np.ndarray(len(files), dtype = float)

    #! Initialization of array index
    j = 0
    for file in files:
        minit = float(file.split("_")[0])
        if minit >= 4.5:
            try:
                # print(os.path.join(source, file))
                #* Get data from profiles and store the initial mass
                profile = mr.MesaData(os.path.join(source, file)).data
                m_init_list[j] = minit
                mass = profile('mass')[::-1]*u.Msun
                mfin = mass[-1].value
                mass = mass.to(u.g).value
                radius = profile('radius')[::-1]*u.Rsun
                radius = radius.to(u.cm).value
                entropy = profile('entropy')[::-1]
                rho=(10**profile('logRho')[::-1])*u.g/(u.cm**3)
                rho=rho.value
                pressure = profile('pressure')[::-1]
                ybind = -(3*c.G.cgs.value*(mass*u.g)**2/(radius*u.cm)/5.).to(u.erg).value
                cs = np.sqrt(4.0/3.0*pressure/rho)
                velocity = profile('velocity')[::-1]
                dm = (profile('dq')*profile('mass')[0])[::-1]

                dm = (dm*u.Msun).to(u.g).value

                tot_energy_int = 0
                total_ener_int = []
                for kk in range(0,len(rho)):
                    tot_energy_int = tot_energy_int - (c.G.cgs.value*(mass[kk]*u.g)*(dm[kk]*u.g)/(radius[kk]*u.cm)).to(u.erg).value + (pressure[kk]/rho[kk])*(dm[kk])+ 0.5*np.power(velocity[kk],2)*dm[kk]
                    total_ener_int.append(tot_energy_int)

                #* Run the evolution model for the specified choice of parameters
                res = criterion_Muller(profile, radius, mass, rho, entropy, total_ener_int, alpha_outflow=alpha_outflow_val, alpha_turb=aturb_val, beta=beta_val, eta_outflow=eta_outflow_val, t_15=t_15_val * u.s, zeta=zeta_val, m_max= m_max_val * u.Msun, addwind=wind_is_on)
                res2 = extra_stuff(profile)

                #* Get the desired results
                ni56[j] = ((res['Ni56']*u.g).to(u.Msun)).value
                E_exp[j] = res['E_expl'].value
                M_ej[j] = ((res['m_ej']*u.g).to(u.Msun)).value
                m_grav[j] = ((res['m_grav']*u.g).to(u.Msun)).value
                xi25[j] = res['xi25']
                mu4[j] = res['mu4']
                mu4_mod[j] = res['mu4']
                mfinal[j] = (res['m_star']*u.g).to(u.Msun).value
                rfinal[j] = (res['r_star']*u.cm).to(u.Rsun).value
                mwind[j] = (res['m_wind']*u.g).to(u.Msun).value
                vkick[j] = (res['v_kick']*(u.cm/u.s)).to(u.km/u.s).value
                co_mass[j] = res2['cocoremass']
                Z[j] = metallicity

                j += 1
            except:
                pass

    #* Store the results in a dictionary for easier access
    outputs_per_Z = {
        'm_init': m_init_list,
        'ni56': ni56,
        'E_exp': E_exp,
        'M_ej': M_ej,
        'm_grav': m_grav,
        'xi25': xi25,
        'mu4': mu4,
        'mu4_mod': mu4_mod,
        'mfinal': mfinal,
        'rfinal': rfinal,
        'mwind': mwind,
        'vkick': vkick,
        'co_mass': co_mass
    }

    return outputs_per_Z
#TODO =====================================================================================================================================
#?=========================================================================================================================================
def Ashman_distance(mu_1:float, sigma_1:float, mu_2:float, sigma_2:float):
    return np.sqrt(2) * abs(mu_1 - mu_2) / np.sqrt(sigma_1**2 + sigma_2**2)


def sample_kurtosis(data: any):
    sample_mean_val = sample_mean(data)

    n = len(data)
    s_1 = 0
    s_2 = 0
    for i in range(n):
        s_1 += (data[i] - sample_mean_val)**4
        s_2 += (data[i] - sample_mean_val)**2
    
    s_1 /= n
    s_2 = s_2**2 / (n**2)

    return s_1/s_2 - 3


def sample_mean(data: any):
    n = len(data)
    s = 0

    for i in range(n):
        s += data[i] 
    
    return s / n

#! remember that sample variance is S^2
def sample_variance(data: any, sample_m: float):
    n = len(data)
    s_sq = 0

    for i in range(n):
        s_sq += (data[i] - sample_m)**2
    
    s_sq /= n - 1
    return s_sq


def compute_GaussianMixture(X: any, N: any,covariance_type='full', max_iter: int =10000):
    models = [None for n in N]
    for i in range(len(N)):
        if X.shape[0] >= N[i]:
            models[i] = GaussianMixture(n_components=N[i], max_iter=max_iter,
                                        covariance_type=covariance_type)  # , warm_start=True
            models[i].fit(X)
    return models


def compute_GaussianMixture_with_var_floor(X: any, N: any, covariance_type='full', max_iter: int =10000, variance_floor: bool = True):
    models = [None for n in N]
    sample_m = sample_mean(X)
    sample_var = sample_variance(X, sample_m)
    for i in range(len(N)):
        models[i] = GaussianMixture(n_components=N[i], max_iter=max_iter,
                                    covariance_type=covariance_type)
        models[i].fit(X)
        model_covariances = np.array([cs[0][0] for cs in models[i].covariances_])
        while variance_floor and np.any(model_covariances > 0.1 * sample_var):
            pass
    return models
#?=========================================================================================================================================
#! this is the main function. Inside it happens everything. If you want to add additional processes, add them inside this function
def main():
    code_path = os.path.dirname(__file__)
    #! select what you want to do (more than one parts can be active at once)
    create_parameter_grid = False
    split_params_file = False
    parameter_study = False
    result_analysis = False
    results_analysis_statistics = True
    #! This part creates the parameter grid and saves it in a .txt file
    if create_parameter_grid:
        grid_resolution = 3
        aout_vals_for_study = np.linspace(0.3, 0.7, grid_resolution, endpoint=True)
        aturb_vals_for_study = np.linspace(1, 1.4, grid_resolution, endpoint=True)
        beta_vals_for_study = np.linspace(3, 7, grid_resolution, endpoint=True)
        zeta_vals_for_study = np.linspace(0.5, 1, grid_resolution, endpoint=True)
        t_15_vals_for_study = np.linspace(0.6, 3, grid_resolution, endpoint=True)
        eta_outflow_vals_for_study = np.linspace(0, 2, grid_resolution, endpoint=True)
        m_max_vals_for_study = np.linspace(2.0, 3.0, grid_resolution, endpoint=True) #! up tp 3.0

        param_config_file = os.path.join(code_path, 'param_configs.txt')
        if os.path.isfile(param_config_file):
            pass
        else:
            with open(param_config_file, 'x') as f:
                pass

        for a_out_for_param_st in aout_vals_for_study:
            for a_turb_for_param_st in aturb_vals_for_study:
                for beta_for_param_st in beta_vals_for_study:
                    for zeta_for_param_st in zeta_vals_for_study:
                        for t_15_for_param_st in t_15_vals_for_study:
                            for eta_for_param_st in eta_outflow_vals_for_study:
                                for m_max_for_param_st in m_max_vals_for_study:
                                    with open(param_config_file, 'a') as f:
                                        f.write(f'{round(a_out_for_param_st, 3)}, {round(a_turb_for_param_st, 3)}, {round(beta_for_param_st, 3)}, {round(zeta_for_param_st, 3)}, {round(t_15_for_param_st, 3)}, {round(eta_for_param_st, 3)}, {round(m_max_for_param_st, 3)}\n')
    #! This part takes the above .txt file and splites it into N other files with smaller size 
    if split_params_file:
        parent_dir = 'Parameter Cases'
        dir_params_path = os.path.join(code_path, parent_dir)
        if not os.path.isdir(dir_params_path):
            os.makedirs(dir_params_path)

        #! loading the parameters file that is created above
        params_configs = np.loadtxt(
            param_config_file, dtype=float, delimiter=',')

        #! number of files you want to create
        N = 10

        length = len(params_configs)
        l = length // N  # ! the number of lines to be stored in a seperate file

        for i in range(1, N + 1):
            file = 'param_configs_' + f'{i}' + '.txt'

            file_path = os.path.join(dir_params_path, file)
            if os.path.isfile(file_path):
                pass
            else:
                with open(file_path, 'x') as f:
                    pass

            for j in range(l):
                ind = min((i - 1) * l + j, len(params_configs)-1)
                with open(file_path, 'a') as f:
                    ln = ''
                    for item in params_configs[ind]:
                        ln += f'{item}, '
                    ln = ln[:-2]
                    ln += '\n'
                    f.write(ln)

        #! check for remaining lines and store them in an additional file
        remaining_inds = abs(N * l - len(params_configs))
        if remaining_inds != 0:
            file = 'param_configs_' + f'{N + 1}' + '.txt'

            file_path = os.path.join(dir_params_path, file)
            if os.path.isfile(file_path):
                pass
            else:
                with open(file_path, 'x') as f:
                    pass

            for j in range(remaining_inds):
                ind = N * l + j
                with open(file_path, 'a') as f:
                    ln = ''
                    for item in params_configs[ind]:
                        ln += f'{item}, '
                    ln = ln[:-2]
                    ln += '\n'
                    f.write(ln)
    #! This part performes the parameter study
    if parameter_study:
        old_model = False #? Uncomment boom and arnett for this to work
        new_model = True
        #! gets the path for the directory
        parent_dir = 'Parameter Study'
        dir_path = os.path.join(code_path, parent_dir)

        #! makes sure the directory that the plots will be saved in is created if it does not exist
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        
        #* Get the parameter file path
        params_parent_dir = 'Parameter Cases'
        dir_params_path = os.path.join(code_path, params_parent_dir)
        ind_of_file = 1 #sys.argv[1]
        param_file = 'param_configs_' + f'{ind_of_file}' + '.txt'
        param_config_file = os.path.join(dir_params_path, param_file)
        wind_on = True
        Z_metallicity = [0.5, 0.75, 1.0]
        #* Additional delimeter to discriminate the directories for the two models
        add_M = 'Muller'
        add_O = 'Old'
        
        params_configs = np.loadtxt(param_config_file, dtype=float, delimiter=',')
        for Z_m in Z_metallicity:
            results_dir = ''
            #* Check whether we use the new or the old model
            if new_model:
                #! set the directory in which the data will be stored
                dir_path_M = os.path.join(dir_path, add_M)
                if not os.path.isdir(dir_path_M):
                    os.makedirs(dir_path_M)
                results_dir = os.path.join(dir_path_M, f'{Z_m} metallicity')
                #! makes sure the directory that the data will be saved in is created if it does not exist
                if not os.path.isdir(results_dir):
                    os.makedirs(results_dir)
            elif old_model:
                #! set the directory in which the data will be stored
                dir_path_M = os.path.join(dir_path, add_O)
                if not os.path.isdir(dir_path_M):
                    os.makedirs(dir_path_M)
                results_dir = os.path.join(dir_path_M, f'{Z_m} metallicity')
                #! makes sure the directory that the data will be saved in is created if it does not exist
                if not os.path.isdir(results_dir):
                    os.makedirs(results_dir)
            
            #? Data directory
            subf_data_dir = os.path.join(
                results_dir, 'Data')
            #! makes sure the directory that the data will be saved in is created if it does not exist
            if not os.path.isdir(subf_data_dir):
                os.makedirs(subf_data_dir)
            
            # ? Monte Carlo directory
            #! set the directory in which the data will be stored
            monte_carlo_dir_path = os.path.join(
                results_dir, 'Monte Carlo Study')
            #! makes sure the directory that the data will be saved in is created if it does not exist
            if not os.path.isdir(monte_carlo_dir_path):
                os.makedirs(monte_carlo_dir_path)
            
            for params in params_configs:
                a_out_for_param_st, a_turb_for_param_st, beta_for_param_st, zeta_for_param_st, t_15_for_param_st, eta_for_param_st, m_max_for_param_st = params 
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                #                                  correlations
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # * creates the specific figure name depending on the name of the parameters
                file_name = f'corr_for_Z_{Z_m}_beta_{beta_for_param_st}_a_out_{a_out_for_param_st}_eta_out_{eta_for_param_st}_zeta_{zeta_for_param_st}_a_turb_{a_turb_for_param_st}_tau_1.5_{t_15_for_param_st}_m_max_{m_max_for_param_st}.txt'

                # * creates the path for saving
                file_name_save = os.path.join(
                    subf_data_dir, file_name)

                if os.path.isfile(file_name_save):
                    pass
                else:
                    with open(file_name_save, 'x') as f:
                        pass
                res = {}
                # * run the model for the imposed values of the parameters
                if old_model:
                    res = model(metallicity=Z_m, beta_val=beta_for_param_st, alpha_outflow_val=a_out_for_param_st, eta_outflow_val=eta_for_param_st,
                            zeta_val=zeta_for_param_st, aturb_val=a_turb_for_param_st, t_15_val=t_15_for_param_st, m_max_val=m_max_for_param_st,
                            wind_is_on=wind_on)
                elif new_model:
                    res = model_Muller(metallicity=Z_m, beta_val=beta_for_param_st, alpha_outflow_val=a_out_for_param_st, eta_outflow_val=eta_for_param_st,
                            zeta_val=zeta_for_param_st, aturb_val=a_turb_for_param_st, t_15_val=t_15_for_param_st, m_max_val=m_max_for_param_st,
                            wind_is_on=wind_on)
                
                with open(file_name_save, 'a') as f:
                    for key in res:
                        line = ''
                        for item in res[key]:
                            line += f'{item}, '
                        line = line[:-2]
                        line += '\n'
                        f.write(line)
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                #                                                    Monte Carlo study
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                metal_file = f'MC_for_metallicity_Z_{Z_m}_beta_{beta_for_param_st}_a_out_{a_out_for_param_st}_eta_out_{eta_for_param_st}_zeta_{zeta_for_param_st}_a_turb_{a_turb_for_param_st}_tau_1.5_{t_15_for_param_st}_m_max_{m_max_for_param_st}.txt'
                metal_file_path = os.path.join(monte_carlo_dir_path, metal_file)

                if os.path.isfile(metal_file_path):
                    pass
                else:
                    with open(metal_file_path, 'x') as f:
                        pass          
                # * number of repetitions 
                N = 10_000
                # * number of repetitions done
                n = 0                
                while n < N:
                    try:
                        # * generate randomly an initial mass for a star
                        randomly_generated_mass = random_power_law(
                            a=4.5, b=70.0, p=-2.35)  # ? Salpeter: p = -2.35
                        # * evole that star up until it reaches the He status
                        He_mass_from_randomly_generated_mass = ZAMS_to_He(
                            randomly_generated_mass)[0]
                        # * mass which will indicate which profile we shall use
                        mass_for_profile = 0
                        if He_mass_from_randomly_generated_mass % int(He_mass_from_randomly_generated_mass) < 0.5:
                            # * np.fix() returns the int part as a float, meaning for 2.3 it returns 2.0
                            mass_for_profile = np.floor(
                                He_mass_from_randomly_generated_mass)
                        elif He_mass_from_randomly_generated_mass % int(He_mass_from_randomly_generated_mass) >= 0.5:
                            mass_for_profile = np.floor(
                                He_mass_from_randomly_generated_mass) + 0.5
                        
                        # * Get the desired result from the already found data
                        m_grav = res['m_grav'][np.where(
                            res['m_init'] == mass_for_profile)][0]
                        # print(
                        #     f'Z: {Z_m}| Initial mass: {randomly_generated_mass[0]} -> He star mass: {mass_for_profile} -> final mass: {m_grav}| n: {n}')
                        with open(metal_file_path, 'a') as metal_f:
                            metal_f.write(f'{m_grav}\n')
                        n += 1
                    except:
                        pass
    #! This part analyses the results from the simulations
    if result_analysis:
        hists_and_fits = False
        random_hists_and_fits = True

        if hists_and_fits:
            # old_model = True
            # new_model = False
            # dir_name = 'Distribution Information'
            # results_dir = os.path.join(code_path, dir_name)
            # if not os.path.isdir(results_dir):
            #     os.makedirs(results_dir)

            # #* Initialization for file with results from fitting in remnant (ns + bh) mass distribution
            # # results_file_both = os.path.join(results_dir, 'Fitting_results_for_both.txt')
            # # if os.path.isfile(results_file_both):
            # #     pass
            # # else:
            # #     with open(results_file_both, 'x') as f:
            # #         pass

            # #* Initialization for file for results from fitting in bh mass distribution
            # results_file_bh = os.path.join(results_dir, 'Fitting_results_for_bh.txt')
            # if os.path.isfile(results_file_bh):
            #     pass
            # else:
            #     with open(results_file_bh, 'x') as f:
            #         pass

            # #* Initialization for file for results from fitting in ns mass distribution
            # # results_file_ns = os.path.join(
            # #     results_dir, 'Fitting_results_for_ns.txt')
            # # if os.path.isfile(results_file_ns):
            # #     pass
            # # else:
            # #     with open(results_file_ns, 'x') as f:
            # #         pass

            # #* Initialization for the mass gap file
            # mass_gap_file = os.path.join(results_dir, 'mass_gap_file.txt')
            # if os.path.isfile(mass_gap_file):
            #     pass
            # else:
            #     with open(mass_gap_file, 'x') as f:
            #         pass
            
            # #* Path for data files
            # starting_dir = "Parameter Study"
            # starting_dir_path = os.path.join(code_path, starting_dir)
            
            # if old_model:
            #     starting_dir_path = os.path.join(starting_dir_path, 'Old')
            # elif new_model:
            #     starting_dir_path = os.path.join(starting_dir_path, 'Muller')
            
            #* Number of Gaussians to fit
            N = np.arange(1, 10)

            #* train test split percentage
            train_test_split_percentage = 0.25
            # metallicities = [0.5, 0.75, 1.0]
            # for Z_m in metallicities:
                #* Get the corresponding directory
                # data_dir = os.path.join(starting_dir_path, "{} metallicity".format(str(Z_m)))

                # #* Focus only on the Monte Carlo subdirectory
                # monte_carlo_dir_path = os.path.join(
                #     data_dir, "Monte Carlo Study")
            
            #     #* Get the directories
            #     dirs = os.listdir(monte_carlo_dir_path)
    
            # # data_dir = os.path.join(code_path, 'Data in new txt')
            # # files = os.listdir(data_dir)
            #     for dirum in dirs:
            #         dir_path = os.path.join(monte_carlo_dir_path, dirum)
            #         #* Get the files
            #         files = os.listdir(dir_path)
                
                #* Get the files
                # files = os.listdir(monte_carlo_dir_path)
    
            data_dir = os.path.join(code_path, 'Data in new txt')
            files = os.listdir(data_dir)
            for file in files:
                file_path = os.path.join(data_dir, file)
                to_remove = ['MC', 'for', 'metallicity', 'Z', 'beta',
                                'a', 'out', 'eta', 'out', 'zeta', 'a', 'turb', 'tau', '1.5', 'm', 'max']
                file_without_extension = file.replace('.txt', '')
                f_n = file_without_extension.split('_')
                for word in to_remove:
                    f_n.remove(word)
                nums = list(map(float, f_n))
                Z, beta, a_out, eta_out, zeta, a_turb, tau_15, m_max = nums
                data_pre = np.loadtxt(file_path, dtype = float)
                ns_data = data_pre[data_pre <= m_max]
                bh_data = data_pre[data_pre > m_max]
                
                # no_nss = 0
                # no_bhs = 0
                # zero_mass_gap = 0
                # if np.any(ns_data) and np.any(bh_data):
                #     mass_gap = abs(max(ns_data) - min(bh_data))
                #     if mass_gap == 0:
                #         zero_mass_gap = 1
                # elif len(ns_data) == 0:
                #     #? Nnoooo, me mass gap is gone
                #     mass_gap = 0
                #     no_nss = 1
                # elif len(bh_data) == 0:
                #     #? Nnoooo, me mass gap is gone
                #     mass_gap = 0
                #     no_bhs = 1
                # with open(mass_gap_file, 'a') as f:
                #     #? File structure: Z, β, α_out, η_out, ζ, α_out, τ_1.5, m_max, mass gap, zero mass gap (1 if true), no NSs (1 if true), no Bhs (1 if true)
                #     f.write(f'{Z}, {beta}, {a_out}, {eta_out}, {zeta}, {a_turb}, {tau_15}, {m_max}, {mass_gap}, {zero_mass_gap}, {no_nss}, {no_bhs}\n')
                # # # data = StandardScaler().fit_transform(data)

                # fig, axs = plt.subplots(2, 2, figsize=(10, 7))
                # fig.suptitle('Mass distributions')
                # fig.delaxes(axs[1,0])

                #* Remnant (ns + bh) data
                data_for_GM = np.reshape(data_pre, (-1, 1))
                data_train, data_test = sklearn.model_selection.train_test_split(data_for_GM, test_size = train_test_split_percentage)
                models = compute_GaussianMixture(data_train, N)
                # index_AIC = [m.aic(data_test) for m in models]                       
                index_BIC = [m.bic(data_test) if m != None else np.inf for m in models]
                # print(index_BIC)
                # best_model_index = min(np.argmin(index_BIC), np.argmin(index_AIC))
                best_model_index = np.argmin(index_BIC)
                best_model = models[best_model_index]

                # modes = 1
                # means = [mn[0] for mn in best_model.means_] #! The means are not in ascending order, meaning the neighbouring Gaussian j in the plot is not j + 1. So we need sort the list to set the order correctly
                # means_sorted = sorted(means)
                # #! Then we need to set the sigmas accordingly
                # sigmas = [sm[0][0] for sm in best_model.covariances_]
                # sigmas_sorted = [0 for i in range(len(means_sorted))]
                # for mu in means_sorted:
                #     sigmas_sorted[means_sorted.index(mu)] = sigmas[means.index(mu)]

                # Ds_list = np.zeros(len(means_sorted)-1, dtype = float)
                # for j in range(len(means_sorted)-1):
                #     Ds_list[j] = Ashman_distance(means_sorted[j], np.sqrt(sigmas_sorted[j]), means_sorted[j + 1], np.sqrt(sigmas_sorted[j + 1]))
                #     if Ds_list[j] > 2:
                #         modes += 1
                        # print(
                        #     f'μ_1 = {means_sorted[j]}, σ_1 = {np.sqrt(sigmas_sorted[j])}, μ_2 = {means_sorted[j + 1]}, σ_2 = {np.sqrt(sigmas_sorted[j + 1])} | D = {Ds}')
                # print(f'NS + BH| The distribution has {modes} modes | kurtosi: {sample_kurtosis(data_pre)}')

                # n_dist_groups = 1
                # n_large_sep = 0
                # for d in Ds_list:
                #     if d > 4:
                #         n_dist_groups += 1
                #     if d > 8:
                #         n_large_sep += 1

                # plot_x = np.linspace(0, max(data_pre), len(data_pre))
                # plot_y = np.exp(best_model.score_samples(plot_x.reshape(-1, 1)))
                # hist_values, bin_edges, _ = axs[0, 0].hist(data_pre, bins=100, density = True, color = 'blue')
                # scale_factor = max(hist_values) / max(plot_y)
                # plot_y *= scale_factor
                # axs[0, 0].plot(plot_x, plot_y, color = 'red')
                # axs[0, 0].set_title('NS + BH')

                #* NS data 
                if np.any(ns_data):
                    ns_data_for_GM = ns_data.reshape(-1, 1)
                    ns_data_train, ns_data_test = sklearn.model_selection.train_test_split(ns_data_for_GM, test_size = train_test_split_percentage)
                    models_for_NS = compute_GaussianMixture(ns_data_train, N)
                    index_BIC_NS = [m.bic(ns_data_test) if m != None else np.inf for m in models_for_NS]
                    # index_AIC_NS = [m.aic(ns_data_test) for m in models_for_NS]
                    # best_model_NS_index = min(
                    #     np.argmin(index_BIC_NS), np.argmin(index_AIC_NS))
                    best_model_NS_index = np.argmin(index_BIC_NS)
                    best_model_NS = models_for_NS[best_model_NS_index]

                    # modes_ns = 1
                    # means_ns = [mn[0] for mn in best_model_NS.means_] #! The means are not in ascending order, meaning the neighbouring Gaussian j in the plot is not j + 1. So we need sort the list to set the order correctly
                    # means_ns_sorted = sorted(means_ns)
                    # #! Then we need to set the sigmas accordingly
                    # sigmas_ns = [sm[0][0] for sm in best_model_NS.covariances_]
                    # sigmas_ns_sorted = [0 for i in range(len(means_ns_sorted))]

                    # for mu_ns in means_ns_sorted:
                    #     sigmas_ns_sorted[means_ns_sorted.index(mu_ns)] = sigmas_ns[means_ns.index(mu_ns)]

                    # Ds_ns_list = np.zeros(len(means_ns_sorted)-1, dtype = float)
                    # for j_ns in range(len(means_ns_sorted)-1):
                    #     Ds_ns_list[j_ns] = Ashman_distance(means_ns_sorted[j_ns], np.sqrt(sigmas_ns_sorted[j_ns]), means_ns_sorted[j_ns + 1], np.sqrt(sigmas_ns_sorted[j_ns + 1]))
                    #     if Ds_ns_list[j_ns] > 2:
                            # modes_ns += 1
                            # print(
                            #     f'μ_1 = {means_ns_sorted[j_ns]}, σ_1 = {np.sqrt(sigmas_ns_sorted[j_ns])}, μ_2 = {means_ns_sorted[j_ns + 1]}, σ_2 = {np.sqrt(sigmas_ns_sorted[j_ns + 1])} | D = {Ds_ns}')
                    # print(f'NS| The distribution has {modes_ns} modes | kurtosi: {sample_kurtosis(ns_data)}')

                    # n_dist_groups_ns = 0
                    # n_large_sep_ns = 0
                    # for d_ns in Ds_ns_list:
                    #     if d_ns > 4:
                    #         n_dist_groups_ns += 1
                    #     if d_ns > 8:
                    #         n_large_sep_ns += 1

                    # hist_values_NS, bin_edges_NS, _ = axs[0, 1].hist(ns_data, bins=40, density = True, color='orange')
                    # plot_x_NS = np.linspace(max(min(bin_edges_NS) - 0.1, 0), max(bin_edges_NS), len(ns_data))
                    # plot_y_NS = np.exp(best_model_NS.score_samples(plot_x_NS.reshape(-1, 1)))
                    # scale_factor_NS = max(hist_values_NS) / max(plot_y_NS)
                    # plot_y_NS *= scale_factor_NS
                    # axs[0, 1].plot(plot_x_NS, plot_y_NS, color = 'red')
                    # axs[0, 1].set_title('NS only')

                #* BH data
                if np.any(bh_data):
                    bh_data_for_GM = bh_data.reshape(-1, 1)
                    data_bh_train, data_bh_test = sklearn.model_selection.train_test_split(bh_data_for_GM, test_size = train_test_split_percentage)               
                    models_for_BH = compute_GaussianMixture(data_bh_train, N)
                    index_BIC_BH = [m.bic(data_bh_test) if m != None else np.inf for m in models_for_BH]
                    # index_AIC_BH = [m.aic(data_bh_test) for m in models_for_BH]
                    # best_model_BH_index = min(
                    #     np.argmin(index_BIC_BH), np.argmin(index_AIC_BH))
                    best_model_BH_index = np.argmin(index_BIC_BH)
                    best_model_BH = models_for_BH[best_model_BH_index]

                    modes_bh = 1
                    means_bh = [mn[0] for mn in best_model_BH.means_] #! The means are not in ascending order, meaning the neighbouring Gaussian j in the plot is not j + 1. So we need sort the list to set the order correctly
                    means_bh_sorted = sorted(means_bh)
                    #! Then we need to set the sigmas accordingly
                    sigmas_bh = [sm[0][0] for sm in best_model_BH.covariances_]
                    sigmas_bh_sorted = [0 for i in range(len(means_bh_sorted))]

                    for mu_bh in means_bh_sorted:
                        sigmas_bh_sorted[means_bh_sorted.index(mu_bh)] = sigmas_bh[means_bh.index(mu_bh)]

                    Ds_bh_list = np.zeros(len(means_bh_sorted)-1, dtype = float)
                    for j_bh in range(len(means_bh_sorted)-1):
                        Ds_bh_list[j_bh] = Ashman_distance(means_bh_sorted[j_bh], np.sqrt(sigmas_bh_sorted[j_bh]), means_bh_sorted[j_bh + 1], np.sqrt(sigmas_bh_sorted[j_bh + 1]))
                        if Ds_bh_list[j_bh] > 2:
                            modes_bh += 1
                    #         print(
                    #             f'μ_1 = {means_bh_sorted[j_bh]}, σ_1 = {np.sqrt(sigmas_bh_sorted[j_bh])}, μ_2 = {means_bh_sorted[j_bh + 1]}, σ_2 = {np.sqrt(sigmas_bh_sorted[j_bh + 1])} | D = {Ds_bh}')
                    # print(f'BH| The distribution has {modes_bh} modes | kurtosi: {sample_kurtosis(bh_data)}')
                    n_dist_groups_bh = 0
                    n_large_sep_bh = 0
                    for d_bh in Ds_bh_list:
                        if d_bh > 4:
                            n_dist_groups_bh += 1
                        if d_bh > 8:
                            n_large_sep_bh += 1

                    # with open(results_file_bh, 'a') as f:
                    #     #? File structure: model has converged (1 = True, 0 = False), N_comp, Z, β, α_out, η_out, ζ, α_turb, τ_1.5, m_max, N_{groups}, N_{large seperation}
                    #     f.write(f'{int(best_model_BH.converged_}, {N[best_model_BH_index]}, {Z}, {beta}, {a_out}, {eta_out}, {zeta}, {a_turb}, {tau_15}, {m_max}, {n_dist_groups_bh}, {n_large_sep_bh}\n')

                    # hist_values_BH, bin_edges_BH, _ = axs[1, 1].hist(bh_data, bins=30, density = True, color='green')
                    # plot_x_BH = np.linspace(
                    #     max(min(bin_edges_BH) - 0.5, 0), max(bin_edges_BH), len(bh_data))
                    # plot_y_BH = np.exp(best_model_BH.score_samples(plot_x_BH.reshape(-1, 1)))
                    # # print(hist_values_BH)
                    # scale_factor_BH = max(hist_values_BH) / max(plot_y_BH)
                    # plot_y_BH *= scale_factor_BH
                    # axs[1, 1].plot(plot_x_BH, plot_y_BH, color = 'red')
                    # axs[1, 1].set_title('BH only')
                
                # plt.show()
                # plt.close()
                break
        
        if random_hists_and_fits:
            old_model = True
            new_model = False
            dir_name = 'Distribution Information'
            results_dir = os.path.join(code_path, dir_name)
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

            #* Initialization for file with results from fitting in remnant (ns + bh) mass distribution
            # results_file_both = os.path.join(results_dir, 'Fitting_results_for_both.txt')
            # if os.path.isfile(results_file_both):
            #     pass
            # else:
            #     with open(results_file_both, 'x') as f:
            #         pass

            #* Initialization for file for results from fitting in bh mass distribution
            results_file_bh = os.path.join(results_dir, 'Fitting_results_for_bh_random.txt')
            if os.path.isfile(results_file_bh):
                pass
            else:
                with open(results_file_bh, 'x') as f:
                    pass

            #* Initialization for file for results from fitting in ns mass distribution
            # results_file_ns = os.path.join(
            #     results_dir, 'Fitting_results_for_ns.txt')
            # if os.path.isfile(results_file_ns):
            #     pass
            # else:
            #     with open(results_file_ns, 'x') as f:
            #         pass

            #* Initialization for the mass gap file
            mass_gap_file = os.path.join(results_dir, 'mass_gap_file_random.txt')
            if os.path.isfile(mass_gap_file):
                pass
            else:
                with open(mass_gap_file, 'x') as f:
                    pass
            
            #* Path for data files
            starting_dir = "Parameter Study"
            starting_dir_path = os.path.join(code_path, starting_dir)
            
            if old_model:
                starting_dir_path = os.path.join(starting_dir_path, 'Old')
            elif new_model:
                starting_dir_path = os.path.join(starting_dir_path, 'Muller')
            
            #* Number of Gaussians to fit
            N = np.arange(1, 10)
            #* train test split percentage
            train_test_split_percentage = 0.25

            metallicities = [0.5, 0.75, 1.0]
            for Z_m in metallicities:
                random_plots = 0
                #* Get the corresponding directory
                data_dir = os.path.join(starting_dir_path, "{} metallicity".format(str(Z_m)))

                #* Focus only on the Monte Carlo subdirectory
                monte_carlo_dir_path = os.path.join(
                    data_dir, "Monte Carlo Study")
                
                #* Get the directories
                dirs = os.listdir(monte_carlo_dir_path)
                while random_plots < 2:
                    #* Get a random dir and its path
                    random_dir = np.random.choice(dirs)
                    random_dir_path = os.path.join(monte_carlo_dir_path, random_dir)
                    #* Get the files
                    file_from_random_dir = np.random.choice(os.listdir(random_dir_path))
                    file_from_random_dir_path = os.path.join(random_dir_path, file_from_random_dir)
                    #* Extract the values of the parameters
                    to_remove = ['MC', 'for', 'metallicity', 'Z', 'beta',
                                        'a', 'out', 'eta', 'out', 'zeta', 'a', 'turb', 'tau', '1.5', 'm', 'max']
                    file_from_random_dir_without_extension = file_from_random_dir.replace(
                        '.txt', '')
                    f_n = file_from_random_dir_without_extension.split('_')
                    for word in to_remove:
                        f_n.remove(word)
                    nums = list(map(float, f_n))
                    Z, beta, a_out, eta_out, zeta, a_turb, tau_15, m_max = nums
                    #* Load the data
                    data_pre = np.loadtxt(file_from_random_dir_path, dtype = float)
                    ns_data = data_pre[data_pre <= m_max]
                    bh_data = data_pre[data_pre > m_max]

                    if np.any(ns_data) and np.any(bh_data):
                        no_nss = 0
                        no_bhs = 0
                        zero_mass_gap = 0
                        if np.any(ns_data) and np.any(bh_data):
                            mass_gap = abs(max(ns_data) - min(bh_data))
                            if mass_gap == 0:
                                zero_mass_gap = 1
                        elif len(ns_data) == 0:
                            # ? Nnoooo, me mass gap is gone
                            mass_gap = 0
                            no_nss = 1
                        elif len(bh_data) == 0:
                            # ? Nnoooo, me mass gap is gone
                            mass_gap = 0
                            no_bhs = 1
                        with open(mass_gap_file, 'a') as f:
                            # ? File structure: Z, β, α_out, η_out, ζ, α_out, τ_1.5, m_max, mass gap, zero mass gap (1 if true), no NSs (1 if true), no Bhs (1 if true)
                            f.write(
                                f'{Z}, {beta}, {a_out}, {eta_out}, {zeta}, {a_turb}, {tau_15}, {m_max}, {mass_gap}, {zero_mass_gap}, {no_nss}, {no_bhs}\n')
                        #* Remnant (ns + bh) data
                        data_for_GM = np.reshape(data_pre, (-1, 1))
                        data_train, data_test = sklearn.model_selection.train_test_split(data_for_GM, test_size = train_test_split_percentage)
                        models = compute_GaussianMixture(data_train, N)                      
                        index_BIC = [m.bic(data_test) if m != None else np.inf for m in models]
                        best_model_index = np.argmin(index_BIC)
                        best_model = models[best_model_index]

                        modes = 1
                        means = [mn[0] for mn in best_model.means_] #! The means are not in ascending order, meaning the neighbouring Gaussian j in the plot is not j + 1. So we need sort the list to set the order correctly
                        means_sorted = sorted(means)
                        #! Then we need to set the sigmas accordingly
                        sigmas = [sm[0][0] for sm in best_model.covariances_]
                        sigmas_sorted = [0 for i in range(len(means_sorted))]
                        for mu in means_sorted:
                            sigmas_sorted[means_sorted.index(mu)] = sigmas[means.index(mu)]

                        Ds_list = np.zeros(len(means_sorted)-1, dtype = float)
                        for j in range(len(means_sorted)-1):
                            Ds_list[j] = Ashman_distance(means_sorted[j], np.sqrt(sigmas_sorted[j]), means_sorted[j + 1], np.sqrt(sigmas_sorted[j + 1]))
                            if Ds_list[j] > 2:
                                modes += 1

                        n_dist_groups = 0
                        n_large_sep = 0
                        for d in Ds_list:
                            if d > 4:
                                n_dist_groups += 1
                            if d > 8:
                                n_large_sep += 1

                        fig = plt.plot(figsize = (8, 7))
                        plot_x = np.linspace(0, max(data_pre), len(data_pre))
                        plot_y = np.exp(best_model.score_samples(plot_x.reshape(-1, 1)))
                        hist_values, bin_edges, _ = plt.hist(data_pre, bins=100, density = True, color = 'blue', label='Actual remnant mass distribution')
                        scale_factor = max(hist_values) / max(plot_y)
                        plot_y *= scale_factor
                        plt.plot(plot_x, plot_y, color = 'red', label='Best Gaussian Mixture fit')
                        plt.legend()
                        plt.title(f'NS + BH mass distribution for Z={Z}, '+r'$\beta$='+f'{beta}, '+r'$\alpha_{out}$='+ f'{a_out}, \n'+r'$\eta_{out}$='+f'{eta_out}, '+r'$\zeta$='+f'{zeta},'+r'$\alpha_{turb}$='+f'{a_turb}, '+r'$\tau_{1.5}$='+f'{tau_15}, '+'$m_{max}$='+f'{m_max}')
                        plt.savefig(os.path.join(
                            results_dir, f'remnant_mass_distr_{Z}_{beta}_{a_out}_{eta_out}_{zeta}_{a_turb}_{tau_15}_{m_max}.pdf'))
                        plt.close()

                        #* BH data
                        bh_data_for_GM = bh_data.reshape(-1, 1)
                        data_bh_train, data_bh_test = sklearn.model_selection.train_test_split(bh_data_for_GM, test_size = train_test_split_percentage)               
                        models_for_BH = compute_GaussianMixture(data_bh_train, N)
                        index_BIC_BH = [m.bic(data_bh_test) if m != None else np.inf for m in models_for_BH]
                        best_model_BH_index = np.argmin(index_BIC_BH)
                        best_model_BH = models_for_BH[best_model_BH_index]

                        modes_bh = 1
                        means_bh = [mn[0] for mn in best_model_BH.means_] #! The means are not in ascending order, meaning the neighbouring Gaussian j in the plot is not j + 1. So we need sort the list to set the order correctly
                        means_bh_sorted = sorted(means_bh)
                        #! Then we need to set the sigmas accordingly
                        sigmas_bh = [sm[0][0] for sm in best_model_BH.covariances_]
                        sigmas_bh_sorted = [0 for i in range(len(means_bh_sorted))]

                        for mu_bh in means_bh_sorted:
                            sigmas_bh_sorted[means_bh_sorted.index(mu_bh)] = sigmas_bh[means_bh.index(mu_bh)]

                        Ds_bh_list = np.zeros(len(means_bh_sorted)-1, dtype = float)
                        for j_bh in range(len(means_bh_sorted)-1):
                            Ds_bh_list[j_bh] = Ashman_distance(means_bh_sorted[j_bh], np.sqrt(sigmas_bh_sorted[j_bh]), means_bh_sorted[j_bh + 1], np.sqrt(sigmas_bh_sorted[j_bh + 1]))
                            if Ds_bh_list[j_bh] > 2:
                                modes_bh += 1

                        n_dist_groups_bh = 0
                        n_large_sep_bh = 0
                        for d_bh in Ds_bh_list:
                            if d_bh > 4:
                                n_dist_groups_bh += 1
                            if d_bh > 8:
                                n_large_sep_bh += 1

                        with open(results_file_bh, 'a') as f:
                            #? File structure: model has converged (1 = True, 0 = False), N_comp, Z, β, α_out, η_out, ζ, α_turb, τ_1.5, m_max, N_{groups}, N_{large seperation}
                            f.write(f'{int(best_model_BH.converged_)}, {N[best_model_BH_index]}, {Z}, {beta}, {a_out}, {eta_out}, {zeta}, {a_turb}, {tau_15}, {m_max}, {n_dist_groups_bh}, {n_large_sep_bh}\n')

                        fig = plt.plot(figsize=(8, 7))
                        plot_x_BH = np.linspace(min(bh_data)-0.5, max(bh_data), len(bh_data))
                        plot_y_BH = np.exp(best_model_BH.score_samples(
                            plot_x_BH.reshape(-1, 1)))
                        hist_values_BH, bin_edges, _ = plt.hist(
                            bh_data, bins=30, density=True, color='blue', label='Actual BH mass distribution')
                        scale_factor_BH = max(hist_values_BH) / max(plot_y_BH)
                        plot_y_BH *= scale_factor_BH
                        plt.plot(plot_x_BH, plot_y_BH, color='red',label='Best Gaussian Mixture fit')
                        plt.legend()
                        plt.title(f'BH mass distribution for Z={Z}, '+r'$\beta$='+f'{beta}, '+r'$\alpha_{out}$=' + f'{a_out}, \n'+r'$\eta_{out}$=' +
                                      f'{eta_out}, '+r'$\zeta$='+f'{zeta},'+r'$\alpha_{turb}$='+f'{a_turb}, '+r'$\tau_{1.5}$='+f'{tau_15}, '+'$m_{max}$='+f'{m_max}')
                        plt.savefig(os.path.join(
                            results_dir, f'BH_mass_distr_{Z}_{beta}_{a_out}_{eta_out}_{zeta}_{a_turb}_{tau_15}_{m_max}.pdf'))
                        plt.close()

                        random_plots += 1

    #! This part produces the desired statistics from the above analysis
    if results_analysis_statistics:       
        histograms = False
        mfinal_mremnant_plot = False

        if histograms: 
            random_files = True
            ordinary_files = False
            if random_files:
                #? Analysis for all values of Z
                #* Create the file where the statistics will be stored
                analysis_results_file_name = 'statistics_random.txt'
                analysis_results_file = os.path.join(code_path, analysis_results_file_name)
                if os.path.isfile(analysis_results_file):
                    with open(analysis_results_file, 'w') as f:
                        pass
                else:
                    with open(analysis_results_file, 'x') as f:
                        pass
                #* Load the analysis data
                data_dir = os.path.join(code_path, 'Distribution Information')
                bh_distr_data = np.loadtxt(os.path.join(data_dir, 'Fitting_results_for_bh_random.txt'), dtype=float, delimiter=',')
                mass_gap_data = np.loadtxt(os.path.join(data_dir, 'mass_gap_file_random.txt'), dtype=float, delimiter=',')
                # print(len(mass_gap_data))
                for i in range(len(mass_gap_data)):
                    mass_gap_value = mass_gap_data[i][-4]
                    param_vals = mass_gap_data[i][0:8]
                    Z, beta, a_out, eta_out, zeta, a_turb, tau_15, m_max = param_vals
                    number_of_components = bh_distr_data[i][1]
                    number_of_dist_groups = bh_distr_data[i][-2] + 1
                    number_of_large_gaps= bh_distr_data[i][-1]

                    with open(analysis_results_file, 'a') as f:
                        f.write(
                            f'{Z}, {beta}, {a_out}, {eta_out}, {zeta}, {a_turb}, {tau_15}, {m_max}| mass gap = {round(mass_gap_value, 1)}, Number of components = {number_of_components}, number of distinct groups = {number_of_dist_groups}, number of large gaps = {number_of_large_gaps}\n')

            if ordinary_files:
                #? Analysis for all values of Z
                #* Create the file where the statistics will be stored
                analysis_results_file_name = 'statistics.txt'
                analysis_results_file = os.path.join(code_path, analysis_results_file_name)
                if os.path.isfile(analysis_results_file):
                    pass
                else:
                    with open(analysis_results_file, 'x') as f:
                        pass
                #* Load the analysis data
                data_dir = os.path.join(code_path, 'Distribution Information')
                bh_distr_data = np.loadtxt(os.path.join(data_dir, 'Fitting_results_for_bh.txt'), dtype=float, delimiter=',')
                mass_gap_data = np.loadtxt(os.path.join(data_dir, 'mass_gap_file.txt'), dtype=float, delimiter=',')

                number_of_no_bhs = 0
                number_of_no_nss = 0
                number_of_zero_mass_gap = 0

                mass_gap_file_length = len(mass_gap_data)
                existent_mass_gap = 0
                non_existent_mass_gap = 0

                for i in range(len(mass_gap_data)):
                    if mass_gap_data[i][-4] > 0:
                        existent_mass_gap +=1
                    else:
                        non_existent_mass_gap += 1
                        if mass_gap_data[i][-1] == 1:
                            number_of_no_bhs += 1
                        elif mass_gap_data[i][-2] == 1:
                            number_of_no_nss += 1
                        elif mass_gap_data[i][-3] == 1:
                            number_of_zero_mass_gap += 1

                mass_gap_data = mass_gap_data[mass_gap_data[:, -1] == 0]
                bh_distr_data = bh_distr_data[mass_gap_data[:, -2] == 0]

                number_of_components = np.zeros(len(bh_distr_data), dtype=int)
                number_of_dist_groups = np.zeros(len(bh_distr_data), dtype=int)
                number_of_large_gaps = np.zeros(len(bh_distr_data), dtype=int)
                bh_data_file_length = len(bh_distr_data)

                for i in range(len(bh_distr_data)):
                    number_of_components[i] = bh_distr_data[i][1]
                    number_of_dist_groups[i] = bh_distr_data[i][-2] + 1
                    number_of_large_gaps[i] = bh_distr_data[i][-1]

                with open(analysis_results_file, 'w') as f:
                    f.write(
                        f'Out of {mass_gap_file_length} remnant mass distributions, {existent_mass_gap} (or {round(existent_mass_gap / mass_gap_file_length, 2) * 100}%) have a mass gap and {non_existent_mass_gap} (or {round(non_existent_mass_gap / mass_gap_file_length, 2) * 100}%) do not have a mass gap. Out of those that do not have a mass gap, {number_of_no_nss} (or {round(number_of_no_nss / non_existent_mass_gap, 2) * 100}%) do not predict any NSs, {number_of_no_bhs} (or {round(number_of_no_bhs / non_existent_mass_gap, 2) * 100}%) do not predict any BHs, and {number_of_zero_mass_gap} (or {round(number_of_zero_mass_gap / non_existent_mass_gap, 2) * 100}%) simply have a zero mass gap. \nOut of {bh_data_file_length} bh mass distributions, {len(number_of_components[number_of_components > 6])} (or {round(len(number_of_components[number_of_components > 6]) / bh_data_file_length, 2) * 100}%) have more that 6 components in their GMM fit, {len(number_of_dist_groups[number_of_dist_groups > 1])} (or {round(len(number_of_dist_groups[number_of_dist_groups > 1]) / bh_data_file_length, 2) * 100}%) have distinct groups and {len(number_of_large_gaps[number_of_large_gaps > 0])} (or {round(len(number_of_large_gaps[number_of_large_gaps > 0]) / len(number_of_dist_groups[number_of_dist_groups > 1]), 2) * 100}%) have large gaps.')

                fig1 = plt.figure()
                hist_1, bin_edges_1, patches_1 = plt.hist(number_of_components, bins = len(np.unique(number_of_components)), edgecolor='black', color='red')
                plt.xlabel('Number of components')
                ticks_1 = [(bin_edges_1[i] + bin_edges_1[i+1])/2 for i in range(len(bin_edges_1) - 1)]
                plt.xticks(ticks_1, np.unique(number_of_components))
                plt.ylabel('Number of distributions')
                plt.title('For all 3 values of $Z_{\odot}$')
                plt.savefig('number_of_comp_hist_all_Z.pdf')
                plt.close()

                fig2 = plt.figure(figsize = (8, 7))
                hist_2, bin_edges_2, patches_2 = plt.hist(number_of_dist_groups, bins = len(np.unique(number_of_dist_groups)), edgecolor='black', color='blue')
                plt.xlabel('Number of distinct groups')
                ticks_2 = [(bin_edges_2[i] + bin_edges_2[i+1])/2 for i in range(len(bin_edges_2) - 1)]
                plt.xticks(ticks_2, np.unique(number_of_dist_groups))
                plt.ylabel('Number of distributions')
                plt.title('For all 3 values of $Z_{\odot}$')
                plt.tight_layout()
                plt.savefig('number_of_dist_groups_hist_all_Z.pdf')
                plt.close()

                fig3 = plt.figure(figsize = (8, 7))
                hist_3, bin_edges_3, patches_3 = plt.hist(number_of_large_gaps, bins=len(np.unique(number_of_large_gaps)), edgecolor='black', color='green')
                plt.xlabel('Number of large gaps')
                ticks_3 = [(bin_edges_3[i] + bin_edges_3[i+1])/2 for i in range(len(bin_edges_3) - 1)]
                plt.xticks(ticks_3, np.unique(number_of_large_gaps))
                plt.ylabel('Number of distributions')
                plt.title('For all 3 values of $Z_{\odot}$')
                plt.tight_layout()
                plt.savefig('number_of_large_gaps_hist_all_Z.pdf')
                plt.close()

                #? Analysis for Z = 0.5 Z_Sun
                #* Create the file where the statistics will be stored
                analysis_results_file_name_Z_0_5 = 'statistics_Z_0.5.txt'
                analysis_results_file_Z_0_5 = os.path.join(code_path, analysis_results_file_name_Z_0_5)
                if os.path.isfile(analysis_results_file_Z_0_5):
                    pass
                else:
                    with open(analysis_results_file_Z_0_5, 'x') as f:
                        pass

                mass_gap_data_Z_0_5 = mass_gap_data[mass_gap_data[:, 0] == 0.5]
                bh_distr_data_Z_0_5 = bh_distr_data[bh_distr_data[:, 2] == 0.5]

                number_of_no_bhs_Z_0_5 = 0
                number_of_no_nss_Z_0_5 = 0
                number_of_zero_mass_gap_Z_0_5 = 0

                mass_gap_file_length_Z_0_5 = len(mass_gap_data_Z_0_5)
                existent_mass_gap_Z_0_5 = 0
                non_existent_mass_gap_Z_0_5 = 0

                for i in range(len(mass_gap_data_Z_0_5)):
                    if mass_gap_data_Z_0_5[i][-4] > 0:
                        existent_mass_gap_Z_0_5 += 1
                    else:
                        non_existent_mass_gap_Z_0_5 += 1
                        if mass_gap_data_Z_0_5[i][-1] == 1:
                            number_of_no_bhs_Z_0_5 += 1
                        elif mass_gap_data_Z_0_5[i][-2] == 1:
                            number_of_no_nss_Z_0_5 += 1
                        elif mass_gap_data_Z_0_5[i][-3] == 1:
                            number_of_zero_mass_gap_Z_0_5 += 1

                number_of_components_Z_0_5 = np.zeros(len(bh_distr_data_Z_0_5), dtype=int)
                number_of_dist_groups_Z_0_5 = np.zeros(len(bh_distr_data_Z_0_5), dtype=int)
                number_of_large_gaps_Z_0_5 = np.zeros(len(bh_distr_data_Z_0_5), dtype=int)
                bh_data_file_length_Z_0_5 = len(bh_distr_data_Z_0_5)

                for i in range(len(bh_distr_data_Z_0_5)):
                    number_of_components_Z_0_5[i] = bh_distr_data_Z_0_5[i][1]
                    number_of_dist_groups_Z_0_5[i] = bh_distr_data_Z_0_5[i][-2] + 1
                    number_of_large_gaps_Z_0_5[i] = bh_distr_data_Z_0_5[i][-1]

                with open(analysis_results_file_Z_0_5, 'w') as f:
                    f.write(
                        f'Out of {mass_gap_file_length_Z_0_5} remnant mass distributions, {existent_mass_gap_Z_0_5} (or {round(existent_mass_gap_Z_0_5 / mass_gap_file_length_Z_0_5, 2) * 100}%) have a mass gap and {non_existent_mass_gap_Z_0_5} (or {round(non_existent_mass_gap_Z_0_5 / mass_gap_file_length_Z_0_5, 2) * 100}%) do not have a mass gap. Out of those that do not have a mass gap, {number_of_no_nss_Z_0_5} (or {round(number_of_no_nss_Z_0_5 / non_existent_mass_gap_Z_0_5, 2) * 100}%) do not predict any NSs, {number_of_no_bhs_Z_0_5} (or {round(number_of_no_bhs_Z_0_5 / non_existent_mass_gap_Z_0_5, 2) * 100}%) do not predict any BHs, and {number_of_zero_mass_gap_Z_0_5} (or {round(number_of_zero_mass_gap_Z_0_5 / non_existent_mass_gap_Z_0_5, 2) * 100}%) simply have a zero mass gap. \nOut of {bh_data_file_length_Z_0_5} bh mass distributions, {len(number_of_components_Z_0_5[number_of_components_Z_0_5 > 6])} (or {round(len(number_of_components_Z_0_5[number_of_components_Z_0_5 > 6]) / bh_data_file_length_Z_0_5, 2) * 100}%) have more that 6 components in their GMM fit, {len(number_of_dist_groups_Z_0_5[number_of_dist_groups_Z_0_5 > 1])} (or {round(len(number_of_dist_groups_Z_0_5[number_of_dist_groups_Z_0_5 > 1]) / bh_data_file_length_Z_0_5, 2) * 100}%) have distinct groups and {len(number_of_large_gaps_Z_0_5[number_of_large_gaps_Z_0_5 > 0])} (or {round(len(number_of_large_gaps_Z_0_5[number_of_large_gaps_Z_0_5 > 0]) / len(number_of_dist_groups_Z_0_5[number_of_dist_groups_Z_0_5 > 1]), 2) * 100}%) have large gaps.')

                fig1 = plt.figure()
                hist_1, bin_edges_1, patches_1 = plt.hist(number_of_components_Z_0_5, bins=len(
                    np.unique(number_of_components_Z_0_5)), edgecolor='black', color='red')
                plt.xlabel('Number of components')
                ticks_1 = [(bin_edges_1[i] + bin_edges_1[i+1]) /
                        2 for i in range(len(bin_edges_1) - 1)]
                plt.xticks(ticks_1, np.unique(number_of_components_Z_0_5))
                plt.ylabel('Number of distributions')
                plt.title('For $Z = 0.5 \ Z_{\odot}$')
                plt.savefig('number_of_comp_hist_Z_0.5.pdf')
                plt.close() 

                fig2 = plt.figure(figsize=(8, 7))
                hist_2, bin_edges_2, patches_2 = plt.hist(number_of_dist_groups_Z_0_5, bins=len(
                    np.unique(number_of_dist_groups_Z_0_5)), edgecolor='black', color='blue')
                plt.xlabel('Number of distinct groups')
                ticks_2 = [(bin_edges_2[i] + bin_edges_2[i+1]) /
                        2 for i in range(len(bin_edges_2) - 1)]
                plt.xticks(ticks_2, np.unique(number_of_dist_groups_Z_0_5))
                plt.ylabel('Number of distributions')
                plt.title('For $Z = 0.5 \ Z_{\odot}$')
                plt.tight_layout()
                plt.savefig('number_of_dist_groups_hist_Z_0.5.pdf')
                plt.close()

                fig3 = plt.figure(figsize=(8, 7))
                hist_3, bin_edges_3, patches_3 = plt.hist(number_of_large_gaps_Z_0_5, bins=len(
                    np.unique(number_of_large_gaps_Z_0_5)), edgecolor='black', color='green')
                plt.xlabel('Number of large gaps')
                ticks_3 = [(bin_edges_3[i] + bin_edges_3[i+1]) / 2 for i in range(len(bin_edges_3) - 1)]
                plt.xticks(ticks_3, np.unique(number_of_large_gaps_Z_0_5))
                plt.ylabel('Number of distributions')
                plt.title('For $Z = 0.5 \ Z_{\odot}$')
                plt.tight_layout()
                plt.savefig('number_of_large_gaps_hist_Z_0.5.pdf')
                plt.close()

                #? Analysis for Z = 0.75 Z_Sun
                analysis_results_file_name_Z_0_75 = 'statistics_Z_0.75.txt'
                analysis_results_file_Z_0_75 = os.path.join(code_path, analysis_results_file_name_Z_0_75)
                if os.path.isfile(analysis_results_file_Z_0_75):
                    pass
                else:
                    with open(analysis_results_file_Z_0_75, 'x') as f:
                        pass
                mass_gap_data_Z_0_75 = mass_gap_data[mass_gap_data[:, 0] == 0.75]
                bh_distr_data_Z_0_75 = bh_distr_data[bh_distr_data[:, 2] == 0.75]

                number_of_no_bhs_Z_0_75 = 0
                number_of_no_nss_Z_0_75 = 0
                number_of_zero_mass_gap_Z_0_75 = 0

                mass_gap_file_length_Z_0_75 = len(mass_gap_data_Z_0_75)
                existent_mass_gap_Z_0_75 = 0
                non_existent_mass_gap_Z_0_75 = 0

                for i in range(len(mass_gap_data_Z_0_75)):
                    if mass_gap_data_Z_0_75[i][-4] > 0:
                        existent_mass_gap_Z_0_75 += 1
                    else:
                        non_existent_mass_gap_Z_0_75 += 1
                        if mass_gap_data_Z_0_75[i][-1] == 1:
                            number_of_no_bhs_Z_0_75 += 1
                        elif mass_gap_data_Z_0_75[i][-2] == 1:
                            number_of_no_nss_Z_0_75 += 1
                        elif mass_gap_data_Z_0_75[i][-3] == 1:
                            number_of_zero_mass_gap_Z_0_75 += 1

                number_of_components_Z_0_75 = np.zeros(len(bh_distr_data_Z_0_75), dtype=int)
                number_of_dist_groups_Z_0_75 = np.zeros(len(bh_distr_data_Z_0_75), dtype=int)
                number_of_large_gaps_Z_0_75 = np.zeros(len(bh_distr_data_Z_0_75), dtype=int)
                bh_data_file_length_Z_0_75 = len(bh_distr_data_Z_0_75)

                for i in range(len(bh_distr_data_Z_0_75)):
                    number_of_components_Z_0_75[i] = bh_distr_data_Z_0_75[i][1]
                    number_of_dist_groups_Z_0_75[i] = bh_distr_data_Z_0_75[i][-2] + 1
                    number_of_large_gaps_Z_0_75[i] = bh_distr_data_Z_0_75[i][-1]

                with open(analysis_results_file_Z_0_75, 'w') as f:
                    f.write(
                        f'Out of {mass_gap_file_length_Z_0_75} remnant mass distributions, {existent_mass_gap_Z_0_75} (or {round(existent_mass_gap_Z_0_75 / mass_gap_file_length_Z_0_75, 2) * 100}%) have a mass gap and {non_existent_mass_gap_Z_0_75} (or {round(non_existent_mass_gap_Z_0_75 / mass_gap_file_length_Z_0_75, 2) * 100}%) do not have a mass gap. Out of those that do not have a mass gap, {number_of_no_nss_Z_0_75} (or {round(number_of_no_nss_Z_0_75 / non_existent_mass_gap_Z_0_75, 2) * 100}%) do not predict any NSs, {number_of_no_bhs_Z_0_75} (or {round(number_of_no_bhs_Z_0_75 / non_existent_mass_gap_Z_0_75, 2) * 100}%) do not predict any BHs, and {number_of_zero_mass_gap_Z_0_75} (or {round(number_of_zero_mass_gap_Z_0_75 / non_existent_mass_gap_Z_0_75, 2) * 100}%) simply have a zero mass gap. \nOut of {bh_data_file_length_Z_0_75} bh mass distributions, {len(number_of_components_Z_0_75[number_of_components_Z_0_75 > 6])} (or {round(len(number_of_components_Z_0_75[number_of_components_Z_0_75 > 6]) / bh_data_file_length_Z_0_75, 2) * 100}%) have more that 6 components in their GMM fit, {len(number_of_dist_groups_Z_0_75[number_of_dist_groups_Z_0_75 > 1])} (or {round(len(number_of_dist_groups_Z_0_75[number_of_dist_groups_Z_0_75 > 1]) / bh_data_file_length_Z_0_75, 2) * 100}%) have distinct groups and {len(number_of_large_gaps_Z_0_75[number_of_large_gaps_Z_0_75 > 0])} (or {round(len(number_of_large_gaps_Z_0_75[number_of_large_gaps_Z_0_75 > 0]) / len(number_of_dist_groups_Z_0_75[number_of_dist_groups_Z_0_75 > 1]), 2) * 100}%) have large gaps.')

                fig1 = plt.figure()
                hist_1, bin_edges_1, patches_1 = plt.hist(number_of_components_Z_0_75, bins=len(
                    np.unique(number_of_components_Z_0_75)), edgecolor='black', color='red')
                plt.xlabel('Number of components')
                ticks_1 = [(bin_edges_1[i] + bin_edges_1[i+1]) / 2 for i in range(len(bin_edges_1) - 1)]
                plt.xticks(ticks_1, np.unique(number_of_components_Z_0_75))
                plt.ylabel('Number of distributions')
                plt.title('For $Z = 0.75 \ Z_{\odot}$')
                plt.savefig('number_of_comp_hist_Z_0.75.pdf')
                plt.close() 

                fig2 = plt.figure(figsize=(8, 7))
                hist_2, bin_edges_2, patches_2 = plt.hist(number_of_dist_groups_Z_0_75, bins=len(
                    np.unique(number_of_dist_groups_Z_0_75)), edgecolor='black', color='blue')
                plt.xlabel('Number of distinct groups')
                ticks_2 = [(bin_edges_2[i] + bin_edges_2[i+1]) /
                        2 for i in range(len(bin_edges_2) - 1)]
                plt.xticks(ticks_2, np.unique(number_of_dist_groups_Z_0_75))
                plt.ylabel('Number of distributions')
                plt.title('For $Z = 0.75 \ Z_{\odot}$')
                plt.tight_layout()
                plt.savefig('number_of_dist_groups_hist_Z_0.75.pdf')
                plt.close()

                fig3 = plt.figure(figsize=(8, 7))
                hist_3, bin_edges_3, patches_3 = plt.hist(number_of_large_gaps_Z_0_75, bins=len(
                    np.unique(number_of_large_gaps_Z_0_75)), edgecolor='black', color='green')
                plt.xlabel('Number of large gaps')
                ticks_3 = [(bin_edges_3[i] + bin_edges_3[i+1]) /
                        2 for i in range(len(bin_edges_3) - 1)]
                plt.xticks(ticks_3, np.unique(number_of_large_gaps_Z_0_75))
                plt.ylabel('Number of distributions')
                plt.title('For $Z = 0.75 \ Z_{\odot}$')
                plt.tight_layout()
                plt.savefig('number_of_large_gaps_hist_Z_0.75.pdf')
                plt.close()

                #? Analysis for Z = Z_Sun
                analysis_results_file_name_Z_1 = 'statistics_Z_1.0.txt'
                analysis_results_file_Z_1 = os.path.join(code_path, analysis_results_file_name_Z_1)
                if os.path.isfile(analysis_results_file_Z_1):
                    pass
                else:
                    with open(analysis_results_file_Z_1, 'x') as f:
                        pass
                mass_gap_data_Z_1 = mass_gap_data[mass_gap_data[:, 0] == 1.0]
                bh_distr_data_Z_1 = bh_distr_data[bh_distr_data[:, 2] == 1.0]

                number_of_no_bhs_Z_1 = 0
                number_of_no_nss_Z_1 = 0
                number_of_zero_mass_gap_Z_1 = 0

                mass_gap_file_length_Z_1 = len(mass_gap_data_Z_1)
                existent_mass_gap_Z_1 = 0
                non_existent_mass_gap_Z_1 = 0

                for i in range(len(mass_gap_data_Z_1)):
                    if mass_gap_data_Z_1[i][-4] > 0:
                        existent_mass_gap_Z_1 += 1
                    else:
                        non_existent_mass_gap_Z_1 += 1
                        if mass_gap_data_Z_1[i][-1] == 1:
                            number_of_no_bhs_Z_1 += 1
                        elif mass_gap_data_Z_1[i][-2] == 1:
                            number_of_no_nss_Z_1 += 1
                        elif mass_gap_data_Z_1[i][-3] == 1:
                            number_of_zero_mass_gap_Z_1 += 1

                number_of_components_Z_1 = np.zeros(len(bh_distr_data_Z_1), dtype=int)
                number_of_dist_groups_Z_1 = np.zeros(len(bh_distr_data_Z_1), dtype=int)
                number_of_large_gaps_Z_1 = np.zeros(len(bh_distr_data_Z_1), dtype=int)
                bh_data_file_length_Z_1 = len(bh_distr_data_Z_1)

                for i in range(len(bh_distr_data_Z_1)):
                    number_of_components_Z_1[i] = bh_distr_data_Z_1[i][1]
                    number_of_dist_groups_Z_1[i] = bh_distr_data_Z_1[i][-2] + 1
                    number_of_large_gaps_Z_1[i] = bh_distr_data_Z_1[i][-1]

                with open(analysis_results_file_Z_1, 'w') as f:
                    f.write(
                        f'Out of {mass_gap_file_length_Z_1} remnant mass distributions, {existent_mass_gap_Z_1} (or {round(existent_mass_gap_Z_1 / mass_gap_file_length_Z_1, 2) * 100}%) have a mass gap and {non_existent_mass_gap_Z_1} (or {round(non_existent_mass_gap_Z_1 / mass_gap_file_length_Z_1, 2) * 100}%) do not have a mass gap. Out of those that do not have a mass gap, {number_of_no_nss_Z_1} (or {round(number_of_no_nss_Z_1 / non_existent_mass_gap_Z_1, 2) * 100}%) do not predict any NSs, {number_of_no_bhs_Z_1} (or {round(number_of_no_bhs_Z_1 / non_existent_mass_gap_Z_1, 2) * 100}%) do not predict any BHs, and {number_of_zero_mass_gap_Z_1} (or {round(number_of_zero_mass_gap_Z_1 / non_existent_mass_gap_Z_1, 2) * 100}%) simply have a zero mass gap. \nOut of {bh_data_file_length_Z_1} bh mass distributions, {len(number_of_components_Z_1[number_of_components_Z_1 > 6])} (or {round(len(number_of_components_Z_1[number_of_components_Z_1 > 6]) / bh_data_file_length_Z_1, 2) * 100}%) have more that 6 components in their GMM fit, {len(number_of_dist_groups_Z_1[number_of_dist_groups_Z_1 > 1])} (or {round(len(number_of_dist_groups_Z_1[number_of_dist_groups_Z_1 > 1]) / bh_data_file_length_Z_1, 2) * 100}%) have distinct groups and {len(number_of_large_gaps_Z_1[number_of_large_gaps_Z_1 > 0])} (or {round(len(number_of_large_gaps_Z_1[number_of_large_gaps_Z_1 > 0]) / len(number_of_dist_groups_Z_1[number_of_dist_groups_Z_1 > 1]), 2) * 100}%) have large gaps.')

                fig1 = plt.figure()
                hist_1, bin_edges_1, patches_1 = plt.hist(number_of_components_Z_1, bins=len(
                    np.unique(number_of_components_Z_1)), edgecolor='black', color='red')
                plt.xlabel('Number of components')
                ticks_1 = [(bin_edges_1[i] + bin_edges_1[i+1]) / 2 for i in range(len(bin_edges_1) - 1)]
                plt.xticks(ticks_1, np.unique(number_of_components_Z_1))
                plt.ylabel('Number of distributions')
                plt.title('For $Z = 1.0 \ Z_{\odot}$')
                plt.savefig('number_of_comp_hist_Z_1.pdf')
                plt.close() 

                fig2 = plt.figure(figsize=(8, 7))
                hist_2, bin_edges_2, patches_2 = plt.hist(number_of_dist_groups_Z_1, bins=len(
                    np.unique(number_of_dist_groups_Z_1)), edgecolor='black', color='blue')
                plt.xlabel('Number of distinct groups')
                ticks_2 = [(bin_edges_2[i] + bin_edges_2[i+1]) /
                        2 for i in range(len(bin_edges_2) - 1)]
                plt.xticks(ticks_2, np.unique(number_of_dist_groups_Z_1))
                plt.ylabel('Number of distributions')
                plt.title('For $Z = 1.0 \ Z_{\odot}$')
                plt.tight_layout()
                plt.savefig('number_of_dist_groups_hist_Z_1.pdf')
                plt.close()

                fig3 = plt.figure(figsize=(8, 7))
                hist_3, bin_edges_3, patches_3 = plt.hist(number_of_large_gaps_Z_1, bins=len(
                    np.unique(number_of_large_gaps_Z_1)), edgecolor='black', color='green')
                plt.xlabel('Number of large gaps')
                ticks_3 = [(bin_edges_3[i] + bin_edges_3[i+1]) /
                        2 for i in range(len(bin_edges_3) - 1)]
                plt.xticks(ticks_3, np.unique(number_of_large_gaps_Z_1))
                plt.ylabel('Number of distributions')
                plt.title('For $Z = 1.0 \ Z_{\odot}$')
                plt.tight_layout()
                plt.savefig('number_of_large_gaps_hist_Z_1.pdf')
                plt.close()

        if mfinal_mremnant_plot:
            old_model = True
            new_model = False
            #* Plots path
            plots_dir = 'Plots'
            plots_path = os.path.join(code_path, plots_dir)
            if not os.path.isdir(plots_path):
                    os.makedirs(plots_path)
            Z_m = 0.75
            # * Path for data files
            starting_dir = 'Parameter Study'
            starting_dir_path = os.path.join(code_path, starting_dir)

            if old_model:
                starting_dir_path = os.path.join(starting_dir_path, 'Old')
            elif new_model:
                starting_dir_path = os.path.join(starting_dir_path, 'Muller')
            
            metallicities = [0.5, 0.75, 1.0]
            for Z_m in metallicities:
                # * Get the corresponding directory
                data_dir = os.path.join(
                    starting_dir_path, f'{Z_m} metallicity')

                # * Focus only on the Data subdirectory
                data_dir_path = os.path.join(
                    data_dir, 'Data')
                dirs = os.listdir(data_dir_path)
                
                m_final = []
                m_remnant = []
                for dirum in dirs:
                    dir_path = os.path.join(data_dir_path, dirum)
                    #* Get the files
                    files = os.listdir(dir_path)
                    for file in files:
                        #* File path
                        file_path = os.path.join(dir_path, file)
                        #* Load the data
                        data_plot = np.loadtxt(file_path, dtype = float, delimiter = ',')

                        for i in range(len(data_plot)):
                            if i == 4:
                                for j in range(len(data_plot[i])):
                                    m_remnant.append(data_plot[i][j])
                            if i == 8:
                                for j in range(len(data_plot[i])):
                                    m_final.append(data_plot[i][j])

                fig = plt.figure()
                plt.plot(m_final, m_remnant, linestyle = '', marker = '.', color = 'blue')
                plt.xlabel(r'Final mass $M_{final}$ $\left[M_{\odot}\right]$')
                plt.ylabel(r'Remnant mass $M_{grav}$ $\left[M_{\odot}\right]$')
                plt.title(f'Final Mass - Remnant Mass for Z = {Z_m}')
                plt.savefig(os.path.join(plots_path, f'mfinal_mremnant_plot_{Z_m}.pdf'))
                plt.close()


if __name__ == '__main__':
    main()
