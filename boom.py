import os, fnmatch 
#import mesaPlot as mp
import numpy as np
import astropy.units as u
import astropy.constants as c 
import mesa_reader as mr
from scipy.interpolate import interp1d
#import corner
from pathlib import Path
import scipy.integrate as integrate
import scipy.special as special


#-------------------------------------------------------------------------------

def rhoav(r, m):
    """
    average density in g/cm**3
    """
    rhoav = (0.75 / np.pi) * m * np.maximum(r,1e-99*u.cm)**(-3)
    return rhoav.to(u.g * u.cm**(-3.))
#-------------------------------------------------------------------------------

def t_infall(r, m):
    """
    infall time in sec
    """
    rho_av = rhoav(r, m)
    time = 0.5 * np.sqrt(np.pi / c.G)* rho_av**(-0.5)
    return time.to(u.s)
#-------------------------------------------------------------------------------

def mdot(r, m, rho):
    """
    accretion rate in g/s
    """
    time = t_infall(r, m)
    rho_av = rhoav(r, m)
    acc = 2 * m / time * (rho/(rho_av-rho))
    return acc.to(u.Msun/u.s)
#-------------------------------------------------------------------------------

def r_gain(macc, m):
    """
    gain radius in cm
    """
    r0=12*u.km
    r1=120*u.km
    
    rg = (r1**3 * macc.to(u.Msun/u.s).value * m.to(u.Msun).value**(-3.) + r0**3.)**(1./3.)
        
    #rg = ((macc.value / m.value**3 * (1.2e7**3) + (1.2e6)**3)**(1/3))*u.cm
    return rg.to(u.km)
#-------------------------------------------------------------------------------

def luminosity(mdot, m, r, r_gain, zeta = 0.7, t_15 = 1.2*u.s,redshift=False):
    """
    luminosity in erg/s
    """

    t_cool = t_15 * (m.to(u.Msun).value / 1.5) ** (5.0/3.0)
    t_cool = np.maximum(t_cool.value, 0.1)*u.s        
    Ebind = (m.to(u.Msun) - (-1. + (1+0.336*m.to(u.Msun).value)**0.5)*u.Msun/0.168)*c.c**2
        
    time = t_infall(r, m)
        
    rpns = 5.*r_gain/7.
        
    lnu = 0.3*Ebind/t_cool
    lacc = zeta*c.G*m*mdot/r_gain
        
    ldiff = lnu*np.exp(-np.minimum((time.to(u.s).value/t_cool.value),100))
    
    #
        
    lum = (lnu+ldiff)
    if redshift:
        alpha=np.maximum(0,(1-2*c.G*mass/rpns/c.c**2))
        lum = alpha**0.5 *lum
        
    return lum.to(u.erg/u.s)
#-------------------------------------------------------------------------------   

def r_shock(mass,radius,rg,md,lum,aturb=1.17):
    rpns = 5*rg/7.
    alpha=np.maximum(0,(1-2.*c.G*mass/rpns/c.c**2))
    lume2 = lum.to(u.erg/u.s).value * 1e-52 * (mass.value)**2 * alpha ** 1.5
    
    rsh=aturb*0.55e5 * lume2**(4 / 9) * (rg.to(u.cm).value * 1.e-6)**(16 / 9) / (md.value**2 * mass.value)**(1/3)
    
    #rsh=55*u.km * aturb* ((lum.value/1e52)*alpha**3.)**(4/9) * mass.value**(5./9.) * (rg.to(u.km).value /10.)**(16./9.)  * md.value**(-2./3.)
    return rsh*u.cm
#-------------------------------------------------------------------------------

def t_adv(rs,mass,rg):
    tadv = 18*u.ms * (np.maximum(rs.to(u.km).value, 0.)/100.)**1.5 * np.log(np.maximum(rs.to(u.km).value/rg.to(u.km).value, 1)) * (mass.value)**(-0.5)
    #tadv = 18e-3 * (np.maximum(rs.to(u.cm), 0.) * 1.e-7)**1.5 * np.log(np.maximum((rs/rg).to(u.dimensionless_unscaled), 1)) * (mass.value)**(-0.5)
    #tadv = 18e-3 * (np.maximum(rs.to(u.cm), 0.) * 1.e-7)**1.5 * np.log(np.maximum(rs/rg, 1)) * (mass.value)**(-0.5)
    return tadv
#-------------------------------------------------------------------------------

def egain(mass,rs,rg,ediss=8.8*u.MeV/u.u):
    r = np.maximum(rs.to(u.km).value,rg.to(u.km).value)*u.km
 #   eg=3./4. * ediss + c.G * mass/4/r
    
    
    eg = (0.25 * c.G) * mass / np.maximum(rs, rg) + ediss * 0.75

    
    return eg.to(u.erg/u.g)
#-------------------------------------------------------------------------------

def t_heat(eg,rg,lum,mass):
    rpns = 5*rg/7.
    alpha=np.maximum(0,(1-2.*c.G*mass/(5*rg/7.)/c.c**2))
    #lume2=lum.to(u.erg/u.s).value/1e52 * (mass.to(u.Msun).value)**2 * np.maximum(0,(1-2.*c.G*mass/rpns/c.c**2))  
    
    lume2 = lum.to(u.erg/u.s).value * 1e-52 * (mass.value)**2 * alpha ** 1.5
    
    
    theat = 150.*u.ms * (eg.to(u.erg/u.g).value / 1e19) * (rg.to(u.km).value / 100)**2 / np.maximum(lume2, 1e-80)

    #theat = 150*u.ms * (eg.to(u.erg/u.g).value / 1e19) * (rg.to(u.km).value / 100)**2 * np.maximum((lum.to(u.erg/u.s).value*1e-52 * alpha**3 * (mass.value)**2),1e-80)**(-1.)
    return theat 
#-------------------------------------------------------------------------------

def m_gain(ma,tadv):
    return (ma*tadv).to(u.Msun)
#-------------------------------------------------------------------------------

def u_pre(mass,rs):
    return (2*c.G*mass/rs).to(u.km/u.s)
#-------------------------------------------------------------------------------

def rho_pre(mass,macc,radius,rs):
    return macc/(4*np.pi*radius**2 *u_pre(mass,rs))
#-------------------------------------------------------------------------------

def add_wind(m_by,lum0,rg,e_expl,m_grav,i_final,radius=1.2e6*u.cm,t_15=1.2*u.s):
    """
    Modifies the explosion energy and remenant mass by adding in the neutrino driven wind phase.

    """
    C=1                                                         # Proportionality constant for neutrino heating i   n the wind phase
    e_rec_alpha = 5 * u.MeV / u.u                               # Binding (recombination energy) ???
    epsilon_v = 9.5 * m_by.to(u.Msun).value *u.MeV                    # Mean neutrino energy

    t_cool = t_15 * (m_by / (1.5 * u.Msun)) ** (5.0/3.0)
    L_0 = lum0[i_final]                                         # Neutrino luminosity at the neutron star surface
  #  radius = rg[i_final]

    e_neutrino = L_0 * t_cool                                   # Total energy released by neutrino luminosity (assuming exp L decay)
    m_dot_0 = 1.14e-10 * C**(5/3) * (L_0.value/1e51)**(5/3) * epsilon_v.value**(10/3) * (radius.value/1e6) **(5/3) * (1.4*u.Msun/m_by)**2 * u.Msun/u.s

    m_wind = m_dot_0 * t_cool                                   # Mass lost in wind (baryonic)
    e_wind = m_wind * e_rec_alpha                               # Energy contribution in wind

    e_expl =  u.Quantity(e_expl,u.erg) + e_wind                               # Modifed explosion energy
    m_by   =  m_by-m_wind                                   # Modified neutron star remenant mass (baryonic)
    m_grav = (-1.0 + np.sqrt (1.0 + 4.0 * 0.084* m_by / u.Msun)) / (2 * 0.084)  * u.Msun

    if e_expl <=0:  # <-- No modification necessary if there is no explosion (i.e. black hole)
        e_wind     =  0. *u.erg
        m_wind     =  0. *u.Msun
        
    return e_expl,m_by,m_grav,e_wind,m_wind
#-------------------------------------------------------------------------------

def nuc_energy(A1,A2,mu1,mu2,X,mtot):
     # A1 -> A2 + energy
    energy_per_reaction = ((A2*mu1 - A1*mu2)*u.mu*c.c**2).to(u.erg)
    reactions_per_gram = (1*u.g / (A2*mu1*u.mu)).to(u.dimensionless_unscaled)/u.g
    mtot = u.Quantity(mtot,u.Msun)
    return (energy_per_reaction*reactions_per_gram *(X*mtot)).to(u.erg)
#-------------------------------------------------------------------------------

def eburn(p,T9,i,rho,beta=4):
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
        if (np.log10(beta*rho) < 11.62 + 1.5 * np.log10(T9)-39.17/T9) and (T9 > 5.):
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
            pass
     
    return E,mni56 *u.Msun
#-------------------------------------------------------------------------------

def criterion(profile,
              threshold = 1.0,
              beta=4.,
              alpha_outflow=0.5,
              eta_outflow = 1.0,
              zeta=0.7,
              aturb=1.18,
              t_15=1.2*u.s,
              m_max = 2.2 * u.Msun,
              wind=True,
              _use_egrav=True,
             _energy_offset = 0*u.erg): 
    
    rho=(10**profile('logRho')[::-1])*u.g/(u.cm**3)
    mass=profile('mass')[::-1]*u.Msun
    radius=profile('radius')[::-1]*u.Rsun
    entropy=profile('entropy')[::-1]
    
    co_mass_cut = np.where(profile('he4')[::-1] <=0.01)
    try:
        co_mass = mass[co_mass_cut][-1] * u.Msun
    except:
        co_mass = 0.0*u.Msun

    
    macc = mdot(radius,mass,rho)
    time = t_infall(radius,mass)
    rg   = r_gain(macc, mass)
    rpns = 5*rg/7.
    alpha= np.maximum(0,(1-2.*c.G*mass/(5*rpns/7.)/c.c**2))
    lum  =luminosity(macc, mass, radius, rg, zeta = zeta, t_15 = t_15)  
    rs   =r_shock(mass,radius,rg,macc,lum,aturb=aturb)
    
    tadv=t_adv(rs,mass,rg)
    eg=egain(mass,rs,rg,ediss=8.8*u.MeV/u.u)
    theat=t_heat(eg,rg,lum*alpha**0.5,mass)
    mgain=m_gain(macc,tadv)
    qheat = (eg * mgain) / theat
    eta_acc = qheat / macc
    tadv_theat = tadv / theat
    tadv_thead_max = tadv_theat.max()
    e_rec = 5 * u.MeV / u.u
    m_max_by  = m_max + 0.084 * m_max **2 / u.Msun
    

    
    
    try:
        jj0 = np.where(np.logical_and (tadv_theat > threshold, mass < m_max_by))[0][0]
        jj = np.arange(jj0, len(mass))
    except IndexError:
        jj = np.array([len(mass)-1], dtype = np.int)

    m_init = 0.0*u.Msun

    if len(jj) > 1:
        i_expl = np.min(jj)
        m_init = mass[i_expl - 1]
    else:
        i_final = len(mass)
        m_grav  = mass[-1]
              
    v_esc = np.sqrt(2 * c.G * mass/radius)

    e_diag = np.zeros_like(mass.value)*u.erg
    v_shock = np.zeros_like(mass.value)*u.km/u.s
    T9 = np.zeros_like(mass.value)*u.K
    e_burn = 0.0 
    con_matzner = 0.794
    m_by_rem = m_init
    e_delayed = 0.
    m_neutrino = 0.0
    i_kick = 0
    if _use_egrav:
        ybind = -(3*c.G*mass**2/radius/5.).to(u.erg) + _energy_offset
        #ybind=profile('total_energy')[::-1]*u.erg
    else:
        ybind=profile('total_energy_integral')[::-1]*u.erg + _energy_offset
        
    m_out_dot = eta_outflow*eta_acc*macc/eg
    e_diag_dot = e_rec*m_out_dot
    
              
    mNi56 = 0.0*u.Msun
    v_kick=0 *u.km/u.s   
    e_expl = 0.0 *u.erg
    for i in jj[:-1]: 
        #eta_outflow = 1.0 - alpha_outflow
        dot_m_sh = (4.0 * np.pi * v_shock[i-1] * radius[i-1]**2 * rho[i-1]).to(u.Msun/u.s)
        scr1 = 1.0
        if (eta_outflow * dot_m_sh > macc[i]):
            scr1 = (macc[i] / dot_m_sh).to(u.dimensionless_unscaled)


        eta_1 = max( [(eta_acc[i] / np.abs(eg[i])), 0.0]).to(u.dimensionless_unscaled) # * alpha_outflow
        eta_2 =  eta_1 * scr1

        m_out = (mass[i] - mass[i-1]) * eta_2
        e_delayed = e_delayed + (mass[i] - mass[i-1]) * (eta_outflow * eta_1 - eta_2) * e_rec
        e_diag[i] = e_diag[i-1] + m_out * e_rec + (ybind [i-1] - ybind[i])  * alpha_outflow
        v_shock[i] = (con_matzner * np.sqrt(max(e_diag[i],0.0*u.erg) / (mass[i] - m_init).cgs) * ((mass[i] - m_init).cgs / (rho[i].cgs * radius[i].cgs**3))**0.19)

        m_by_rem = m_by_rem + (mass[i] - mass[i-1]) * (1 - alpha_outflow)*(1.0-eta_1) #- m_out
        T9[i] = ((3/(c.sigma_sb*4./c.c))*((beta)/(beta-1))*rho[i]*v_shock[i]**2)**0.25*1e-9

            
        #T9=((((3*beta-1.)/(beta*c.sigma_sb*4./c.c)*rho[i]*v_shock[i]**2)**(1./4.)).to(u.K)/1e9)
        eburn_i,dmni56 = eburn(profile,T9[i].value,i,rho[i].value,beta=beta)
        mNi56 += dmni56*alpha_outflow
        e_diag[i] += eburn_i*alpha_outflow
        m_grav = (-1 + np.sqrt (1 + beta * 0.084 * m_by_rem.value)) / (2 * 0.084) *u.Msun
    
        if (((beta-1)/beta) * v_shock[i] > v_esc[i]):
                i_final = i
                e_diag [i_final] = e_diag [i_final] #+ e_delayed
                e_expl = e_diag[i_final]
                i_kick = i_final
                break
        if (m_by_rem > m_max_by):
                m_by_rem = np.max(mass)
                m_grav = np.max(mass)
                e_expl = 0. *u.erg
                e_delayed = 0.
                i_final = jj[-1]
                mNi56 = 0. *u.Msun
                break
        #m_grav = (-1 + np.sqrt (1 + beta * 0.084 * m_by_rem.value)) / (2 * 0.084) *u.Msun
        m_wind = 0.0*u.Msun
        e_wind = 0.0*u.erg
        if m_by_rem > m_max_by:
            m_by_rem = np.max(mass)
            m_grav = m_by_rem
            e_expl = 0.*u.erg
            mNi56 = 0.*u.Msun
            i_final = jj[-1]
            
    for i in range(i_final+1, jj[-1]):
        e_diag[i] = u.Quantity(e_diag[i-1],u.erg)
        eexp = e_diag[i] + (ybind[i-1] - ybind[i])
        v_shock[i] = (con_matzner * np.sqrt(max([eexp, 0*u.erg]) / (mass[i] - m_init).cgs) * ((mass[i] - m_init).cgs / (rho[i].cgs * radius[i].cgs**3))**0.19)
        T9[i] = ((3/(c.sigma_sb*4./c.c))*((beta)/(beta-1))*rho[i]*v_shock[i]**2)**0.25*1e-9

        eburn_i,dmni56 = eburn(profile,T9[i].value,i,rho[i].value,beta=beta)
        mNi56 += dmni56
        #*alpha_outflow
        e_diag[i] = eexp + eburn_i
        #*alpha_outflow
        e_expl = e_diag[i]
        e_burn += eburn_i
         

    if (m_init >= m_max_by):
        m_init = 0.0
        m_grav = mass[-1]
        m_by_rem=mass[-1]
    if m_by_rem < m_max_by:
        e_expl = e_expl + e_delayed
        if m_by_rem.value != 0:
            v_kick = 211 * (0.05/0.1)*(e_expl.value/1e51)*(m_by_rem.value/1.5)**(-1.) * u.km/u.s
            if wind:
                e_expl,m_by_rem,m_grav,e_wind,m_wind=add_wind(m_by_rem,lum,rg,e_expl,m_grav,i_final) 
        else:
            v_kick= 0 *u.km/u.s
            if wind:
                m_wind = 0 * u.Msun
    if e_expl.value < 1e45:
        m_by_rem = np.max(mass)
        m_grav = mass[-1]
        e_expl = 0.0*u.erg
        mNi56 = 0.0*u.Msun
        v_kick = 0 *u.km/u.s
        i_final = jj[-1]

    if mass[-1].value > 2.5:
            xi25 = interp1d(mass.value,mass.value/(radius.to(u.km).value/1000))(2.5).tolist()
    else:
            xi25 = 0.0
    if mass[-1].value > 1.7:
        xi17 = interp1d(mass.value,mass.value/(radius.to(u.km).value/1000))(1.7).tolist()
    else:
            xi17 = 0.0
                
    mc4  = interp1d(entropy,mass.value)(4.0).tolist()
    mu4  =0.3/((interp1d(mass.value,radius.to(u.km).value)(mc4+0.3)-radius[entropy>4][0].to(u.km).value)/1000.)
    mu4_mod = 0.33*mu4*mc4+0.09
    if wind: 
        m_ej=mass[-1]-m_by_rem+m_wind
    else:
        m_ej=mass[-1]-m_by_rem
        m_wind = 0*u.Msun

    
    result = {
        'm_grav': m_grav,
        'M_bar': m_by_rem,
        'M_ni56':mNi56,
        'm_co':co_mass,
        'E_expl':e_expl,
        'M_ej':m_ej,
        'M_wind':m_wind,
        'xi17':xi17,
        'xi25':xi25,
        'mu4':mu4,
        'mu4_mod':mu4_mod,
        'vkick':v_kick,
        'M_star':mass[-1],
        'R_star':radius[-1],
        'Rho_c':rho[0],
        'icut':i_final,
        'jj':jj[0],
        'tadv_thead_max':tadv_thead_max,
        't9':T9,
        'vshock':v_shock,
        'rg':rg,
        'rs':rs,
        'lum':lum,
        'time':time,
        'macc':macc
    }
    
    return result
#-------------------------------------------------------------------------------

def probabilistic(Mco = 1.5*u.Msun,
                  Mstar = 9* u.Msun,
                  M1 = 2.0 * u.Msun,
                 M2 = 3.0* u.Msun,
                 M3 = 7.0 * u.Msun,
                 M4 = 8.0 * u.Msun,
                 mu1 = 1.2 *u.Msun,
                 sigma1 = 0.02*u.Msun,
                 mu2a = 1.4*u.Msun,
                 mu2b = 0.5,
                 sigma2 = 0.05*u.Msun,
                 mu3a = 1.4 *u.Msun,
                 mu3b = 0.4,
                 sigma3 = 0.05*u.Msun,
                 muBH = 0.8,
                 sigmaBH = 0.5*u.Msun,
                 Mnsmin = 1.13*u.Msun,
                 Mnsmax = 2.0*u.Msun,
                 vns = 400 * u.km / u.s,
                 vbh = 200 * u.km / u.s,
                 sigmakick = 0.3):
    remnant_mass = 0.0*u.Msun
    mukick = 0.0 * u.km / u.s
    if Mco < M1:
        remnant_type = 0.0 #0.0 == NS, 1.0 == BH 
        mu = mu1.value
        sigma = sigma1.value
    elif (Mco >= M1) & (Mco < M2):
        pbh = (Mco - M1).value/(M3-M1).value
        remnant_type = np.random.binomial(1,pbh)
        mu = mu2a.value + mu2b*(Mco - M1).value/(M2-M1).value
        sigma = sigma2.value
    elif (Mco >= M2) & (Mco < M3):
        pbh = (Mco - M1).value/(M3-M1).value
        remnant_type = np.random.binomial(1,pbh)
        mu = mu3a.value + mu3b*(Mco - M2).value/(M3-M2).value
        sigma = sigma3.value       
   # elif Mco >= M3:
   #     remnant_type = 1.0
    else:
        remnant_type = 1.0
    if remnant_type == 0.0:
        while (remnant_mass >= Mnsmax) or (remnant_mass <= Mnsmin):
            remnant_mass = np.random.normal(mu,sigma)*u.Msun
        mukick = vns*(Mco-remnant_mass).value/remnant_mass.value
            
    if remnant_type == 1.0:
        if Mco >= M4:
            fallback = 1.0
        else:
            fallback = np.random.binomial(1,(Mco-M1).value/(M4-M1).value)
        if fallback:
            remnant_mass = Mstar 
            mukick = 0.0 * u.km/u.s 
        else:
            while (remnant_mass <= Mnsmax) or (remnant_mass >= Mstar):
                remnant_mass = np.random.normal(muBH*Mco.value,sigmaBH.value)*u.Msun
            mukick = vbh*(Mco-remnant_mass).value/remnant_mass.value
                
    result = {'m_CO': Mco,'Remnant_Type': remnant_type,'remnant_mass': remnant_mass,'mukick': mukick}

                
    return result
#-------------------------------------------------------------------------------            
    
