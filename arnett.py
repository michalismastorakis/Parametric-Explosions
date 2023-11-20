import astropy.units as u
import astropy.constants as c
import numpy as np


class Lightcurve:
    def __init__(self,t,energy,ejecta,beta=13.8,electron_opacity=0.1,g_trap=0.03,m_Ni=2.5e-03):
        self.t    = (t * u.day).to(u.s)
        self.e_ej = energy * u.erg 
        self.m_ej = ejecta * u.Msun
        self.m_Ni = m_Ni * u.Msun 
        self.b_c  = beta 
        self.g_trap = g_trap
        self.k_e  = electron_opacity *  u.cm**2 * u.g**(-1.)
        self.usc  = ((10.*self.e_ej/(3.*self.m_ej))**0.5).to(u.cm/u.s)
        self.tau_m = 10*(self.m_ej.value)**(3./4.) * (self.e_ej.value/1e51)**(-1./4.) *(self.k_e.value/0.1)**(1./2.) * u.day
        self.tau_m = self.tau_m.to(u.s)
        self.x     = self.t/self.tau_m
        self.y  = self.tau_m/(2*(8.7*u.day).to(u.s))
        self.xy = self.x*self.y
        self.l_bol = self.lc()
        self.mag_bol = -2.5*np.log10((self.l_bol/u.Lsun).to(u.dimensionless_unscaled)) +4.74
        self.teff = (((3./4.)*self.l_bol / (np.pi * (self.t*self.usc)**2. * c.sigma_sb))**0.25).to(u.K)


    def D(self,decay_time,t):
        dt = (decay_time * u.day).to(u.s)
        x     = t/self.tau_m
        y  = self.tau_m/(2*dt)
        xy = x*y
        tau_g = 55.3*(0.1/self.g_trap)*y**2. /((self.usc.to(u.m/u.s)/c.c)*(0.1 + 2*xy)**2.)
        G = tau_g.value/(tau_g.value + 1.6)
        D = G*(1. + 2.*G*(1.-G)*(1.-0.75*G))
        return D


    def luminosity(self,t):
        t = t * u.s
        t_Ni = (8.7*u.day).to(u.s)
        t_co = (111.*u.day).to(u.s)
        l_Ni = 4.78e10 * (self.m_Ni.to(u.g).value) * np.exp(-t.value/(t_Ni.value))
        d_Ni = self.D(8.7,t)
        l_co = 2.561e8 *  (self.m_Ni.to(u.g).value) * (np.exp(-t.value/(t_co.value)) - np.exp(-t.value/(t_Ni.value)))/(1.-t_Ni/t_co)
        d_co = self.D(111.,t)
        return (l_Ni*d_Ni + l_co*d_co) * (u.erg/u.s)
            


  

    def lc(self):
        x  = self.t/self.tau_m
        out = np.zeros_like(x)
        fac = np.exp(-x.value**2.)
        for i in range(x.value.shape[0]):
            l = self.luminosity(self.t[:i].value).value*2.*x[:i]*np.exp(x[:i].value**2)/self.tau_m.value
            out[i] = np.trapz(l,self.t[:i].value)

        return fac*out * u.erg/ u.s
