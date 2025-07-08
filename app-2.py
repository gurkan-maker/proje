from valve_database import load_valves_from_excel, add_valve_to_database, delete_valve_from_database, delete_valve_from_database
from valve import Valve
from scipy.interpolate import CubicSpline
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import math
import base64
import tempfile
import os
from datetime import datetime
from fpdf import FPDF
from io import BytesIO
import requests
from PIL import Image
import traceback
import kaleido
import CoolProp.CoolProp as CP

# ========================
# CONSTANTS & UNIT CONVERSION
# ========================
BAR_TO_KPA = 100
KPA_TO_BAR = 0.01
C_TO_K = 273.15
F_TO_R = 459.67
PSI_TO_BAR = 0.0689476
G_CONST = 9.80665
MMHG_TO_BAR = 0.00133322

CONSTANTS = {
    "N1": {"gpm, psia": 1.00, "m³/h, bar": 0.865, "m³/h, kPa": 0.0865},
    "N2": {"mm": 0.00214, "inch": 890},
    "N4": {"mm": 76000, "inch": 17300},
    "N5": {"mm": 0.00241, "inch": 1000},
    "N6": {"kg/h, kPa, kg/m³": 2.73, "kg/h, bar, kg/m³": 27.3, "lb/h, psia, lb/ft³": 63.3},
    "N7": {"m³/h, kPa, K (standard)": 4.17, "m³/h, bar, K (standard)": 417, "scfh, psia, R": 1360},
    "N8": {"kg/h, kPa, K": 0.948, "kg/h, bar, K": 94.8, "lb/h, psia, R": 19.3},
    "N9": {"m³/h, kPa, K (standard)": 22.4, "m³/h, bar, K (standard)": 2240, "scfh, psia, R": 7320}
}

WATER_DENSITY_4C = 999.97
AIR_DENSITY_0C = 1.293

# ========================
# FLUID LIBRARY
# ========================
FLUID_LIBRARY = {
    "Water": {
        "type": "liquid",
        "coolprop_name": "Water",
        "visc_func": lambda t, p: calculate_kinematic_viscosity("Water", t, p),
        "k_func": None,
        "pv_func": lambda t, p: calculate_vapor_pressure("Water", t, p),
        "pc_func": lambda: CP.PropsSI('Pcrit', 'Water') / 1e5,
        "rho_func": lambda t, p: calculate_density("Water", t, p)
    },
    "Methane": {
        "type": "gas",
        "coolprop_name": "Methane",
        "sg": 0.554,
        "visc_func": None,
        "k_func": lambda t, p: calculate_specific_heat_ratio("Methane", t, p),
        "z_func": lambda t, p: calculate_compressibility_factor("Methane", t, p),
        "pv_func": None,
        "rho_func": lambda t, p: calculate_density("Methane", t, p)
    },
    "Steam": {
        "type": "steam",
        "coolprop_name": "Water",
        "sg": None,
        "visc_func": None,
        "k_func": lambda t, p: calculate_specific_heat_ratio("Water", t, p),
        "z_func": None,  # Steam doesn't use Z in the same way
        "pv_func": None,
        "rho_func": lambda t, p: calculate_density("Water", t, p)
    },
    "CarbonDioxide": {
        "type": "gas",
        "coolprop_name": "CarbonDioxide",
        "sg": 1.52,
        "visc_func": None,
        "k_func": lambda t, p: calculate_specific_heat_ratio("CarbonDioxide", t, p),
        "z_func": lambda t, p: calculate_compressibility_factor("CarbonDioxide", t, p),
        "pv_func": None,
        "rho_func": lambda t, p: calculate_density("CarbonDioxide", t, p)
    },
    "Ammonia": {
        "type": "gas",
        "coolprop_name": "Ammonia",
        "sg": 0.59,
        "visc_func": None,
        "k_func": lambda t, p: calculate_specific_heat_ratio("Ammonia", t, p),
        "z_func": lambda t, p: calculate_compressibility_factor("Ammonia", t, p),
        "pv_func": None,
        "rho_func": lambda t, p: calculate_density("Ammonia", t, p)
    },
    "Hydrogen": {
        "type": "gas",
        "coolprop_name": "Hydrogen",
        "sg": 0.0696,
        "visc_func": None,
        "k_func": lambda t, p: calculate_specific_heat_ratio("Hydrogen", t, p),
        "z_func": lambda t, p: calculate_compressibility_factor("Hydrogen", t, p),
        "pv_func": None,
        "rho_func": lambda t, p: calculate_density("Hydrogen", t, p)
    },
    "20% Hydrogen + 80% Methane": {
        "type": "gas",
        "coolprop_name": None,
        "sg": 0.455,  # Calculated as (0.2*0.0696 + 0.8*0.554)
        "visc_func": None,
        "k_func": lambda t, p: calculate_specific_heat_ratio_mixture(t, p, 0.2, 0.8, "Hydrogen", "Methane"),
        "z_func": lambda t, p: calculate_compressibility_factor_mixture(t, p, 0.2, 0.8, "Hydrogen", "Methane"),
        "pv_func": None,
        "rho_func": lambda t, p: calculate_density_mixture(t, p, 0.2, 0.8, "Hydrogen", "Methane")
    },
    "Octane": {
        "type": "liquid",
        "coolprop_name": "Octane",
        "sg": 0.74,
        "visc_func": lambda t, p: calculate_kinematic_viscosity("Octane", t, p),
        "k_func": None,
        "pv_func": lambda t, p: calculate_vapor_pressure("Octane", t, p),
        "pc_func": lambda: CP.PropsSI('Pcrit', 'Octane') / 1e5,
        "rho_func": lambda t, p: calculate_density("Octane", t, p)
    }
}

# ========================
# FLUID PROPERTY FUNCTIONS
# ========================
def calculate_compressibility_factor(fluid: str, temp_c: float, press_bar: float) -> float:
    """Calculate compressibility factor Z using CoolProp"""
    try:
        if fluid in FLUID_LIBRARY and FLUID_LIBRARY[fluid].get("coolprop_name"):
            fluid_name = FLUID_LIBRARY[fluid]["coolprop_name"]
            T = temp_c + C_TO_K
            P = press_bar * 1e5
            Z = CP.PropsSI('Z', 'T', T, 'P', P, fluid_name)
            return Z
    except:
        pass
    return 1.0

def calculate_compressibility_factor_mixture(temp_c: float, press_bar: float, frac1: float, frac2: float, fluid1: str, fluid2: str) -> float:
    """Calculate compressibility factor for gas mixtures"""
    try:
        Z1 = calculate_compressibility_factor(fluid1, temp_c, press_bar)
        Z2 = calculate_compressibility_factor(fluid2, temp_c, press_bar)
        return frac1 * Z1 + frac2 * Z2
    except:
        pass
    return 1.0

def calculate_vapor_pressure(fluid: str, temp_c: float, press_bar: float) -> float:
    try:
        if fluid in FLUID_LIBRARY and FLUID_LIBRARY[fluid].get("coolprop_name"):
            fluid_name = FLUID_LIBRARY[fluid]["coolprop_name"]
            T = temp_c + C_TO_K
            pv = CP.PropsSI('P', 'T', T, 'Q', 0, fluid_name) / 1e5
            return pv
    except:
        pass
    return 0.0

def calculate_density(fluid: str, temp_c: float, press_bar: float) -> float:
    try:
        if fluid in FLUID_LIBRARY and FLUID_LIBRARY[fluid].get("coolprop_name"):
            fluid_name = FLUID_LIBRARY[fluid]["coolprop_name"]
            T = temp_c + C_TO_K
            P = press_bar * 1e5
            rho = CP.PropsSI('D', 'T', T, 'P', P, fluid_name)
            return rho
    except:
        pass
    if fluid == "water":
        return WATER_DENSITY_4C
    elif fluid == "air":
        return (press_bar * 1e5) * 28.97 / (8.314462 * (temp_c + C_TO_K))
    return 1000

def calculate_density_mixture(temp_c: float, press_bar: float, frac1: float, frac2: float, fluid1: str, fluid2: str) -> float:
    try:
        T = temp_c + C_TO_K
        P = press_bar * 1e5
        rho1 = calculate_density(fluid1, temp_c, press_bar)
        rho2 = calculate_density(fluid2, temp_c, press_bar)
        return frac1 * rho1 + frac2 * rho2
    except:
        pass
    return 1.0

def calculate_kinematic_viscosity(fluid: str, temp_c: float, press_bar: float) -> float:
    try:
        if fluid in FLUID_LIBRARY and FLUID_LIBRARY[fluid].get("coolprop_name"):
            fluid_name = FLUID_LIBRARY[fluid]["coolprop_name"]
            T = temp_c + C_TO_K
            P = press_bar * 1e5
            mu = CP.PropsSI('VISCOSITY', 'T', T, 'P', P, fluid_name)
            rho = CP.PropsSI('D', 'T', T, 'P', P, fluid_name)
            nu = mu / rho * 1e6
            return nu
    except:
        pass
    if fluid == "water":
        return 1.79 / (1 + 0.0337 * temp_c + 0.00022 * temp_c**2)
    elif fluid == "propane":
        return 0.2 * math.exp(-0.02 * (temp_c - 20))
    elif fluid == "Octane":
        return 0.6 * math.exp(-0.04 * (temp_c - 20))
    return 1.0

def calculate_specific_heat_ratio(fluid: str, temp_c: float, press_bar: float) -> float:
    try:
        if fluid in FLUID_LIBRARY and FLUID_LIBRARY[fluid].get("coolprop_name"):
            fluid_name = FLUID_LIBRARY[fluid]["coolprop_name"]
            T = temp_c + C_TO_K
            P = press_bar * 1e5
            Cp = CP.PropsSI('CPMASS', 'T', T, 'P', P, fluid_name)
            Cv = CP.PropsSI('CVMASS', 'T', T, 'P', P, fluid_name)
            return Cp / Cv
    except:
        pass
    if fluid == "air":
        return 1.4 - 0.0001 * temp_c
    elif fluid == "methane":
        return 1.31 - 0.00008 * temp_c
    elif fluid == "steam":
        return 1.33 - 0.0001 * temp_c
    elif fluid == "CarbonDioxide":
        return 1.28 - 0.00005 * temp_c
    elif fluid == "ammonia":
        return 1.32 - 0.00007 * temp_c
    elif fluid == "hydrogen":
        return 1.41 - 0.0001 * temp_c
    return 1.4

def calculate_specific_heat_ratio_mixture(temp_c: float, press_bar: float, frac1: float, frac2: float, fluid1: str, fluid2: str) -> float:
    try:
        k1 = calculate_specific_heat_ratio(fluid1, temp_c, press_bar)
        k2 = calculate_specific_heat_ratio(fluid2, temp_c, press_bar)
        return frac1 * k1 + frac2 * k2
    except:
        pass
    return 1.4

def calculate_ff(pv: float, pc: float) -> float:
    if pc <= 0:
        return 0.96
    return 0.96 - 0.28 * math.sqrt(pv / pc)
# ========================
# VALVE DATABASE
# ========================
VALVE_DATABASE = load_valves_from_excel()
# ========================
# VALVE MODELS (static mapping)
# ========================

VALVE_MODELS = {
    "1\" E33": "https://example.com/models/0_5E31.glb",
    "2\" E33": "https://example.com/models/1E31.glb",
    "4\" E33": "https://raw.githubusercontent.com/gurkan-maker/demo2/main/obje8e43.glb",
    "8\" E33": "https://github.com/gurkan-maker/proje/raw/94a24395c07603d1f94114fb1d12672a5dcdc50b/8e43.glb",
    "8\" E43": "https://h6zcfagabwtbjri5.public.blob.vercel-storage.com/8e43-wg0eEqrfslcYlhRqoQhYcNbNpSDBqK.glb",
    "12\" E33": "https://example.com/models/3E32.glb",
    "16\" E33": "https://example.com/models/4E32.glb",
    "20\" E33": "https://example.com/models/6E32.glb",
    "30\" E33": "https://example.com/models/8E32.glb",
}

# ========================
# CV CALCULATION MODULE
# ========================
# ========================
# REYNOLDS NUMBER CALCULATION (FIXED)
# ========================
def reynolds_number(flow_m3h: float, d_m: float, visc_cst: float, 
                   F_d: float, F_L: float, C_v: float) -> float:
    """
    Calculate Reynolds number according to IEC standard formula:
    
    """
    if visc_cst < 0.1:
        return 1e6
    
    N2 = CONSTANTS["N2"]["mm"]  # Use mm constant
    N4 = CONSTANTS["N4"]["mm"]  # Use mm constant

    D_mm = d_m * 1000  # Convert diameter to mm
    term = ((F_L**2 * C_v**2) / (N2 * D_mm**4)) + 1
    denominator = visc_cst * math.sqrt(F_L) * math.sqrt(C_v)
    
    return (N4 * F_d * flow_m3h) / denominator * term

# ========================
# REYNOLDS NUMBER FACTOR CALCULATION (FIXED)
# ========================
def viscosity_correction(rev: float, method: str = "size_selection") -> float:
    """
    Calculate FR based on Reynolds number and method:
    - 'size_selection': For valve size selection
    - 'flow_prediction': For flow rate prediction
    - 'pressure_prediction': For pressure drop prediction
    """
    if rev >= 40000:
        return 1.0
    
    if method == "size_selection":
        if rev < 56:
            return 0.019 * (rev ** 0.67)
        else:
            # Use table values with linear interpolation
            rev_points = [56, 66, 79, 94, 110, 130, 154, 188, 230, 278, 340, 
                         471, 620, 980, 1560, 2470, 4600, 10200, 40000]
            fr_points = [0.284, 0.32, 0.36, 0.40, 0.44, 0.48, 0.52, 0.56, 0.60,
                        0.64, 0.68, 0.72, 0.76, 0.80, 0.84, 0.88, 0.92, 0.96, 1.00]
            return np.interp(rev, rev_points, fr_points)
    
    return 1.0

def calculate_piping_factor_fp(valve_d_inch: float, pipe_d_inch: float, cv_100: float) -> float:
    if pipe_d_inch <= valve_d_inch or abs(pipe_d_inch - valve_d_inch) < 0.01:
        return 1.0
    d_ratio = valve_d_inch / pipe_d_inch
    sumK = 1.5 * (1 - d_ratio**2)**2
    term = 1 + (sumK / CONSTANTS["N2"]["inch"]) * (cv_100 / valve_d_inch**2)**2
    Fp = 1 / math.sqrt(term)
    return Fp

def calculate_flp(valve, valve_d_inch: float, pipe_d_inch: float, cv_100: float) -> float:
    if abs(pipe_d_inch - valve_d_inch) < 0.01:
        return valve.get_fl_at_opening(100)  # Use max opening FL for FLP calculation
    d_ratio = valve_d_inch / pipe_d_inch
    K1 = 0.5 * (1 - d_ratio**2)**2
    KB1 = 1 - d_ratio**4
    Ki = K1 + KB1
    term = (Ki / CONSTANTS["N2"]["inch"]) * (cv_100 / valve_d_inch**2)**2 + 1/valve.get_fl_at_opening(100)**2
    FLP = 1 / math.sqrt(term)
    return FLP

def calculate_x_tp(valve, valve_d_inch: float, pipe_d_inch: float, Fp: float) -> float:
    if abs(pipe_d_inch - valve_d_inch) < 0.01:
        return valve.get_xt_at_opening(100)
    xT = valve.get_xt_at_opening(100)
    d_ratio = valve_d_inch / pipe_d_inch
    K1 = 0.5 * (1 - d_ratio**2)**2
    KB1 = 1 - d_ratio**4
    Ki = K1 + KB1
    cv_100 = valve.get_cv_at_opening(100)
    term = 1 + (xT * Ki / CONSTANTS["N5"]["inch"]) * (cv_100 / valve_d_inch**2)**2
    xTP = xT / Fp**2 * (1 / term)
    return xTP

# ========================
# UPDATED CV_LIQUID FUNCTION (FIXED)
# ========================
def cv_liquid(flow: float, p1: float, p2: float, sg: float, fl_at_op: float, 
              pv: float, pc: float, visc_cst: float, d_m: float, 
              valve, fp: float = 1.0) -> tuple:
    if p1 <= 0 or p2 < 0 or p1 <= p2:
        return 0, {'error': 'Invalid pressures', 'theoretical_cv': 0, 'fp': fp, 'fr': 1.0, 'reynolds': 0, 'is_choked': False, 'ff': 0, 'dp_max': 0}
    
    N1 = CONSTANTS["N1"]["m³/h, bar"]
    dp = p1 - p2
    if dp <= 0:
        return 0, {'theoretical_cv': 0, 'fp': fp, 'fr': 1.0, 'reynolds': 0, 'is_choked': False, 'ff': 0, 'dp_max': 0}
    
    # Calculate Ff (fluid property only)
    ff = calculate_ff(pv, pc)
    
    # Calculate theoretical Cv (valve-independent)
    dp_max_fluid = p1 - ff * pv
    if dp_max_fluid <= 0:
        dp_max_fluid = float('inf')  # No choking possible
    
    if dp < dp_max_fluid:
        theoretical_cv = (flow / N1) * math.sqrt(sg / dp)
    else:
        theoretical_cv = (flow / N1) * math.sqrt(sg) / math.sqrt(dp_max_fluid)
    
    # Calculate max allowable pressure drop for the valve (uses Fl)
    dp_max_valve = fl_at_op**2 * (p1 - ff * pv)
    if dp_max_valve <= 0:
        dp_max_valve = float('inf')  # No choking possible
    
    # Calculate pseudo Cv (valve-specific but without viscosity correction)
    if dp < dp_max_valve:
        cv_pseudo = (flow / N1) * math.sqrt(sg / dp)
    else:
        cv_pseudo = (flow / N1) * math.sqrt(sg) / (fl_at_op * math.sqrt(p1 - ff * pv))

    # Calculate Reynolds number with correct formula
    rev = reynolds_number(
        flow_m3h=flow,
        d_m=d_m,
        visc_cst=visc_cst,
        F_d=valve.fd,
        F_L=fl_at_op,
        C_v=cv_pseudo
    )
    
    # Use viscosity correction for valve size selection
    fr = viscosity_correction(rev, method="size_selection")
    
    # Apply corrections
    cv_after_fp = cv_pseudo / fp
    corrected_cv = cv_after_fp / fr
    
    details = {
        'theoretical_cv': theoretical_cv,  # This is now valve-independent
        'fp': fp,
        'fr': fr,
        'reynolds': rev,
        'is_choked': (dp >= dp_max_valve),
        'ff': ff,
        'dp_max': dp_max_valve,
        'fl_at_op': fl_at_op
    }
    
    return corrected_cv, details

def cv_gas(flow: float, p1: float, p2: float, sg: float, t: float, k: float, 
           xt_at_op: float, z: float, fp: float = 1.0) -> tuple:
    if p1 <= 0 or p2 < 0 or p1 <= p2:
        return 0, {'error': 'Invalid pressures', 'theoretical_cv': 0, 'fp': fp, 'expansion_factor': 0, 'is_choked': False, 'x_crit': 0, 'x_actual': 0, 'xt_at_op': xt_at_op}
    
    x_actual = (p1 - p2) / p1
    if x_actual <= 0:
        return 0, {'error': 'Negative pressure drop', 'theoretical_cv': 0, 'fp': fp, 'expansion_factor': 0, 'is_choked': False, 'x_crit': 0, 'x_actual': x_actual, 'xt_at_op': xt_at_op}
    
    fk = k / 1.4
    x_crit = fk * xt_at_op
    
    if x_actual >= x_crit:
        y = 0.667
        x = x_crit
        is_choked = True
    else:
        x = x_actual
        y = 1 - x / (3 * fk * xt_at_op)
        is_choked = False
    
    N7 = CONSTANTS["N7"]["m³/h, bar, K (standard)"]
    term = (sg * (t + C_TO_K) * z) / x
    if term < 0:
        return 0, {'error': 'Negative value in sqrt', 'theoretical_cv': 0, 'fp': fp, 'expansion_factor': y, 'is_choked': is_choked, 'x_crit': x_crit, 'x_actual': x_actual, 'xt_at_op': xt_at_op}
    
    theoretical_cv = (flow / (N7 * fp * p1 * y)) * math.sqrt(term)
    corrected_cv = theoretical_cv
    
    details = {
        'theoretical_cv': theoretical_cv,
        'fp': fp,
        'expansion_factor': y,
        'is_choked': is_choked,
        'x_crit': x_crit,
        'x_actual': x_actual,
        'xt_at_op': xt_at_op
    }
    
    return corrected_cv, details

def cv_steam(flow: float, p1: float, p2: float, rho: float, k: float, 
             xt_at_op: float, fp: float = 1.0) -> tuple:
    if p1 <= 0 or p2 < 0 or p1 <= p2:
        return 0, {'error': 'Invalid pressures', 'theoretical_cv': 0, 'fp': fp, 'expansion_factor': 0, 'is_choked': False, 'x_crit': 0, 'x_actual': 0, 'xt_at_op': xt_at_op}
    
    x_actual = (p1 - p2) / p1
    if x_actual <= 0:
        return 0, {'error': 'Negative pressure drop', 'theoretical_cv': 0, 'fp': fp, 'expansion_factor': 0, 'is_choked': False, 'x_crit': 0, 'x_actual': x_actual, 'xt_at_op': xt_at_op}
    
    fk = k / 1.4
    x_crit = fk * xt_at_op
    
    if x_actual >= x_crit:
        y = 0.667
        x = x_crit
        is_choked = True
    else:
        x = x_actual
        y = 1 - x / (3 * fk * xt_at_op)
        is_choked = False
    
    N6 = CONSTANTS["N6"]["kg/h, bar, kg/m³"]
    term = x * p1 * rho
    if term <= 0:
        return 0, {'error': 'Invalid term in sqrt', 'theoretical_cv': 0, 'fp': fp, 'expansion_factor': y, 'is_choked': is_choked, 'x_crit': x_crit, 'x_actual': x_actual, 'xt_at_op': xt_at_op}
    
    theoretical_cv = flow / (N6 * y * math.sqrt(term))
    corrected_cv = theoretical_cv / fp
    
    details = {
        'theoretical_cv': theoretical_cv,
        'fp': fp,
        'expansion_factor': y,
        'is_choked': is_choked,
        'x_crit': x_crit,
        'x_actual': x_actual,
        'xt_at_op': xt_at_op
    }
    
    return corrected_cv, details

def check_cavitation(p1: float, p2: float, pv: float, fl_at_op: float, pc: float) -> tuple:
    if pc <= 0:
        return False, 0, 0, "Critical pressure not available"
    if p1 <= 0 or p2 < 0 or p1 <= p2:
        return False, 0, 0, "Invalid pressures"
    ff = calculate_ff(pv, pc)
    dp = p1 - p2
    if dp <= 0:
        return False, 0, 0, "No pressure drop"
    dp_max = fl_at_op**2 * (p1 - ff * pv)
    km = fl_at_op**2
    sigma = (p1 - pv) / dp
    if dp >= dp_max:
        return True, sigma, km, "Choked flow - cavitation likely"
    elif sigma < 1.5 * km:
        return True, sigma, km, "Severe cavitation risk"
    elif sigma < 2 * km:
        return False, sigma, km, "Moderate cavitation risk"
    elif sigma < 4 * km:
        return False, sigma, km, "Mild cavitation risk"
    return False, sigma, km, "Minimal cavitation risk"


# ========================
# ENHANCED PDF REPORT GENERATION
# ========================
class EnhancedPDFReport(FPDF):
    def __init__(self, logo_bytes=None, logo_type=None):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.logo_bytes = logo_bytes
        self.logo_type = logo_type
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(15, 15, 15)
        self.set_title("Control Valve Sizing Report")
        self.set_author("VASTAŞ Valve Sizing Software")
        self.alias_nb_pages()
        self.set_compression(True)
        
        # Add Unicode support (DejaVuSans font supports most characters)
        self.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
        self.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)
        self.add_font('DejaVu', 'I', 'DejaVuSans-Oblique.ttf', uni=True)
        self.add_font('DejaVu', 'BI', 'DejaVuSans-BoldOblique.ttf', uni=True)
    
    def header(self):
        # Header only from second page onward
        if self.page_no() == 1:
            return
            
        # Draw top border
        self.set_draw_color(0, 51, 102)
        self.set_line_width(0.5)
        self.line(10, 15, 200, 15)
        
        # Logo
        if self.logo_bytes and self.logo_type:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{self.logo_type.lower()}") as tmpfile:
                    tmpfile.write(self.logo_bytes)
                    tmpfile_path = tmpfile.name
                self.image(tmpfile_path, x=15, y=8, w=20)
                os.unlink(tmpfile_path)
            except Exception as e:
                pass
        
        # Title
        self.set_font('DejaVu', 'B', 10)
        self.set_text_color(0, 51, 102)
        self.set_y(10)
        self.cell(0, 10, 'Control Valve Sizing Report', 0, 0, 'C')
        
        # Page number
        self.set_font('DejaVu', 'I', 8)
        self.set_text_color(100)
        self.set_y(10)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'R')
        
        # Line break
        self.ln(15)
        
    def footer(self):
        # Footer only from second page onward
        if self.page_no() == 1:
            return
            
        self.set_y(-15)
        self.set_font('DejaVu', 'I', 8)
        self.set_text_color(100)
        self.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 0, 'L')
        self.cell(0, 10, 'Confidential - VASTAŞ Valve Technologies', 0, 0, 'R')
    
    def cover_page(self, title, subtitle, project_info=None):
        self.add_page()
        
        # Background rectangle
        self.set_fill_color(0, 51, 102)
        self.rect(0, 0, 210, 297, 'F')
        
        # Main content area
        self.set_fill_color(255, 255, 255)
        self.rect(15, 15, 180, 267, 'F')
        
        # Logo at top
        if self.logo_bytes and self.logo_type:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{self.logo_type.lower()}") as tmpfile:
                    tmpfile.write(self.logo_bytes)
                    tmpfile_path = tmpfile.name
                self.image(tmpfile_path, x=80, y=40, w=50)
                os.unlink(tmpfile_path)
            except Exception as e:
                pass
        
        # Title
        self.set_y(120)
        self.set_font('DejaVu', 'B', 24)
        self.set_text_color(0, 51, 102)
        self.cell(0, 15, title, 0, 1, 'C')
        
        # Subtitle
        self.set_font('DejaVu', 'I', 18)
        self.set_text_color(70, 70, 70)
        self.cell(0, 10, subtitle, 0, 1, 'C')
        
        # Project info
        if project_info:
            self.set_font('DejaVu', '', 14)
            self.set_text_color(0, 0, 0)
            self.ln(20)
            self.cell(0, 10, project_info, 0, 1, 'C')
        
        # Company info
        self.set_y(220)
        self.set_font('DejaVu', 'B', 14)
        self.set_text_color(0, 51, 102)
        self.cell(0, 10, 'VASTAŞ Valve Technologies', 0, 1, 'C')
        
        # Date
        self.set_font('DejaVu', 'I', 12)
        self.set_text_color(70, 70, 70)
        self.cell(0, 10, datetime.now().strftime("%B %d, %Y"), 0, 1, 'C')
        
        # Confidential notice
        self.set_y(270)
        self.set_font('DejaVu', 'I', 10)
        self.set_text_color(150, 0, 0)
        self.cell(0, 5, 'CONFIDENTIAL - For internal use only', 0, 0, 'C')
    
    def chapter_title(self, title):
        self.set_font('DejaVu', 'B', 14)
        self.set_text_color(0, 51, 102)
        self.set_fill_color(230, 240, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(5)
    
    def chapter_body(self, body, font_size=12):
        self.set_font('DejaVu', '', font_size)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 6, body)
        self.ln()
    
    def add_table(self, headers, data, col_widths=None, header_color=(0, 51, 102), 
                  row_colors=[(255, 255, 255), (240, 248, 255)]):
        if col_widths is None:
            col_widths = [self.w / len(headers)] * len(headers)
        
        # Table header
        self.set_font('DejaVu', 'B', 10)
        self.set_text_color(255, 255, 255)
        self.set_fill_color(*header_color)
        
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, header, 1, 0, 'C', 1)
        self.ln()
        
        # Table data
        self.set_font('DejaVu', '', 10)
        self.set_text_color(0, 0, 0)
        
        for row_idx, row in enumerate(data):
            # Alternate row colors
            fill_color = row_colors[row_idx % len(row_colors)]
            self.set_fill_color(*fill_color)
            
            for i, item in enumerate(row):
                self.cell(col_widths[i], 6, str(item), 1, 0, 'C', 1)
            self.ln()
    
    def add_key_value_table(self, data, col_widths=[70, 130], font_size=10):
        self.set_font('DejaVu', 'B', font_size)
        self.set_text_color(0, 51, 102)
        self.set_fill_color(240, 248, 255)
        
        for key, value in data:
            self.cell(col_widths[0], 7, key, 1, 0, 'L', 1)
            self.set_font('DejaVu', '', font_size)
            self.set_text_color(0, 0, 0)
            self.set_fill_color(255, 255, 255)
            self.multi_cell(col_widths[1], 7, str(value), 1, 'L', 1)
            self.set_font('DejaVu', 'B', font_size)
            self.set_text_color(0, 51, 102)
            self.set_fill_color(240, 248, 255)
    
    def add_image(self, image_bytes, width=180, caption=None):
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_plot:
                tmp_plot.write(image_bytes)
                tmp_plot_path = tmp_plot.name
            x = (self.w - width) / 2
            self.image(tmp_plot_path, x=x, w=width)
            os.unlink(tmp_plot_path)
            
            if caption:
                self.set_font('DejaVu', 'I', 8)
                self.set_text_color(100)
                self.cell(0, 5, caption, 0, 1, 'C')
                self.ln(3)
        except Exception as e:
            self.cell(0, 10, f"Failed to insert image: {str(e)}", 0, 1)

def generate_pdf_report(scenarios, valve, op_points, req_cvs, warnings, cavitation_info, 
                        plot_bytes=None, flow_dp_plot_bytes=None, logo_bytes=None, logo_type=None):
    try:
        # Create PDF with enhanced features
        pdf = EnhancedPDFReport(logo_bytes=logo_bytes, logo_type=logo_type)
        
        # Cover page
        project_name = "Valve Sizing Project"
        if scenarios and scenarios[0].get("name"):
            project_name = scenarios[0]["name"]
        pdf.cover_page(
            title="CONTROL VALVE SIZING REPORT",
            subtitle=project_name,
            project_info=f"Prepared by VASTAŞ Engineering Department"
        )
        
        # Table of Contents
        pdf.add_page()
        pdf.chapter_title('Table of Contents')
        toc = [
            ("1. Project Information", 3),
            ("2. Valve Specifications", 4),
            ("3. Sizing Results Summary", 5),
            ("4. Detailed Calculations", 6),
            ("5. Valve Characteristics", 8),
            ("6. Cavitation Analysis", 10),
            ("7. Appendices", 12)
        ]
        
        pdf.set_font('DejaVu', '', 12)
        pdf.set_text_color(0, 0, 0)
        
        for title, page in toc:
            # Add dot leaders
            pdf.cell(0, 10, title, 0, 0, 'L')
            dot_leader = '.' * (70 - len(title))
            pdf.cell(0, 10, dot_leader, 0, 0, 'L')
            pdf.cell(0, 10, str(page), 0, 1, 'R')
        
        # Project Information
        pdf.add_page()
        pdf.chapter_title('1. Project Information')
        pdf.chapter_body('This report contains the sizing calculations for the control valve based on the provided operational scenarios.', 12)
        
        project_info = [
            ("Project Name:", project_name),
            ("Report Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("Prepared By:", "VASTAŞ Valve Sizing Software"),
            ("Valve Model:", get_valve_display_name(valve)),
            ("Number of Scenarios:", str(len(scenarios)))
        ]
        pdf.add_key_value_table(project_info)
        
        # Valve Specifications
        pdf.add_page()
        pdf.chapter_title('2. Valve Specifications')
        
        valve_specs = [
            ("Valve Size:", f"{valve.size}\""),
            ("Type:", "Globe" if valve.valve_type == 3 else "Axial"),
            ("Rating Class:", str(valve.rating_class)),
            ("Fl (Liquid Recovery):", f"{valve.get_fl_at_opening(100):.3f}"),
            ("Xt (Pressure Drop Ratio):", f"{valve.get_xt_at_opening(100):.3f}"),
            ("Fd (Valve Style Modifier):", f"{valve.fd:.2f}"),
            ("Internal Diameter:", f"{valve.diameter:.2f} in"),
            ("Manufacturer:", "VASTAŞ Valves")
        ]
        pdf.add_key_value_table(valve_specs)
        
        pdf.chapter_title('Valve Cv Characteristics')
        cv_table_data = []
        for open_percent, cv in valve.cv_table.items():
            cv_table_data.append([f"{open_percent}%", f"{cv:.1f}"])
        pdf.add_table(['Opening %', 'Cv Value'], cv_table_data, col_widths=[50, 50])
        
        pdf.chapter_title('Valve Fl Characteristics')
        fl_table_data = []
        for open_percent, fl in valve.fl_table.items():
            fl_table_data.append([f"{open_percent}%", f"{fl:.3f}"])
        pdf.add_table(['Opening %', 'Fl Value'], fl_table_data, col_widths=[50, 50])
        
        pdf.chapter_title('Valve Xt Characteristics')
        xt_table_data = []
        for open_percent, xt in valve.xt_table.items():
            xt_table_data.append([f"{open_percent}%", f"{xt:.3f}"])
        pdf.add_table(['Opening %', 'Xt Value'], xt_table_data, col_widths=[50, 50])
        
        # Sizing Results Summary
        pdf.add_page()
        pdf.chapter_title('3. Sizing Results Summary')
        
        results_data = []
        for i, scenario in enumerate(scenarios):
            actual_cv = valve.get_cv_at_opening(op_points[i])
            margin = (actual_cv / req_cvs[i] - 1) * 100 if req_cvs[i] > 0 else 0
            
            status = "✅ Optimal"
            if "Severe" in cavitation_info[i]:
                status = "⚠️ Severe Cavitation"
            elif "High opening" in warnings[i]:
                status = "⚠️ High Opening"
            elif "Low opening" in warnings[i]:
                status = "⚠️ Low Opening"
            elif "Choked" in cavitation_info[i]:
                status = "❌ Choked Flow"
            elif "Insufficient" in warnings[i]:
                status = "❌ Insufficient Capacity"
            
            results_data.append([
                scenario["name"],
                f"{req_cvs[i]:.1f}",
                f"{valve.size}\"",
                f"{op_points[i]:.1f}%",
                f"{actual_cv:.1f}",
                f"{margin:.1f}%",
                status
            ])
        
        pdf.add_table(
            ['Scenario', 'Req Cv', 'Valve Size', 'Opening %', 'Actual Cv', 'Margin %', 'Status'],
            results_data,
            col_widths=[30, 25, 25, 25, 25, 25, 40]
        )
        
        # Status legend
        pdf.set_font('DejaVu', 'I', 9)
        pdf.set_text_color(100)
        pdf.cell(0, 5, "Status Legend: ✅ Optimal | ⚠️ Warning | ❌ Critical Issue", 0, 1)
        
        # Detailed Calculations
        for i, scenario in enumerate(scenarios):
            pdf.add_page()
            pdf.chapter_title(f'4. Detailed Calculations: Scenario {i+1} - {scenario["name"]}')
            
            # Scenario parameters
            pdf.set_font('DejaVu', 'B', 12)
            pdf.cell(0, 8, "Process Conditions", 0, 1)
            scenario_params = [
                ("Fluid Type:", scenario['fluid_type'].title()),
                ("Flow Rate:", f"{scenario['flow']} "
                 f"{'m³/h' if scenario['fluid_type']=='liquid' else 'kg/h' if scenario['fluid_type']=='steam' else 'std m³/h'}"),
                ("Inlet Pressure (P1):", f"{scenario['p1']:.2f} bar a"),
                ("Outlet Pressure (P2):", f"{scenario['p2']:.2f} bar a"),
                ("Pressure Drop (dP):", f"{scenario['p1'] - scenario['p2']:.2f} bar"),
                ("Temperature:", f"{scenario['temp']}°C"),
                ("Pipe Diameter:", f"{scenario['pipe_d']} in")
            ]
            pdf.add_key_value_table(scenario_params)
            
            # Fluid properties
            pdf.set_font('DejaVu', 'B', 12)
            pdf.cell(0, 8, "Fluid Properties", 0, 1)
            fluid_props = []
            if scenario["fluid_type"] == "liquid":
                fluid_props.extend([
                    ("Specific Gravity:", f"{scenario['sg']:.3f}"),
                    ("Viscosity:", f"{scenario['visc']} cSt"),
                    ("Vapor Pressure:", f"{scenario['pv']:.4f} bar a"),
                    ("Critical Pressure:", f"{scenario['pc']:.2f} bar a")
                ])
            elif scenario["fluid_type"] == "gas":
                fluid_props.extend([
                    ("Specific Gravity (air=1):", f"{scenario['sg']:.3f}"),
                    ("Specific Heat Ratio (k):", f"{scenario['k']:.3f}"),
                    ("Compressibility Factor (Z):", f"{scenario['z']:.3f}")
                ])
            else:
                fluid_props.extend([
                    ("Density:", f"{scenario['rho']:.3f} kg/m³"),
                    ("Specific Heat Ratio (k):", f"{scenario['k']:.3f}")
                ])
            pdf.add_key_value_table(fluid_props)
            
            # Calculation results
            result = {
                "op_point": op_points[i],
                "req_cv": req_cvs[i],
                "warning": warnings[i],
                "cavitation_info": cavitation_info[i]
            }
            actual_cv = valve.get_cv_at_opening(op_points[i])
            
            pdf.set_font('DejaVu', 'B', 12)
            pdf.cell(0, 8, "Sizing Results", 0, 1)
            sizing_results = [
                ("Theoretical Cv:", f"{result.get('theoretical_cv', 0):.1f}"),
                ("Corrected Cv:", f"{result['req_cv']:.1f}"),
                ("Operating Point:", f"{result['op_point']:.1f}% open"),
                ("Actual Cv at Operating Point:", f"{actual_cv:.1f}"),
                ("Margin:", f"{(actual_cv / result['req_cv'] - 1) * 100:.1f}%"),
                ("Warnings:", result['warning']),
                ("Cavitation Status:", result['cavitation_info'])
            ]
            pdf.add_key_value_table(sizing_results)
            
            # Flow vs DP graph
            if i == 0 and flow_dp_plot_bytes:  # Only include for first scenario to save space
                pdf.add_page()
                pdf.chapter_title('Flow Rate vs Pressure Drop')
                pdf.chapter_body('The graph below shows the relationship between flow rate and pressure drop for the selected valve at the operating point.', 10)
                pdf.add_image(flow_dp_plot_bytes, width=150, caption=f"Flow vs Pressure Drop - {scenario['name']}")
        
        # Valve Characteristics
        pdf.add_page()
        pdf.chapter_title('5. Valve Characteristics')
        pdf.chapter_body('The following graph shows the Cv characteristic curve of the selected valve with operating points for each scenario.', 12)
        
        if plot_bytes:
            pdf.add_image(plot_bytes, width=150, caption="Valve Cv Characteristic Curve")
        
        # Cavitation Analysis
        pdf.add_page()
        pdf.chapter_title('6. Cavitation Analysis')
        pdf.chapter_body('Cavitation occurs when the pressure drops below the vapor pressure of the liquid, causing vapor bubbles to form and collapse. This can cause damage to the valve and piping system.', 12)
        
        cavitation_table = [
            ["Risk Level", "Sigma (σ) Range", "Description"],
            ["Minimal", "σ > 4Km", "Little to no cavitation damage expected"],
            ["Mild", "2Km < σ ≤ 4Km", "Some cavitation may occur, but damage is limited"],
            ["Moderate", "1.5Km < σ ≤ 2Km", "Moderate cavitation, potential for damage over time"],
            ["Severe", "σ ≤ 1.5Km", "Severe cavitation, immediate damage likely"],
            ["Choked", "ΔP ≥ Fl²(P1 - FfPv)", "Choked flow - cavitation inevitable"]
        ]
        
        pdf.add_table(cavitation_table[0], cavitation_table[1:], col_widths=[40, 60, 90])
        
        pdf.ln(10)
        pdf.set_font('DejaVu', 'B', 12)
        pdf.cell(0, 8, "Cavitation Mitigation Strategies", 0, 1)
        strategies = [
            "1. Use valves with higher pressure recovery factors (Fl)",
            "2. Increase upstream pressure if possible",
            "3. Reduce pressure drop across the valve",
            "4. Use multi-stage pressure reduction for high pressure drops",
            "5. Select cavitation-resistant trim materials",
            "6. Use anti-cavitation trim designs"
        ]
        for strategy in strategies:
            pdf.set_font('DejaVu', '', 10)
            pdf.multi_cell(0, 6, strategy)
        
        # Appendices
        pdf.add_page()
        pdf.chapter_title('7. Appendices')
        
        pdf.set_font('DejaVu', 'B', 12)
        pdf.cell(0, 8, "Calculation Standards", 0, 1)
        pdf.set_font('DejaVu', '', 10)
        pdf.multi_cell(0, 6, "This sizing report is based on the ISA-75.01.01 (IEC 60534-2-1) standard for control valve sizing equations. The calculations consider fluid properties, piping geometry, and valve characteristics to determine the optimal valve size and operating conditions.")
        
        pdf.ln(5)
        pdf.set_font('DejaVu', 'B', 12)
        pdf.cell(0, 8, "Symbols and Abbreviations", 0, 1)
        abbreviations = [
            ("Cv:", "Valve flow coefficient"),
            ("Fl:", "Liquid pressure recovery factor"),
            ("Fd:", "Valve style modifier"),
            ("Xt:", "Pressure drop ratio factor"),
            ("dP:", "Pressure drop (P1 - P2)"),
            ("P1:", "Inlet pressure"),
            ("P2:", "Outlet pressure"),
            ("SG:", "Specific gravity (liquid) or specific gravity (gas)"),
            ("T:", "Temperature"),
            ("PV:", "Vapor pressure"),
            ("PC:", "Critical pressure"),
            ("Rev:", "Reynolds number"),
            ("σ:", "Cavitation index"),
            ("Km:", "Valve recovery coefficient")
        ]
        pdf.add_key_value_table(abbreviations, col_widths=[20, 170], font_size=10)
        
        pdf.ln(5)
        pdf.set_font('DejaVu', 'B', 12)
        pdf.cell(0, 8, "Disclaimer", 0, 1)
        pdf.set_font('DejaVu', '', 9)
        pdf.multi_cell(0, 5, "This report is generated by VASTAŞ Valve Sizing Software and is provided for informational purposes only. While every effort has been made to ensure the accuracy of the calculations, VASTAŞ makes no warranties or representations regarding the completeness or accuracy of this information. Final valve selection should be verified by a qualified engineer.")
        
        # Generate PDF in memory
        pdf_bytes_io = BytesIO()
        pdf.output(pdf_bytes_io)
        pdf_bytes_io.seek(0)
        return pdf_bytes_io
        
    except Exception as e:
        error_bytes_io = BytesIO()
        error_pdf = FPDF()
        error_pdf.add_page()
        error_pdf.set_font('Arial', 'B', 16)
        error_pdf.cell(0, 10, 'PDF Generation Error', 0, 1)
        error_pdf.set_font('Arial', '', 12)
        error_pdf.multi_cell(0, 10, f"An error occurred while generating the PDF report: {str(e)}\n\n{traceback.format_exc()}")
        error_pdf.output(error_bytes_io)
        error_bytes_io.seek(0)
        return error_bytes_io

# ========================
# SIMULATION RESULTS
# ========================
def get_simulation_image(valve_name):
    simulation_images = {
        
        "2\" E33": "https://raw.githubusercontent.com/gurkan-maker/demo2/main/2e33.png",
        "4\" E33": "https://raw.githubusercontent.com/gurkan-maker/demo2/main/4e33.png",
        "8\" E33": "https://raw.githubusercontent.com/gurkan-maker/demo2/main/8e33.png",
        "8\" E43": "https://raw.githubusercontent.com/gurkan-maker/demo2/main/8e43.png",
        "12\" E33": "https://raw.githubusercontent.com/gurkan-maker/demo2/main/12e33.png",
        "16\" E33": "https://raw.githubusercontent.com/gurkan-maker/demo2/main/16e33.png",
        "20\" E33": "https://raw.githubusercontent.com/gurkan-maker/demo2/main/20e33.png",
        "30\" E33": "https://raw.githubusercontent.com/gurkan-maker/demo2/main/30e33.png",
    }
    return simulation_images.get(valve_name, "https://via.placeholder.com/1200x900.png?text=Simulation+Not+Available")

# ========================
# FLOW RATE VS PRESSURE DROP GRAPH
# ========================
def generate_flow_vs_dp_graph(scenario, valve, op_point, details, req_cv):
    # Get actual Cv at operating point
    actual_cv = valve.get_cv_at_opening(op_point)
    valve_cv_effective = actual_cv * details.get('fp', 1.0)
    
    # Determine max pressure drop
    if scenario['fluid_type'] == "liquid":
        max_dp = details.get('dp_max', scenario['p1'] - scenario['p2'])
    elif scenario['fluid_type'] in ["gas", "steam"]:
        # Safely get x_crit with fallback
        x_crit = details.get('x_crit', 0)
        if x_crit <= 0:
            # Calculate from k and xt if available
            k = scenario.get('k', 1.4)
            xt = details.get('xt_at_op', 0.5)
            fk = k / 1.4
            x_crit = fk * xt
        max_dp = x_crit * scenario['p1']
    else:
        max_dp = scenario['p1'] - scenario['p2']
    
    # Create pressure drop range (from 1/10 max to max)
    min_dp = max(0.1, max_dp / 10)  # Ensure min_dp is at least 0.1 bar
    dp_range = np.linspace(min_dp, max_dp, 200)  # More points for smoother curve
    flow_rates = []
    
    # Calculate flow rates for each dp
    for dp in dp_range:
        if scenario["fluid_type"] == "liquid":
            if dp <= details.get('dp_max', dp):
                flow = valve_cv_effective * CONSTANTS["N1"]["m³/h, bar"] * math.sqrt(dp / scenario['sg'])
            else:
                flow = valve_cv_effective * CONSTANTS["N1"]["m³/h, bar"] * details.get('fl_at_op', 0.9) * math.sqrt(
                    (scenario['p1'] - details.get('ff', 0.96) * scenario.get('pv', 0)) / scenario['sg'])
            flow_rates.append(flow)
            
        elif scenario["fluid_type"] == "gas":
            x = dp / scenario['p1']
            x_crit = details.get('x_crit', 0.5)
            fk = scenario['k'] / 1.4
            if x < x_crit:
                Y = 1 - x / (3 * fk * details.get('xt_at_op', 0.5))
            else:
                Y = 0.667
                x = x_crit
            flow = valve_cv_effective * CONSTANTS["N7"]["m³/h, bar, K (standard)"] * scenario['p1'] * Y * math.sqrt(
                x / (scenario['sg'] * (scenario['temp'] + C_TO_K) * scenario['z']))
            flow_rates.append(flow)
            
        elif scenario["fluid_type"] == "steam":
            x = dp / scenario['p1']
            x_crit = details.get('x_crit', 0.5)
            fk = scenario['k'] / 1.4
            if x < x_crit:
                Y = 1 - x / (3 * fk * details.get('xt_at_op', 0.5))
            else:
                Y = 0.667
                x = x_crit
            flow = valve_cv_effective * CONSTANTS["N6"]["kg/h, bar, kg/m³"] * Y * math.sqrt(
                x * scenario['p1'] * scenario['rho'])
            flow_rates.append(flow)
        else:
            # Fallback for unknown fluid type
            flow_rates.append(0)
    
    # Current operating point
    current_dp = scenario['p1'] - scenario['p2']
    current_flow = scenario['flow']
    
    # Create plot with smooth curve using polynomial interpolation
    if len(dp_range) > 3 and len(flow_rates) > 3:
        # Create polynomial fit for smooth curve
        z = np.polyfit(dp_range, flow_rates, 3)
        p = np.poly1d(z)
        x_smooth = np.linspace(min_dp, max_dp, 300)
        y_smooth = p(x_smooth)
    else:
        x_smooth = dp_range
        y_smooth = flow_rates
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_smooth, 
        y=y_smooth, 
        mode='lines',
        name='Flow Rate',
        line=dict(color='blue', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[current_dp], 
        y=[current_flow], 
        mode='markers',
        name='Operating Point',
        marker=dict(size=12, color='red')
    ))
    
    # Add max flow annotation
    if max_dp > 0 and flow_rates:
        max_flow = flow_rates[-1]
        fig.add_annotation(
            x=max_dp,
            y=max_flow,
            text=f'Max Flow: {max_flow:.1f}',
            showarrow=True,
            arrowhead=1,
            ax=-50,
            ay=-30
        )
    
    fig.update_layout(
        title=f'Flow Rate vs Pressure Drop - {scenario["name"]}',
        xaxis_title='Pressure Drop (bar)',
        yaxis_title=f'Flow Rate ({"m³/h" if scenario["fluid_type"]=="liquid" else "std m³/h" if scenario["fluid_type"]=="gas" else "kg/h"})',
        legend_title='Legend',
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    fig.update_xaxes(range=[0, max_dp * 1.1])
    if flow_rates:
        fig.update_yaxes(range=[0, max(flow_rates) * 1.1])
    
    return fig

# ========================
# MATPLOTLIB PLOT FOR PDF
# ========================
def plot_cv_curve_matplotlib(valve, op_points, req_cvs, theoretical_cvs, scenario_names):
    # Generate smooth curve using polynomial interpolation
    openings = sorted(valve.cv_table.keys())
    cv_values = np.array([valve.get_cv_at_opening(op) for op in openings])
    
    # Create dense x values for smooth curve
    x_smooth = np.linspace(0, 100, 300)
    
    # Create polynomial fit (degree 3 for smooth curve)
    if len(openings) > 3:
        z = np.polyfit(openings, cv_values, 3)
        p = np.poly1d(z)
        y_smooth = p(x_smooth)
    else:
        # Fallback to linear interpolation
        y_smooth = np.interp(x_smooth, openings, cv_values)
    
    plt.figure(figsize=(10, 6))
    
    # Valve Cv curve (smooth)
    plt.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='Valve Cv')
    
    # Operating points
    for i, op in enumerate(op_points):
        actual_cv = valve.get_cv_at_opening(op)
        plt.plot(op, actual_cv, 'ro', markersize=8)
        plt.text(op + 2, actual_cv, f'S{i+1}', fontsize=10, color='red')
    
    # Required Cv lines
    for i, cv in enumerate(req_cvs):
        plt.axhline(y=cv, color='r', linestyle='--', linewidth=1)
        plt.text(100, cv, f'Corrected S{i+1}: {cv:.1f}', 
                 fontsize=9, color='red', ha='right', va='bottom')
    
    # Theoretical Cv lines
    for i, cv in enumerate(theoretical_cvs):
        plt.axhline(y=cv, color='g', linestyle=':', linewidth=1)
        plt.text(100, cv, f'Theoretical S{i+1}: {cv:.1f}', 
                 fontsize=9, color='green', ha='right', va='top')
    
    plt.title(f'{valve.size}" Valve Cv Characteristic')
    plt.xlabel('Opening Percentage (%)')
    plt.ylabel('Cv Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.xlim(0, 100)
    plt.ylim(0, max(cv_values) * 1.1)
    
    # Save to bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.getvalue()

def plot_flow_vs_dp_matplotlib(scenario, valve, op_point, details, req_cv):
    # Get actual Cv at operating point
    actual_cv = valve.get_cv_at_opening(op_point)
    valve_cv_effective = actual_cv * details.get('fp', 1.0)
    
    # Determine max pressure drop
    if scenario['fluid_type'] == "liquid":
        max_dp = details.get('dp_max', scenario['p1'] - scenario['p2'])
    elif scenario['fluid_type'] in ["gas", "steam"]:
        x_crit = details.get('x_crit', 0)
        if x_crit <= 0:
            k = scenario.get('k', 1.4)
            xt = details.get('xt_at_op', 0.5)
            fk = k / 1.4
            x_crit = fk * xt
        max_dp = x_crit * scenario['p1']
    else:
        max_dp = scenario['p1'] - scenario['p2']
    
    # Create pressure drop range
    min_dp = max(0.1, max_dp / 10)
    dp_range = np.linspace(min_dp, max_dp, 200)  # More points for smoother curve
    flow_rates = []
    
    # Calculate flow rates
    for dp in dp_range:
        if scenario['fluid_type'] == "liquid":
            if dp <= details.get('dp_max', dp):
                flow = valve_cv_effective * CONSTANTS["N1"]["m³/h, bar"] * math.sqrt(dp / scenario['sg'])
            else:
                flow = valve_cv_effective * CONSTANTS["N1"]["m³/h, bar"] * details.get('fl_at_op', 0.9) * math.sqrt(
                    (scenario['p1'] - details.get('ff', 0.96) * scenario.get('pv', 0)) / scenario['sg'])
            flow_rates.append(flow)
        elif scenario['fluid_type'] == "gas":
            x = dp / scenario['p1']
            x_crit = details.get('x_crit', 0.5)
            fk = scenario['k'] / 1.4
            if x < x_crit:
                Y = 1 - x / (3 * fk * details.get('xt_at_op', 0.5))
            else:
                Y = 0.667
                x = x_crit
            flow = valve_cv_effective * CONSTANTS["N7"]["m³/h, bar, K (standard)"] * scenario['p1'] * Y * math.sqrt(
                x / (scenario['sg'] * (scenario['temp'] + C_TO_K) * scenario['z']))
            flow_rates.append(flow)
        elif scenario['fluid_type'] == "steam":
            x = dp / scenario['p1']
            x_crit = details.get('x_crit', 0.5)
            fk = scenario['k'] / 1.4
            if x < x_crit:
                Y = 1 - x / (3 * fk * details.get('xt_at_op', 0.5))
            else:
                Y = 0.667
                x = x_crit
            flow = valve_cv_effective * CONSTANTS["N6"]["kg/h, bar, kg/m³"] * Y * math.sqrt(
                x * scenario['p1'] * scenario['rho'])
            flow_rates.append(flow)
        else:
            flow_rates.append(0)
    
    # Current operating point
    current_dp = scenario['p1'] - scenario['p2']
    current_flow = scenario['flow']
    
    # Create smooth curve using polynomial interpolation
    if len(dp_range) > 3 and len(flow_rates) > 3:
        # Create polynomial fit (degree 3 for smooth curve)
        z = np.polyfit(dp_range, flow_rates, 3)
        p = np.poly1d(z)
        x_smooth = np.linspace(min_dp, max_dp, 300)
        y_smooth = p(x_smooth)
    else:
        x_smooth = dp_range
        y_smooth = flow_rates
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='Flow Rate')
    plt.plot(current_dp, current_flow, 'ro', markersize=8, label='Operating Point')
    
    # Mark max flow
    if flow_rates:
        plt.plot(max_dp, flow_rates[-1], 'go', markersize=8)
        plt.annotate(f'Max Flow: {flow_rates[-1]:.1f}', 
                     xy=(max_dp, flow_rates[-1]), 
                     xytext=(-50, -30), 
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='green'))
    
    plt.title(f'Flow Rate vs Pressure Drop - {scenario["name"]}')
    plt.xlabel('Pressure Drop (bar)')
    plt.ylabel(f'Flow Rate ({"m³/h" if scenario["fluid_type"]=="liquid" else "std m³/h" if scenario["fluid_type"]=="gas" else "kg/h"})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.xlim(0, max_dp * 1.1)
    if flow_rates:
        plt.ylim(0, max(flow_rates) * 1.1)
    
    # Save to bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.getvalue()

# ========================
# RECOMMENDED VALVE LOGIC
# ========================
def evaluate_valve_for_scenario(valve, scenario):
    # Determine pipe diameter based on selection
    use_valve_size = scenario.get("use_valve_size", True)
    if use_valve_size:
        pipe_d = valve.diameter
    else:
        pipe_d = scenario["pipe_d"]
        
    valve_d = valve.diameter
    cv_100 = valve.get_cv_at_opening(100)
    fp = calculate_piping_factor_fp(valve_d, pipe_d, cv_100)
    
    if scenario["fluid_type"] in ["gas", "steam"]:
        if abs(pipe_d - valve_d) > 0.01:
            xt = calculate_x_tp(valve, valve_d, pipe_d, fp)
        else:
            xt = valve.get_xt_at_opening(100)
    else:
        xt = valve.get_xt_at_opening(100)
    
   # Calculate velocity
    valve_diameter_m = valve.diameter * 0.0254
    valve_area = math.pi * (valve_diameter_m/2)**2
    
    if scenario['fluid_type'] == "liquid":
        flow_m3s = scenario['flow'] / 3600
        velocity = flow_m3s / valve_area
        
    elif scenario['fluid_type'] == "gas":
        # Convert standard flow to actual flow at valve inlet conditions
        T_actual = scenario['temp'] + C_TO_K  # Actual temperature in K
        P_actual = scenario['p1'] * 1e5       # Actual pressure in Pa
        R = 8.314462618  # Universal gas constant
        
        # Standard conditions (ISA standard: 15°C, 1.01325 bar)
        T_std = 288.15  # K (15°C)
        P_std = 1.01325 * 1e5  # Pa (1 atm)
        
        # Calculate actual volumetric flow
        Q_std = scenario['flow']  # Standard volumetric flow
        Q_actual = Q_std * (P_std / P_actual) * (T_actual / T_std) * (scenario['z'] / 1.0)
        
        # Convert to m³/s and calculate velocity
        Q_actual_m3s = Q_actual / 3600
        velocity = Q_actual_m3s / valve_area
        
    elif scenario['fluid_type'] == "steam":
        volume_flow_m3h = scenario['flow'] / scenario['rho']
        volume_flow_m3s = volume_flow_m3h / 3600
        velocity = volume_flow_m3s / valve_area
    
    # Calculate required Cv
    if scenario["fluid_type"] == "liquid":
        if scenario.get('fluid_library') in FLUID_LIBRARY:
            fluid_data = FLUID_LIBRARY[scenario['fluid_library']]
            scenario["visc"] = fluid_data["visc_func"](scenario["temp"], scenario["p1"])
            scenario["pv"] = fluid_data["pv_func"](scenario["temp"], scenario["p1"])
            if "pc_func" in fluid_data:
                scenario["pc"] = fluid_data["pc_func"]()
        
        # Get Fl at operating point (will be updated in second pass)
        fl_at_op = valve.get_fl_at_opening(50)  # Initial guess at 50% opening
        
        cv_req, details = cv_liquid(
            flow=scenario["flow"],
            p1=scenario["p1"],
            p2=scenario["p2"],
            sg=scenario["sg"],
            fl_at_op=fl_at_op,
            pv=scenario["pv"],
            pc=scenario["pc"],
            visc_cst=scenario["visc"],
            d_m=valve.diameter * 0.0254,
            valve=valve,
            fp=fp
        )
        
        # First pass: calculate operating point with initial Fl
        open_percent = 10
        while open_percent <= 100:
            cv_valve = valve.get_cv_at_opening(open_percent)
            if cv_valve >= cv_req:
                break
            open_percent += 1
        
        # Second pass: recalculate with Fl at actual operating point
        fl_at_op = valve.get_fl_at_opening(open_percent)
        cv_req, details = cv_liquid(
            flow=scenario["flow"],
            p1=scenario["p1"],
            p2=scenario["p2"],
            sg=scenario["sg"],
            fl_at_op=fl_at_op,
            pv=scenario["pv"],
            pc=scenario["pc"],
            visc_cst=scenario["visc"],
            d_m=valve.diameter * 0.0254,
            valve=valve,
            fp=fp
        )
        
        # Recalculate operating point with updated cv_req
        open_percent = 10
        while open_percent <= 100:
            cv_valve = valve.get_cv_at_opening(open_percent)
            if cv_valve >= cv_req:
                break
            open_percent += 1
            
        if scenario["pc"] > 0:
            choked, sigma, km, cav_msg = check_cavitation(
                scenario["p1"], scenario["p2"], scenario["pv"], fl_at_op, scenario["pc"]
            )
            details['is_choked'] = choked
            details['cavitation_severity'] = cav_msg
        else:
            details['cavitation_severity'] = "Critical pressure not available"
        
    elif scenario["fluid_type"] == "gas":
        if scenario.get('fluid_library') in FLUID_LIBRARY:
            fluid_data = FLUID_LIBRARY[scenario['fluid_library']]
            scenario["k"] = fluid_data["k_func"](scenario["temp"], scenario["p1"])
            if "z_func" in fluid_data:
                scenario["z"] = fluid_data["z_func"](scenario["temp"], scenario["p1"])
        
        # First pass: calculate with Xt at 100% opening
        cv_req, details = cv_gas(
            flow=scenario["flow"],
            p1=scenario["p1"],
            p2=scenario["p2"],
            sg=scenario["sg"],
            t=scenario["temp"],
            k=scenario["k"],
            xt_at_op=xt,
            z=scenario["z"],
            fp=fp
        )
        
        # First pass: calculate operating point
        open_percent = 10
        while open_percent <= 100:
            cv_valve = valve.get_cv_at_opening(open_percent)
            if cv_valve >= cv_req:
                break
            open_percent += 1
        
        # Second pass: recalculate with Xt at actual operating point
        xt_at_op = valve.get_xt_at_opening(open_percent)
        if abs(pipe_d - valve_d) > 0.01:
            xt_at_op = calculate_x_tp(valve, valve_d, pipe_d, fp)
        
        cv_req, details = cv_gas(
            flow=scenario["flow"],
            p1=scenario["p1"],
            p2=scenario["p2"],
            sg=scenario["sg"],
            t=scenario["temp"],
            k=scenario["k"],
            xt_at_op=xt_at_op,
            z=scenario["z"],
            fp=fp
        )
        
        # Recalculate operating point with updated cv_req
        open_percent = 10
        while open_percent <= 100:
            cv_valve = valve.get_cv_at_opening(open_percent)
            if cv_valve >= cv_req:
                break
            open_percent += 1
            
        # Set cavitation severity based on choked flow
        if details.get('is_choked', False):
            details['cavitation_severity'] = "Choked flow detected"
        else:
            details['cavitation_severity'] = "No choked flow"
        
    else:
        if scenario.get('fluid_library') in FLUID_LIBRARY:
            fluid_data = FLUID_LIBRARY[scenario['fluid_library']]
            scenario["rho"] = fluid_data["rho_func"](scenario["temp"], scenario["p1"])
            scenario["k"] = fluid_data["k_func"](scenario["temp"], scenario["p1"])
        
        # First pass: calculate with Xt at 100% opening
        cv_req, details = cv_steam(
            flow=scenario["flow"],
            p1=scenario["p1"],
            p2=scenario["p2"],
            rho=scenario["rho"],
            k=scenario["k"],
            xt_at_op=xt,
            fp=fp
        )
        
        # First pass: calculate operating point
        open_percent = 10
        while open_percent <= 100:
            cv_valve = valve.get_cv_at_opening(open_percent)
            if cv_valve >= cv_req:
                break
            open_percent += 1
        
        # Second pass: recalculate with Xt at actual operating point
        xt_at_op = valve.get_xt_at_opening(open_percent)
        if abs(pipe_d - valve_d) > 0.01:
            xt_at_op = calculate_x_tp(valve, valve_d, pipe_d, fp)
        
        cv_req, details = cv_steam(
            flow=scenario["flow"],
            p1=scenario["p1"],
            p2=scenario["p2"],
            rho=scenario["rho"],
            k=scenario["k"],
            xt_at_op=xt_at_op,
            fp=fp
        )
        
        # Recalculate operating point with updated cv_req
        open_percent = 10
        while open_percent <= 100:
            cv_valve = valve.get_cv_at_opening(open_percent)
            if cv_valve >= cv_req:
                break
            open_percent += 1
            
        # Set cavitation severity based on choked flow
        if details.get('is_choked', False):
            details['cavitation_severity'] = "Choked flow detected"
        else:
            details['cavitation_severity'] = "No choked flow"
    
    if 'error' in details:
        return {
            "op_point": 0,
            "req_cv": 0,
            "theoretical_cv": 0,
            "warning": details['error'],
            "cavitation_info": "N/A",
            "status": "red",
            "margin": 0,
            "details": details,
            "velocity": velocity
        }
    
    warn = ""
    # Check for insufficient capacity
    if open_percent >= 100 and cv_valve < cv_req:
        warn = "Insufficient Capacity – Valve is undersized"
        status = "red"
    elif open_percent < 20:
        warn = "Low opening (<20%)"
        status = "yellow"
    elif open_percent > 80:
        warn = "High opening (>80%)"
        status = "yellow"
    else:
        status = "green"
    
    # Override status based on flow conditions
    if details.get('is_choked', False):
        status = "red"
    elif "Severe" in details.get('cavitation_severity', ""):
        status = "orange"
    elif "Moderate" in details.get('cavitation_severity', ""):
        status = "yellow"
    
    return {
        "op_point": open_percent,
        "req_cv": cv_req,
        "theoretical_cv": details.get('theoretical_cv', 0),
        "warning": warn,
        "cavitation_info": details.get('cavitation_severity', "N/A"),
        "status": status,
        "margin": (cv_valve / cv_req - 1) * 100 if cv_req > 0 else 0,
        "details": details,
        "velocity": velocity
    }

def find_recommended_valve(scenarios):
    best_valve = None
    best_score = float('-inf')
    all_valve_results = []
    
    for valve in VALVE_DATABASE:
        valve_results = []
        valve_score = 0
        is_suitable = True
        
        for scenario in scenarios:
            result = evaluate_valve_for_scenario(valve, scenario)
            valve_results.append(result)
            
            if result["status"] == "red":
                valve_score -= 100
                is_suitable = False
            elif result["status"] == "orange":
                valve_score -= 50
            elif result["status"] == "yellow":
                valve_score -= 10
            else:
                valve_score += 20
                
            valve_score += max(0, 10 - abs(result["op_point"] - 50)/5)
        
        valve_display_name = get_valve_display_name(valve)
        all_valve_results.append({
            "valve": valve,
            "results": valve_results,
            "score": valve_score,
            "display_name": valve_display_name
        })
        
        if is_suitable and valve_score > best_score:
            best_valve = {
                "valve": valve,
                "results": valve_results,
                "score": valve_score,
                "display_name": valve_display_name
            }
            best_score = valve_score
    
    if best_valve is None and all_valve_results:
        all_valve_results.sort(key=lambda x: x["score"], reverse=True)
        best_valve = all_valve_results[0]
    
    return best_valve, all_valve_results

# ========================
# STREAMLIT APPLICATION
# ========================
def get_valve_display_name(valve):
    rating_code_map = {
        150: 1,
        300: 2,
        600: 3,
        900: 4,
        1500: 5,
        2500: 6
    }
    rating_code = rating_code_map.get(valve.rating_class, valve.rating_class)
    return f"{valve.size}\" E{valve.valve_type}{rating_code}"

def create_valve_dropdown():
    valves = sorted(st.session_state.valve_database, key=lambda v: (v.size, v.rating_class, v.valve_type))
    valve_options = {get_valve_display_name(v): v for v in valves}
    return valve_options

def create_fluid_dropdown():
    return ["Select Fluid Library..."] + list(FLUID_LIBRARY.keys())

def scenario_input_form(scenario_num, scenario_data=None):
    default_values = {
        "use_valve_size": True,  # Default to True
        "sg": 1.0,
        "visc": 1.0,
        "pv": 0.023,
        "pc": 220.55,
        "k": 1.4,
        "z": 1.0,
        "rho": 1.0,
        "fluid_type": "liquid",
        "pipe_d": 2.0  # Default pipe diameter
    }
    
    if scenario_data is None:
        scenario_data = {
            "name": f"Scenario {scenario_num}",
            "fluid_type": "liquid",
            "flow": 10.0 if scenario_num == 1 else 50.0,
            "p1": 10.0,
            "p2": 6.0,
            "temp": 20.0,
            "pipe_d": 2.0
        }
        scenario_data = {**default_values, **scenario_data}
    else:
        for key, default in default_values.items():
            if key not in scenario_data:
                scenario_data[key] = default
    
    st.subheader(f"Scenario {scenario_num}")
    scenario_name = st.text_input("Scenario Name", value=scenario_data["name"], key=f"name_{scenario_num}")
    
    col1, col2 = st.columns(2)
    with col1:
        fluid_library = st.selectbox(
            "Fluid Library", 
            create_fluid_dropdown(), 
            key=f"fluid_library_{scenario_num}"
        )
    
    with col2:
        if fluid_library != "Select Fluid Library...":
            fluid_data = FLUID_LIBRARY[fluid_library]
            fluid_type = fluid_data["type"]
            st.text_input("Fluid Type", value=fluid_type.capitalize(), disabled=True, key=f"fluid_type_text_{scenario_num}")
        else:
            try:
                index_val = ["Liquid", "Gas", "Steam"].index(scenario_data["fluid_type"].capitalize())
            except (ValueError, AttributeError):
                index_val = 0
            fluid_type = st.selectbox(
                "Fluid Type", 
                ["Liquid", "Gas", "Steam"], 
                index=index_val,
                key=f"fluid_type_{scenario_num}"
            ).lower()
    
    col1, col2 = st.columns(2)
    with col1:
        flow_label = "Flow Rate (m³/h)" if fluid_type == "liquid" else "Flow Rate (std m³/h)" if fluid_type == "gas" else "Flow Rate (kg/h)"
        flow_value = st.number_input(
            flow_label, 
            min_value=0.0, 
            max_value=1000000.0, 
            value=scenario_data["flow"], 
            step=0.1,
            key=f"flow_{scenario_num}"
        )
        p1 = st.number_input(
            "Inlet Pressure (bar a)", 
            min_value=0.0, 
            max_value=1000.0, 
            value=scenario_data["p1"], 
            step=0.1,
            key=f"p1_{scenario_num}"
        )
        p2 = st.number_input(
            "Outlet Pressure (bar a)", 
            min_value=0.0, 
            max_value=1000.0, 
            value=scenario_data["p2"], 
            step=0.1,
            key=f"p2_{scenario_num}"
        )
        temp = st.number_input(
            "Temperature (°C)", 
            min_value=-200.0, 
            max_value=1000.0, 
            value=scenario_data["temp"], 
            step=1.0,
            key=f"temp_{scenario_num}"
        )
    
    with col2:
        if fluid_library != "Select Fluid Library...":
            fluid_data = FLUID_LIBRARY[fluid_library]
            scenario_data["fluid_type"] = fluid_data["type"]
            if fluid_data.get("visc_func") and fluid_data["type"] == "liquid":
                scenario_data["visc"] = fluid_data["visc_func"](temp, p1)
            if fluid_data.get("k_func") and fluid_data["type"] in ["gas", "steam"]:
                scenario_data["k"] = fluid_data["k_func"](temp, p1)
            if fluid_data.get("pv_func") and fluid_data["type"] == "liquid":
                scenario_data["pv"] = fluid_data["pv_func"](temp, p1)
            if fluid_data.get("pc_func") and fluid_data["type"] == "liquid":
                scenario_data["pc"] = fluid_data["pc_func"]()
            if fluid_data.get("rho_func") and fluid_data["type"] == "steam":
                scenario_data["rho"] = fluid_data["rho_func"](temp, p1)
            if fluid_data.get("z_func") and fluid_data["type"] == "gas":
                scenario_data["z"] = fluid_data["z_func"](temp, p1)
            if fluid_data.get("sg") is not None:
                scenario_data["sg"] = fluid_data["sg"]
        
        if fluid_type in ["liquid", "gas"]:
            sg = st.number_input(
                "Specific Gravity (water=1)" if fluid_type == "liquid" else "Specific Gravity (air=1)",
                min_value=0.01, 
                max_value=10.0, 
                value=scenario_data["sg"], 
                step=0.01,
                key=f"sg_{scenario_num}",
                disabled=(fluid_library != "Select Fluid Library...")
            )
        
        if fluid_type == "liquid":
            visc = st.number_input(
                "Viscosity (cSt)", 
                min_value=0.01, 
                max_value=10000.0, 
                value=scenario_data["visc"], 
                step=0.1,
                key=f"visc_{scenario_num}",
                disabled=(fluid_library != "Select Fluid Library...")
            )
            pv = st.number_input(
                "Vapor Pressure (bar a)", 
                min_value=0.0, 
                max_value=100.0, 
                value=scenario_data["pv"], 
                step=0.0001,
                format="%.4f",
                key=f"pv_{scenario_num}",
                disabled=(fluid_library != "Select Fluid Library...")
            )
            pc = st.number_input(
                "Critical Pressure (bar a)", 
                min_value=0.0, 
                max_value=1000.0, 
                value=scenario_data["pc"], 
                step=0.1,
                key=f"pc_{scenario_num}",
                disabled=(fluid_library != "Select Fluid Library...")
            )
        
        if fluid_type in ["gas", "steam"]:
            k = st.number_input(
                "Specific Heat Ratio (k=Cp/Cv)", 
                min_value=1.0, 
                max_value=2.0, 
                value=scenario_data["k"], 
                step=0.01,
                key=f"k_{scenario_num}",
                disabled=(fluid_library != "Select Fluid Library...")
            )
        
        if fluid_type == "gas":
            # Show Z as disabled if fluid library is selected, else editable
            if fluid_library != "Select Fluid Library...":
                z = st.number_input(
                    "Compressibility Factor (Z)", 
                    min_value=0.1, 
                    max_value=2.0, 
                    value=scenario_data["z"], 
                    step=0.01,
                    key=f"z_{scenario_num}",
                    disabled=True  # Disabled for library fluids
                )
            else:
                z = st.number_input(
                    "Compressibility Factor (Z)", 
                    min_value=0.1, 
                    max_value=2.0, 
                    value=scenario_data["z"], 
                    step=0.01,
                    key=f"z_{scenario_num}",
                    disabled=False
                )
        
        if fluid_type == "steam":
            rho = st.number_input(
                "Density (kg/m³)", 
                min_value=0.01, 
                max_value=2000.0, 
                value=scenario_data["rho"], 
                step=0.1,
                key=f"rho_{scenario_num}",
                disabled=(fluid_library != "Select Fluid Library...")
            )
        
        # Add checkbox for pipe size option - default to True
        use_valve_size = st.checkbox(
            "Use valve size for pipe diameter?",
            value=scenario_data.get("use_valve_size", True),
            key=f"use_valve_size_{scenario_num}"
        )
        
        # Only show pipe diameter input if not using valve size
        if not use_valve_size:
            pipe_d = st.number_input(
                "Pipe Diameter (inch)", 
                min_value=0.1, 
                max_value=100.0, 
                value=scenario_data["pipe_d"], 
                step=0.1,
                key=f"pipe_d_{scenario_num}"
            )
        else:
            pipe_d = scenario_data["pipe_d"]  # Keep existing value but won't be used
    
    return {
        "name": scenario_name,
        "fluid_type": fluid_type,
        "flow": flow_value,
        "p1": p1,
        "p2": p2,
        "temp": temp,
        "sg": sg if fluid_type in ["liquid", "gas"] else scenario_data["sg"],
        "visc": visc if fluid_type == "liquid" else scenario_data["visc"],
        "pv": pv if fluid_type == "liquid" else scenario_data["pv"],
        "pc": pc if fluid_type == "liquid" else scenario_data["pc"],
        "k": k if fluid_type in ["gas", "steam"] else scenario_data["k"],
        "z": z if fluid_type == "gas" else scenario_data["z"],
        "rho": rho if fluid_type == "steam" else scenario_data["rho"],
        "pipe_d": pipe_d,
        "use_valve_size": use_valve_size,
        "fluid_library": fluid_library
    }

def plot_cv_curve(valve, op_points, req_cvs, theoretical_cvs, scenario_names):
    # Get valve's Cv characteristics
    openings = sorted(valve.cv_table.keys())
    cv_values = np.array([valve.get_cv_at_opening(op) for op in openings])
    
    # Create cubic spline interpolation for smooth curve
    cs = CubicSpline(openings, cv_values)
    x_smooth = np.linspace(0, 100, 300)
    y_smooth = cs(x_smooth)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_smooth, 
        y=y_smooth, 
        mode='lines',
        name='Valve Cv',
        line=dict(color='blue', width=3)
    ))
    
    # Add actual CV points from valve table (for reference)
    fig.add_trace(go.Scatter(
        x=openings,
        y=cv_values,
        mode='markers',
        name='Actual CV Points',
        marker=dict(size=8, color='black', symbol='x'),
        showlegend=True
    ))
    
    for i, (op, req_cv, theoretical_cv) in enumerate(zip(op_points, req_cvs, theoretical_cvs)):
        actual_cv = valve.get_cv_at_opening(op)
        fig.add_trace(go.Scatter(
            x=[op], 
            y=[actual_cv], 
            mode='markers+text',
            name=f'Scenario {i+1} Operating Point',
            marker=dict(size=12, color='red'),
            text=[f'S{i+1}'],
            textposition="top center"
        ))
        fig.add_trace(go.Scatter(
            x=[0, 100],
            y=[req_cv, req_cv],
            mode='lines',
            line=dict(color='red', dash='dash', width=1),
            name=f'Corrected Cv S{i+1}',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[0, 100],
            y=[theoretical_cv, theoretical_cv],
            mode='lines',
            line=dict(color='green', dash='dot', width=1),
            name=f'Theoretical Cv S{i+1}',
            showlegend=False
        ))
    
    for i, (req_cv, theoretical_cv) in enumerate(zip(req_cvs, theoretical_cvs)):
        fig.add_annotation(
            x=100,
            y=req_cv,
            text=f'Corrected S{i+1}: {req_cv:.1f}',
            showarrow=False,
            xshift=-10,
            yshift=10,
            align='right',
            font=dict(color='red')
        )
        fig.add_annotation(
            x=100,
            y=theoretical_cv,
            text=f'Theoretical S{i+1}: {theoretical_cv:.1f}',
            showarrow=False,
            xshift=-10,
            yshift=-10,
            align='right',
            font=dict(color='green')
        )
    
    fig.update_layout(
        title=f'{valve.size}" Valve Cv Characteristic',
        xaxis_title='Opening Percentage (%)',
        yaxis_title='Cv Value',
        legend_title='Legend',
        hovermode='x unified',
        height=600,
        template='plotly_white'
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_xaxes(range=[0, 100])
    fig.update_yaxes(range=[0, max(cv_values) * 1.1])
    return fig

def valve_3d_viewer(valve_name, model_url):
    html_code = f"""
    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
    <model-viewer src="{model_url}"
                  alt="{valve_name}"
                  auto-rotate
                  camera-controls
                  autoplay
                  animation-name="valve_animation"
                  style="width: 100%; height: 1000px;">
    </model-viewer>
    """
    components.html(html_code, height=1000)

def main():
    # Global değişkeni kullanacağımızı belirtiyoruz
    global VALVE_DATABASE
    st.set_page_config(
        page_title="Control Valve Sizing",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        html {
            font-size: 18px;
        }
        .stApp {
            background-color: #f0f2f6;
        }
        .block-container {
            padding-top: 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 15px 25px;
            border-radius: 10px 10px 0 0;
            font-size: 18px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1f77b4;
            color: white;
        }
        .stButton button {
            width: 100%;
            font-weight: bold;
            font-size: 18px;
        }
        .result-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            font-size: 18px;
        }
        .warning-card {
            background-color: #fff3cd;
            border-left: 5px solid #ffc107;
        }
        .success-card {
            background-color: #d4edda;
            border-left: 5px solid #28a745;
        }
        .danger-card {
            background-color: #f8d7da;
            border-left: 5px solid #dc3545;
        }
        .insufficient-card {
            background-color: #f8d7da;
            border-left: 5px solid #8b0000; /* Dark red for insufficient capacity */
        }
        .cavitation-card {
            background-color: #ffe8cc;
            border-left: 5px solid #fd7e14;
        }
        .logo-container {
            display: flex;
            justify-content: center;
            padding: 10px 0;
        }
        .simulation-modal {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            z-index: 1000;
            width: 80%;
            max-width: 900px;
            max-height: 80vh;
            overflow: auto;
        }
        .modal-backdrop {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.5);
            z-index: 999;
        }
        .stMetric {
            font-size: 20px !important;
        }
        .stNumberInput, .stTextInput, .stSelectbox {
            font-size: 18px;
        }
        .stMarkdown {
            font-size: 18px;
        }
        .valve-table {
            width: 100%;
            border-collapse: collapse;
        }
        .valve-table th {
            background-color: #2c3e50;
            color: white;
            padding: 10px;
            text-align: center;
        }
        .valve-table td {
            padding: 8px;
            text-align: center;
            border: 1px solid #ddd;
        }
        .status-green {
            background-color: #d4edda;
        }
        .status-yellow {
            background-color: #fff3cd;
        }
        .status-orange {
            background-color: #ffe8cc;
        }
        .status-red {
            background-color: #f8d7da;
        }
        .status-insufficient {
            background-color: #f8d7da;
            border: 2px solid #8b0000; /* Dark red border for insufficient capacity */
        }
        </style>
    """, unsafe_allow_html=True)
    if 'valve_database' not in st.session_state:
        st.session_state.valve_database =   VALVE_DATABASE
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'valve' not in st.session_state:
        st.session_state.valve = None
    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = None
    if 'logo_bytes' not in st.session_state:
        st.session_state.logo_bytes = None
    if 'logo_type' not in st.session_state:
        st.session_state.logo_type = None
    if 'show_simulation' not in st.session_state:
        st.session_state.show_simulation = False
    if 'show_3d_viewer' not in st.session_state:
        st.session_state.show_3d_viewer = False
    if 'recommended_valve' not in st.session_state:
        st.session_state.recommended_valve = None
    if 'all_valve_results' not in st.session_state:
        st.session_state.all_valve_results = None
    
    col1, col2 = st.columns([1, 4])
    with col1:
        default_logo = "logo.png"
        if os.path.exists(default_logo):
            st.image(default_logo, width=100)
        else:
            st.image("https://via.placeholder.com/100x100?text=LOGO", width=100)
    with col2:
        st.title("Control Valve Sizing Program")
        st.markdown("**ISA/IEC Standards Compliant Valve Sizing with Enhanced Visualization**")
    
    with st.sidebar:
        st.header("VASTAŞ Logo")
        logo_upload = st.file_uploader("Upload VASTAŞ logo", type=["png", "jpg", "jpeg"], key="logo_uploader")
        if logo_upload is not None:
            st.session_state.logo_bytes = logo_upload.getvalue()
            st.session_state.logo_type = "PNG"
            st.success("Logo uploaded successfully!")
        if st.session_state.logo_bytes:
            st.image(Image.open(BytesIO(st.session_state.logo_bytes)), use_container_width=True)
        elif os.path.exists("logo.png"):
            st.image(Image.open("logo.png"), use_container_width=True)
        else:
            st.image("https://via.placeholder.com/300x100?text=VASTAŞ+Logo", use_container_width=True)
        
        st.header("Valve Selection")
        valve_options = create_valve_dropdown()
        selected_valve_name = st.selectbox("Select Valve", list(valve_options.keys()))
        selected_valve = valve_options[selected_valve_name]
        
        st.header("Actions")
        calculate_btn = st.button("Calculate Opening", type="primary", use_container_width=True)
        export_btn = st.button("Export PDF Report", use_container_width=True)
        view_3d_btn = st.button("View 3D Model", use_container_width=True)
        show_simulation_btn = st.button("Show Simulation Results", use_container_width=True)
        
        st.header("Valve Details")
        st.markdown(f"**Size:** {selected_valve.size}\"")
        st.markdown(f"**Type:** {'Globe' if selected_valve.valve_type == 3 else 'Axial'}")
        st.markdown(f"**Rating Class:** {selected_valve.rating_class}")
        st.markdown(f"**Fl (Liquid Recovery):** {selected_valve.get_fl_at_opening(100):.3f}")
        st.markdown(f"**Xt (Pressure Drop Ratio):** {selected_valve.get_xt_at_opening(100):.3f}")
        st.markdown(f"**Fd (Style Modifier):** {selected_valve.fd:.2f}")
        st.markdown(f"**Internal Diameter:** {selected_valve.diameter:.2f} in")
        
        st.subheader("Cv Characteristics")
        cv_data = {"Opening %": list(selected_valve.cv_table.keys()), "Cv": list(selected_valve.cv_table.values())}
        cv_df = pd.DataFrame(cv_data)
        st.dataframe(cv_df, hide_index=True, height=300)
        
        st.subheader("Fl Characteristics")
        fl_data = {"Opening %": list(selected_valve.fl_table.keys()), "Fl": list(selected_valve.fl_table.values())}
        fl_df = pd.DataFrame(fl_data)
        st.dataframe(fl_df, hide_index=True, height=300)
        
        st.subheader("Xt Characteristics")
        xt_data = {"Opening %": list(selected_valve.xt_table.keys()), "Xt": list(selected_valve.xt_table.values())}
        xt_df = pd.DataFrame(xt_data)
        st.dataframe(xt_df, hide_index=True, height=300)
    
    if view_3d_btn:
        st.session_state.show_3d_viewer = True
        st.session_state.show_simulation = False
    if show_simulation_btn:
        st.session_state.show_simulation = True
        st.session_state.show_3d_viewer = False
    
    with st.sidebar:
        st.header("Valve Management")
        
        if st.button("Reload Valve Database"):
            VALVE_DATABASE = load_valves_from_excel()
            st.session_state.valve_database = VALVE_DATABASE
            st.success("Valve database reloaded!")
        
        with st.expander("Add New Valve"):
            size = st.number_input("Size (inch)", min_value=0.5, step=0.5)
            rating_class = st.selectbox("Rating Class", [150, 300, 600, 900, 1500])
            valve_type = st.selectbox("Valve Type", [3, 4], format_func=lambda x: "Globe" if x == 3 else "Axial")
            fd = st.number_input("Fd", value=1.0)
            diameter = st.number_input("Diameter (inch)", min_value=0.1)
            
            st.subheader("Valve Characteristics Tables")
            
            # Default values for each characteristic
            default_openings = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            default_cv = [0.0, 3.28, 7.39, 12.0, 14.2, 14.9, 15.3, 15.7, 16.0, 16.4, 16.8]
            default_fl = [0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68]
            default_xt = [0.581, 0.605, 0.617, 0.644, 0.764, 0.790, 0.809, 0.813, 0.795, 0.768, 0.74]
            
            # Create editable dataframes
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Cv Table**")
                cv_data = pd.DataFrame({
                    "Opening (%)": default_openings,
                    "Cv": default_cv
                })
                cv_edited = st.data_editor(
                    cv_data,
                    num_rows="dynamic",
                    hide_index=True,
                    use_container_width=True,
                    key=f"cv_editor_{size}_{rating_class}_{valve_type}"
                )
            
            with col2:
                st.markdown("**Fl Table**")
                fl_data = pd.DataFrame({
                    "Opening (%)": default_openings,
                    "Fl": default_fl
                })
                fl_edited = st.data_editor(
                    fl_data,
                    num_rows="dynamic",
                    hide_index=True,
                    use_container_width=True,
                    key=f"fl_editor_{size}_{rating_class}_{valve_type}"
                )
            
            with col3:
                st.markdown("**Xt Table**")
                xt_data = pd.DataFrame({
                    "Opening (%)": default_openings,
                    "Xt": default_xt
                })
                xt_edited = st.data_editor(
                    xt_data,
                    num_rows="dynamic",
                    hide_index=True,
                    use_container_width=True,
                    key=f"xt_editor_{size}_{rating_class}_{valve_type}"
                )
            
            if st.button("Add Valve to Database"):
                try:
                    # Convert edited data to dictionaries
                    cv_dict = {}
                    for _, row in cv_edited.iterrows():
                        opening = int(row['Opening (%)'])
                        cv = float(row['Cv'])
                        cv_dict[opening] = cv
                    
                    fl_dict = {}
                    for _, row in fl_edited.iterrows():
                        opening = int(row['Opening (%)'])
                        fl = float(row['Fl'])
                        fl_dict[opening] = fl
                    
                    xt_dict = {}
                    for _, row in xt_edited.iterrows():
                        opening = int(row['Opening (%)'])
                        xt = float(row['Xt'])
                        xt_dict[opening] = xt
                    
                    new_valve = Valve(
                        size_inch=size,
                        rating_class=rating_class,
                        cv_table=cv_dict,
                        fl_table=fl_dict,
                        xt_table=xt_dict,
                        fd=fd,
                        d_inch=diameter,
                        valve_type=valve_type
                    )
                    add_valve_to_database(new_valve)
                    VALVE_DATABASE = load_valves_from_excel()
                    st.session_state.valve_database = VALVE_DATABASE
                    st.success("Valve added to database!")
                except Exception as e:
                    st.error(f"Error adding valve: {str(e)}")
        
        with st.expander("Delete Valve"):
            valve_options = {get_valve_display_name(v): v for v in VALVE_DATABASE}
            valve_to_delete = st.selectbox("Select Valve to Delete", list(valve_options.keys()))
            
            if st.button("Delete Valve"):
                valve = valve_options[valve_to_delete]
                delete_valve_from_database(valve.size, valve.rating_class, valve.valve_type)
                VALVE_DATABASE = load_valves_from_excel()
                st.session_state.valve_database = VALVE_DATABASE
                st.success("Valve deleted from database!")
    
    tab1, tab2, tab3, tab_results = st.tabs(["Scenario 1", "Scenario 2", "Scenario 3", "Results"])
    
    with tab1:
        scenario1 = scenario_input_form(1)
    with tab2:
        scenario2 = scenario_input_form(2)
    with tab3:
        scenario3 = scenario_input_form(3)
    
    scenarios = []
    if scenario1["flow"] > 0:
        scenarios.append(scenario1)
    if scenario2["flow"] > 0:
        scenarios.append(scenario2)
    if scenario3["flow"] > 0:
        scenarios.append(scenario3)
    st.session_state.scenarios = scenarios
    
    if calculate_btn:
        if not scenarios:
            st.error("Please define at least one scenario with flow > 0.")
            st.stop()
        try:
            selected_valve_results = []
            for scenario in scenarios:
                result = evaluate_valve_for_scenario(selected_valve, scenario)
                selected_valve_results.append(result)
            recommended_valve, all_valve_results = find_recommended_valve(scenarios)
            st.session_state.results = {
                "selected_valve": selected_valve,
                "selected_valve_results": selected_valve_results,
                "recommended_valve": recommended_valve,
                "all_valve_results": all_valve_results
            }
        except Exception as e:
            st.error(f"Calculation error: {str(e)}")
            st.error(traceback.format_exc())
    
    with tab_results:
        if st.session_state.results:
            results = st.session_state.results
            selected_valve = results["selected_valve"]
            selected_valve_results = results["selected_valve_results"]
            recommended_valve = results["recommended_valve"]
            
            if recommended_valve:
                st.subheader("Recommended Valve")
                st.markdown(f"**{recommended_valve['display_name']}** - Score: {recommended_valve['score']:.1f}")
                
                # Show each scenario result for recommended valve in card layout
                for i, scenario in enumerate(scenarios):
                    result = recommended_valve["results"][i]
                    actual_cv = recommended_valve["valve"].get_cv_at_opening(result["op_point"])
                    
                    # Determine status class
                    status_class = ""
                    if "Insufficient" in result["warning"]:
                        status_class = "insufficient-card"
                    elif result["status"] == "green":
                        status_class = "success-card"
                    elif result["status"] == "yellow":
                        status_class = "warning-card"
                    elif result["status"] == "orange":
                        status_class = "cavitation-card"
                    elif result["status"] == "red":
                        status_class = "danger-card"
                    
                    # Combine warnings
                    warn_msgs = []
                    if result["warning"]:
                        warn_msgs.append(result["warning"])
                    if result["cavitation_info"]:
                        warn_msgs.append(result["cavitation_info"])
                    warn_text = ", ".join(warn_msgs)
                    
                    with st.container():
                        st.markdown(f"<div class='result-card {status_class}'>", unsafe_allow_html=True)
                        cols = st.columns([1.8, 1, 1, 1, 1, 1, 1, 1.5])
                        cols[0].markdown(f"**{scenario['name']}**")
                        cols[1].metric("Req Cv", f"{result['req_cv']:.1f}")
                        cols[2].metric("Theo Cv", f"{result['theoretical_cv']:.2f}")
                        cols[3].metric("Valve Cv", f"{actual_cv:.1f}")
                        cols[4].metric("Valve Size", f"{recommended_valve['valve'].size}\"")
                        cols[5].metric("Opening", f"{result['op_point']:.1f}%")
                        cols[6].metric("Margin", f"{result['margin']:.1f}%", 
                                      delta_color="inverse" if result['margin'] < 0 else "normal")
                        cols[7].markdown(f"**{warn_text}**")
                        st.markdown("</div>", unsafe_allow_html=True)
                
                with st.expander("Why this valve is recommended"):
                    st.markdown("""
                    - **Green status**: Optimal performance (good operating range, no cavitation)
                    - **Yellow status**: Acceptable but suboptimal (moderate cavitation or bad opening)
                    - **Orange status**: Severe cavitation risk
                    - **Red status**: Choked flow (unacceptable)
                    - **Dark red**: Insufficient capacity (valve undersized)
                    """)
                    st.markdown(f"""
                    **Selection criteria**:
                    - Highest overall score considering operating point, cavitation risk, and flow conditions
                    - Valve size: {recommended_valve['valve'].size}\"
                    - Valve type: {'Globe' if recommended_valve['valve'].valve_type == 3 else 'Axial'}
                    """)
            else:
                st.warning("No suitable valve found for all scenarios. Consider modifying your scenarios.")
            
            st.subheader(f"Selected Valve: {get_valve_display_name(selected_valve)} Cv Characteristic")
            fig = plot_cv_curve(
                selected_valve, 
                [r["op_point"] for r in selected_valve_results],
                [r["req_cv"] for r in selected_valve_results],
                [r["theoretical_cv"] for r in selected_valve_results],
                [s["name"] for s in scenarios]
            )
            st.plotly_chart(fig, use_container_width=True, key="main_cv_curve")
            
            st.subheader("Selected Valve Performance")
            for i, scenario in enumerate(scenarios):
                result = selected_valve_results[i]
                actual_cv = selected_valve.get_cv_at_opening(result["op_point"])
                status = "success-card"
                if "Insufficient" in result["warning"]:
                    status = "insufficient-card"
                elif result["status"] == "yellow":
                    status = "warning-card"
                elif result["status"] == "orange":
                    status = "cavitation-card"
                elif result["status"] == "red":
                    status = "danger-card"
                warn_msgs = []
                if result["warning"]:
                    warn_msgs.append(result["warning"])
                if result["cavitation_info"]:
                    warn_msgs.append(result["cavitation_info"])
                warn_text = ", ".join(warn_msgs)
                
                with st.container():
                    st.markdown(f"<div class='result-card {status}'>", unsafe_allow_html=True)
                    cols = st.columns([1.8, 1, 1, 1, 1, 1, 1, 1.5])
                    cols[0].markdown(f"**{scenario['name']}**")
                    cols[1].metric("Req Cv", f"{result['req_cv']:.1f}")
                    cols[2].metric("Theo Cv", f"{result['theoretical_cv']:.2f}")
                    cols[3].metric("Valve Cv", f"{actual_cv:.1f}")
                    cols[4].metric("Valve Size", f"{selected_valve.size}\"")
                    cols[5].metric("Opening", f"{result['op_point']:.1f}%")
                    cols[6].metric("Margin", f"{result['margin']:.1f}%", 
                                  delta_color="inverse" if result['margin'] < 0 else "normal")
                    cols[7].markdown(f"**{warn_text}**")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    with st.expander(f"Detailed Calculations for {scenario['name']}"):
                        st.subheader("Calculation Parameters")
                        col1, col2 = st.columns(2)
                        with col1:
                            # Fixed Fl display
                            fl_at_op = result['details'].get('fl_at_op', 'N/A')
                            if isinstance(fl_at_op, (int, float)):
                                st.markdown(f"**Fl (Liquid Recovery):** {fl_at_op:.3f}")
                            else:
                                st.markdown(f"**Fl (Liquid Recovery):** {fl_at_op}")
                                
                            st.markdown(f"**Fd (Valve Style Modifier):** {selected_valve.fd:.2f}")
                            
                            # Fixed Fp display
                            fp_val = result['details'].get('fp', 1.0)
                            if isinstance(fp_val, (int, float)):
                                st.markdown(f"**Fp (Piping Factor):** {fp_val:.4f}")
                            else:
                                st.markdown(f"**Fp (Piping Factor):** {fp_val}")
                            
                        with col2:
                            if scenario["fluid_type"] == "liquid":
                                # Fixed FF display
                                ff_val = result['details'].get('ff', 0.96)
                                if isinstance(ff_val, (int, float)):
                                    st.markdown(f"**FF (Critical Pressure Ratio):** {ff_val:.4f}")
                                else:
                                    st.markdown(f"**FF (Critical Pressure Ratio):** {ff_val}")
                                    
                                # Fixed Fr display
                                fr_val = result['details'].get('fr', 1.0)
                                if isinstance(fr_val, (int, float)):
                                    st.markdown(f"**Fr (Viscosity Correction):** {fr_val:.4f}")
                                else:
                                    st.markdown(f"**Fr (Viscosity Correction):** {fr_val}")
                                
                                # Fixed Reynolds display
                                reynolds_val = result['details'].get('reynolds', 0)
                                if isinstance(reynolds_val, (int, float)):
                                    st.markdown(f"**Reynolds Number:** {reynolds_val:.0f}")
                                else:
                                    st.markdown(f"**Reynolds Number:** {reynolds_val}")
                        
                        # Fixed ΔPmax display
                        dp_max_val = result['details'].get('dp_max', 0)
                        if isinstance(dp_max_val, (int, float)):
                            st.markdown(f"**Max Pressure Drop (ΔPmax):** {dp_max_val:.2f} bar")
                        else:
                            st.markdown(f"**Max Pressure Drop (ΔPmax):** {dp_max_val} bar")
                            
                        # Fixed velocity display
                        velocity_val = result.get('velocity', 0)
                        if isinstance(velocity_val, (int, float)):
                            st.markdown(f"**Average Velocity in Valve:** {velocity_val:.2f} m/s")
                        else:
                            st.markdown(f"**Average Velocity in Valve:** {velocity_val} m/s")
                        
                        if scenario["fluid_type"] == "liquid":
                            if result["details"].get('cavitation_severity'):
                                st.subheader("Cavitation Analysis")
                                st.markdown(f"**Status:** {result['details']['cavitation_severity']}")
                                
                                # Fixed sigma display
                                sigma_val = result['details'].get('sigma', 0)
                                if isinstance(sigma_val, (int, float)):
                                    st.markdown(f"**Sigma (σ):** {sigma_val:.2f}")
                                else:
                                    st.markdown(f"**Sigma (σ):** {sigma_val}")
                                
                                # Fixed Km display
                                km_val = result['details'].get('km', 0)
                                if isinstance(km_val, (int, float)):
                                    st.markdown(f"**Km (Valve Recovery Coefficient):** {km_val:.2f}")
                                else:
                                    st.markdown(f"**Km (Valve Recovery Coefficient):** {km_val}")
                        
                        if scenario["fluid_type"] in ["gas", "steam"]:
                            st.subheader("Choked Flow Analysis")
                            st.markdown(f"**Status:** {result['cavitation_info']}")
                            
                            # Fixed x_actual display
                            x_actual_val = result['details'].get('x_actual', 0)
                            if isinstance(x_actual_val, (int, float)):
                                st.markdown(f"**Pressure Drop Ratio (x):** {x_actual_val:.4f}")
                            else:
                                st.markdown(f"**Pressure Drop Ratio (x):** {x_actual_val}")
                            
                            # Fixed x_crit display
                            x_crit_val = result['details'].get('x_crit', 0)
                            if isinstance(x_crit_val, (int, float)):
                                st.markdown(f"**Critical Pressure Drop Ratio (x_crit):** {x_crit_val:.4f}")
                            else:
                                st.markdown(f"**Critical Pressure Drop Ratio (x_crit):** {x_crit_val}")
                            
                            # Fixed xt_at_op display
                            xt_at_op_val = result['details'].get('xt_at_op', 0)
                            if isinstance(xt_at_op_val, (int, float)):
                                st.markdown(f"**Pressure Drop Ratio Factor (xT or xTP):** {xt_at_op_val:.4f}")
                            else:
                                st.markdown(f"**Pressure Drop Ratio Factor (xT or xTP):** {xt_at_op_val}")
                            
                            # Fixed choked pressure drop display
                            if isinstance(x_crit_val, (int, float)) and isinstance(scenario['p1'], (int, float)):
                                choked_dp = x_crit_val * scenario['p1']
                                st.markdown(f"**Choked Pressure Drop:** {choked_dp:.2f} bar")
                            else:
                                st.markdown("**Choked Pressure Drop:** N/A")
                        
                        st.subheader("Flow Rate vs Pressure Drop")
                        flow_fig = generate_flow_vs_dp_graph(
                            scenario,
                            selected_valve,
                            result["op_point"],
                            result["details"],
                            result["req_cv"]
                        )
                        st.plotly_chart(flow_fig, use_container_width=True, key=f"flow_dp_{i}")
            
            st.subheader("All Valves Evaluation")
            st.markdown("""
            **Status colors**:
            - <span style="background-color:#d4edda; padding:2px 5px;">Green</span>: Optimal
            - <span style="background-color:#fff3cd; padding:2px 5px;">Yellow</span>: Warning (moderate issue)
            - <span style="background-color:#ffe8cc; padding:2px 5px;">Orange</span>: Severe cavitation
            - <span style="background-color:#f8d7da; padding:2px 5px;">Red</span>: Choked flow (unacceptable)
            - <span style="background-color:#f8d7da; border:2px solid #8b0000; padding:2px 5px;">Dark Red</span>: Insufficient capacity
            """, unsafe_allow_html=True)
            all_valves_table_html = """
            <table class="valve-table">
                <thead>
                    <tr>
                        <th>Valve</th>
            """
            for i, scenario in enumerate(scenarios):
                all_valves_table_html += f'<th>{scenario["name"]} Status</th>'
            all_valves_table_html += """
                        <th>Score</th>
                    </tr>
                </thead>
                <tbody>
            """
            all_valve_results = sorted(
                st.session_state.results["all_valve_results"], 
                key=lambda x: x["score"], 
                reverse=True
            )
            for valve_result in all_valve_results:
                all_valves_table_html += f'<tr><td>{valve_result["display_name"]}</td>'
                for result in valve_result["results"]:
                    status_class = ""
                    if "Insufficient" in result["warning"]:
                        status_class = "status-insufficient"
                    elif result["status"] == "green":
                        status_class = "status-green"
                    elif result["status"] == "yellow":
                        status_class = "status-yellow"
                    elif result["status"] == "orange":
                        status_class = "status-orange"
                    elif result["status"] == "red":
                        status_class = "status-red"
                    all_valves_table_html += f'<td class="{status_class}">{result["status"]}</td>'
                all_valves_table_html += f'<td>{valve_result["score"]:.1f}</td></tr>'
            all_valves_table_html += "</tbody></table>"
            st.markdown(all_valves_table_html, unsafe_allow_html=True)
            
            st.subheader("Detailed Results")
            for i, scenario in enumerate(scenarios):
                with st.expander(f"Scenario {i+1}: {scenario['name']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Process Conditions**")
                        st.markdown(f"- Fluid Type: {scenario['fluid_type'].title()}")
                        st.markdown(f"- Flow Rate: {scenario['flow']} "
                                    f"{'m³/h' if scenario['fluid_type']=='liquid' else 'kg/h' if scenario['fluid_type']=='steam' else 'std m³/h'}")
                        st.markdown(f"- Inlet Pressure (P1): {scenario['p1']:.2f} bar a")
                        st.markdown(f"- Outlet Pressure (P2): {scenario['p2']:.2f} bar a")
                        st.markdown(f"- Pressure Drop (dP): {scenario['p1'] - scenario['p2']:.2f} bar")
                        st.markdown(f"- Temperature: {scenario['temp']}°C")
                    with col2:
                        st.markdown("**Fluid Properties**")
                        if scenario["fluid_type"] == "liquid":
                            st.markdown(f"- Specific Gravity: {scenario['sg']:.3f}")
                            st.markdown(f"- Viscosity: {scenario['visc']} cSt")
                            st.markdown(f"- Vapor Pressure: {scenario['pv']:.4f} bar a")
                            st.markdown(f"- Critical Pressure: {scenario['pc']:.2f} bar a")
                        elif scenario["fluid_type"] == "gas":
                            st.markdown(f"- Specific Gravity (air=1): {scenario['sg']:.3f}")
                            st.markdown(f"- Specific Heat Ratio (k): {scenario['k']:.3f}")
                            st.markdown(f"- Compressibility Factor (Z): {scenario['z']:.3f}")
                        else:
                            st.markdown(f"- Density: {scenario['rho']:.3f} kg/m³")
                            st.markdown(f"- Specific Heat Ratio (k): {scenario['k']:.3f}")
                        st.markdown(f"- Pipe Diameter: {scenario['pipe_d']} in")
                    st.markdown("**Sizing Results**")
                    st.markdown(f"- Theoretical Cv: {selected_valve_results[i]['theoretical_cv']:.1f}")
                    st.markdown(f"- Corrected Cv: {selected_valve_results[i]['req_cv']:.1f}")
                    st.markdown(f"- Operating Point: {selected_valve_results[i]['op_point']:.1f}% open")
                    st.markdown(f"- Actual Cv at Operating Point: {selected_valve.get_cv_at_opening(selected_valve_results[i]['op_point']):.1f}")
                    st.markdown(f"- Margin: {selected_valve_results[i]['margin']:.1f}%")
                    status_msg = "Optimal" if selected_valve_results[i]['status'] == "green" else "Warning" if selected_valve_results[i]['status'] == "yellow" else "Severe Cavitation" if selected_valve_results[i]['status'] == "orange" else "Critical (Choked)" if selected_valve_results[i]['status'] == "red" else "Insufficient Capacity"
                    st.markdown(f"- Status: {status_msg}")
                    if selected_valve_results[i]['warning']:
                        st.markdown(f"- Warning: {selected_valve_results[i]['warning']}")
                    if selected_valve_results[i]['cavitation_info']:
                        st.markdown(f"- Flow Status: {selected_valve_results[i]['cavitation_info']}")
                    
                    # Add detailed calculation parameters
                    st.markdown("**Calculation Parameters**")
                    col1, col2 = st.columns(2)
                    with col1:
                        fl_at_op = result['details'].get('fl_at_op', 'N/A')
                        if isinstance(fl_at_op, (int, float)):
                            st.markdown(f"**Fl (Liquid Recovery):** {fl_at_op:.3f}")
                        else:
                            st.markdown(f"**Fl (Liquid Recovery):** {fl_at_op}")
                        st.markdown(f"**Fd (Valve Style Modifier):** {selected_valve.fd:.2f}")
                        st.markdown(f"**Fp (Piping Factor):** {result['details'].get('fp', 1.0):.4f}")
                        
                    with col2:
                        if scenario["fluid_type"] == "liquid":
                            st.markdown(f"**FF (Critical Pressure Ratio):** {result['details'].get('ff', 0.96):.4f}")
                            st.markdown(f"**Fr (Viscosity Correction):** {result['details'].get('fr', 1.0):.4f}")
                            st.markdown(f"**Reynolds Number:** {result['details'].get('reynolds', 0):.0f}")
                    
                    st.markdown(f"**Max Pressure Drop (ΔPmax):** {result['details'].get('dp_max', 0):.2f} bar")
                    st.markdown(f"**Average Velocity in Valve:** {result.get('velocity', 0):.2f} m/s")
                    
                    if scenario["fluid_type"] == "liquid":
                        if result["details"].get('cavitation_severity'):
                            st.subheader("Cavitation Analysis")
                            st.markdown(f"**Status:** {result['details']['cavitation_severity']}")
                            st.markdown(f"**Sigma (σ):** {result['details'].get('sigma', 0):.2f}")
                            st.markdown(f"**Km (Valve Recovery Coefficient):** {result['details'].get('km', 0):.2f}")
                    
                    if scenario["fluid_type"] in ["gas", "steam"]:
                        st.subheader("Choked Flow Analysis")
                        st.markdown(f"**Status:** {result['cavitation_info']}")
                        st.markdown(f"**Pressure Drop Ratio (x):** {result['details'].get('x_actual', 0):.4f}")
                        st.markdown(f"**Critical Pressure Drop Ratio (x_crit):** {result['details'].get('x_crit', 0):.4f}")
                        st.markdown(f"**Pressure Drop Ratio Factor (xT or xTP):** {result['details'].get('xt_at_op', 0):.4f}")
                        st.markdown(f"**Choked Pressure Drop:** {result['details'].get('x_crit', 0) * scenario['p1']:.2f} bar")
                    
                    st.subheader("Flow Rate vs Pressure Drop")
                    flow_fig = generate_flow_vs_dp_graph(
                        scenario,
                        selected_valve,
                        result["op_point"],
                        result["details"],
                        result["req_cv"]
                    )
                    st.plotly_chart(flow_fig, use_container_width=True, key=f"detailed_flow_dp_{i}")
    
    # Handle export button
    if export_btn:
        if st.session_state.results is None:
            st.error("Please run the calculation first.")
        else:
            with st.spinner("Generating PDF report..."):
                # Prepare data for PDF
                scenarios = st.session_state.scenarios
                valve = st.session_state.results["selected_valve"]
                op_points = [r["op_point"] for r in st.session_state.results["selected_valve_results"]]
                req_cvs = [r["req_cv"] for r in st.session_state.results["selected_valve_results"]]
                warnings = [r["warning"] for r in st.session_state.results["selected_valve_results"]]
                cavitation_info = [r["cavitation_info"] for r in st.session_state.results["selected_valve_results"]]
                theoretical_cvs = [r["theoretical_cv"] for r in st.session_state.results["selected_valve_results"]]
                
                # Generate the Cv curve plot for PDF
                plot_bytes = plot_cv_curve_matplotlib(valve, op_points, req_cvs, theoretical_cvs, [s["name"] for s in scenarios])
                
                # Generate one Flow vs DP plot (for the first scenario) for PDF
                flow_dp_plot_bytes = None
                if scenarios:
                    flow_dp_plot_bytes = plot_flow_vs_dp_matplotlib(
                        scenarios[0],
                        valve,
                        op_points[0],
                        st.session_state.results["selected_valve_results"][0]["details"],
                        req_cvs[0]
                    )
                
                # Generate PDF
                logo_bytes = st.session_state.logo_bytes
                logo_type = st.session_state.logo_type
                pdf_bytes_io = generate_pdf_report(
                    scenarios, valve, op_points, req_cvs, warnings, cavitation_info, 
                    plot_bytes, flow_dp_plot_bytes, logo_bytes, logo_type
                )
                
                # Offer download
                st.success("PDF report generated!")
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes_io,
                    file_name=f"Valve_Sizing_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
    
    # Handle 3D viewer and simulation display
    if st.session_state.show_3d_viewer:
        valve_name = get_valve_display_name(selected_valve)
        model_url = VALVE_MODELS.get(valve_name, None)
        if model_url:
            st.subheader(f"3D Model: {valve_name}")
            valve_3d_viewer(valve_name, model_url)
        else:
            st.warning(f"3D model not available for {valve_name}")
        # Add a button to close the viewer
        if st.button("Close 3D Viewer"):
            st.session_state.show_3d_viewer = False
    
    if st.session_state.show_simulation:
        valve_name = get_valve_display_name(selected_valve)
        sim_image_url = get_simulation_image(valve_name)
        st.subheader(f"CFD Simulation Results: {valve_name}")
        st.image(sim_image_url, use_container_width=True)
        # Add a button to close the simulation
        if st.button("Close Simulation"):
            st.session_state.show_simulation = False

# Run the main function
if __name__ == "__main__":
    main()
