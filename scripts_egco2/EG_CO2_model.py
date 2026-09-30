# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 11:09:26 2025

@author: Nikolai P. Ponomarev
"""
#%% Modeling of pulping process using Ethylene Glycole (EG) and CO2 
# The process is based on unpublished data
# Author of script: Nikolai P. Ponomarev
print("\nModeling of the Ethylene Glycole (EG) - CO2 pulping.")
print("\nThe process is based on unpublished data.")
print("\nAuthor of script: Nikolai P. Ponomarev.")
#%% Needed libraries 
import pandas as pd
import numpy as np
from pathlib import Path
import olca_ipc as ipc
import olca_schema as o
import olca_ipc.utree as utree
import random as rand
import statistics as stats
import random
import time
import warnings
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from typing import Dict # to avoid having repeating "dict'
from scipy.stats import linregress
#import math
#import matplotlib.pyplot as plt
#%% Given data used for calculation
yP = 0.6436 # (%) Pulp Yield, [40-70%]
chips = 27.3 # wet wooden chips amount of experiment 3 (see notes) (g)
dry_chips = 15 # air dried chips (g)
eg = 12 # amount of EG used for experiment 3 (mL)
dEG = 1.1132 # density of EG (g/cm3)
h2o = 95.7 # amount of added water of exp.3 (mL)
H2O = (h2o + eg * dEG + (chips - dry_chips)) / dry_chips # LtoW or just set the value (5-10)
EG = eg * dEG ** (-1) / dry_chips # EG amount per mass of wood
lig = 0.271 # lignin content: hardwood 18-25%, softwood 25-33%
#hemi = 0.25 # hemicelluloses content: hardwood 20-30%, softwood 20-35%
WoodH2O = 0.45 # water content of wood is 45% (Gustafson2011, p.279)[40-50%]
sw_wood_logs = 0.5 #specific weight of wooden logs [t/m3]
PulpH2O = 0.65 # assumed high consistency pulp 30-35%, i.e. water 70%. (Gustafson2011, p.248)[30-35%]
Pulp_dried = 0.12 # assumed water content in the resultant pulp (%) 
LigH2O = 0.7 # assumed water content in the resultant lignin 70% [70-80%]
Loss_EG = 0.05 # assumed 5% losses of EG [0-100%]
Loss_H2O = 0.1 # assumed 10% losses of H2O [0-100%]
Loss_CO2 = 0.05 # assumed 5% losses of CO2 for pulping
pCO2 = 0.8 # MPa, carbon dioxide partial pressure [0.8- MPa]
R = 8.3145 # (J / mol * K), gas constant 
MCO2 = 44 # molar mass of carbon dioxide (g/mol) 
MC = 12 # molar mass of carbon g/mol 
MEG = 62.068 # Molar mass of EG (g/mol)
Head = 50 # meters, head of pump for ultrafiltration, i.e. 5 bar (Servaes2017)
g = 9.81 # m/s2
eff_pump = 0.7 # efficiency of the pump and motor, % [0.5-0.7]
eff_boiler = 0.8 # boiler efficiency, % (85-90%)
ED = 0.75 # Evaporation degree of MVR evaporator, % [0.5-0.9]
SP = 0.9 # Separation performance of the membrane, % (Servaes2017) [0.75-0.94]
Tdl = 170 # temperature of delignification 120-180, degrees Celsius
Tdef = 165 # defibration stage temperature 160-170, degrees Celsius
Tcinf = 10 # temperature of cold influent, degress Celcius
Thinf = 80 # temperature of hot influent, degress Celcius
t1 = 2.5 # delignification time in hours, 1-2 hours
Edef = 0.050 # energy for defibration at consistency 30-35%, MWh/tp, i.e. product pulp. (Gustafson2011, p.252)  
Ewh = 0.045 # electricuty for wood handling (FRAM2011)
Emvr_per_tH2O = 0.012 # energy for MVR [MWh/tH2Oevap] (Parviainen2008, P = 2.2 * dT, dT = 5-10 C)
heat_loss = 0.04 # 4% heat losses (Gustafson2011)
C_H2O = 4.19 # heat capacity of water (kJ/C*kg)
C_w = 1.53 # heat capacity of wood (kJ/C*kg)
C_co2 = 0.85 # heat capacity of CO2 (kJ/c*kg)
C_EG = 2.41 # heat capacity of EG (kJ/(kg*oC)) CPEg = 149.5 / MEG 
hev = 2256 # Enthalpy of vaporization of water, i.e latent heat (kJ/kg)
biomass_cal = 11 # lignin or wood LHV (MJ/kg) [FRAM11]
Kb = 0.512 # ebullioscopic constant (boiling-point elevation constant) of water (i.e. solvent) (K*kg/mol) 
wood_carbon = 0.5 # Stoicheometric carbon content in wood (%)
data_given_EGCO2 = [
    ["Description", "Value", "Variation limit", "Unit", "Variable"],
    ["Pulp yield", yP, "40-70%", "fraction", "yP"],
    ["Liqour-to-wood (LtoW) (calculated)", H2O, "5-10", "L/kg", "H2O"],
    ["Ethylene Glycol (EG) (calculated", EG, "", "t/t", "EG"],
    ["Wet wooden chips amount (experiment 3)", chips, "", "g", "chips"],
    ["Air dried chips", dry_chips, "", "g", "dry_chips"],
    ["Ethylene glycol (EG) used (experiment 3)", eg, "", "mL", "eg"],
    ["Density of EG", dEG, "", "g/cm3", "dEG"],
    ["Added water (experiment 3)", h2o, "", "mL", "h2o"],
    ["Lignin content in Norway spruce", lig, "27-30%", "fraction", "lig"],
    ["Water content of wood", WoodH2O, "40-50%", "fraction", "WoodH2O"],
    ["Specific weight of wooden logs", sw_wood_logs, "", "t/m3", "sw_wood_logs"],
    ["Water content of high-consistency pulp (30-35% consistency)", PulpH2O, "65-70% (water)", "fraction", "PulpH2O"],
    ["Water content in resultant pulp (dried)", Pulp_dried, "", "fraction", "Pulp_dried"],
    ["Water content in resultant lignin", LigH2O, "70-80%", "fraction", "LigH2O"],
    ["Assumed EG losses", Loss_EG, "0-100%", "fraction", "Loss_EG"],
    ["Assumed H2O losses", Loss_H2O, "0-100%", "fraction", "Loss_H2O"],
    [" Assumed CO2 losses for pulping", Loss_CO2,"0-100%", "fraction", "Loss_CO2" ],
    ["CO2 partial pressure", pCO2, ">=0.8 MPa", "MPa", "pCO2"],
    ["Gas constant", R, "", "J/(mol*K)", "R"],
    ["Molar mass of CO2", MCO2, "", "g/mol", "MCO2"],
    ["Molar mass of carbon", MC, "", "g/mol", "MC"],
    ["Molar mass of EG", MEG, "", "g/mol", "MEG"],
    ["Pump head for ultrafiltration (~5 bar)", Head, "", "m", "Head"],
    ["Gravitational acceleration", g, "", "m/s2", "g"],
    ["Efficiency of pump+motor", eff_pump, "0.5-0.7", "fraction", "eff"],
    ["Efficiency of boiler", eff_boiler, "0.85-0.90", "fraction", "eff_boiler"],
    ["Evaporation Degree of MVR", ED, "0.5-0.9", "fraction", "ED"],
    ["Separation Performance of membrane", SP, "0.75-0.94", "fraction", "SP"],
    ["Temperature of delignification", Tdl, "120-180", "°C", "Tdl"],
    ["Temperature of defibration", Tdef, "160-170", "°C", "Tdef"],
    ["Temperature of cold influent", Tcinf, "", "°C", "Tcinf"],
    ["Temperature of hot influent", Thinf, "", "°C", "Thinf"],
    ["Time of delignification", t1, "1-2 h", "h", "t1"],
    ["Electricity for wood handling", Ewh, "", "MWh/tp", "Ewh"],
    ["Electricity for defibration (per t pulp)", Edef, "", "MWh/tp", "Edef"],
    ["Electricity for MVR (per t H2O evaporated)", Emvr_per_tH2O, "", "MWh/tH2Oevap", "Emvr_per_tH2O"],
    ["Heat capacity of water", C_H2O, "", "kJ/(kg*°C)", "C_H2O"],
    ["Heat capacity of wood", C_w, "", "kJ/(kg*°C)", "C_w"],
    ["Heat capacity of CO2", C_co2, "", "kJ/(kg*°C)", "C_co2"],
    ["Heat capacity of EG", C_EG, "", "kJ/(kg*°C)", "C_EG"],
    ["Enthalpy of vaporization of water", hev, "", "kJ/kg", "hev"],
    ["Lower heating value of biomass (lignin/wood)", biomass_cal, "", "MJ/kg", "biomass_cal"],
    ["Heat losses", heat_loss, "", "fraction", "heat_loss"],
    ["Ebullioscopic constant of water", Kb, "", "K*kg/mol", "Kb"],
    ["Stoichiometric carbon content in wood", wood_carbon, "", "fraction", "wood_carbon"],
]
df_given_data_EGCO2 = pd.DataFrame(data_given_EGCO2).round(3)
print("\nGiven data was summarized as a data frame.")
#%% Creating one master function to unite all functions
def egco2(gd):
    # Unpack the given data
    h2o       = gd["h2o"]
    H2O = gd["H2O"]
    eg        = gd["eg"]
    EG = gd["EG"]
    # dEG       = gd["dEG"]
    # chips     = gd["chips"]
    # dry_chips = gd["dry_chips"]
    lig      = gd["lig"]
    yP       = gd["yP"]
    WoodH2O  = gd["WoodH2O"]
    PulpH2O  = gd["PulpH2O"]
    pCO2     = gd["pCO2"]
    MCO2     = gd["MCO2"]
    R        = gd["R"]
    Tdl      = gd["Tdl"]
    Tcinf    = gd["Tcinf"]
    Thinf    = gd["Thinf"]
    C_H2O    = gd["C_H2O"]
    C_w      = gd["C_w"]
    C_EG     = gd["C_EG"]
    C_co2    = gd["C_co2"]
    heat_loss= gd["heat_loss"]
    Ewh      = gd["Ewh"]
    Edef     = gd["Edef"]
    SP       = gd["SP"]
    LigH2O   = gd["LigH2O"]
    Loss_H2O = gd["Loss_H2O"]
    Loss_EG  = gd["Loss_EG"]
    Loss_CO2  = gd["Loss_CO2"]
    g        = gd["g"]
    Head     = gd["Head"]
    eff_pump = gd["eff_pump"]
    ED       = gd["ED"]
    MEG      = gd["MEG"]
    Kb       = gd["Kb"]
    Pulp_dried = gd["Pulp_dried"]
    hev        = gd["hev"]
    biomass_cal = gd["biomass_cal"]
    eff_boiler  = gd["eff_boiler"]
    sw_wood_logs = gd["sw_wood_logs"]
    wood_carbon  = gd["wood_carbon"]
    MC           = gd["MC"]
    # # Estimation of LtoW and EG dose per mass of wood
    # def LtoW_EG (h2o,eg,dEG,chips,dry_chips):
    #     # Liqour-to-wood ratio
    #     H2O = (h2o + eg * dEG + (chips - dry_chips)) / dry_chips # or just set the value (5-10)
    #     EG = eg * dEG ** (-1) / dry_chips    
    #     return H2O, EG
    # H2O, EG = LtoW_EG(h2o, eg, dEG, chips, dry_chips)
    # print("\n L-to-W ratio", round(H2O,2))
    # print("\n EG (g/g-wood)", round(EG,2))
    # Delignification (DL), Defibration (DF) and Dewatering (DW)
    def DL(lig, yP, EG, H2O, WoodH2O, PulpH2O, pCO2, MCO2, R, Tdl, Tcinf, Thinf, C_H2O, C_w, heat_loss,Ewh,Edef):
        """
        Compute mass balance for the Delignification (DL) stage.
        Including Defibration (DF) and Dewatering (DW)
        Returns:
          - rounded dataframe of streams
          - dict with key totals (inf_DL, eff_DL)
        """
        # Convenience terms
        carb = 1 - lig                  # carbohydrates in wood
        wood11 = 1 / yP                 # total dry wood needed for process (carb+lig = 1)
        # Stream 1.1 (influent from wood)
        carb11 = carb / yP
        lig11  = lig / yP
        h2o11  = (wood11 * WoodH2O) / (1 - WoodH2O)
        tot11  = wood11 + h2o11
        # Stream 2.1 (chemicals/water make-up)
        eg21  = wood11 * EG
        h2o21 = wood11 * H2O - h2o11
        tot21 = eg21 + h2o21
        # Stream 2.2 (CO2 in)
        co2_required = (pCO2 * (tot11 + tot21) * MCO2) / (R * (Tdl + 273))
        co2_22 = co2_required * Loss_CO2 # taking into account the losses. The rest is came from fluegas.
        tot22  = co2_22
        # Total influent
        inf_DL = tot11 + tot21 + tot22
        # Effluent
        carb12 = (carb11 + lig11) * yP * carb
        lig12  = (carb11 + lig11) * yP * lig
        h2o12  = ((carb12 + lig12) * PulpH2O) / (1 - PulpH2O)
        pulpDL   = carb12 + lig12
        tot12  = pulpDL + h2o12
        # Stream 2.3 (CO2 out)
        co2_23 = co2_22
        tot23  = co2_23
        # Stream 2.4 (liquor)
        carb24 = (carb / yP) - ((carb11 + lig11) * yP * carb) #carb11 - carb12: 1st term in brackets - 2nd term in brackets
        lig24  = (lig / yP) - ((carb11 + lig11) * yP * lig) #lig11 - lig12: 1st term in brackets - 2nd term in brackets
        h2o24  = (h2o11 + h2o21) - h2o12
        eg24   = eg21
        tot24  = carb24 + lig24 + h2o24 + eg24
        # Total effluent
        eff_DL = tot12 + tot23 + tot24
        # Build dataframe (use NaN for missing to keep numeric dtype)
        data_mass_bal_DL = {
            "stream": ["1.1", "1.2", "2.1", "2.2", "2.3", "2.4"],
            "carbs":  [carb11, carb12, np.nan, np.nan, np.nan, carb24],
            "lignin": [lig11,  lig12,  np.nan, np.nan, np.nan, lig24],
            "H2O":    [h2o11,  h2o12,  h2o21,  np.nan, np.nan, h2o24],
            "EG":     [np.nan, np.nan, eg21,   np.nan, np.nan, eg24],
            "CO2":    [np.nan, np.nan, np.nan, co2_22, co2_23, np.nan],
            "inf":    [inf_DL, np.nan, np.nan, np.nan, np.nan, np.nan],
            "eff":    [eff_DL, np.nan, np.nan, np.nan, np.nan, np.nan],
        }
        mass_bal_DL = pd.DataFrame(data_mass_bal_DL)
        # Round numeric values to 3 decimals
        df_mat_bal_DL = mass_bal_DL.round(3)
        """
        Compute heat balance in MJ/t for the Delignification (DL) stage.
        Only sensible heat. No latent heat. Enthalpy of cooking using EG is unknown.
        """
        #Sensible heat
        Q_sens_DL = (
                C_H2O * h2o11 * (Tdl - Tcinf) + 
                C_H2O * h2o21 * (Tdl - Thinf) + 
                C_w * wood11 * (Tdl - Tcinf) + 
                C_EG * eg21 * (Tdl - Thinf) +
                C_co2 * co2_22 * (Tdl - Tcinf)
                )
        #Heat losses
        Q_loss_DL = Q_sens_DL * heat_loss
        # Total heat and converted to GJ/t
        Q_DL = (Q_sens_DL + Q_loss_DL) * 1e-3
        data_heat_bal_DL = [
            ["Item", "Value", "Unit"],
            ["Sensible heat", round(Q_sens_DL * 1e-3, 3), "GJ/t"],
            ["Heat losses", round(Q_loss_DL * 1e-3, 3), "GJ/t"],
            ["Total heat for DL", round(Q_DL, 3), "GJ/t"]
            ]
        df_heat_bal_DL = pd.DataFrame(data_heat_bal_DL[1:], columns=data_heat_bal_DL[0])
        # Electricity for wood handling
        E_wh = Ewh * 1e+3
        # Electricity for defibration
        E_df = Edef * 1e+3    
        return (df_mat_bal_DL,df_heat_bal_DL,carb24,lig24,h2o24,eg24,tot24,h2o21,tot12,
                h2o12,carb12,lig12,tot21,eg21,h2o11,Q_DL,carb11,lig11,co2_22,co2_23,
                tot11,tot22,tot23,wood11,E_wh,E_df)
    #Calling variables from function using parantheses:
    (df_mat_bal_DL,df_heat_bal_DL,carb24,lig24,h2o24,eg24,tot24,h2o21,tot12,
    h2o12,carb12,lig12,tot21,eg21,h2o11,Q_DL,carb11,lig11,co2_22,co2_23,
    tot11,tot22,tot23,wood11,E_wh,E_df) = DL(lig, yP, EG, H2O, WoodH2O, PulpH2O, pCO2, 
    MCO2, R, Tdl, Tcinf, Thinf, C_H2O, C_w, heat_loss,Ewh,Edef)
    # print("\n Material balance of delignification (DL) (t/t)\n", df_mat_bal_DL)
    # print("\n Heat balance of DL\n", df_heat_bal_DL)
    # print("\n Electrical energy consumption for wood handling (WH) (kWh/t)\n", round(E_wh, 3))
    # print("\n Electrical energy consumption for defibration (DF) (kWh/t)\n", round(E_df, 3))
    #Membrane separation (MEM)
    def MEM (carb24,lig24,h2o24,tot24,SP,LigH2O,Loss_H2O,Loss_EG,g,Head,eff_pump):
        """
        Compute mass balance for the Membrane separation (MEM) stage.
        Returns:
          - rounded dataframe of streams
          - dict with key totals (inf_MEM, eff_MEM)
        """
        lig26 = lig24 * SP # lignin out of the membrane
        
        h2o26 = (lig26 * LigH2O) / (1 - LigH2O) # water of lignin
        tot26 = h2o26 + lig26
        # Stream 2.7
        lig27 = lig24 * (1 - SP) # Lignin losses
        carb27 = carb24 * (1 - SP) # Carbohydrates losses
        h2o27 = h2o21 * Loss_H2O # Water losses. Should be compensated using 2.12
        eg27 = eg24 * Loss_EG # EG losses. Should be compensated using 2.12
        tot27 = eg27 + h2o27 + lig27 + carb27
        # Stream 2.5
        eg25 = eg24 - eg27
        carb25 = carb24 * SP 
        h2o25 = h2o24 - h2o27 - h2o26
        tot25 = eg25 + h2o25 + carb25
        # Influent and effluent
        inf_MEM = tot24
        eff_MEM = tot26 + tot27 + tot25
        # Build dataframe (use NaN for missing to keep numeric dtype)
        data_mass_bal_MEM = {
            "stream": ["2.4", "2.5","2.6","2.7"],
            "carbs":  [carb24, carb25, np.nan, carb27],
            "lignin": [lig24, np.nan, lig26, lig27],
            "H2O":    [h2o24, h2o25, h2o26, h2o27],
            "EG":     [eg24, eg25, np.nan, eg27],
            "CO2":    [np.nan, np.nan, np.nan, np.nan],
            "inf":    [inf_MEM, np.nan, np.nan, np.nan],
            "eff":    [eff_MEM, np.nan, np.nan, np.nan],
        }
        mass_bal_MEM = pd.DataFrame(data_mass_bal_MEM)
        # Round numeric values to 3 decimals
        df_mat_bal_MEM = mass_bal_MEM.round(3)
        # Required energy for pumping
        """
        Compute consumption for the Membrane separation (MEM) stage.
        """
        E_mem = tot24 * g * Head * 2.777778e-4 * eff_pump **-1
        return df_mat_bal_MEM, E_mem, h2o25, eg25, carb25, tot25, h2o27, h2o26, lig26, carb27, lig27, eg27, tot26, tot27
    df_mat_bal_MEM, E_mem, h2o25, eg25, carb25, tot25, h2o27, h2o26, lig26, carb27, lig27, eg27, tot26, tot27 = MEM(carb24, lig24, h2o24, tot24, SP, LigH2O, Loss_H2O, Loss_EG, g, Head, eff_pump)
    # print("\n Material balance of membrane separation (MEM) (t/t)\n", df_mat_bal_MEM)
    # print("\n Electrical energy consumption for MEM (kWh/t)\n", round(E_mem, 3))
    #Mechanical Vapor Recompression (MVR)
    def MVR(h2o25, eg25, carb25, ED, MEG, Kb):
        #better tot use empirical MVR equation from Parviainen2008, p.43.
        # P = 2.2 * dT
        # P - energy for MVR kWh/t-distilate, typically 11-15 kWh/tH2O
        # dT - steam saturation temperature increase in the compressor (oC)
        # Based on Raoult and Clausius–Clapeyron equations
        # Boiling-point elevation (BPE)
        #Initial composition (from earlier): EG 0.215 mol; water 95.7 g → 5.31 mol.
        #Evaporate 75% of water: remaining water ≈ 0.25 × 95.7 g = 23.9 g → 1.33 mol (0.0239 kg).
        #Molality m ≈ 0.215 mol / 0.0239 kg ≈ 9.0 m.
        #ΔTb ≈ Kb m ≈ 0.512 × 9.0 ≈ 4.6 K. => dT = 5 oC
        #Final boiling temperature at 1 atm ≈ 100 + 4.6 ≈ 104.6 °C.
        #MVR temperature lift
        #To transfer heat, the compressed vapor must condense a few degrees hotter than the boiling liquor. 
        #A typical approach (pinch) is 4–6 K; So 5 K is used.
        #Required discharge saturation temperature ≈ 104.6 + 5 ≈ 109.6 °C.
        #So the compressor must raise saturation temperature by ≈ 5 K above its suction saturation 
        #(which is ≈ the liquor boiling temperature).
        #Equivalent pressure ratio (for reference)
        #Psat at 104.6 °C ≈ 1.20 bar(a); at 109.6 °C ≈ 1.36 bar(a).
        #Pressure ratio ≈ 1.36/1.20 ≈ 1.13.
        # Molality of EG
        # m = (eg / MEG) / ((1-ED) * h2o * 1e-3), simplifying
        #Mass balance
        h2o28 = h2o25 * (1 - ED)
        eg28 = eg25
        carb28 = carb25
        tot28 = h2o28 + eg28 + carb28
        h2o29 = h2o25 * ED
        tot29 = h2o29
        inf_MVR = tot25
        eff_MVR = tot28 + tot29
        # Build dataframe (use NaN for missing to keep numeric dtype)
        data_mass_bal_MVR = {
             "stream": ["2.5", "2.8","2.9"],
             "carbs":  [carb25, carb28, np.nan],
             "lignin": [np.nan, np.nan, np.nan],
             "H2O":    [h2o25, h2o28, h2o29],
             "EG":     [eg25, eg28, np.nan],
             "CO2":    [np.nan, np.nan, np.nan],
             "inf":    [inf_MVR, np.nan, np.nan],
             "eff":    [eff_MVR, np.nan, np.nan],
         }
        mass_bal_MVR = pd.DataFrame(data_mass_bal_MVR)
        # Round numeric values to 3 decimals
        df_mat_bal_MVR = mass_bal_MVR.round(3)
        #Energy consumption
        m = eg * 1e+3 / (MEG * (1-ED) * h2o)
        dT = Kb * m
        E_mvr = 2.2 * dT * h2o29
        return df_mat_bal_MVR, E_mvr, carb28, h2o28, eg28, tot28, tot29, h2o29
    df_mat_bal_MVR, E_mvr, carb28, h2o28, eg28, tot28, tot29, h2o29 = MVR(h2o25, eg25, carb25, ED, MEG, Kb)
    # print("\n Material balance of mechanical vapor recompression (MVR) (t/t)\n", df_mat_bal_MVR)
    # print("\n Electrical energy consumption for MVR (kWh/t)\n", round(E_mvr, 3))
    #Ultrafiltration (UF)
    def UF(carb28, h2o28, eg28, tot28, g, Head, eff_pump):
        #Mass balance
        carb210 = carb28
        h2o210 = h2o28
        tot210 = carb28 + h2o28
        eg211 = eg28
        tot211 = eg211
        inf_UF = tot28
        eff_UF = tot210 + tot211
        # Dataframe
        data_mass_bal_UF = {
                "stream": ["2.8","2.10", "2.11"],
                "carbs":  [carb28, carb210, np.nan],
                "lignin": [np.nan, np.nan, np.nan],
                "H2O":    [h2o28, h2o210, np.nan],
                "EG":     [eg28, np.nan, eg211],
                "CO2":    [np.nan, np.nan, np.nan],
                "inf":    [inf_UF, np.nan, np.nan],
                "eff":    [eff_UF, np.nan, np.nan],
            }
        mass_bal_UF = pd.DataFrame(data_mass_bal_UF)
        df_mat_bal_UF = mass_bal_UF.round(3)
        #Energy consumption
        E_uf = tot28 * g * Head * 2.777778e-4 * eff_pump **-1
        return df_mat_bal_UF, E_uf, tot211, eg211, h2o210, carb210, tot210
    df_mat_bal_UF, E_uf, tot211, eg211, h2o210, carb210, tot210 = UF(carb28, h2o28, eg28, tot28, g, Head, eff_pump)
    # print("\n Material balance of ultrafiltration (UF)(t/t)\n", df_mat_bal_UF)
    # print("\n Electrical energy consumption for UF (kWh/t)\n", round(E_uf, 3))
    #Drying (DR)
    def DR(h2o12, carb12, lig12):
        # Mass balance
        carb13 = carb12
        lig13 = lig12
        h2o13 = h2o12 * Pulp_dried
        pulpDR = carb13 + lig13 # should be 1 since calcualtion is t/t-product
        tot13 = carb13 + lig13 + h2o13
        h2o212 = h2o12 * (1-Pulp_dried)
        tot212 = h2o212
        inf_DR = tot12
        eff_DR = tot13 + tot212
        # Dataframe
        data_mass_bal_DR = {
                "stream": ["1.2","1.3", "2.12"],
                "carbs":  [carb12, carb13, np.nan],
                "lignin": [lig12, lig12, np.nan],
                "H2O":    [h2o12, h2o13, h2o21],
                "EG":     [np.nan, np.nan, np.nan],
                "CO2":    [np.nan, np.nan, np.nan],
                "inf":    [inf_DR, np.nan, np.nan],
                "eff":    [eff_DR, np.nan, np.nan],
            }
        mass_bal_DR = pd.DataFrame(data_mass_bal_DR)
        df_mat_bal_DR = mass_bal_DR.round(3)
        #Energy balance
        Q_sens_DR = (
                C_H2O * h2o12 * (100 - Thinf) + 
                C_w * (carb12 + lig12) * (100 - Thinf)
                )
        Q_lat_DR = hev * (h2o12 - h2o13)
        Q_loss_DR = (Q_sens_DR + Q_lat_DR) * heat_loss
        # Total heat and converted to GJ/t
        Q_DR = (Q_sens_DR + Q_lat_DR + Q_loss_DR) * 1e-3
        data_heat_bal_DR = [
            ["Item", "Value", "Unit"],
            ["Sensible heat", round(Q_sens_DR * 1e-3, 3), "GJ/t"],
            ["Latent heat", round(Q_lat_DR * 1e-3, 3), "GJ/t"],
            ["Heat losses", round(Q_loss_DR * 1e-3, 3), "GJ/t"],
            ["Total heat for DR", round(Q_DR, 3), "GJ/t"]
            ]
        df_heat_bal_DR = pd.DataFrame(data_heat_bal_DR[1:], columns=data_heat_bal_DR[0])
        return df_mat_bal_DR, df_heat_bal_DR, pulpDR, h2o13, tot212, h2o212, Q_DR, carb13, lig13, tot13
    df_mat_bal_DR, df_heat_bal_DR, pulpDR, h2o13, tot212, h2o212, Q_DR, carb13, lig13, tot13 = DR(h2o12, carb12, lig12)
    # print("\n Material balance of drying (DR) (t/t)\n", df_mat_bal_DR)
    # print("\n Heat balance of DR\n", df_heat_bal_DR)
    # Buffer Tank (BT)
    def BT(h2o29, h2o212, eg211, h2o21, eg21, h2o13, h2o11, Loss_EG):
        # Mass balance
        eg213 = eg21 * Loss_EG # EG make-up
        h2o213 =  h2o13 + h2o26 + h2o27 + h2o210 - h2o11 # Water make-up
        tot213 = eg213 + h2o213
        inf_BT = tot213 + tot29 + tot211 + tot212
        eff_BT = tot21 
        #Data frame
        data_mass_bal_BT = {
            "stream": ["2.1", "2.9", "2.11", "2.12", "2.13"],
            "carbs":  [np.nan, np.nan, np.nan, np.nan, np.nan],
            "lignin": [np.nan,  np.nan,  np.nan, np.nan, np.nan],
            "H2O":    [h2o21,  h2o29,  np.nan,  h2o212, h2o213],
            "EG":     [eg21, np.nan, eg211, np.nan, eg213],
            "CO2":    [np.nan, np.nan, np.nan, np.nan, np.nan],
            "inf":    [inf_BT, np.nan, np.nan, np.nan, np.nan],
            "eff":    [eff_BT, np.nan, np.nan, np.nan, np.nan],
        }
        mass_bal_BT = pd.DataFrame(data_mass_bal_BT)
        df_mat_bal_BT = mass_bal_BT.round(3)
        return df_mat_bal_BT, h2o213, eg213, tot213
    df_mat_bal_BT, h2o213, eg213, tot213 = BT(h2o29, h2o212, eg211, h2o21, eg21, h2o13, h2o11, Loss_EG)
    # print("\n Material balance of Buffer Tank (BT) (t/t)\n", df_mat_bal_BT)
    # Boiler (BLR)
    def BLR (lig26, biomass_cal, eff_boiler, heat_loss):
        Q_BLR = lig26 * biomass_cal * eff_boiler * (1 - heat_loss)
        biomass_egco2 = (Q_DL + Q_DR) * 1e+3 / (biomass_cal * eff_boiler * (1 - heat_loss))
        co2_261 = biomass_egco2 * wood_carbon * MCO2 * 1e-3 / MC
        tot261 = co2_261
        return Q_BLR, co2_261, tot261
    Q_BLR, co2_261, tot261 = BLR(lig26, biomass_cal, eff_boiler, heat_loss)
    # print("\nFrom the boiler (BLR) you can get:")
    # print("Heat from lignin (GJ/t) \n", round(Q_BLR, 3))
    # print("CO2 non-fossil while covering entire heat demand (t/t) \n", round(co2_261, 0))
    #Overall material balance of EG-CO2 pulping
    data_mass_bal_EGCO2 = {
        "stream": ["1.1","1.2","1.3","2.1","2.2","2.3","2.4","2.5","2.6","2.6.1","2.7","2.8","2.9","2.1o","2.11","2.12","2.13"],
        "carbs":  [carb11, carb12, carb13, np.nan, np.nan, np.nan, carb24, carb25, np.nan, np.nan, carb27, carb28, np.nan, carb210, np.nan, np.nan, np.nan],
        "lignin": [lig11,  lig12,  lig13,  np.nan, np.nan, np.nan, lig24,  np.nan, lig26,  np.nan, lig27,  np.nan, np.nan, np.nan,  np.nan, np.nan, np.nan],
        "H2O":    [h2o11,  h2o12,  h2o13,  h2o21, np.nan, np.nan, h2o24, h2o25, h2o26, np.nan, h2o27, h2o28, h2o29, h2o210, np.nan, h2o212, h2o213],
        "EG":     [np.nan, np.nan, np.nan, eg21,  np.nan, np.nan, eg24,  eg25,  np.nan, np.nan, eg27,  eg28,  np.nan, np.nan,  eg211, np.nan,  eg213],
        "CO2":    [np.nan, np.nan, np.nan, np.nan, co2_22, co2_23, np.nan, np.nan, np.nan, co2_261, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "total":  [tot11, tot12, tot13, tot21, tot22, tot23, tot24, tot25, tot26, tot261, tot27, tot28, tot29, tot210, tot211, tot212, tot213],
        "influent":[(tot11 + tot22 + tot213)] + [np.nan]*16,
        "effluent":[(tot13 + tot26 + tot27 + tot23 + tot210)] + [np.nan]*16,
    }
    mass_bal_EGCO2 = pd.DataFrame(data_mass_bal_EGCO2)
    df_mat_bal_EGCO2 = mass_bal_EGCO2.round(3).set_index("stream")
    #print("\n Overall material balance of EG-CO2 pulping (t/t)\n", df_mat_bal_EGCO2)
    #Summirizing total inputs and outputs for the process
    def summary_EG_CO2 (wood11,sw_wood_logs,eg213,co2_22,h2o213,Q_DL,Q_DR,E_wh,E_df,E_mem,E_mvr,E_uf,pulpDR,lig26,Q_BLR,carb210,co2_261,tot27): 
        #inputs
        wood_logs_egco2 = wood11 / sw_wood_logs # Wood logs (m3/t) 
        eg_egco2 = eg213 # Ethylene glycol (t/t)
        co2_egco2 = co2_22 # CO2 for delignification (t/t)
        h2o_egco2 = h2o213 # Process water (t/t)
        heat_egco2 = Q_DL + Q_DR # Heat demand (GJ/t)
        elect_egco2 = E_wh + E_df + E_mem + E_mvr + E_uf # Electricity demand (kWh/t)
        #outputs
        pulp_egco2 = pulpDR # Produced pulp (t/t)
        lignin_egco2 = lig26 # Lignin (t/t)
        lignin_heat_egco2 = Q_BLR # Lignin as heat (GJ/t)
        hemi_egco2 = carb210 # Hemicellusloses (t/t)
        #wastes
        co2_non_fos_egco2 = co2_261 # Non-fossil CO2 (t/t)
        wastewater_egco2 = tot27 # Wastewater (t/t)
        # Build summary DataFrame
        data_summary = {
            "Category": [
                "given data","properties","properties","input", "input", "input", "input", "input", "input",
                "output", "output", "output", "output",
                "waste", "waste"
            ],
            "Item": [
                "Pulp yield","Kappa Number","Fibre length","Wood logs", "Ethylene glycol", "CO2 (delignification)", "Process water",
                "Heat demand", "Electricity demand",
                "Pulp", "Lignin", "Lignin as heat", "Hemicelluloses",
                "Non-fossil CO2", "Wastewater"
            ],
            "Value": [
                yP*100,"110","1.8-2.0",wood_logs_egco2, eg_egco2, co2_egco2, h2o_egco2, heat_egco2, elect_egco2,
                pulp_egco2, lignin_egco2, lignin_heat_egco2, hemi_egco2,
                co2_non_fos_egco2, wastewater_egco2
            ],
            "Unit": [
                "%","-","mm","m3/t", "t/t", "t/t", "t/t", "GJ/t", "kWh/t",
                "t/t", "t/t", "GJ/t", "t/t",
                "t/t", "t/t"
            ],
        }
        df_summary_egco2 = pd.DataFrame(data_summary).round(3)
        return (eg_egco2,
                heat_egco2,
                wood_logs_egco2,
                co2_egco2,
                h2o_egco2,
                elect_egco2,
                co2_non_fos_egco2,
                wastewater_egco2,
                lignin_heat_egco2,
                hemi_egco2,
                df_summary_egco2)
    (eg_egco2,
    heat_egco2,
    wood_logs_egco2,
    co2_egco2,
    h2o_egco2,
    elect_egco2,
    co2_non_fos_egco2,
    wastewater_egco2,
    lignin_heat_egco2,
    hemi_egco2,
    df_summary_egco2) = summary_EG_CO2(wood11, sw_wood_logs, eg213, co2_22, h2o213, 
                                       Q_DL, Q_DR, E_wh, E_df, E_mem, E_mvr, E_uf, pulpDR, lig26, Q_BLR, carb210, co2_261, tot27)
    # print("\n Summary of total inputs, outputs, and wastes for EG-CO2 process\n", df_summary_egco2)
    # Return what you need from egco2 function
    return (
        eg_egco2,
        heat_egco2,
        wood_logs_egco2,
        co2_egco2,
        h2o_egco2,
        elect_egco2,
        co2_non_fos_egco2,
        wastewater_egco2,
        lignin_heat_egco2,
        hemi_egco2#,
        #df_mat_bal_EGCO2
        #df_summary_egco2
    )
# used parameters from given data
giv_dat = {
    "h2o": h2o, "eg": eg, "dEG": dEG, "chips": chips, "dry_chips": dry_chips,
    "lig": lig, "yP": yP, "WoodH2O": WoodH2O, "PulpH2O": PulpH2O, "pCO2": pCO2, "MCO2": MCO2, "R": R,
    "Tdl": Tdl, "Tcinf": Tcinf, "Thinf": Thinf, "C_H2O": C_H2O, "C_w": C_w, "C_EG": C_EG, "C_co2": C_co2,
    "heat_loss": heat_loss, "Ewh": Ewh, "Edef": Edef,
    "SP": SP, "LigH2O": LigH2O, "Loss_H2O": Loss_H2O, "Loss_EG": Loss_EG, "Loss_CO2": Loss_CO2, "g": g, "Head": Head, "eff_pump": eff_pump,
    "ED": ED, "MEG": MEG, "Kb": Kb,
    "Pulp_dried": Pulp_dried, "hev": hev,
    "biomass_cal": biomass_cal, "eff_boiler": eff_boiler,
    "sw_wood_logs": sw_wood_logs, "wood_carbon": wood_carbon, "MC": MC, "H2O":H2O, "EG":EG       
}

# Using the results from function egco2
(eg_egco2,
 heat_egco2,
 wood_logs_egco2,
 co2_egco2,
 h2o_egco2,
 elect_egco2,
 co2_non_fos_egco2,
 wastewater_egco2,
 lignin_heat_egco2, 
 hemi_egco2) = egco2(giv_dat)
 # df_mat_bal_EGCO2,
 # df_summary_egco2) 

# sum inputs for LCA in one list
inp_base = (eg_egco2, 
            heat_egco2,
            wood_logs_egco2,
            co2_egco2,
            h2o_egco2,
            elect_egco2,
            co2_non_fos_egco2,
            wastewater_egco2,
            lignin_heat_egco2,
            hemi_egco2
            )
#%% LCA with OpenLCA using olca-ipc
# For EG-CO2
# The main idea is that global parameters (GB) are set in the LCA database,
# and equal 1. Then thes GBs are multiplied by corresponding values obtained
#in material and energy balances, and in OVAT part of the program.
# it is here just to understand the logic of the calcualtion
# Process should be created in advance in the exisiting database
# Kraft process is also added for the comparision purposes
# Start timing
t0 = time.perf_counter()
print("\nConnecting to openLCA IPC server on port 8080 ...", end=" ")
try:
    # Try to connect
    client = ipc.Client(8080)
    # Light check that the server responds (a simple call)
    _ = client.find(o.ImpactMethod, "EF v3.1")
    print("connected.\n")
except Exception as e:
    warnings.warn(f"Could not connect to openLCA IPC server on port 8080: {e}")
    # Stop here if not connected
    elapsed = time.perf_counter() - t0
    total_seconds = int(elapsed)
    centi = int((elapsed - total_seconds) * 100)
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    print(f"Execution aborted. Elapsed time: {h:02d}:{m:02d}:{s:02d}:{centi:02d}")
    raise SystemExit(1)
# Find process of EG-CO2
model = client.find(o.Process, "EG-CO2v2.0")
# To get kraft pulping
model_kraft = client.find(o.Process, "sulfate pulp production, from softwood, unbleached | sulfate pulp, unbleached | Cutoff, U")
# Create a method for both processes
method = client.find(o.ImpactMethod, "EF v3.1") # change it to 'EF v3.1' or 'EF v3.1 (CO2only)' which should be created in database.
# To get global parameters
params = client.get_all(o.Parameter)
def g(name):
    for p in params:
        if p.name == name and getattr(p, "context", None) is None:
            return float(p.value)
    raise KeyError(f"Global parameter '{name}' not found")
# Recalculated values:
egco2_eg_r           = g("egco2_eg") * eg_egco2
egco2_heat_r         = g("egco2_heat") * heat_egco2
egco2_wood_r         = g("egco2_wood") * wood_logs_egco2
egco2_co2_r          = g("egco2_co2") * co2_egco2
egco2_h2o_r          = g("egco2_h2o") * h2o_egco2
egco2_elect_r        = g("egco2_elect") * elect_egco2
egco2_co2_non_fos_r  = g("egco2_co2_non_fos") * co2_non_fos_egco2
egco2_wastewater_r   = g("egco2_wastewater") * wastewater_egco2
egco2_lignin_heat_r  = g("egco2_lignin_heat") * lignin_heat_egco2
egco2_hemi_r         = g("egco2_hemi") * hemi_egco2
# Setup calculation method
setup = o.CalculationSetup(
    target=model,
    impact_method=method,
    parameters=[
        o.ParameterRedef(name="egco2_eg",             value=egco2_eg_r),
        o.ParameterRedef(name="egco2_heat",           value=egco2_heat_r),
        o.ParameterRedef(name="egco2_wood",           value=egco2_wood_r),
        o.ParameterRedef(name="egco2_co2",            value=egco2_co2_r),
        o.ParameterRedef(name="egco2_h2o",            value=egco2_h2o_r),
        o.ParameterRedef(name="egco2_elect",          value=egco2_elect_r),
        o.ParameterRedef(name="egco2_co2_non_fos",    value=egco2_co2_non_fos_r),
        o.ParameterRedef(name="egco2_wastewater",     value=egco2_wastewater_r),
        o.ParameterRedef(name="egco2_lignin_heat",    value=egco2_lignin_heat_r),
        o.ParameterRedef(name="egco2_hemi",           value=egco2_hemi_r),
    ],
)
print("Getting values has started... Wait...\n")
# To get only climate change
result = client.calculate(setup)
result.wait_until_ready()
climate_change = next(float(i.amount) for i in result.get_total_impacts() if i.impact_category.name == "Climate change")
result.dispose()
print(f"Climate change (EFv3.1): {climate_change:.3f} kg CO2 eq per kg of pulp\n")
print("Wait...\n")
# To get all results
result = client.calculate(setup)
result.wait_until_ready()
impacts = result.get_total_impacts()
egco2_dict = {i.impact_category.name: (i.amount, i.impact_category.ref_unit) for i in impacts}
result.dispose()
# Setup only for Kraft
setup_kraft = o.CalculationSetup(
    target=model_kraft,
    impact_method=method)
# Run calculations for kraft
result_kraft = client.calculate(setup_kraft)
result_kraft.wait_until_ready()
impacts_kraft = result_kraft.get_total_impacts()
kraft_dict = {i.impact_category.name: (i.amount, i.impact_category.ref_unit) for i in impacts_kraft}
result_kraft.dispose()
# # Build a unified list of categories
# all_categories = sorted(set(impact_dict.keys()))
# # Assemble rows: Impact category, Amount (EG-CO2 pulping), Unit
# rows = []
# for cat in all_categories:
#     amt, unit = impact_dict.get(cat, (float('nan'), ""))
#     rows.append([cat, amt, unit])

# df_egco2_lca = pd.DataFrame(
#     rows,
#     columns=["Impact category", "EG-CO2 pulping", "Unit"]
# )
# Build a unified list of categories
all_categories = sorted(set(egco2_dict.keys()) | set(kraft_dict.keys()))
# Assemble rows: Impact category, Amount (lupine pulping), Amount (reed pulping), Unit
rows = []
for cat in all_categories:
    egco2_amt, egco2_unit = egco2_dict.get(cat, (float('nan'), ""))
    kraft_amt, kraft_unit = kraft_dict.get(cat, (float('nan'), ""))
    unit = egco2_unit if egco2_unit else kraft_unit
    rows.append([cat, egco2_amt, kraft_amt, unit])
df_egco2_lca = pd.DataFrame(
    rows,
    columns=["Impact category", "EG-CO2 pulping", "Kraft pulping", "Unit"]
)
# Print timing summary
elapsed = time.perf_counter() - t0
total_seconds = int(elapsed)
centi = int((elapsed - total_seconds) * 100)
h, rem = divmod(total_seconds, 3600)
m, s = divmod(rem, 60)
print(f"Execution completed. Total elapsed time (hh:mm:ss:cs): {h:02d}:{m:02d}:{s:02d}:{centi:02d}\n")
#print("\nLCA impacts (EFv3.1) per kg of pulp") # or TRACI2.1
#print(df_lca)
#%% LCA: Contribution tree
# Optimized upstream-tree extraction for
# Tuning parameters for the upstream tree expansion
MAX_EXPAND_LEVELS = 1#5     # maximum recursion depth
MAX_EXPAND_NODES = 8#50     # max number of children per node to traverse
def expand(node: utree.Node, level: int, rows: list, impact_idx: int, impact_name: str,
           unit: str, process_label: str):
    """Recursively expands an upstream tree and collects rows."""
    # Append to rows for DataFrame
    rows.append([
        process_label,
        impact_idx,
        impact_name,
        unit,
        level,
        node.provider.name if node.provider else "",
        node.result
    ])
    # Recurse
    if level < MAX_EXPAND_LEVELS:
        for c in node.childs[0:MAX_EXPAND_NODES]:
            expand(c, level + 1, rows, impact_idx, impact_name, unit, process_label)
def collect_upstream_for_process(client, process_ref, method_ref, process_label: str):
    """
    Runs an OpenLCA calculation, iterates all impact categories of the method,
    expands the upstream tree.
    """
    setup = o.CalculationSetup(target=process_ref, impact_method=method_ref,
                               parameters=[
                                   o.ParameterRedef(name="egco2_eg",             value=egco2_eg_r),
                                   o.ParameterRedef(name="egco2_heat",           value=egco2_heat_r),
                                   o.ParameterRedef(name="egco2_wood",           value=egco2_wood_r),
                                   o.ParameterRedef(name="egco2_co2",            value=egco2_co2_r),
                                   o.ParameterRedef(name="egco2_h2o",            value=egco2_h2o_r),
                                   o.ParameterRedef(name="egco2_elect",          value=egco2_elect_r),
                                   o.ParameterRedef(name="egco2_co2_non_fos",    value=egco2_co2_non_fos_r),
                                   o.ParameterRedef(name="egco2_wastewater",     value=egco2_wastewater_r),
                                   o.ParameterRedef(name="egco2_lignin_heat",    value=egco2_lignin_heat_r),
                                   o.ParameterRedef(name="egco2_hemi",           value=egco2_hemi_r),
                               ])
    result = client.calculate(setup)
    result.wait_until_ready()
    rows = []
    cats = result.get_impact_categories()
    for idx, ref in enumerate(cats):
        root = utree.of(result, ref)
        expand(root, 0, rows, idx, ref.name, ref.ref_unit, process_label)
    result.dispose()
    df_interim = pd.DataFrame(rows, columns=[
        "process",
        "impact_index",
        "impact_name",
        "unit",
        "level",
        "provider",
        "result"
    ])
    return df_interim
def get_pulping_upstream_dfs():
    """
    Runs upstream-tree calculations for EG-CO2 pulping
    and returns DataFrame: (df_egco2_lca_tree).
    """
    print("Getting values for contribution tree has started... Wait...\n")
    t0 = time.perf_counter()
    client = ipc.Client(8080)
    # Resolve processes by name (adjust if you use UUIDs instead)
    proc_egco2 = client.find(o.Process, "EG-CO2v2.0")
    # Impact method
    method = client.find(o.ImpactMethod, "EF v3.1") # or EF v3.1 (CO2only) or TRACI 2.1
    df_egco2_lca_tree = collect_upstream_for_process(client, proc_egco2, method, process_label="EG-CO2v2.0")
    #Timing summary
    elapsed = time.perf_counter() - t0
    total_seconds = int(elapsed)
    centi = int((elapsed - total_seconds) * 100)
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    print(f"Execution for contribution tree completed. Total elapsed time (hh:mm:ss:cs): {h:02d}:{m:02d}:{s:02d}:{centi:02d}")
    return df_egco2_lca_tree, method.name
df_egco2_lca_tree, method.name = get_pulping_upstream_dfs()
#%% LCA: Sankey diagram
# Only Climate Change
# Tuning parameters for the upstream tree expansion
MAX_EXPAND_LEVELS = 3#5     # number of subprocesses 
MAX_EXPAND_NODES = 6#50     # number of stages
def expand(node: utree.Node, level: int, rows: list, impact_idx: int, impact_name: str,
           unit: str, process_label: str):
    """Recursively expands an upstream tree and collects rows."""
    # Append to rows for DataFrame
    rows.append([
        process_label,
        impact_idx,
        impact_name,
        unit,
        level,
        node.provider.name if node.provider else "",
        node.result
    ])
    # Recurse
    if level < MAX_EXPAND_LEVELS:
        for c in node.childs[0:MAX_EXPAND_NODES]:
            expand(c, level + 1, rows, impact_idx, impact_name, unit, process_label)
def collect_upstream_for_process(client, process_ref, method_ref, process_label: str):
    """
    Runs an OpenLCA calculation, iterates all impact categories of the method,
    expands the upstream tree.
    """
    setup = o.CalculationSetup(target=process_ref, impact_method=method_ref,
                               parameters=[
                                   o.ParameterRedef(name="egco2_eg",             value=egco2_eg_r),
                                   o.ParameterRedef(name="egco2_heat",           value=egco2_heat_r),
                                   o.ParameterRedef(name="egco2_wood",           value=egco2_wood_r),
                                   o.ParameterRedef(name="egco2_co2",            value=egco2_co2_r),
                                   o.ParameterRedef(name="egco2_h2o",            value=egco2_h2o_r),
                                   o.ParameterRedef(name="egco2_elect",          value=egco2_elect_r),
                                   o.ParameterRedef(name="egco2_co2_non_fos",    value=egco2_co2_non_fos_r),
                                   o.ParameterRedef(name="egco2_wastewater",     value=egco2_wastewater_r),
                                   o.ParameterRedef(name="egco2_lignin_heat",    value=egco2_lignin_heat_r),
                                   o.ParameterRedef(name="egco2_hemi",           value=egco2_hemi_r),
                               ])
    result = client.calculate(setup)
    result.wait_until_ready()
    rows = []
    cats = result.get_impact_categories()
    for idx, ref in enumerate(cats):
        root = utree.of(result, ref)
        expand(root, 0, rows, idx, ref.name, ref.ref_unit, process_label)
    result.dispose()
    df_interim = pd.DataFrame(rows, columns=[
        "process",
        "impact_index",
        "impact_name",
        "unit",
        "level",
        "provider",
        "result"
    ])
    return df_interim
def get_pulping_upstream_dfs():
    """
    Runs upstream-tree calculations for EG-CO2 pulping
    and returns DataFrame: (df_egco2_lca_tree).
    """
    print("Getting values for Sankey diagram has started... Wait...\n")
    t0 = time.perf_counter()
    client = ipc.Client(8080)
    # Resolve processes by name (adjust if you use UUIDs instead)
    proc_egco2 = client.find(o.Process, "EG-CO2v2.0")
    # Impact method
    method = client.find(o.ImpactMethod, "EF v3.1 (CO2only)") # or EF v3.1 (CO2only) or TRACI 2.1
    df_egco2_lca_sankey = collect_upstream_for_process(client, proc_egco2, method, process_label="EG-CO2v2.0")
    #Timing summary
    elapsed = time.perf_counter() - t0
    total_seconds = int(elapsed)
    centi = int((elapsed - total_seconds) * 100)
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    print(f"Execution for contribution tree completed (Sankey). Total elapsed time (hh:mm:ss:cs): {h:02d}:{m:02d}:{s:02d}:{centi:02d}")
    return df_egco2_lca_sankey, method.name
df_egco2_lca_sankey, method.name = get_pulping_upstream_dfs()
#%% Linear Regression of Climate Change from OpenLCA
#To avoid interaction with OpenLCA for OVAT and MonteCarlo.
#Validate linearity and extract regression coefficients.
# See detailed explanation in Backup.
# Start timing
t0 = time.perf_counter()
# Connect to openLCA IPC
client = ipc.Client(8080)
# Find model and method
model = client.find(o.Process, "EG-CO2v2.0")
method = client.find(o.ImpactMethod, "EF v3.1 (CO2only)")
# Get global parameters once
print("\nGetting values for regression coefficients has started... Wait... \n")
params = client.get_all(o.Parameter)
def g(name: str) -> float:
    for p in params:
        if p.name == name and getattr(p, "context", None) is None:
            return float(p.value)
    raise KeyError(f"Global parameter '{name}' not found")
# Map global parameter names in the LCA model to i_egco2 variable names
param_map = [
    ("egco2_eg",            "eg_egco2"),
    ("egco2_heat",          "heat_egco2"),
    ("egco2_wood",          "wood_logs_egco2"),
    ("egco2_co2",           "co2_egco2"),
    ("egco2_h2o",           "h2o_egco2"),
    ("egco2_elect",         "elect_egco2"),
    # ("egco2_co2_non_fos",   "co2_non_fos_egco2"), # no effect on CC
    # ("egco2_wastewater",    "wastewater_egco2"), # no effect on CC
    ("egco2_lignin_heat",   "lignin_heat_egco2"),
    ("egco2_hemi",          "hemi_egco2"),
]
# Alter one variable while others = 0
base_multipliers = {v: 0.0 for _, v in param_map}
def build_setup(mult_overrides):
    mults = base_multipliers.copy()
    mults.update(mult_overrides)
    redefs = []
    for gp_name, mult_key in param_map:
        redefs.append(
            o.ParameterRedef(
                name=gp_name,
                value=g(gp_name) * mults[mult_key]
            )
        )
    return o.CalculationSetup(
        target=model,
        impact_method=method,
        parameters=redefs,
    )
# Values to alter for each i_egco2
test_values = [0.1, 1, 10] # keep "1" always in the middle of the array!!!
# Run "OVAT" for each variable and collect 10 DataFrames
dfs = {}  # dict: key=i_egco2 variable, value=DataFrame with columns ["i_egco2", "Climate Change"]
for _, var_key in param_map:
    results = []
    for val in test_values:
        setup = build_setup({var_key: val})
        result = client.calculate(setup)
        result.wait_until_ready()
        climate_change = next(
            float(i.amount)
            for i in result.get_total_impacts()
            if i.impact_category.name == "Climate change"
        )
        results.append(climate_change)
        result.dispose()
    df = pd.DataFrame({
        "i_egco2": test_values,
        "Climate Change": results
    })
    dfs[var_key] = df
# Create individual variables like df_eg_egco2, df_heat_egco2, ... for each dataframe
for k, df in dfs.items():
    globals()[f"df_{k}"] = df
# To get statistical validation to prove that Linear Regression of Climate Change from OpenLCA is linear
# Intercept should be zero, RSQ = 1, Slope is a value of Climate Change at i_egco2 = 1. 
for name, df in dfs.items():
    res = linregress(df["i_egco2"], df["Climate Change"])
    slope = res.slope
    intercept = res.intercept
    r2 = res.rvalue ** 2
    print(f"{name}: Slope={slope:.6f}, Intercept={intercept:.3f}, RSQ={r2:.6g}")
# To extract regressions coefficients, i.e. weights using values of slope.
# See back up to see how To extract regressions coefficients, i.e. weights at i_egco2 = 1 (the same results).
# Map DataFrame keys to desired variable names
var_map = [
    ("eg_egco2",          "w_eg"),
    ("heat_egco2",        "w_heat"),
    ("wood_logs_egco2",   "w_wood"),
    ("co2_egco2",         "w_co2"),
    ("h2o_egco2",         "w_h2o"),
    ("elect_egco2",       "w_elect"),
    # ("co2_non_fos_egco2", "w_co2_non_fos"), # no effect on CC
    # ("wastewater_egco2",  "w_wastewater"), # no effect on CC
    ("lignin_heat_egco2", "w_lignin_heat"),
    ("hemi_egco2",        "w_hemi"),
]
# Compute slopes once using linregress
slope_map = {}
for name, df in dfs.items():
    res = linregress(df["i_egco2"], df["Climate Change"])
    slope_map[name] = res.slope  # reuse this slope
# Assign slopes to the requested variables
for df_key, var_name in var_map:
    globals()[var_name] = float(slope_map[df_key])
# Optional: print results
for _, var_name in var_map:
    print(f"{var_name} = {globals()[var_name]:.6f}")
# to avoid "undefined names"
w_eg           = float(slope_map["eg_egco2"])
w_heat         = float(slope_map["heat_egco2"])
w_wood         = float(slope_map["wood_logs_egco2"])
w_co2          = float(slope_map["co2_egco2"])
w_h2o          = float(slope_map["h2o_egco2"])
w_elect        = float(slope_map["elect_egco2"])
# w_co2_non_fos  = float(slope_map["co2_non_fos_egco2"])
# w_wastewater   = float(slope_map["wastewater_egco2"])
w_lignin_heat  = float(slope_map["lignin_heat_egco2"])
w_hemi         = float(slope_map["hemi_egco2"])
# Execution time summary
elapsed = time.perf_counter() - t0
ts = int(elapsed); cs = int((elapsed - ts) * 100)
h, rem = divmod(ts, 3600); m, s = divmod(rem, 60)
print(f"\nExecution completed for regression coefficients. Total elapsed time (hh:mm:ss:cs): {h:02d}:{m:02d}:{s:02d}:{cs:02d}")
#%% Validation that Climate Change is similar to that obtained from OpenLCA
# The Climate Change linear regression is
cc = (w_eg * eg_egco2 + 
      w_heat * heat_egco2 +
      w_wood * wood_logs_egco2 + 
      w_co2 * co2_egco2 +
      w_h2o * h2o_egco2 +
      w_elect * elect_egco2 + 
      w_lignin_heat * lignin_heat_egco2 +
      w_hemi * hemi_egco2)
print("\nValue of Climate Change obtained by linear regression equation (kgCO2eq/kgPulp)", round(cc, 3))
# To print value obtained through the interaction with OpenLCA run cell '#%% LCA with OpenLCA using olca-ipc'.
# The values should be identical.
#%% LCA: OVAT without interaction with OpenLCA
# to see the old version, see backup files
# creating and collecting set of data for simulation
# min and max variation
min_var = 0.5
max_var = 1.5
# Define variables ranges
Loss_EG_i    = [0.02,        Loss_EG,      0.10]
yP_i         = [0.40,        yP,           0.70]
H2O_i        = [5.0,         H2O,          10.0]
Ewh_i        = [Ewh*min_var, Ewh,          Ewh*max_var]
Edef_i       = [Edef*min_var, Edef,        Edef*max_var]
eff_boiler_i = [0.60,        eff_boiler,   0.95]
lig_i = [0.2, lig, 0.35]
Loss_CO2_i = [0.02, Loss_CO2, 0.05]
SP_i = [0.75, SP, 0.94]
# Build a DataFrame with min, base, max for each OVAT variable
data_ovat_ranges = [
    {"variable": "Loss_EG",   "min": Loss_EG_i[0],   "base": Loss_EG_i[1],   "max": Loss_EG_i[2]},
    {"variable": "yP",        "min": yP_i[0],        "base": yP_i[1],        "max": yP_i[2]},
    {"variable": "H2O",       "min": H2O_i[0],       "base": H2O_i[1],       "max": H2O_i[2]},
    {"variable": "Ewh",       "min": Ewh_i[0],       "base": Ewh_i[1],       "max": Ewh_i[2]},
    {"variable": "Edef",      "min": Edef_i[0],      "base": Edef_i[1],      "max": Edef_i[2]},
    {"variable": "eff_boiler","min": eff_boiler_i[0],"base": eff_boiler_i[1],"max": eff_boiler_i[2]},
    {"variable": "lig",       "min": lig_i[0],       "base": lig_i[1],       "max": lig_i[2]},
    {"variable": "Loss_CO2",  "min": Loss_CO2_i[0],  "base": Loss_CO2_i[1],  "max": Loss_CO2_i[2]},
    {"variable": "SP",        "min": SP_i[0],        "base": SP_i[1],        "max": SP_i[2]},
]
df_egco2_ovat_ranges = pd.DataFrame(data_ovat_ranges, columns=["variable", "min", "base", "max"])
#print(df_ranges_ovat.round(3).to_string(index=False))
# Helper: compute Climate Change from a 10-tuple returned by egco2
# Tuple order: (eg_egco2, heat_egco2, wood_logs_egco2, co2_egco2, h2o_egco2,
#               elect_egco2, co2_non_fos_egco2, wastewater_egco2, lignin_heat_egco2, hemi_egco2)
def cc_from_inp(inp):
    eg, heat, wood, co2, h2o, elect, co2_non_fos, wastewater, lignin_heat, hemi = inp
    return (w_eg * eg +
            w_heat * heat +
            w_wood * wood +
            w_co2 * co2 +
            w_h2o * h2o +
            w_elect * elect +
            w_lignin_heat * lignin_heat +
            w_hemi * hemi)
# Base tuple and its CC
inp_base = (
    eg_egco2,
    heat_egco2,
    wood_logs_egco2,
    co2_egco2,
    h2o_egco2,
    elect_egco2,
    co2_non_fos_egco2,
    wastewater_egco2,
    lignin_heat_egco2,
    hemi_egco2
)
cc_base = cc_from_inp(inp_base)
# Minimum effects (OVAT)
in_EG_min          = egco2({**giv_dat, "Loss_EG":    Loss_EG_i[0]})
in_yP_min          = egco2({**giv_dat, "yP":         yP_i[0]})
in_H2O_min         = egco2({**giv_dat, "H2O":        H2O_i[0]})
in_Ewh_min         = egco2({**giv_dat, "Ewh":        Ewh_i[0]})
in_Edef_min        = egco2({**giv_dat, "Edef":       Edef_i[0]})
in_eff_boiler_min  = egco2({**giv_dat, "eff_boiler": eff_boiler_i[0]})
in_lig_min         = egco2({**giv_dat, "lig":        lig_i[0]})
in_Loss_CO2_min    = egco2({**giv_dat, "Loss_CO2":   Loss_CO2_i[0]})
in_SP_min          = egco2({**giv_dat, "SP":         SP_i[0]})

cc_EG_min         = cc_from_inp(in_EG_min)
cc_yP_min         = cc_from_inp(in_yP_min)
cc_H2O_min        = cc_from_inp(in_H2O_min)
cc_Ewh_min        = cc_from_inp(in_Ewh_min)
cc_Edef_min       = cc_from_inp(in_Edef_min)
cc_eff_boiler_min = cc_from_inp(in_eff_boiler_min)
cc_lig_min        = cc_from_inp(in_lig_min)
cc_Loss_CO2_min   = cc_from_inp(in_Loss_CO2_min)
cc_SP_min         = cc_from_inp(in_SP_min)

min_effect = [cc_EG_min, cc_yP_min, cc_H2O_min, cc_Ewh_min, cc_Edef_min, cc_eff_boiler_min,
              cc_lig_min, cc_Loss_CO2_min, cc_SP_min]
# No effect (baseline CC for each slot)
no_effect = [cc_base] * len(min_effect) # repeating i-times cc_base, while i is a number of altered given data.
# Maximum effects (OVAT)
in_EG_max          = egco2({**giv_dat, "Loss_EG":    Loss_EG_i[2]})
in_yP_max          = egco2({**giv_dat, "yP":         yP_i[2]})
in_H2O_max         = egco2({**giv_dat, "H2O":        H2O_i[2]})
in_Ewh_max         = egco2({**giv_dat, "Ewh":        Ewh_i[2]})
in_Edef_max        = egco2({**giv_dat, "Edef":       Edef_i[2]})
in_eff_boiler_max  = egco2({**giv_dat, "eff_boiler": eff_boiler_i[2]})
in_lig_max         = egco2({**giv_dat, "lig":        lig_i[2]})
in_Loss_CO2_max    = egco2({**giv_dat, "Loss_CO2":   Loss_CO2_i[2]})
in_SP_max          = egco2({**giv_dat, "SP":         SP_i[2]})

cc_EG_max         = cc_from_inp(in_EG_max)
cc_yP_max         = cc_from_inp(in_yP_max)
cc_H2O_max        = cc_from_inp(in_H2O_max)
cc_Ewh_max        = cc_from_inp(in_Ewh_max)
cc_Edef_max       = cc_from_inp(in_Edef_max)
cc_eff_boiler_max = cc_from_inp(in_eff_boiler_max)
cc_lig_max        = cc_from_inp(in_lig_max)
cc_Loss_CO2_max   = cc_from_inp(in_Loss_CO2_max)
cc_SP_max         = cc_from_inp(in_SP_max)

max_effect = [cc_EG_max, cc_yP_max, cc_H2O_max, cc_Ewh_max, cc_Edef_max, cc_eff_boiler_max,
              cc_lig_max, cc_Loss_CO2_max, cc_SP_max]
# Build DataFrame 
labels = ["EG losses", "yP", "H2O", "Ewh", "Edef", "eff_boiler", "lig", "Loss_CO2", "SP"]
df_egco2_ovat_results = pd.DataFrame({
    "varied parameter": labels,
    "min effect": min_effect,
    "no effect": no_effect,
    "max effect": max_effect,
})
print("\nOVAT simulation has been completed (no interaction with OpenLCA). The results are: \n")
print(df_egco2_ovat_results.round(3))
#%% LCA, Monte Carlo: no iteraction with OpenLCA
# To see old version see backup
# Helper already defined earlier; repeat here if needed
# Tuple order: (eg_egco2, heat_egco2, wood_logs_egco2, co2_egco2, h2o_egco2,
#               elect_egco2, co2_non_fos_egco2, wastewater_egco2, lignin_heat_egco2, hemi_egco2)
print("\nMonte Carlo simulation has been started (no interaction with OpenLCA).\n")
def cc_from_inp(inp):
    eg, heat, wood, co2, h2o, elect, co2_non_fos, wastewater, lignin_heat, hemi = inp
    return (w_eg * eg +
            w_heat * heat +
            w_wood * wood +
            w_co2 * co2 +
            w_h2o * h2o +
            w_elect * elect +
            w_lignin_heat * lignin_heat +
            w_hemi * hemi)
# LCA, Monte Carlo (no interaction with OpenLCA)
np.random.seed(42)# to have predictable entropy, the random sequence deterministic and reproducible. 
CI = 0.95
num_samples = 10000
var = 0.5
# Draws
Loss_EG_j    = np.random.triangular(Loss_EG_i[0],    Loss_EG_i[1],    Loss_EG_i[2],    size=num_samples)
yP_j         = np.random.triangular(yP_i[0],         yP_i[1],         yP_i[2],         size=num_samples)
H2O_j        = np.random.triangular(H2O_i[0],        H2O_i[1],        H2O_i[2],        size=num_samples)
Ewh_j        = np.random.normal(loc=Ewh_i[1],  scale=Ewh_i[1]*var,   size=num_samples)
Edef_j       = np.random.normal(loc=Edef_i[1], scale=Edef_i[1]*var,  size=num_samples)
eff_boiler_j = np.random.triangular(eff_boiler_i[0], eff_boiler_i[1], eff_boiler_i[2], size=num_samples)
# NEW: additional draws
lig_j        = np.random.triangular(lig_i[0],        lig_i[1],        lig_i[2],        size=num_samples)
Loss_CO2_j   = np.random.triangular(Loss_CO2_i[0],   Loss_CO2_i[1],   Loss_CO2_i[2],   size=num_samples)
SP_j         = np.random.triangular(SP_i[0],         SP_i[1],         SP_i[2],         size=num_samples)
# Run egco2 per sample; collect Climate Change values
cc_mc = np.empty(num_samples, dtype=float)
print_every = 500  # print progress every N iterations
t0 = time.perf_counter()
for k in range(num_samples):
    gd = {
        **giv_dat,
        "Loss_EG":    float(Loss_EG_j[k]),
        "yP":         float(yP_j[k]),
        "H2O":        float(H2O_j[k]),
        "Ewh":        float(Ewh_j[k]),
        "Edef":       float(Edef_j[k]),
        "eff_boiler": float(eff_boiler_j[k]),
        # NEW: pass added variables
        "lig":        float(lig_j[k]),
        "Loss_CO2":   float(Loss_CO2_j[k]),
        "SP":         float(SP_j[k]),
    }
    inp = egco2(gd)                # tuple of 10 i_egco2 values
    cc_mc[k] = cc_from_inp(inp)    # scalar Climate Change
    # progress print
    if (k + 1) % print_every == 0 or k == 0 or (k + 1) == num_samples:
        elapsed = time.perf_counter() - t0
        total_seconds = int(elapsed)
        centi = int((elapsed - total_seconds) * 100)  # centiseconds
        h, rem = divmod(total_seconds, 3600)
        m, s = divmod(rem, 60)
        print(
            f"Iteration {k+1}/{num_samples} | elapsed {h:02d}:{m:02d}:{s:02d}:{centi:02d} | "
            f"CC={cc_mc[k]:.3f}"
        )
# Summary and total time
elapsed_total = time.perf_counter() - t0
# Calculating mean 
mean_cc = float(np.mean(cc_mc))
# Calculation empirical quantilies
low, high = np.percentile(cc_mc, [(1 - CI) / 2 * 100, (1 + CI) / 2 * 100])
ts = int(elapsed_total)
centi = int((elapsed_total - ts) * 100)
h, rem = divmod(ts, 3600)
m, s = divmod(rem, 60)
print("\nMonte Carlo simulation has been completed (no connection to OpenLCA).\n")
print(f"MC Climate Change: mean={mean_cc:.3f}, {int(CI*100)}% CI=({low:.3f}, {high:.3f})\n")
print(f"Total elapsed time: {h:02d}:{m:02d}:{s:02d}:{centi:02d}\n")
#DataFrame
df_mc = pd.DataFrame({"cc_mc": cc_mc})
print(df_mc.head())
#%% To save data frames
# csv_path1 = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/EG_CO2/tables_egco2/df_given_data_egco2.csv')
# df_given_data_EGCO2.to_csv(csv_path1, index=False)
# # csv_path2 = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/EG_CO2/tables_egco2/df_material_balance_egco2.csv')
# # df_mat_bal_EGCO2.to_csv(csv_path2, index=True)
# csv_path3 = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/EG_CO2/tables_egco2/df_summary_egco2.csv')
# df_summary_egco2.to_csv(csv_path3, index=False)
# csv_path4 = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/EG_CO2/tables_egco2/df_egco2_lca.csv')
# df_egco2_lca.to_csv(csv_path4, index=False)
# csv_path5 = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/EG_CO2/tables_egco2/df_egco2_lca_tree.csv')
# df_egco2_lca_tree.to_csv(csv_path5, index=False)
# csv_path6 = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/EG_CO2/tables_egco2/df_egco2_ovat_ranges.csv')
# df_egco2_ovat_ranges.to_csv(csv_path6, index=False)
# csv_path7 = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/EG_CO2/tables_egco2/df_egco2_ovat_results.csv')
# df_egco2_ovat_results.to_csv(csv_path7, index=False)
# csv_path8 = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/EG_CO2/tables_egco2/df_mc.csv')  # change path/name if needed
# df_mc.to_csv(csv_path8, index=False)
# csv_path9 = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/EG_CO2/tables_egco2/df_egco2_lca_sankey.csv')
# df_egco2_lca_sankey.to_csv(csv_path9, index=False)

































