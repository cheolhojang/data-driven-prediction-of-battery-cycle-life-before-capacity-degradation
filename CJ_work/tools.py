import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import re
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy import optimize
from scipy import interpolate
from sympy.solvers import solve
from sympy import Symbol

qa = pd.read_csv('QA_metadata_tabDelimited.txt', sep = "\t")
simplified = qa[['ProcessDataID', 'CodeName', 'cathodeMass']]
simplified = simplified.dropna()

def path_listing(path):
    path_files = [f for f in listdir(path) if isfile(join(path, f)) if f[:13] == "ProcessDataID"]
    for i in path_files:
        print(i)
    return path_files

def test_func(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x**1 + e

def pred_diff_plot(input):
    if type(input) == list:
        df = pd.read_csv(input, sep = "\t")
        
    elif type(input) == str:
        assert(input[:13] == "ProcessDataID")
        df = pd.read_csv(input, sep = "\t")
    else:
        print("Wrong Input")
    simplified = df[['ProcessDataID', 'CodeName', 'cathodeMass']]
    simplified = simplified.dropna()
    i = list(simplified['ProcessDataID'])

    df['cathodeMass'] = np.ones(len(df)) * simplified.iloc[i.index(df['ProcessDataID'][0])].cathodeMass
    df['divided'] = df['StateCapacity_mAh']/df['cathodeMass']
    si = df[['ProcessDataID', 'StepNumber', 'Cycle', 'Step','State', 'Voltage_V', 'cathodeMass', 'StateCapacity_mAh', 'divided']]


    tenth_dis = si[(si['Cycle'] == 10) & (si['State'] == 'Discharge')]
    #tenth cycle plot
    tenth_charge = si[(si['Cycle'] == 10) & (si['State'] == 'Charge')]

    #plt.plot(tenth_charge['divided'], tenth_charge['Voltage_V'])
    #plt.plot(tenth_dis['divided'], tenth_dis['Voltage_V'])

    si = df[['ProcessDataID', 'StepNumber', 'Cycle', 'Step','State', 'Voltage_V', 'cathodeMass', 'StateCapacity_mAh', 'divided']]
    hundth_dis = si[(si['Cycle'] == 100) & (si['State'] == 'Discharge')]
    #tenth cycle plot
    hundth_charge = si[(si['Cycle'] == 100) & (si['State'] == 'Charge')]

    #plt.plot(hundth_charge['divided'], hundth_charge['Voltage_V'])
    #plt.plot(hundth_dis['divided'], hundth_dis['Voltage_V'])
    y_data = tenth_charge['Voltage_V']
    x_data = tenth_charge['divided']
    params, params_covariance = optimize.curve_fit(test_func, x_data, y_data,p0=[0, 0, 0, 0, 0])
    sols = []
    for i in hundth_charge['Voltage_V']:
        x = Symbol('x', real=True)
        solutions = solve(params[0]*x**4 + params[1]*x**3 + params[2]*x**2 + params[3]*x**1 + params[4] - i, x)
        #print(solutions[0])
        sols.append(solutions[0])
    hundth_charge_preds = {'voltage': hundth_charge['Voltage_V'], 'hund mass': hundth_charge['divided'], 'tenth mass': sols}
    df1 = pd.DataFrame(data = hundth_charge_preds)
    df1['difference'] = abs(df1['hund mass'] - df1['tenth mass'])
    fig1, ax1 = plt.subplots() 
    ax1.set_ylabel("Difference between the 10th and 100th Capacity")
    ax1.set_xlabel("Voltage")
    ax1.plot(df1['voltage'], df1['difference'])

def volt_statecap(lst_input):
    
    df = pd.read_csv(lst_input, sep = "\t")
    simplified = df[['ProcessDataID', 'CodeName', 'cathodeMass']]
    simplified = simplified.dropna()
    i = list(simplified['ProcessDataID'])

    df['cathodeMass'] = np.ones(len(df)) * simplified.iloc[i.index(df['ProcessDataID'][0])].cathodeMass
    df['divided'] = df['StateCapacity_mAh']/df['cathodeMass']
    si = df[['ProcessDataID', 'StepNumber', 'Cycle', 'Step','State', 'Voltage_V', 'cathodeMass', 'StateCapacity_mAh', 'divided']]


    tenth_dis = si[(si['Cycle'] == 10) & (si['State'] == 'Discharge')]
    #tenth cycle plot
    tenth_charge = si[(si['Cycle'] == 10) & (si['State'] == 'Charge')]

    si = df[['ProcessDataID', 'StepNumber', 'Cycle', 'Step','State', 'Voltage_V', 'cathodeMass', 'StateCapacity_mAh', 'divided']]
    hundth_dis = si[(si['Cycle'] == 100) & (si['State'] == 'Discharge')]
    #tenth cycle plot
    hundth_charge = si[(si['Cycle'] == 100) & (si['State'] == 'Charge')]

    fig1, ax1 = plt.subplots() 
    ax1.plot(tenth_charge['divided'], tenth_charge['Voltage_V'])
    ax1.plot(tenth_dis['divided'], tenth_dis['Voltage_V'])
    ax1.plot(hundth_charge['divided'], hundth_charge['Voltage_V'])
    ax1.plot(hundth_dis['divided'], hundth_dis['Voltage_V'])
    ax1.set_title("Tenth Charge and Discharge for " + lst_input)
    ax1.set_xlabel("State Capacity(mAh/g")
    ax1.set_ylabel("Voltage")

def interp(inpt):
    df = pd.read_csv(inpt, sep = "\t")
    i = list(simplified['ProcessDataID'])
    df['cathodeMass'] = np.ones(len(df)) * simplified.iloc[i.index(df['ProcessDataID'][1])].cathodeMass
    df['divided'] = df['StateCapacity_mAh']/df['cathodeMass']
    df = df[['ProcessDataID', 'StepNumber', 'Cycle', 'Step','State', 'Voltage_V', 'cathodeMass', 'StateCapacity_mAh', 'divided']]
    df = df[df['Step'] == "ApplyCurrent"]
    cycle_ten = df[df['Cycle'] == 10]
    cycle_hun = df[df['Cycle'] == 100]

    tenth_dis = cycle_ten[(cycle_ten['State'] == 'Discharge')]
    tenth_charge = cycle_ten[(cycle_ten['State'] == 'Charge')]

    hundth_dis = cycle_hun[(cycle_hun['State'] == 'Discharge')]
    hundth_charge = cycle_hun[(cycle_hun['State'] == 'Charge')]

    figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')

    fp = tenth_dis['divided']
    xp = tenth_dis['Voltage_V']
    new_vals = hundth_dis['Voltage_V']
    f = interpolate.interp1d(xp,fp,fill_value='extrapolate')
    hundth_dis['interp_divided'] = list(f(new_vals))
    hundth_dis['difference'] = abs(hundth_dis['divided'] - hundth_dis['interp_divided'])

    fpc = tenth_charge['divided']
    xpc = tenth_charge['Voltage_V']
    new_valsc = hundth_charge['Voltage_V']
    fc = interpolate.interp1d(xpc,fpc,fill_value='extrapolate')
    hundth_charge['interp_divided'] = list(fc(new_valsc))
    hundth_charge['difference'] = abs(hundth_charge['divided'] - hundth_charge['interp_divided'])

    plt.plot(hundth_charge['Voltage_V'], hundth_charge['difference'])
    plt.plot(hundth_dis['Voltage_V'], hundth_dis['difference'])

    #plt.plot(hundth_charge['Voltage_V'], hundth_charge['divided'])
    #plt.plot(hundth_dis['Voltage_V'], hundth_dis['divided'])

    #plt.plot(tenth_charge['Voltage_V'], tenth_charge['divided'])
    #plt.plot(tenth_dis['Voltage_V'], tenth_dis['divided'])