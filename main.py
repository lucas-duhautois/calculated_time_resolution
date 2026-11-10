from data_read import fast_file_reader
from mean_pulse import get_mean_pulse_strontium, plot_pulse, plot_save, plot_save_strontium, get_folder_name, mean_pulse, standardize
from HV_analysis import plot_HV_from_save, function_max, function_time_down, function_integrate, function_time_up, plot_function_HV_from_save, get_HV, plot_function_from_save_strontium
from peak_analysis import plot_difference, plot_scatter_max, plot_max_std, get_pulse, get_stats, mean_max, void_parameters, get_mean_max_from_save, get_t_arrival_dif, get_t_arrival, get_multiple_max, get_t_arrival_1_pulse
from noise_analysis import plot_noise_std, noise_std, get_limit_noise_once, get_noise_mean_from_save, multiple_noise_std, get_limit_multiple_noise_once
from strontium import pulse_distribution
from utils import test_noise, test_saturation, test_noise_old

from scipy.stats import ks_1samp, norm
from scipy.optimize import curve_fit
from scipy.special import erf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import zipfile

# General informations
n_seg = -1
len_seg = -1
# n_seg = 100           # always use this (with the correct value) for zip
# len_seg = 50002        # always use this (with the correct value) for zip


# SantaCruz laser

HV_value = 100
ampl_value = 1500
offset_value = 750

global_folder_path = "../Data/SantaCruz/laser/"
folder_list = ["SantaCruz_epi_HG_hv" + str(HV_value) + "_opv300_amp_"+ str(ampl_value)+"_off_"+str(offset_value)+".zip"]
HV_list = [HV_value]
save_path = "new_saves/SantaCruz_laser/amp_" + str(ampl_value) + "_off_" + str(offset_value)
save_path_mean = "new_saves/SantaCruz_laser/amp_" + str(ampl_value) + "_off_" + str(offset_value) + "/mean_pulse"
channel = ("C2")
exp = "IR"


# Parameters for some plots
title_name = "SantaCruz laser"               # for the titles of plots
sub_string = ""                             # if particles, indicates the segment of amplitude for HV_analysis
normalized = False                          # whether you want some results from peak_analysis/noise_analysis to be divided by the max of the mean pulse
standard = False                            # whether you want the curves to be standardized between 0 and 1
min_ampl = 0                                # minimal amplitude for plots of particle mean pulses 
max_ampl = -1                              # maximal amplitude for plots of particle mean pulses (-1 to see all curves)
bins = 100                                  # nb of segments in histogram for pulse distribution
log = True                                  # whether the scale of the histogram for pulse distribution is in log scale
mini_t_arrival = 0.0e-9
maxi_t_arrival = 2.5e-9
window = []
verbose = 1

intervals = []                              # intervals of amplitude distribution (every end included)

#intervals = [0.4,2.0,2.5,3.0,3.5,4.2,5.2,7.0,10.5,18.0,69.0] # Santa Cruz Sr 1mVp

# intervals = [10.1,22.2,24.4,26.2,27.9,29.7,31.7,34.2,38.1,47.4,315.0] #HRHG_structure2 HV60 [27.9,29.7]
# intervals = [17.2,36.0,40.9,44.2,47.3,50.2,53.7,57.8,64.4,77.5,365.0] #HRHG_structure2 HV75 [44.2,47.3]
# intervals = [16.2,42.9,61.0,67.4,72.5,77.1,82.4,89.0,99.2,121.2,345.0] #HRHG_structure2 HV100 [72.5,77.1]
# intervals = [15.2,51.1,69.7,78.8,84.6,90.0,96.9,104.9,116.8,139.9,368.0] #HRHG_structure2 HV125 [84.6,90.0]
# intervals = [15.4,43.9,76.3,87.3,95.3,102.6,109.6,118.7,132.2,159.8,345.0] #HRHG_structure2 HV150 [95.3,102.6]

# intervals = [12.8,19.3,20.6,21.7,22.8,24.1,25.7,28.2,32.5,43.3,150.0] #epiHG_structure5 HV60 [21.7,22.8]
# intervals = [3.7,19.7,21.2,22.4,23.7,25.3,27.2,29.6,33.8,43.2,150.0] #epiHG_structure5 HV75 [21.2,22.4]
# intervals = [15.2,23.4,26.2,28.5,30.7,33.2,36.0,39.6,45.0,57.1,385.0] #epiHG_structure5 HV100 [26.2,28.5]
# intervals = [17.5,35.0,39.7,43.4,46.6,50.1,54.0,59.1,66.8,82.8,385.0] #epiHG_structure5 HV120 [43.4,46.6]
# intervals = [16.4,46.1,52.8,57.5,61.9,66.6,71.6,78.4,87.8,106.2,386.0] #epiHG_structure5 HV130 [61.9,66.6]

# intervals = [0.9,1.8,2.0,2.2,2.4,2.6,2.8,3.1,3.6,4.5,52.0] #santa cruz PM new 2.2 in [2.2,2.4]
# intervals = [0.7,1.9,2.2,2.4,2.6,2.8,3.0,3.3,3.7,4.5,15.2] #santa cruz 2 PMs new 2.5 in [2.4,2.6]

thresholds = [str(0.5), str(0.5)]
HV_plot_list = []
ampl_plot_list = []


# Check list
check_invert = False                        # plots some pulses for each channel to check if it is upside down + tune the filter for testbeam
check_noise = False
check_noise_old = False
check_sat = False
calculations = False                        # mean pulse calculations + amplitude distribution + noise standard deviation
plot_mean = False                           # plots the mean pulse(s)
HV_analysis = False                         # plots HV related curves : mean pulse, max, area, time up and time down
peak_analysis = False                       # plots peak related curves : difference mean(max)-max(mean), scatter plot of max, standard deviation
noise_analysis = False                      # plots noise related curves : standard deviation
pulse_distrib = False                       # plots the distribution of pulse's amplitude
dif_time_arrival = False                    # plots the distribution and standard deviation of differences of time of arrival between channels
dif_time_arrival_distrib = False             # plots the distribution of differences of time of arrival of adjacent pulses on the same channel
study_time_arrival_HV = False
study_time_arrival_ampl = False
width_distribution = False
study_amplitude = False
dif_time_arrival_distrib_2_channels = False
dif_time_end_distrib_2_channels = False


#########################################################################################################################################################

if global_folder_path[-1] != "/" and global_folder_path[-4:] != ".zip" :
    global_folder_path += "/"
folder_name = get_folder_name(global_folder_path)

if save_path[-1] != "/" :
    save_path += "/"

if type(channel) == str :
    channel = (channel,)
nb_channel = len(channel)

invert = np.array([False] * nb_channel)
t_limit = np.array([0] * nb_channel)
filter_noise = np.array([1.0] * nb_channel)
filter_noise_old = np.array([0.0] * nb_channel)
filter_saturation = np.array([100.0] * nb_channel)

if check_invert :
    folder_test = os.listdir(global_folder_path)[0]

    if os.path.isfile(global_folder_path + folder_test) :
        folder_test = ""
    if len(folder_list) > 0 :
        folder_test = folder_list[0]
    for i in range(nb_channel) :
        if (global_folder_path + folder_test)[-4:] == ".zip" :
            with zipfile.ZipFile(global_folder_path + folder_test) as z:
                for file in z.namelist():
                    with z.open(file) as f:
                        if channel[i] in f.name :
                            
                            n_seg, len_seg, data_signal = fast_file_reader(f, n_seg=n_seg, len_seg=len_seg)
                            pulse0 = get_pulse(data_signal, len_seg, 0)
                            pulse1 = get_pulse(data_signal, len_seg, 1)
                            pulse2 = get_pulse(data_signal, len_seg, 2)
                            pulse3 = get_pulse(data_signal, len_seg, 3)
                            pulse4 = get_pulse(data_signal, len_seg, 4)
                            plot_pulse(pulse0[0], pulse0[1], title="Test " + channel[i], show = False)
                            plot_pulse(pulse1[0], pulse1[1], title="Test " + channel[i], show = False)
                            plot_pulse(pulse2[0], pulse2[1], title="Test " + channel[i], show = False)
                            plot_pulse(pulse3[0], pulse3[1], title="Test " + channel[i], show = False)
                            plot_pulse(pulse4[0], pulse4[1], title="Test " + channel[i], show = True)
                            test = input("Is the signal upside down ? [y/n] ")
                            if test == "y" :
                                invert[i] = True
                            break
        else :
            for f in os.scandir (global_folder_path + folder_test) :
                if channel[i] in f.name :
                    
                    n_seg, len_seg, data_signal = fast_file_reader(f.path)
                    pulse0 = get_pulse(data_signal, len_seg, 0)
                    pulse1 = get_pulse(data_signal, len_seg, 1)
                    pulse2 = get_pulse(data_signal, len_seg, 2)
                    pulse3 = get_pulse(data_signal, len_seg, 3)
                    pulse4 = get_pulse(data_signal, len_seg, 4)
                    plot_pulse(pulse0[0], pulse0[1], title="Test " + channel[i], show = False)
                    plot_pulse(pulse1[0], pulse1[1], title="Test " + channel[i], show = False)
                    plot_pulse(pulse2[0], pulse2[1], title="Test " + channel[i], show = False)
                    plot_pulse(pulse3[0], pulse3[1], title="Test " + channel[i], show = False)
                    plot_pulse(pulse4[0], pulse4[1], title="Test " + channel[i], show = True)
                    test = input("Is the signal upside down ? [y/n] ")
                    if test == "y" :
                        invert[i] = True
                    break
    print(invert)

if check_noise :
    if len(folder_list) == 0 :
        folder_test_list = [os.listdir(global_folder_path)[0]]
    else :
        folder_test_list = folder_list
    plt.clf()
    test_channel = [False] * nb_channel
    if os.path.isfile(global_folder_path + folder_test_list[0]) and (global_folder_path + folder_test_list[0])[-4:] != ".zip" :
            folder_test_list = [""] * len(folder_test_list)
    for i in range (nb_channel) :
        print("Starting check_noise for channel " + channel[i])

        test_filter_noise = False
        useless_files =  []
        while not(test_filter_noise) :
            for folder_test in folder_test_list :
                if (global_folder_path + folder_test)[-4:] == ".zip" :
                    with zipfile.ZipFile(global_folder_path + folder_test) as z:
                        for file in z.namelist():
                            with z.open(file) as f:
                                if (channel[i] in f.name) and not(test_channel[i]) and not(str(folder_test) + str(f.name) in useless_files) :
                                    file_name = f.name
                                    file_folder = folder_test
                                    
                                    # check the noise filter by plotting correct signals
                                    n_seg, len_seg, data_signal = fast_file_reader(f, n_seg=n_seg, len_seg=len_seg)
                                    if invert[i] :
                                        data_signal = - data_signal
                                    nb_plot = 0
                                    for j in range (min(n_seg,100)):
                                        t, signal = get_pulse(data_signal, len_seg, j)
                                        if test_noise(signal, filter_noise[i]) :
                                            signal = standardize(signal)
                                            plot_pulse(t, signal, title="Test filter noise " + channel[i], show = False)
                                            nb_plot += 1

                                    if nb_plot > 0 :
                                        print(nb_plot)
                                        plt.show()
                                        test_channel[i] = True
                else :
                    for f in os.scandir(global_folder_path + folder_test) :
                        if (channel[i] in f.name) and not(test_channel[i]) and not(str(folder_test) + str(f.name) in useless_files) :
                            file_name = f.name
                            file_folder = folder_test
                            
                            # check the noise filter by plotting correct signals
                            n_seg, len_seg, data_signal = fast_file_reader(f.path)
                            if invert[i] :
                                data_signal = - data_signal
                            nb_plot = 0
                            for j in range (min(n_seg,100)):
                                t, signal = get_pulse(data_signal, len_seg, j)
                                if test_noise(signal, filter_noise[i]) :
                                    signal = standardize(signal)
                                    plot_pulse(t, signal, title="Test filter noise " + channel[i], show = False)
                                    nb_plot += 1

                            # check the noise filter by plotting incorrect signals
                            # n_seg, len_seg, data_signal = fast_file_reader(f.path)
                            # j = 0
                            # nb_plot = 0
                            # while (j < n_seg) and (nb_plot < 50) :
                            #     t, signal = get_pulse(data_signal, len_seg, j)
                            #     if not(test_noise(signal, filter_noise[i])) :
                            #         signal = standardize(signal)
                            #         plot_pulse(t, signal, title="Test filter noise " + channel[i], show = False)
                            #         nb_plot += 1
                            #     j += 1
                            if nb_plot > 0 :
                                print(nb_plot)
                                plt.show()
                                test_channel[i] = True
                        
            print("Current filter value : " + str(filter_noise[i]))
            test = input("Was this file useful ? [y/n] ")
            if test == "n" :
                useless_files.append(str(file_folder) + str(file_name))
                print(useless_files)
                test_channel[i] = False
            else :
                test = input("What value do you want filter to be ? ")
                filter_noise[i] = float(test)
    
                test = input("Do you want to check the filter again ? [y/n] ")
                if test == "n" :
                    test_filter_noise = True
                else :
                    test_channel[i] = False
    print(filter_noise)

if check_noise_old :
    if len(folder_list) == 0 :
        folder_test_list = [os.listdir(global_folder_path)[0]]
    else :
        folder_test_list = folder_list
    plt.clf()
    test_channel = [False] * nb_channel
    if os.path.isfile(global_folder_path + folder_test_list[0]) and (global_folder_path + folder_test_list[0])[-4:] != ".zip" :
            folder_test_list = [""] * len(folder_test_list)
    for i in range (nb_channel) :
        print("Starting check_noise for channel " + channel[i])

        test_filter_noise = False
        useless_files =  []
        while not(test_filter_noise) :
            for folder_test in folder_test_list :
                print(folder_test)
                if (global_folder_path + folder_test)[-4:] == ".zip" :
                    with zipfile.ZipFile(global_folder_path + folder_test) as z:
                        for file in z.namelist():
                            with z.open(file) as f:
                                if (channel[i] in f.name) and not(test_channel[i]) and not(str(folder_test) + str(f.name) in useless_files) :
                                    file_name = f.name
                                    file_folder = folder_test
                                    
                                    # check the noise filter by plotting correct signals
                                    n_seg, len_seg, data_signal = fast_file_reader(f, n_seg=n_seg, len_seg=len_seg)
                                    if invert[i] :
                                        data_signal = - data_signal
                                    nb_plot = 0
                                    for j in range (min(n_seg,100)):
                                        t, signal = get_pulse(data_signal, len_seg, j)
                                        if test_noise_old(signal, filter_noise_old[i]) :
                                            signal = standardize(signal)
                                            plot_pulse(t, signal, title="Test filter noise " + channel[i], show = False)
                                            nb_plot += 1

                                    if nb_plot > 0 :
                                        print(nb_plot)
                                        plt.show()
                                        test_channel[i] = True
                else :
                    for f in os.scandir(global_folder_path + folder_test) :
                        if (channel[i] in f.name) and not(test_channel[i]) and not(str(folder_test) + str(f.name) in useless_files) :
                            file_name = f.name
                            file_folder = folder_test
                            
                            # check the noise filter by plotting correct signals
                            n_seg, len_seg, data_signal = fast_file_reader(f.path)
                            if invert[i] :
                                data_signal = - data_signal
                            nb_plot = 0
                            for j in range (min(n_seg,100)):
                                t, signal = get_pulse(data_signal, len_seg, j)
                                if test_noise_old(signal, filter_noise_old[i]) :
                                    signal = standardize(signal)
                                    plot_pulse(t, signal, title="Test filter noise " + channel[i], show = False)
                                    nb_plot += 1

                            # check the noise filter by plotting incorrect signals
                            # n_seg, len_seg, data_signal = fast_file_reader(f.path)
                            # j = 0
                            # nb_plot = 0
                            # while (j < n_seg) and (nb_plot < 50) :
                            #     t, signal = get_pulse(data_signal, len_seg, j)
                            #     if not(test_noise(signal, filter_noise[i])) :
                            #         signal = standardize(signal)
                            #         plot_pulse(t, signal, title="Test filter noise " + channel[i], show = False)
                            #         nb_plot += 1
                            #     j += 1
                            if nb_plot > 0 :
                                print(nb_plot)
                                plt.show()
                                test_channel[i] = True
                        
            print("Current filter value : " + str(filter_noise_old[i]))
            test = input("Was this file useful ? [y/n] ")
            if test == "n" :
                useless_files.append(str(file_folder) + str(file_name))
                print(useless_files)
                test_channel[i] = False
            else :
                test = input("What value do you want filter to be ? ")
                filter_noise_old[i] = float(test)
    
                test = input("Do you want to check the filter again ? [y/n] ")
                if test == "n" :
                    test_filter_noise = True
                else :
                    test_channel[i] = False
    print(filter_noise)


if check_sat :
    if len(folder_list) == 0 :
        folder_test_list = [os.listdir(global_folder_path)[0]]
    else :
        folder_test_list = folder_list
    plt.clf()
    test_channel = [False] * nb_channel
    if os.path.isfile(global_folder_path + folder_test_list[0]) and (global_folder_path + folder_test_list[0])[-4:] != ".zip" :
            folder_test_list = [""] * len(folder_test_list)
    for i in range (nb_channel) :
        print("Starting check_sat for channel " + channel[i])
        test_filter_saturation = False
        useless_files = []
        while not(test_filter_saturation) :
            for folder_test in folder_test_list:
                if (global_folder_path + folder_test)[-4:] == ".zip" :
                    with zipfile.ZipFile(global_folder_path + folder_test) as z:
                        for file in z.namelist():
                            with z.open(file) as f:
                                if (channel[i] in f.name) and not(test_channel[i]) and not(str(folder_test) + str(f.name) in useless_files) :
                                    file_name = f.name
                                    file_folder = folder_test
                                    
                                    # check the saturation filter by plotting correct signals
                                    n_seg, len_seg, data_signal = fast_file_reader(f, n_seg=n_seg, len_seg=len_seg)
                                    if invert[i] :
                                        data_signal = - data_signal
                                    nb_plot = 0
                                    for j in range (min(n_seg,100)):
                                        t, signal = get_pulse(data_signal, len_seg, j)
                                        if test_saturation(signal, filter_saturation[i]) :
                                            plot_pulse(t, signal, title="Test filter saturation " + channel[i], show = False)
                                            nb_plot += 1
        
                                    # check the saturation filter by plotting incorrect signals
                                    # n_seg, len_seg, data_signal = fast_file_reader(f.path)
                                    # j = 0
                                    # nb_plot = 0
                                    # while (j < n_seg) and (nb_plot < 50) :
                                    #     t, signal = get_pulse(data_signal, len_seg, j)
                                    #     if not(test_saturation(signal, filter_saturation[i])) :
                                    #         plot_pulse(t, signal, title="Test filter saturation " + channel[i], show = False)
                                    #         nb_plot += 1
                                    #     j += 1
            
                                    if nb_plot > 0 :
                                        print(nb_plot)
                                        plt.show()
                                        test_channel[i] = True
                else :
                    for f in os.scandir(global_folder_path + folder_test) :
                        if (channel[i] in f.name) and not(test_channel[i]) and not(str(folder_test) + str(f.name) in useless_files) :
                            file_name = f.name
                            file_folder = folder_test
                            
                            # check the saturation filter by plotting correct signals
                            n_seg, len_seg, data_signal = fast_file_reader(f.path)
                            if invert[i] :
                                data_signal = - data_signal
                            nb_plot = 0
                            for j in range (min(n_seg,100)):
                                t, signal = get_pulse(data_signal, len_seg, j)
                                if test_saturation(signal, filter_saturation[i]) :
                                    plot_pulse(t, signal, title="Test filter saturation " + channel[i], show = False)
                                    nb_plot += 1

                            # check the saturation filter by plotting incorrect signals
                            # n_seg, len_seg, data_signal = fast_file_reader(f.path)
                            # j = 0
                            # nb_plot = 0
                            # while (j < n_seg) and (nb_plot < 50) :
                            #     t, signal = get_pulse(data_signal, len_seg, j)
                            #     if not(test_saturation(signal, filter_saturation[i])) :
                            #         plot_pulse(t, signal, title="Test filter saturation " + channel[i], show = False)
                            #         nb_plot += 1
                            #     j += 1

                            if nb_plot > 0 :
                                print(nb_plot)
                                plt.show()
                                test_channel[i] = True
                        
            print("Current filter value : " + str(filter_saturation[i]))

            test = input("Was this file useful ? [y/n] ")
            if test == "n" :
                useless_files.append(str(file_folder) + str(file_name))
                print(useless_files)
                test_channel[i] = False
            else :
                test = input("What value do you want filter to be ? ")
                filter_saturation[i] = float(test)
                
                test = input("Do you want to check the filter again ? [y/n] ")
                if test == "n" :
                    test_filter_saturation = True
                else :
                    test_channel[i] = False
    print(filter_saturation)

if calculations :
    if exp == "IR" :
        def parameters_t_arrival (global_folder_path, channel, save_path, invert, verbose, folder_list = [], HV_list = [], n_seg=-1, len_seg=-1) :
            return (5,) # à améliorer : demander combien il y a de pulses... (si pas de threshold => tous les threshold entre 0.1 et 0.9...)

        get_stats(global_folder_path, channel, (get_t_arrival, get_t_arrival_dif, mean_pulse, multiple_noise_std, get_multiple_max), (parameters_t_arrival, parameters_t_arrival, void_parameters, get_limit_multiple_noise_once, parameters_t_arrival), folder_list=folder_list, HV_list=HV_list, save = True, save_path=save_path, invert=tuple(invert), n_seg=n_seg, len_seg=len_seg)
        
    elif exp == "particles" :
        get_stats(global_folder_path, channel, (mean_max, get_t_arrival_1_pulse, noise_std), (void_parameters, void_parameters, get_limit_noise_once), folder_list=folder_list, HV_list=HV_list, save = True, save_path=save_path, filter_noise = tuple(filter_noise), filter_saturation=tuple(filter_saturation), invert=tuple(invert), multiple_channel_analysis=True, n_seg=n_seg, len_seg=len_seg)
        
        # mean pulse
        if len(intervals) != 0 :    # use the pulse distribution to get the intervals
            t_start = time.time()
            if folder_list == [] :
                for d in os.scandir(global_folder_path) :
                    folder_list.append(d.name)
            if HV_list == [] :
                for f in folder_list :
                    HV_list.append(get_HV(f))

            nb_folder = len(HV_list)

            for i in range(nb_folder) :

                completion = int((i/nb_folder)*100)
                print("Mean pulse calculations completed by : " + str(completion) + "%")
                if completion != 0 :
                    remaining = (100 - completion) * (time.time() - t_start) / completion
                    print("Estimated remaining time : " + str(round(remaining, 2)) + " seconds")

                get_mean_pulse_strontium(tuple([global_folder_path + f for f in folder_list]), channel, 4, intervals=intervals, size_interval=size_ampl, plot=False, save=True, invert = tuple(invert), save_path=save_path, min_ampl=min_ampl, max_ampl=max_ampl, filter_noise=tuple(filter_noise), filter_saturation=tuple(filter_saturation), len_seg=len_seg, n_seg=n_seg)
        
if plot_mean :
    if exp == "IR" :
        HV_value = input ("Which HV value do you want to plot ? ")
        for ch in channel :
            for f in os.scandir(save_path_mean) :
                file_name = f.name
                if (ch in file_name) and (int(HV_value) == get_HV(file_name)) :
                    plot_save(str(f.path), title=file_name, standard=standard)
    
    elif exp == "particles" :
        for ch in channel :
            HV_value = input ("Which HV value do you want to plot ? ")
            plot_save_strontium(save_path_mean, ch, int(HV_value), title = title_name, standard=standard, min_ampl=min_ampl, max_ampl=max_ampl, window=window)

if HV_analysis :
    if exp == "particles" :
        HV_value = input ("Which HV value do you want to test ? ")
        plot_function_from_save_strontium(save_path_mean, channel, function_time_up, int(HV_value), title="Convolved signal " + title_name, save_list=[], window=window)
        plot_function_from_save_strontium(save_path_mean, channel, function_time_down, int(HV_value), title="Convolved signal " + title_name, save_list=[], window=window)
    else :
        plot_HV_from_save(save_path_mean, channel, title = title_name, sub_string=sub_string, standard=standard, save_list=[], HV_list=[])
        plot_function_HV_from_save(save_path_mean, channel, function_max, title="Convolved signal " + title_name, sub_string=sub_string, save_list=[], HV_list=[], n_pulse=1)
        plot_function_HV_from_save(save_path_mean, channel, function_integrate, title="Convolved signal " + title_name, sub_string=sub_string, save_list=[], HV_list=[])
        plot_function_HV_from_save(save_path_mean, channel, function_time_up, title="Convolved signal " + title_name, sub_string=sub_string, save_list=[], HV_list=[], n_pulse=1)
        plot_function_HV_from_save(save_path_mean, channel, function_time_down, title="Convolved signal " + title_name, sub_string=sub_string, save_list=[], HV_list=[])

if peak_analysis :
    for i in range(nb_channel) :
        get_mean_max_from_save(save_path, folder_name, channel[i], tbeam = False)
        plot_difference(global_folder_path, save_path_mean, channel[i], normalized=normalized, invert=invert[i], save_path=save_path, verbose=(verbose>=2))
        HV_value = input ("Which HV value do you want for scatter plot ? ")
        plot_scatter_max(global_folder_path, save_path_mean, channel[i], int(HV_value), save_path=save_path, invert=invert[i], verbose=(verbose>=2))
        plot_max_std(global_folder_path, save_path_mean, channel[i], normalized=normalized, invert=invert[i], save_path=save_path, verbose=(verbose>=2))

if noise_analysis :
    for i in range(nb_channel) :
        get_noise_mean_from_save(save_path, folder_name, channel[i], tbeam= False)
        plot_noise_std(global_folder_path, save_path_mean, channel[i], normalized=normalized, save_path=save_path, verbose=(verbose>=1))

if pulse_distrib :
    for i in range (nb_channel) :
        test_distribution = False
        while not(test_distribution) :
            HV_value = input ("Which HV value do you want to test the amplitude distribution ? ")
            pulse_distribution(global_folder_path, channel[i], int(HV_value), save_path=save_path, bins=int(bins), log=log, invert = invert[i], folder_list=folder_list, HV_list=HV_list, verbose=(verbose>=2), intervals=intervals)
            test = input("Do you need to run the histogram again ? [y/n] ")
            if test == "n" :
                test_distribution = True

if dif_time_arrival_distrib :
    # thresholds = [str(0.3), str(0.5), str(0.5)]
    #px12 HV : C2=0.3, C3=0.5
    #px12 ampl : C2=0.3, C3=0.5 (0.4 moche pour C2 1100mV)
    # bof à partir de 1450mV sur C2 (mieux à thr=0.4 pour 1450 et à thr=0.5 pour 1500) et faux à partir de 1400mV sur C3
    #px22 ampl : C2=0.5, C3=0.5
    #px22 ampl C3 : bof...

    def gaus(x,c,mu,sigma) :
        #return (c/(sigma*np.sqrt(2*np.pi)))*np.exp(-((x-mu)**2) / (2*sigma**2))
        return c*np.exp(-((x-mu)**2) / (2*sigma**2))
    
    def gaus_cumulative(x,c,mu,sigma) :
        return ((c*sigma*np.sqrt(2*np.pi)/2) * (1 + erf((x-mu) / (sigma * np.sqrt(2)))))
    
    for i in range(len(channel)) :
        threshold = thresholds[i]
        df = pd.read_csv(save_path + "dif_t_arrival_" + folder_name + "_" + channel[i] + "_" + threshold)
        data = np.array(df["dif_t_arr"])
        # print(save_path + "dif_t_arrival_" + folder_name + "_" + channel[i] + "_" + threshold)
        print()
        print("Ecart-type : " + str(np.std(data)))
        print("Nb mesures : " + str(len(data)))
        print()

        n, bin_edges, patches = plt.hist(np.array(df["dif_t_arr"]), log=log, bins=bins)

        x_hist = np.zeros(len(n[n!=0.]), dtype=float)
        k = 0
        for j in range (len(n)) :
            if n[j] != 0. :
                x_hist[k] = (bin_edges[j+1] + bin_edges[j]) /2
                k += 1

        param_optimised,param_covariance_matrix = curve_fit(gaus,x_hist,n[n!=0],p0=[max(n),np.mean(np.array(df["dif_t_arr"])),np.std(np.array(df["dif_t_arr"]))])

        # x_hist = np.zeros(len(n), dtype=float)
        # k = 0
        # for j in range (len(n)) :
        #     x_hist[j] = (bin_edges[j+1] + bin_edges[j]) /2

        # param_optimised,param_covariance_matrix = curve_fit(gaus,x_hist,n,p0=[max(n),np.mean(np.array(df["dif_t_arr"])),np.std(np.array(df["dif_t_arr"]))])
        
        c, mu, sigma = param_optimised
        print("C = " + str(c))
        print("mu = " + str(mu))
        print("sigma = " + str(sigma))
        print("Différence moyennes : " + str(np.abs(mu - np.mean(data))))
        print("Différence écart-type : " + str(np.abs(sigma - np.std(data))))

        x_hist_2=np.linspace(np.min(x_hist),np.max(x_hist),1000)
        plt.plot(x_hist_2,gaus(x_hist_2,*param_optimised),'r--',label='Gaussian fit')
        plt.legend()

        #res = ks_2samp(n,gaus(x_hist_2,*param_optimised))
        #res = ks_1samp(n,lambda x : gaus_cumulative(x,*param_optimised))

        data = np.array(df["dif_t_arr"])

        res = ks_1samp(data,lambda x : norm.cdf(x, mu, sigma))
        # print(res.statistic)
        # print(res.pvalue)

        # D, p = lilliefors(data, dist='norm', pvalmethod='table')
        # print(D)
        # print(p)

        # res = anderson(data, dist='norm', method='interpolate')
        # print(res.statistic)
        # print(res.pvalue)

        plt.title("Difference of time of arrival for MiniCactus v2 " + channel[i] + " (threshold " + threshold +")")
        plt.xlabel("Time (s)")
        plt.ylim(bottom=0.5, top=max(n)*1.1)
        plt.show()

        # probplot(data, dist='norm', plot=plt)
        # plt.show()

        # alpha = np.arange(1e-10,1e-8,1e-10)
        # plt.plot(alpha, np.sqrt(np.log(2/alpha) / (2*len(data))))
        # plt.scatter(np.array([res.pvalue]), np.array([res.statistic]))
        # plt.show()

if study_time_arrival_HV :
    # thresholds = [str(0.3), str(0.5), str(0.5)]

    for i in range(len(channel)) :
        threshold = thresholds[i]
        #HV_plot_list = [50, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280,300,320,340,360,380,400,420,440,460] # px12
        #HV_plot_list = [120, 140, 160, 180, 200, 220, 240, 260, 280,300,320,340,360,380,400,420,440,460] # px22
        std_plot_list = [0 for j in range (len(HV_plot_list))]
        for path in os.scandir(save_path) :
            hv = get_HV(path.name)
            for j in range (len(HV_plot_list)) :
                if hv == HV_plot_list[j] :
                    df = pd.read_csv(path.path + "/" + "dif_t_arrival_" + folder_name + "_" + channel[i] + "_" + threshold)
                    data = np.array(df["dif_t_arr"])
                    std_plot_list[j] = (np.std(data)/np.sqrt(2))
        print(HV_plot_list)
        print([int(x*1e12) for x in std_plot_list])
        plt.plot(HV_plot_list, std_plot_list)
        plt.xlabel("HV (V)")
        plt.ylabel("Time resolution (s)")
        plt.title("MiniV2 IR " + save_path.split("/")[-2].split("_")[0] + " " + channel[i] + " (threshold " + threshold +")")
        plt.show()

if study_time_arrival_ampl :
    # thresholds = [str(0.5), str(0.5), str(0.5)]
    #px22 ampl : C2=0.5, C3=0.5 (bruit à basse amplitude sur C2)

    for i in range(len(channel)) :
        # plot_thresholds = set()
        threshold = thresholds[i]
        #ampl_plot_list = [1100,1150,1200,1250,1300,1350,1400,1450,1500] # px12
        #ampl_plot_list = [1150,1200,1250,1300,1350,1400,1450,1500,1550,1600,1650,1700] # px22
        std_plot_list = [0 for j in range (len(ampl_plot_list))]
        for path in os.scandir(save_path) :
            for j in range (len(ampl_plot_list)) :
                # if ampl_plot_list[j] in alt_thresholds.keys():
                #     threshold = alt_thresholds[ampl_plot_list[j]][i]
                #     plot_thresholds.add(threshold)
                # else :
                #     threshold = thresholds[i]
                #     plot_thresholds.add(threshold)
                if str(ampl_plot_list[j]) in path.name :
                    df = pd.read_csv(path.path + "/" + "dif_t_arrival_" + folder_name + "_" + channel[i] + "_" + threshold)
                    data = np.array(df["dif_t_arr"])
                    std_plot_list[j] = (np.std(data)/np.sqrt(2))
        print(ampl_plot_list)
        print([int(x*1e12) for x in std_plot_list])
        plt.plot(ampl_plot_list, std_plot_list)
        plt.xlabel("LED input amplitude (mV)")
        plt.ylabel("Time resolution (s)")
        # plt.title("MiniV2 IR " + save_path.split("/")[-2].split("_")[0] + " " + channel[i] + " (thresholds " + str(plot_thresholds) + ")")
        plt.title("MiniV2 IR " + save_path.split("/")[-2].split("_")[0] + " " + channel[i] + " (threshold " + threshold + ")")
        plt.show()

if width_distribution :
    # thresholds = [str(0.5), str(0.5), str(0.5)]
    for i,c in enumerate(channel) :
        t_arr = pd.read_csv(save_path + "t_arrival_Miniv2_ledIR_" + c + "_" + thresholds[i])
        t_end = pd.read_csv(save_path + "t_end_Miniv2_ledIR_" + c + "_" + thresholds[i])
        plt.hist(np.array(t_end["t_end"]) - np.array(t_arr["t_arr"]), bins=bins, log=log)
        plt.title("Pulse width " + c + " threshold " + thresholds[i])
        plt.show()

if study_amplitude :
    for i in range(len(channel)) :
        #ampl_plot_list = [1100,1150,1200,1250,1300,1350,1400,1450,1500] # px12
        #ampl_plot_list = [1150,1200,1250,1300,1350,1400,1450,1500,1550,1600,1650,1700] # px22
        max_plot_list = [0 for j in range (len(ampl_plot_list))]
        for path in os.scandir(save_path) :
            for j in range (len(ampl_plot_list)) :
                if str(ampl_plot_list[j]) in path.name :
                    df = pd.read_csv(path.path + "/" + "multiple_max_" + folder_name + "_" + channel[i])
                    data = np.array(df["max"])
                    max_plot_list[j] = np.mean(data)
        plt.plot(ampl_plot_list, max_plot_list)
        plt.xlabel("amplitude led (mV)")
        plt.ylabel("amplitude signal (V)")
        plt.title("MiniV2 IR " + save_path.split("/")[-2].split("_")[0] + " " + channel[i])
        plt.show()
    
if dif_time_end_distrib_2_channels :
    def gaus(x,c,mu,sigma) :
        return c*np.exp(-((x-mu)**2) / (2*sigma**2))
    
    def gaus_cumulative(x,c,mu,sigma) :
        return ((c*sigma*np.sqrt(2*np.pi)/2) * (1 + erf((x-mu) / (sigma * np.sqrt(2)))))

    df_ch1 = pd.read_csv(save_path + "t_end_" + folder_name + "_" + channel[0] + "_" + thresholds[0])
    data_ch1 = np.array(df_ch1["t_end"])
    df_ch2 = pd.read_csv(save_path + "t_end_" + folder_name + "_" + channel[1] + "_" + thresholds[1])
    data_ch2 = np.array(df_ch2["t_end"])

    mask = np.logical_and(data_ch1 != -1, data_ch2 != -1)

    data_ch1 = data_ch1[mask]
    data_ch2 = data_ch2[mask]

    print()
    print("Ecart-type : " + str(np.std(data_ch1 - data_ch2)))
    print("Nb mesures : " + str(len(data_ch1 - data_ch2)))
    print()

    n, bin_edges, patches = plt.hist(data_ch1 - data_ch2, log=log, bins=bins)
    plt.show()

    x_hist = np.zeros(len(n[n!=0.]), dtype=float)
    k = 0
    for j in range (len(n)) :
        if n[j] != 0. :
            x_hist[k] = (bin_edges[j+1] + bin_edges[j]) /2
            k += 1

    param_optimised,param_covariance_matrix = curve_fit(gaus,x_hist,n[n!=0],p0=[max(n),np.mean(data_ch1 - data_ch2),np.std(data_ch1 - data_ch2)])
  
    c, mu, sigma = param_optimised
    print("C = " + str(c))
    print("mu = " + str(mu))
    print("sigma = " + str(sigma))
    print("Différence moyennes : " + str(np.abs(mu - np.mean(data_ch1 - data_ch2))))
    print("Différence écart-type : " + str(np.abs(sigma - np.std(data_ch1 - data_ch2))))

    x_hist_2=np.linspace(np.min(x_hist),np.max(x_hist),1000)
    plt.plot(x_hist_2,gaus(x_hist_2,*param_optimised),'r--',label='Gaussian fit')
    plt.legend()

    plt.title("Difference of time end for MiniCactus v2 (" + channel[0] + "-" + channel[1] + ") (thresholds " + thresholds[0] + " and " + thresholds[1] +")")
    plt.xlabel("Time (s)")
    plt.ylim(bottom=0.5, top=max(n)*1.1)
    plt.show()

    # probplot(data_ch1 - data_ch2, dist='norm', plot=plt)
    # plt.show()

if dif_time_arrival_distrib_2_channels :
    def gaus(x,c,mu,sigma) :
        return c*np.exp(-((x-mu)**2) / (2*sigma**2))
    
    def gaus_cumulative(x,c,mu,sigma) :
        return ((c*sigma*np.sqrt(2*np.pi)/2) * (1 + erf((x-mu) / (sigma * np.sqrt(2)))))

    df_ch1 = pd.read_csv(save_path + "t_arrival_" + folder_name + "_" + channel[0] + "_" + thresholds[0])
    data_ch1 = np.array(df_ch1["t_arr"])
    df_ch2 = pd.read_csv(save_path + "t_arrival_" + folder_name + "_" + channel[1] + "_" + thresholds[1])
    data_ch2 = np.array(df_ch2["t_arr"])

    if exp=="particles":
        max_PM = np.array(pd.read_csv(save_path + "max_" + folder_name + "_" + channel[1])["max"])
        mask = max_PM>0.7

        data_ch1 = data_ch1[mask]
        data_ch2 = data_ch2[mask]

    len_min = min(len(data_ch1), len(data_ch2))
    data_ch1 = data_ch1[:len_min]
    data_ch2 = data_ch2[:len_min]

    print()
    print("Ecart-type : " + str(np.std(data_ch1 - data_ch2)))
    print("Nb mesures : " + str(len(data_ch1 - data_ch2)))
    print()

    n, bin_edges, patches = plt.hist(data_ch1 - data_ch2, log=log, bins=bins)

    x_hist = np.zeros(len(n[n!=0.]), dtype=float)
    k = 0
    for j in range (len(n)) :
        if n[j] != 0. :
            x_hist[k] = (bin_edges[j+1] + bin_edges[j]) /2
            k += 1

    param_optimised,param_covariance_matrix = curve_fit(gaus,x_hist,n[n!=0],p0=[max(n),np.mean(data_ch1 - data_ch2),np.std(data_ch1 - data_ch2)])
  
    c, mu, sigma = param_optimised
    print("C = " + str(c))
    print("mu = " + str(mu))
    print("sigma = " + str(sigma))
    print("Différence moyennes : " + str(np.abs(mu - np.mean(data_ch1 - data_ch2))))
    print("Différence écart-type : " + str(np.abs(sigma - np.std(data_ch1 - data_ch2))))

    x_hist_2=np.linspace(np.min(x_hist),np.max(x_hist),1000)
    plt.plot(x_hist_2,gaus(x_hist_2,*param_optimised),'r--',label='Gaussian fit')
    plt.legend()

    plt.title("Difference of time of arrival for MiniCactus v2 (" + channel[0] + "-" + channel[1] + ") (thresholds " + thresholds[0] + " and " + thresholds[1] +")")
    plt.xlabel("Time (s)")
    plt.ylim(bottom=0.5, top=max(n)*1.1)
    plt.show()

    # probplot(data_ch1 - data_ch2, dist='norm', plot=plt)
    # plt.show()