import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from math import floor
import os
import time

from HV_analysis import plot_function_HV_from_save, function_max, get_HV
from data_read import fast_folder_reader
from mean_pulse import sort_multiple_lists, get_folder_name, group_multiple_lists

def get_pulse (data_signal : pd.DataFrame, len_seg : int, pulse_number : int = 0) -> tuple[np.ndarray, np.ndarray] :
    '''
    Extracts one pulse from data_signal.

    Arguments :
    data_signal - pandas.DataFrame : dataframe with signal data from one file
    len_seg - int : number of frame per segment
    pulse_number - (optional) int : index of desired pulse (default is 0)

    Returns : ([float], [float])
    t - numpy.array of float : array of time
    i - numpy.array of float : array of values
    '''
    begin = pulse_number*len_seg
    end = (pulse_number+1)*len_seg
    t = np.array(data_signal['Time'])[begin : end]
    i = np.array(data_signal['Ampl'])[begin : end]
    return(t,i)

def void_parameters (global_folder_path : str, channel : str | tuple[str, ...], save_path : str, invert : bool | tuple[bool, ...] = False, verbose : bool = False, folder_list : list = [], n_seg=-1, len_seg=-1) -> tuple :
    return ()

def mean_max (folder_path : str | tuple[str, ...], channel : str | tuple[str, ...], parameters : tuple, global_name : str = "", HV_value : int = 0, invert : bool | tuple[bool, ...]  = False, verbose : bool = False, n_seg : int = -1, len_seg : int = -1, reader_result : tuple = ()) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[str], list[str]] :
    '''
    Calculate information about the maxima of each pulse, from a folder containing data files. Can be used with get_stats function.

    Arguments :
    folder_path - str or tuple of str : path of the folder(s)
    channel - str or tuple of str : desired channel of the oscilloscope (ex : "C1" or "C3") (1 or more channels accepted)
    parameters - tuple : None
    global_name - (optional) str : name of the global working folder
    HV_value - (optional) int : HV value corresponding to the working folder
    invert - (optional) bool : whether you want the signal to be inverted (default is False)
    n_seg - (optional) int : number of segments
    len_seg - (optional) int : number of frame per segment
    reader_result - (optional) tuple : result of the fast_folder_reader function

    Returns : ([np.array], [np.array], [str], [str], [str])
    stat1_list - list of np.array : array of HV values for each pulse
    stat2_list - list of np.array : array of stats for max, argmax, time_max, name of the data file
    stat1_name_list - list of str : "HV"
    stat2_name_list - list of str : names of the corresponding array
    save_name_list - list of str : names of the save file
    '''
    if type(channel) == str :
        channel = (channel,)
    nb_channels = len(channel)

    if type(invert) == bool :
        invert = (invert,) * nb_channels

    if len(reader_result) == 0 :
        n_seg_list, len_seg_list, data_signal_list, name_list = fast_folder_reader(folder_path, channel, n_seg=n_seg, len_seg=len_seg)
    else :
        n_seg_list, len_seg_list, data_signal_list, name_list = reader_result

    max_list = [[] for i in range (nb_channels)]
    argmax_list = [[] for i in range (nb_channels)]
    time_max_list = [[] for i in range (nb_channels)]
    name_full_list = [[] for i in range (nb_channels)]

    for j in range(nb_channels) :
        if invert[j] :
            for df in data_signal_list[j] :
                df["Ampl"] = - df["Ampl"]

    for i in range (nb_channels) :
        for j in range(len(n_seg_list[i])) :
            df = data_signal_list[i][j]
            for k in range (int(df.shape[0]/len_seg_list[i][j])) :
                curve = get_pulse(df, len_seg_list[i][j], k)
                signal = curve[1]
                max_list[i].append(max(signal) - np.mean(signal[:len(signal)//10]))
                argmax_list[i].append(np.argmax(signal))
                time_max_list[i].append(curve[0][argmax_list[i][-1]])
                name_full_list[i].append(name_list[i][j])

    stat1_list = []
    stat2_list = []
    stat1_name_list = ["HV"] * (4*nb_channels)
    stat2_name_list = ["max", "argmax", "time_max", "name"] * nb_channels
    save_name_list = []
    for i in range (nb_channels) :
        stat1_list += [np.array([HV_value] * len(max_list[i]))] * 4
        stat2_list += [np.array(max_list[i]), np.array(argmax_list[i]), np.array(time_max_list[i]), np.array(name_full_list[i])]
        save_name_list += ["max_" + global_name + "_" + channel[i], "argmax_" + global_name + "_" + channel[i], "time_max_" + global_name + "_" + channel[i], "file_names_" + global_name + "_" + channel[i]]
    return (stat1_list, stat2_list, stat1_name_list, stat2_name_list, save_name_list)

def get_t_arrival_dif (folder_path : str | tuple[str, ...], channel : str | tuple[str, ...], parameters : tuple, global_name : str = "", HV_value : int = 0, invert : bool | tuple[bool, ...]  = False, verbose : bool = False, n_seg : int = -1, len_seg : int = -1, reader_result : tuple = ()) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[str], list[str]] :
    '''
    Calculate the difference between time of arrival of two adjacent pulses from the same frame.

    Arguments :
    folder_path - str or tuple of str : path of the folder(s)
    channel - str or tuple of str : desired channel of the oscilloscope (ex : "C1" or "C3") (1 or more channels accepted)
    parameters - tuple : [0] (int) = number of adjacent pulses considered in calculations, [1, ...] (float) = threshold used to determine time of arrival (for each channel) / if no threshold given, run all threshold between 0.1 and 0.9
    global_name - (optional) str : name of the global working folder
    HV_value - (optional) int : HV value corresponding to the working folder
    invert - (optional) bool : whether you want the signal to be inverted (default is False)
    n_seg - (optional) int : number of segments
    len_seg - (optional) int : number of frame per segment
    reader_result - (optional) tuple : result of the fast_folder_reader function

    Returns : ([np.array], [np.array], [str], [str], [str])
    stat1_list - list of np.array : difference between the time of arrival of two adjacent pulse
    stat2_list - list of np.array : None
    stat1_name_list - list of str : "dif_t_arr"
    stat2_name_list - list of str : None
    save_name_list - list of str : global_name + channel
    '''
    if type(channel) == str :
        channel = (channel,)
    nb_channels = len(channel)

    if type(invert) == bool :
        invert = (invert,) * nb_channels

    if len(reader_result) == 0 :
        n_seg_list, len_seg_list, data_signal_list, name_list = fast_folder_reader(folder_path, channel, n_seg=n_seg, len_seg=len_seg)
    else :
        n_seg_list, len_seg_list, data_signal_list, name_list = reader_result

    if len(parameters) == 1 :
        dif_t_arr = [[] for i in range (nb_channels * 9)]
    else :
        dif_t_arr = [[] for i in range (nb_channels)]
    names = []

    for j in range(nb_channels) :
        if invert[j] :
            for df in data_signal_list[j] :
                df["Ampl"] = - df["Ampl"]
    
    for i in range (nb_channels) :
        for j in range(len(n_seg_list[i])) :
            df = data_signal_list[i][j]
            for k in range (n_seg_list[i][j]) :
                curve = get_pulse(df, len_seg_list[i][j], k)
                t = curve[0]
                signal = curve[1]
                start = np.mean(signal[:int(len(signal)/(20*parameters[0]))])
                
                if len(parameters) == 1 :
                    for threshold in range (1, 10) :
                        threshold = threshold / 10
                        n = len(signal)
                        t_arrival = []

                        #if start < np.mean(signal) :
                        for m in range (parameters[0]) :
                            signal_crop = signal[int(n * m / parameters[0]) : int(n * (m+1) / parameters[0])]
                            start = np.mean(signal_crop[:int(len(signal_crop)/20)])
                            t_crop = t[int(n * m / parameters[0]) : int(n * (m+1) / parameters[0])]
                            maxi = max(signal_crop)
                            t_after_arrival = t_crop[signal_crop > start + threshold * (maxi - start)]
                            t_arrival.append(t_after_arrival[0])

                            if m > 0 :
                                dif_t_arr[9*i + int(threshold*10 - 1)].append(t_arrival[m] - t_arrival[m-1])

                        # else :
                        #     print(str(threshold) + " " + channel[i])
                        #     for m in range (parameters[0]-1) :
                        #         signal_crop = signal[int((n * m / parameters[0]) + (n / (2*parameters[0]))) : int((n * (m+1) / parameters[0]) + (n / (2*parameters[0])))]
                        #         start = np.mean(signal_crop[:int(len(signal_crop)/20)])
                        #         t_crop = t[int((n * m / parameters[0]) + (n / (2*parameters[0]))) : int((n * (m+1) / parameters[0]) + (n / (2*parameters[0])))]
                        #         maxi = max(signal_crop)
                        #         t_after_arrival = t_crop[signal_crop > start + threshold * (maxi - start)]
                        #         t_arrival.append(t_after_arrival[0])

                        #         if m > 0 :
                        #             dif_t_arr[9*i + int(threshold*10 - 1)].append(t_arrival[m] - t_arrival[m-1])

                        #             plt.plot(curve[0], curve[1])
                        #             plt.grid()
                        #             plt.xlabel("time (s)")
                        #             plt.ylabel("ampl (V)")
                        #             plt.title("px22 " + channel[i] + " " + str(int(dif_t_arr[i][-1] * 1e9)) + " ns")
                        #             plt.axvline(t_arrival[m], color = "green", linestyle=":")
                        #             plt.axvline(t_arrival[m-1], color = "green", linestyle=":")
                        #             plt.axhline(start + threshold * (maxi - start), color = "green", linestyle="--")
                        #             plt.show()
                                    
                else :
                    threshold = parameters[1 + i]
                    n = len(signal)
                    t_arrival = []

                    # if start < np.mean(signal) :
                    for m in range (parameters[0]) :
                        signal_crop = signal[int(n * m / parameters[0]) : int(n * (m+1) / parameters[0])]
                        start = np.mean(signal_crop[:int(len(signal_crop)/20)])
                        t_crop = t[int(n * m / parameters[0]) : int(n * (m+1) / parameters[0])]
                        maxi = max(signal_crop)
                        t_after_arrival = t_crop[signal_crop > start + threshold * (maxi - start)]
                        t_arrival.append(t_after_arrival[0])
                    
                        if m > 0 :
                            dif_t_arr[i].append(t_arrival[m] - t_arrival[m-1])

                            #if (channel[i] != "C1") and (dif_t_arr[i][-1] < 5e-7) or (dif_t_arr[i][-1] > 5.072e-7) : # debug for px12
                            #if ((channel[i] == "C2") and ((dif_t_arr[i][-1] < 1.96e-7) or (dif_t_arr[i][-1] > 2.06e-7))) or ((channel[i] == "C4") and (dif_t_arr[i][-1] < 1.983e-7)) : # debug for px22
                            # if channel[i] == "C4" :
                            #     plt.plot(curve[0], curve[1])
                            #     plt.grid()
                            #     plt.xlabel("time (s)")
                            #     plt.ylabel("ampl (V)")
                            #     plt.title("px22 " + channel[i] + " " + str(int(dif_t_arr[i][-1] * 1e9)) + " ns")
                            #     plt.axvline(t_arrival[m], color = "green", linestyle=":")
                            #     plt.axvline(t_arrival[m-1], color = "green", linestyle=":")
                            #     plt.axhline(start + threshold * (maxi - start), color = "red", linestyle="--")
                            #     plt.show()

                    # else :
                    #     print(channel[i])
                    #     for m in range (parameters[0]-1) :
                    #         signal_crop = signal[int((n * m / parameters[0]) + (n / (2*parameters[0]))) : int((n * (m+1) / parameters[0]) + (n / (2*parameters[0])))]
                    #         start = np.mean(signal_crop[:int(len(signal_crop)/20)])
                    #         t_crop = t[int((n * m / parameters[0]) + (n / (2*parameters[0]))) : int((n * (m+1) / parameters[0]) + (n / (2*parameters[0])))]
                    #         maxi = max(signal_crop)
                    #         t_after_arrival = t_crop[signal_crop > start + threshold * (maxi - start)]
                    #         t_arrival.append(t_after_arrival[0])
                        
                    #         if m > 0 :
                    #             dif_t_arr[i].append(t_arrival[m] - t_arrival[m-1])

                    #             #if (channel[i] != "C1") and (dif_t_arr[i][-1] < 5e-7) or (dif_t_arr[i][-1] > 5.072e-7) : # debug for px12
                    #             #if ((channel[i] == "C2") and ((dif_t_arr[i][-1] < 1.96e-7) or (dif_t_arr[i][-1] > 2.06e-7))) or ((channel[i] == "C4") and (dif_t_arr[i][-1] < 1.983e-7)) : # debug for px22
                    #             # plt.plot(curve[0], curve[1])
                    #             # plt.grid()
                    #             # plt.xlabel("time (s)")
                    #             # plt.ylabel("ampl (V)")
                    #             # plt.title("px22 " + channel[i] + " " + str(int(dif_t_arr[i][-1] * 1e9)) + " ns")
                    #             # plt.axvline(t_arrival[m], color = "green", linestyle=":")
                    #             # plt.axvline(t_arrival[m-1], color = "green", linestyle=":")
                    #             # plt.axhline(start + threshold * (maxi - start), color = "green", linestyle="--")
                    #             # plt.show()


        if len(parameters) == 1 :
            for threshold in range (1, 10) :
                threshold = threshold / 10
                names.append("dif_t_arrival_" + global_name + "_" + channel[i] + "_" + str(threshold))
        else :
            names.append("dif_t_arrival_" + global_name + "_" + channel[i])

    dif_t_arr = [np.array(x) for x in dif_t_arr]

    if len(parameters) == 1 :
        return (dif_t_arr, [np.array([]) for i in range (nb_channels*9)], ["dif_t_arr" for i in range (nb_channels*9)], ["" for i in range (nb_channels*9)], names)
    else :
        return (dif_t_arr, [np.array([]) for i in range (nb_channels)], ["dif_t_arr" for i in range (nb_channels)], ["" for i in range (nb_channels)], names)


def get_multiple_max (folder_path : str | tuple[str, ...], channel : str | tuple[str, ...], parameters : tuple, global_name : str = "", HV_value : int = 0, invert : bool | tuple[bool, ...]  = False, verbose : bool = False, n_seg : int = -1, len_seg : int = -1, reader_result : tuple = ()) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[str], list[str]] :
    '''
    Calculate the maximum of the signal with multiple pulses per frame.

    Arguments :
    folder_path - str or tuple of str : path of the folder(s)
    channel - str or tuple of str : desired channel of the oscilloscope (ex : "C1" or "C3") (1 or more channels accepted)
    parameters - tuple : [0] (int) = number of adjacent pulses considered in calculations
    global_name - (optional) str : name of the global working folder
    HV_value - (optional) int : HV value corresponding to the working folder
    invert - (optional) bool : whether you want the signal to be inverted (default is False)
    n_seg - (optional) int : number of segments
    len_seg - (optional) int : number of frame per segment
    reader_result - (optional) tuple : result of the fast_folder_reader function

    Returns : ([np.array], [np.array], [str], [str], [str])
    stat1_list - list of np.array : list of maxima
    stat2_list - list of np.array : None
    stat1_name_list - list of str : "max"
    stat2_name_list - list of str : None
    save_name_list - list of str : global_name + channel
    '''
    if type(channel) == str :
        channel = (channel,)
    nb_channels = len(channel)

    if type(invert) == bool :
        invert = (invert,) * nb_channels

    if len(reader_result) == 0 :
        n_seg_list, len_seg_list, data_signal_list, name_list = fast_folder_reader(folder_path, channel, n_seg=n_seg, len_seg=len_seg)
    else :
        n_seg_list, len_seg_list, data_signal_list, name_list = reader_result

    maxi = [[] for i in range (nb_channels)]
    names = []

    for j in range(nb_channels) :
        if invert[j] :
            for df in data_signal_list[j] :
                df["Ampl"] = - df["Ampl"]
    
    for i in range (nb_channels) :
        for j in range(len(n_seg_list[i])) :
            df = data_signal_list[i][j]
            for k in range (n_seg_list[i][j]) :
                curve = get_pulse(df, len_seg_list[i][j], k)
                signal = curve[1]

                n = len(signal)

                for m in range (parameters[0]) :
                    signal_crop = signal[int(n * m / parameters[0]) : int(n * (m+1) / parameters[0])]
                    start = np.mean(signal_crop[:int(len(signal_crop)/20)])
                    maxi[i].append(max(signal_crop)-start)

        names.append("multiple_max_" + global_name + "_" + channel[i])

    maxi = [np.array(x) for x in maxi]

    return (maxi, [np.array([]) for i in range (nb_channels)], ["max" for i in range (nb_channels)], ["" for i in range (nb_channels)], names)


def get_t_arrival (folder_path : str | tuple[str, ...], channel : str | tuple[str, ...], parameters : tuple, global_name : str = "", HV_value : int = 0, invert : bool | tuple[bool, ...]  = False, verbose : bool = False, n_seg : int = -1, len_seg : int = -1, reader_result : tuple = ()) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[str], list[str]] :
    '''
    Calculate the time of arrival of pulses.

    Arguments :
    folder_path - str or tuple of str : path of the folder(s)
    channel - str or tuple of str : desired channel of the oscilloscope (ex : "C1" or "C3") (1 or more channels accepted)
    parameters - tuple : [0] (int) = number of adjacent pulses considered in calculations, [1, ...] (float) = threshold used to determine time of arrival (for each channel) / if no threshold given, run all threshold between 0.1 and 0.9
    global_name - (optional) str : name of the global working folder
    HV_value - (optional) int : HV value corresponding to the working folder
    invert - (optional) bool : whether you want the signal to be inverted (default is False)
    n_seg - (optional) int : number of segments
    len_seg - (optional) int : number of frame per segment
    reader_result - (optional) tuple : result of the fast_folder_reader function

    Returns : ([np.array], [np.array], [str], [str], [str])
    stat1_list - list of np.array : difference between the time of arrival of two adjacent pulse
    stat2_list - list of np.array : None
    stat1_name_list - list of str : "dif_t_arr"
    stat2_name_list - list of str : None
    save_name_list - list of str : global_name + channel
    '''
    if type(channel) == str :
        channel = (channel,)
    nb_channels = len(channel)

    if type(invert) == bool :
        invert = (invert,) * nb_channels

    if len(reader_result) == 0 :
        n_seg_list, len_seg_list, data_signal_list, name_list = fast_folder_reader(folder_path, channel, n_seg=n_seg, len_seg=len_seg)
    else :
        n_seg_list, len_seg_list, data_signal_list, name_list = reader_result

    if len(parameters) == 1 :
        t_arr = [[] for i in range (nb_channels * 9)]
    else :
        t_arr = [[] for i in range (nb_channels)]
    names = []

    for j in range(nb_channels) :
        if invert[j] :
            for df in data_signal_list[j] :
                df["Ampl"] = - df["Ampl"]
    
    for i in range (nb_channels) :
        for j in range(len(n_seg_list[i])) :
            df = data_signal_list[i][j]
            for k in range (n_seg_list[i][j]) :
                curve = get_pulse(df, len_seg_list[i][j], k)
                t = curve[0]
                signal = curve[1]
                start = np.mean(signal[:int(len(signal)/(20*parameters[0]))])
                
                if len(parameters) == 1 :
                    for threshold in range (1, 10) :
                        threshold = threshold / 10
                        n = len(signal)

                        # if start < np.mean(signal) :
                        for m in range (parameters[0]) :
                            signal_crop = signal[int(n * m / parameters[0]) : int(n * (m+1) / parameters[0])]
                            start = np.mean(signal_crop[:int(len(signal_crop)/20)])
                            t_crop = t[int(n * m / parameters[0]) : int(n * (m+1) / parameters[0])]
                            maxi = max(signal_crop)
                            t_after_arrival = t_crop[signal_crop > start + threshold * (maxi - start)]
                            t_arr[9*i + int(threshold*10 - 1)].append(t_after_arrival[0])

                        # else :
                        #     for m in range (parameters[0]-1) :
                        #         signal_crop = signal[int((n * m / parameters[0]) + (n / (2*parameters[0]))) : int((n * (m+1) / parameters[0]) + (n / (2*parameters[0])))]
                        #         start = np.mean(signal_crop[:int(len(signal_crop)/20)])
                        #         t_crop = t[int((n * m / parameters[0]) + (n / (2*parameters[0]))) : int((n * (m+1) / parameters[0]) + (n / (2*parameters[0])))]
                        #         maxi = max(signal_crop)
                        #         t_after_arrival = t_crop[signal_crop > start + threshold * (maxi - start)]
                        #         t_arr[9*i + int(threshold*10 - 1)].append(t_after_arrival[0])
                                    
                else :
                    threshold = parameters[1 + i]
                    n = len(signal)

                    if start < np.mean(signal) :
                        for m in range (parameters[0]) :
                            signal_crop = signal[int(n * m / parameters[0]) : int(n * (m+1) / parameters[0])]
                            start = np.mean(signal_crop[:int(len(signal_crop)/20)])
                            t_crop = t[int(n * m / parameters[0]) : int(n * (m+1) / parameters[0])]
                            maxi = max(signal_crop)
                            t_after_arrival = t_crop[signal_crop > start + threshold * (maxi - start)]
                            t_arr[i].append(t_after_arrival[0])

                                #if (channel[i] != "C1") and (dif_t_arr[i][-1] < 5e-7) or (dif_t_arr[i][-1] > 5.072e-7) : # debug for px12
                                #if ((channel[i] == "C2") and ((dif_t_arr[i][-1] < 1.96e-7) or (dif_t_arr[i][-1] > 2.06e-7))) or ((channel[i] == "C4") and (dif_t_arr[i][-1] < 1.983e-7)) : # debug for px22
                                # if channel[i] == "C4" :
                                #     plt.plot(curve[0], curve[1])
                                #     plt.grid()
                                #     plt.xlabel("time (s)")
                                #     plt.ylabel("ampl (V)")
                                #     plt.title("px22 " + channel[i] + " " + str(int(dif_t_arr[i][-1] * 1e9)) + " ns")
                                #     plt.axvline(t_arrival[m], color = "green", linestyle=":")
                                #     plt.axvline(t_arrival[m-1], color = "green", linestyle=":")
                                #     plt.axhline(start + threshold * (maxi - start), color = "red", linestyle="--")
                                #     plt.show()

                    else :
                        for m in range (parameters[0]-1) :
                            signal_crop = signal[int((n * m / parameters[0]) + (n / (2*parameters[0]))) : int((n * (m+1) / parameters[0]) + (n / (2*parameters[0])))]
                            start = np.mean(signal_crop[:int(len(signal_crop)/20)])
                            t_crop = t[int((n * m / parameters[0]) + (n / (2*parameters[0]))) : int((n * (m+1) / parameters[0]) + (n / (2*parameters[0])))]
                            maxi = max(signal_crop)
                            t_after_arrival = t_crop[signal_crop > start + threshold * (maxi - start)]
                            t_arr[i].append(t_after_arrival[0])

                                #if (channel[i] != "C1") and (dif_t_arr[i][-1] < 5e-7) or (dif_t_arr[i][-1] > 5.072e-7) : # debug for px12
                                #if ((channel[i] == "C2") and ((dif_t_arr[i][-1] < 1.96e-7) or (dif_t_arr[i][-1] > 2.06e-7))) or ((channel[i] == "C4") and (dif_t_arr[i][-1] < 1.983e-7)) : # debug for px22
                                # plt.plot(curve[0], curve[1])
                                # plt.grid()
                                # plt.xlabel("time (s)")
                                # plt.ylabel("ampl (V)")
                                # plt.title("px22 " + channel[i] + " " + str(int(dif_t_arr[i][-1] * 1e9)) + " ns")
                                # plt.axvline(t_arrival[m], color = "green", linestyle=":")
                                # plt.axvline(t_arrival[m-1], color = "green", linestyle=":")
                                # plt.axhline(start + threshold * (maxi - start), color = "green", linestyle="--")
                                # plt.show()


        if len(parameters) == 1 :
            for threshold in range (1, 10) :
                threshold = threshold / 10
                names.append("t_arrival_" + global_name + "_" + channel[i] + "_" + str(threshold))
        else :
            names.append("t_arrival_" + global_name + "_" + channel[i])

    t_arr = [np.array(x) for x in t_arr]

    if len(parameters) == 1 :
        return (t_arr, [np.array([]) for i in range (nb_channels*9)], ["t_arr" for i in range (nb_channels*9)], ["" for i in range (nb_channels*9)], names)
    else :
        return (t_arr, [np.array([]) for i in range (nb_channels)], ["t_arr" for i in range (nb_channels)], ["" for i in range (nb_channels)], names)

def get_t_arrival_1_pulse (folder_path : str | tuple[str, ...], channel : str | tuple[str, ...], parameters : tuple, global_name : str = "", HV_value : int = 0, invert : bool | tuple[bool, ...]  = False, verbose : bool = False, n_seg : int = -1, len_seg : int = -1, reader_result : tuple = ()) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[str], list[str]] :
    '''
    Calculate the time of arrival of pulses.

    Arguments :
    folder_path - str or tuple of str : path of the folder(s)
    channel - str or tuple of str : desired channel of the oscilloscope (ex : "C1" or "C3") (1 or more channels accepted)
    parameters - tuple : [0] (int) = number of adjacent pulses considered in calculations, [1, ...] (float) = threshold used to determine time of arrival (for each channel) / if no threshold given, run all threshold between 0.1 and 0.9
    global_name - (optional) str : name of the global working folder
    HV_value - (optional) int : HV value corresponding to the working folder
    invert - (optional) bool : whether you want the signal to be inverted (default is False)
    n_seg - (optional) int : number of segments
    len_seg - (optional) int : number of frame per segment
    reader_result - (optional) tuple : result of the fast_folder_reader function

    Returns : ([np.array], [np.array], [str], [str], [str])
    stat1_list - list of np.array : difference between the time of arrival of two adjacent pulse
    stat2_list - list of np.array : None
    stat1_name_list - list of str : "dif_t_arr"
    stat2_name_list - list of str : None
    save_name_list - list of str : global_name + channel
    '''
    if type(channel) == str :
        channel = (channel,)
    nb_channels = len(channel)

    if type(invert) == bool :
        invert = (invert,) * nb_channels

    if len(reader_result) == 0 :
        n_seg_list, len_seg_list, data_signal_list, name_list = fast_folder_reader(folder_path, channel, n_seg=n_seg, len_seg=len_seg)
    else :
        n_seg_list, len_seg_list, data_signal_list, name_list = reader_result

    t_arr = [[] for i in range (nb_channels * 9)]
    names = []

    for j in range(nb_channels) :
        if invert[j] :
            for df in data_signal_list[j] :
                df["Ampl"] = - df["Ampl"]
    
    for i in range (nb_channels) :
        for j in range(len(n_seg_list[i])) :
            df = data_signal_list[i][j]
            for k in range (int(df.shape[0]/len_seg_list[i][j])) :
                curve = get_pulse(df, len_seg_list[i][j], k)
                t = curve[0]
                signal = curve[1]
                start = np.mean(signal[:int(len(signal)/20)])

                for threshold in range (1, 10) :
                    threshold = threshold / 10
                    n = len(signal)

                    if start < np.mean(signal) :
                        maxi = max(signal)
                        t_after_arrival = t[signal > start + threshold * (maxi - start)]
                        t_arr[9*i + int(threshold*10 - 1)].append(t_after_arrival[0])

                    else :
                        end = np.mean(signal[int(len(signal)*19/20):])
                        maxi = max(signal)
                        t_after_arrival = t[signal > end + threshold * (maxi - end)]
                        t_arr[9*i + int(threshold*10 - 1)].append(t_after_arrival[0])

        for threshold in range (1, 10) :
            threshold = threshold / 10
            names.append("t_arrival_" + global_name + "_" + channel[i] + "_" + str(threshold))
    
    t_arr = [np.array(x) for x in t_arr]

    return (t_arr, [np.array([]) for i in range (nb_channels*9)], ["t_arr" for i in range (nb_channels*9)], ["" for i in range (nb_channels*9)], names)


def get_t_end (folder_path : str | tuple[str, ...], channel : str | tuple[str, ...], parameters : tuple, global_name : str = "", HV_value : int = 0, invert : bool | tuple[bool, ...]  = False, verbose : bool = False, n_seg : int = -1, len_seg : int = -1, reader_result : tuple = ()) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[str], list[str]] :
    '''
    Calculate the time of arrival of pulses.

    Arguments :
    folder_path - str or tuple of str : path of the folder(s)
    channel - str or tuple of str : desired channel of the oscilloscope (ex : "C1" or "C3") (1 or more channels accepted)
    parameters - tuple : [0] (int) = number of adjacent pulses considered in calculations, [1, ...] (float) = threshold used to determine time of arrival (for each channel) / if no threshold given, run all threshold between 0.1 and 0.9
    global_name - (optional) str : name of the global working folder
    HV_value - (optional) int : HV value corresponding to the working folder
    invert - (optional) bool : whether you want the signal to be inverted (default is False)
    n_seg - (optional) int : number of segments
    len_seg - (optional) int : number of frame per segment
    reader_result - (optional) tuple : result of the fast_folder_reader function

    Returns : ([np.array], [np.array], [str], [str], [str])
    stat1_list - list of np.array : difference between the time of arrival of two adjacent pulse
    stat2_list - list of np.array : None
    stat1_name_list - list of str : "dif_t_arr"
    stat2_name_list - list of str : None
    save_name_list - list of str : global_name + channel
    '''
    if type(channel) == str :
        channel = (channel,)
    nb_channels = len(channel)

    if type(invert) == bool :
        invert = (invert,) * nb_channels

    if len(reader_result) == 0 :
        n_seg_list, len_seg_list, data_signal_list, name_list = fast_folder_reader(folder_path, channel, n_seg=n_seg, len_seg=len_seg)
    else :
        n_seg_list, len_seg_list, data_signal_list, name_list = reader_result

    if len(parameters) == 1 :
        t_end = [[] for i in range (nb_channels * 9)]
    else :
        t_end = [[] for i in range (nb_channels)]
    names = []

    for j in range(nb_channels) :
        if invert[j] :
            for df in data_signal_list[j] :
                df["Ampl"] = - df["Ampl"]
    
    for i in range (nb_channels) :
        for j in range(len(n_seg_list[i])) :
            df = data_signal_list[i][j]
            for k in range (n_seg_list[i][j]) :
                curve = get_pulse(df, len_seg_list[i][j], k)
                t = curve[0]
                signal = curve[1]
                start = np.mean(signal[:int(len(signal)/(20*parameters[0]))])
                
                if len(parameters) == 1 :
                    for threshold in range (1, 10) :
                        threshold = threshold / 10
                        n = len(signal)

                        # if start < np.mean(signal) :
                        for m in range (parameters[0]) :
                            signal_crop = signal[int(n * m / parameters[0]) : int(n * (m+1) / parameters[0])]
                            start = np.mean(signal_crop[:int(len(signal_crop)/20)])
                            t_crop = t[int(n * m / parameters[0]) : int(n * (m+1) / parameters[0])]
                            maxi = max(signal_crop)
                            i_max = np.argmax(signal_crop)
                            t_after_max = t_crop[i_max:]
                            signal_after_max = signal_crop[i_max:]
                            if len(t_after_max[signal_after_max < start + threshold * (maxi - start)]) > 0 :
                                t_end[9*i + int(threshold*10 - 1)].append(t_after_max[signal_after_max < start + threshold * (maxi - start)][0])
                            else :
                                t_end[9*i + int(threshold*10 - 1)].append(-1.0)

                        # else :
                        #     for m in range (parameters[0]-1) :
                        #         signal_crop = signal[int((n * m / parameters[0]) + (n / (2*parameters[0]))) : int((n * (m+1) / parameters[0]) + (n / (2*parameters[0])))]
                        #         start = np.mean(signal_crop[:int(len(signal_crop)/20)])
                        #         t_crop = t[int((n * m / parameters[0]) + (n / (2*parameters[0]))) : int((n * (m+1) / parameters[0]) + (n / (2*parameters[0])))]
                        #         maxi = max(signal_crop)
                        #         i_max = np.argmax(signal_crop)
                        #         t_after_max = t_crop[i_max:]
                        #         signal_after_max = signal_crop[i_max:]
                        #         if len(t_after_max[signal_after_max < start + threshold * (maxi - start)]) > 0 :
                        #             t_end[9*i + int(threshold*10 - 1)].append(t_after_max[signal_after_max < start + threshold * (maxi - start)][0])
                        #         else :
                        #             t_end[9*i + int(threshold*10 - 1)].append(-1.0)

                else :
                    threshold = parameters[1 + i]
                    n = len(signal)

                    if start < np.mean(signal) :
                        for m in range (parameters[0]) :
                            signal_crop = signal[int(n * m / parameters[0]) : int(n * (m+1) / parameters[0])]
                            start = np.mean(signal_crop[:int(len(signal_crop)/20)])
                            t_crop = t[int(n * m / parameters[0]) : int(n * (m+1) / parameters[0])]
                            maxi = max(signal_crop)
                            i_max = np.argmax(signal_crop)
                            t_after_max = t_crop[i_max:]
                            signal_after_max = signal_crop[i_max:]
                            if len(t_after_max[signal_after_max < start + threshold * (maxi - start)]) > 0 :
                                t_end[i].append(t_after_max[signal_after_max < start + threshold * (maxi - start)][0])
                            else :
                                t_end[i].append(-1.0)

                                #if (channel[i] != "C1") and (dif_t_arr[i][-1] < 5e-7) or (dif_t_arr[i][-1] > 5.072e-7) : # debug for px12
                                #if ((channel[i] == "C2") and ((dif_t_arr[i][-1] < 1.96e-7) or (dif_t_arr[i][-1] > 2.06e-7))) or ((channel[i] == "C4") and (dif_t_arr[i][-1] < 1.983e-7)) : # debug for px22
                                # if channel[i] == "C4" :
                                #     plt.plot(curve[0], curve[1])
                                #     plt.grid()
                                #     plt.xlabel("time (s)")
                                #     plt.ylabel("ampl (V)")
                                #     plt.title("px22 " + channel[i] + " " + str(int(dif_t_arr[i][-1] * 1e9)) + " ns")
                                #     plt.axvline(t_arrival[m], color = "green", linestyle=":")
                                #     plt.axvline(t_arrival[m-1], color = "green", linestyle=":")
                                #     plt.axhline(start + threshold * (maxi - start), color = "red", linestyle="--")
                                #     plt.show()

                    else :
                        for m in range (parameters[0]-1) :
                            signal_crop = signal[int((n * m / parameters[0]) + (n / (2*parameters[0]))) : int((n * (m+1) / parameters[0]) + (n / (2*parameters[0])))]
                            start = np.mean(signal_crop[:int(len(signal_crop)/20)])
                            t_crop = t[int((n * m / parameters[0]) + (n / (2*parameters[0]))) : int((n * (m+1) / parameters[0]) + (n / (2*parameters[0])))]
                            maxi = max(signal_crop)
                            i_max = np.argmax(signal_crop)
                            t_after_max = t_crop[i_max:]
                            signal_after_max = signal_crop[i_max:]
                            if len(t_after_max[signal_after_max < start + threshold * (maxi - start)]) > 0 :
                                t_end[i].append(t_after_max[signal_after_max < start + threshold * (maxi - start)][0])
                            else :
                                t_end[i].append(-1.0)

                                #if (channel[i] != "C1") and (dif_t_arr[i][-1] < 5e-7) or (dif_t_arr[i][-1] > 5.072e-7) : # debug for px12
                                #if ((channel[i] == "C2") and ((dif_t_arr[i][-1] < 1.96e-7) or (dif_t_arr[i][-1] > 2.06e-7))) or ((channel[i] == "C4") and (dif_t_arr[i][-1] < 1.983e-7)) : # debug for px22
                                # plt.plot(curve[0], curve[1])
                                # plt.grid()
                                # plt.xlabel("time (s)")
                                # plt.ylabel("ampl (V)")
                                # plt.title("px22 " + channel[i] + " " + str(int(dif_t_arr[i][-1] * 1e9)) + " ns")
                                # plt.axvline(t_arrival[m], color = "green", linestyle=":")
                                # plt.axvline(t_arrival[m-1], color = "green", linestyle=":")
                                # plt.axhline(start + threshold * (maxi - start), color = "green", linestyle="--")
                                # plt.show()


        if len(parameters) == 1 :
            for threshold in range (1, 10) :
                threshold = threshold / 10
                names.append("t_end_" + global_name + "_" + channel[i] + "_" + str(threshold))
        else :
            names.append("t_end_" + global_name + "_" + channel[i])

    t_end = [np.array(x) for x in t_end]

    if len(parameters) == 1 :
        return (t_end, [np.array([]) for i in range (nb_channels*9)], ["t_end" for i in range (nb_channels*9)], ["" for i in range (nb_channels*9)], names)
    else :
        return (t_end, [np.array([]) for i in range (nb_channels)], ["t_end" for i in range (nb_channels)], ["" for i in range (nb_channels)], names)

def get_t_end_1_pulse (folder_path : str | tuple[str, ...], channel : str | tuple[str, ...], parameters : tuple, global_name : str = "", HV_value : int = 0, invert : bool | tuple[bool, ...]  = False, verbose : bool = False, n_seg : int = -1, len_seg : int = -1, reader_result : tuple = ()) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[str], list[str]] :
    '''
    Calculate the time of arrival of pulses.

    Arguments :
    folder_path - str or tuple of str : path of the folder(s)
    channel - str or tuple of str : desired channel of the oscilloscope (ex : "C1" or "C3") (1 or more channels accepted)
    parameters - tuple : nothing
    global_name - (optional) str : name of the global working folder
    HV_value - (optional) int : HV value corresponding to the working folder
    invert - (optional) bool : whether you want the signal to be inverted (default is False)
    n_seg - (optional) int : number of segments
    len_seg - (optional) int : number of frame per segment
    reader_result - (optional) tuple : result of the fast_folder_reader function

    Returns : ([np.array], [np.array], [str], [str], [str])
    stat1_list - list of np.array : difference between the time of arrival of two adjacent pulse
    stat2_list - list of np.array : None
    stat1_name_list - list of str : "dif_t_arr"
    stat2_name_list - list of str : None
    save_name_list - list of str : global_name + channel
    '''
    if type(channel) == str :
        channel = (channel,)
    nb_channels = len(channel)

    if type(invert) == bool :
        invert = (invert,) * nb_channels

    if len(reader_result) == 0 :
        n_seg_list, len_seg_list, data_signal_list, name_list = fast_folder_reader(folder_path, channel, n_seg=n_seg, len_seg=len_seg)
    else :
        n_seg_list, len_seg_list, data_signal_list, name_list = reader_result

    t_end = [[] for i in range (nb_channels * 9)]

    names = []

    for j in range(nb_channels) :
        if invert[j] :
            for df in data_signal_list[j] :
                df["Ampl"] = - df["Ampl"]
    
    for i in range (nb_channels) :
        for j in range(len(n_seg_list[i])) :
            df = data_signal_list[i][j]
            for k in range (int(df.shape[0]/len_seg_list[i][j])) :
                curve = get_pulse(df, len_seg_list[i][j], k)
                t = curve[0]
                signal = curve[1]
                start = np.mean(signal[:int(len(signal)/20)])

                for threshold in range (1, 10) :
                    threshold = threshold / 10
                    n = len(signal)

                    if start < np.mean(signal) :
                        maxi = max(signal)
                        i_max = np.argmax(signal)
                        t_after_max = t[i_max:]
                        signal_after_max = signal[i_max:]
                        if len(t_after_max[signal_after_max < start + threshold * (maxi - start)]) > 0 :
                            t_end[9*i + int(threshold*10 - 1)].append(t_after_max[signal_after_max < start + threshold * (maxi - start)][0])
                        else :
                            t_end[9*i + int(threshold*10 - 1)].append(-1.0)

                    else :
                        end = np.mean(signal[:int(len(signal)*19/20)])
                        maxi = max(signal)
                        i_max = np.argmax(signal)
                        t_after_max = t[i_max:]
                        signal_after_max = signal[i_max:]
                        if len(t_after_max[signal_after_max < end + threshold * (maxi - start)]) > 0 :
                            t_end[9*i + int(threshold*10 - 1)].append(t_after_max[signal_after_max < end + threshold * (maxi - start)][0])
                        else :
                            t_end[9*i + int(threshold*10 - 1)].append(-1.0)

        for threshold in range (1, 10) :
            threshold = threshold / 10
            names.append("t_end_" + global_name + "_" + channel[i] + "_" + str(threshold))

    t_end = [np.array(x) for x in t_end]

    return (t_end, [np.array([]) for i in range (nb_channels*9)], ["t_end" for i in range (nb_channels*9)], ["" for i in range (nb_channels*9)], names)
    
def get_stats (global_folder_path : str, channel : str | tuple[str, ...], stat_func, get_parameters_func, folder_list : list[str] = [], HV_list : list[int] = [], save : bool = False, save_path : str = "saves_test", invert : bool | tuple[bool, ...] = False, filter_noise : float | tuple[float, ...] = 1.2, filter_saturation : float | tuple[float, ...] = 100.0,  verbose : bool = False, n_seg : int = -1, len_seg : int = -1, multiple_channel_analysis = False) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[str]] :
    '''
    Get information about the stats of each pulse, from a global folder.

    Arguments :
    global_folder_path - str : path of the folder containing the data folders for each HV value
    channel - str : desired channel of the oscilloscope (ex : "C1" or "C3") (only one channel accepted)
    stat_func - function or tuple of functions : function(s) that extract the desired stats from files (max / noise_std / mean_pulse). 
        Structure : (folder_path : str | tuple[str, ...], channel : str | tuple[str, ...], parameters : tuple, global_name : str, HV_value : int, verbose : bool, n_seg : int, len_seg : int, reader_result : tuple) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[str], list[str]]
    get_parameters_func - function or tuple of functions : function(s) that ask the user to give values to the parameters used in the corresponding stat_func
        Structure : (global_folder_path : str, channel : str | tuple[str, ...], save_path : str, invert : bool | tuple[bool, ...], verbose : bool) -> tuple
    folder_list - (optional) list of str : the names of the folders for each HV value
    HV_list - (optional) list of int : the HV values (needs to be in the same order as folder_list)
    save - (optional) bool : whether you want to save the results in a new .csv file (defaut is False)
    save_path - (optional) str : where you want to save (default is "saves_test")
    invert - (optional) bool : whether you want the signal to be inverted (default is False)
    verbose - (optional) bool : whether you want the function to print its status (default is False)
    n_seg - (optional) int : number of segments
    len_seg - (optional) int : number of frame per segment

    Returns : ([np.array], [np.array], [str], [str])
    stat1_list - list of np.array : array of stats (column 1)
    stat2_list - list of np.array : array of stats (column 2)
    stat1_name_list - list of str : names of the corresponding array (column 1)
    stat2_name_list - list of str : names of the corresponding array (column 2)
    '''
    t_start = time.time()
    stat1_list = []
    stat2_list = []
    stat1_name_list = []
    stat2_name_list = []
    save_name_list = []

    folder_name = get_folder_name(global_folder_path)
    if verbose :
        print("Starting calculations on " + folder_name)

    if folder_list == [] :
        for d in os.scandir(global_folder_path) :
            folder_list.append(d.name)
    if HV_list == [] :
        for f in folder_list :
            print(f)
            HV_list.append(get_HV(f))

    if global_folder_path[-1] != "/" :
        global_folder_path += "/"

    if save_path[-1] != "/" :
        save_path += "/"
    
    if type(channel) == str :
        channel = (channel,)
    nb_channels = len(channel)
    
    if type(stat_func) != tuple :
        stat_func = (stat_func,)
    nb_func = len(stat_func)

    if type(get_parameters_func) != tuple :
        get_parameters_func = (get_parameters_func,)
    
    if type(invert) == bool :
        invert = (invert,) * nb_channels
    
    if type(filter_noise) == float :
        filter_noise = (filter_noise,) * nb_channels

    if type(filter_saturation) == float :
        filter_saturation = (filter_saturation,) * nb_channels

    sort_multiple_lists(HV_list, [folder_list])
    group_multiple_lists(HV_list, [folder_list])

    nb_folders = len(HV_list)

    param = []
    for f in get_parameters_func :
        param.append(f(global_folder_path, channel, save_path, invert= invert, verbose=verbose, folder_list=folder_list, n_seg=n_seg, len_seg=len_seg))
    
    for i in range (nb_folders) :
        if verbose :
            completion = int((i/nb_folders)*100)
            print("Calculations completed by : " + str(completion) + "%")
            if completion != 0 :
                remaining = (100 - completion) * (time.time() - t_start) / completion
                print("Estimated remaining time : " + str(round(remaining, 2)) + " seconds")
            print("Current folder for calculations : " + str(folder_list[i]))
        n_seg_list, len_seg_list, data_signal_list, name_list = fast_folder_reader(tuple([global_folder_path + f for f in folder_list[i]]), channel, n_seg=n_seg, len_seg=len_seg, invert=invert, filter_noise=filter_noise, filter_saturation=filter_saturation, multiple_channel_analysis=multiple_channel_analysis)

    for j in range(nb_func) :
        stat1, stat2, stat1_name, stat2_name, save_name = stat_func[j](tuple([global_folder_path + f for f in folder_list[i]]), channel, param[j], global_name=folder_name, HV_value=HV_list[i], verbose = verbose, reader_result = (n_seg_list, len_seg_list, data_signal_list, name_list), n_seg=n_seg, len_seg=len_seg)
        stat1_list += stat1
        stat2_list += stat2
        stat1_name_list += stat1_name
        stat2_name_list += stat2_name
        save_name_list += save_name

    i = 0
    while i < len(stat1_list):
        j = i+1
        while j < len(stat1_list) : # stacks stats with the same save_name (to integrate with stat_func ??)
            if save_name_list[i] == save_name_list[j] :
                stat1 = stat1_list.pop(j)
                stat2 = stat2_list.pop(j)
                stat1_name = stat1_name_list.pop(j)
                stat2_name = stat2_name_list.pop(j)
                save_name = save_name_list.pop(j)
                stat1_list[i] = np.hstack((stat1_list[i], stat1))
                stat2_list[i] = np.hstack((stat2_list[i], stat2))
            else :
                j += 1
        if save :
            if len(stat2_list[i]) == 0 :
                df = pd.DataFrame(stat1_list[i], columns=[stat1_name_list[i]])
                df.to_csv(save_path + save_name_list[i])
            else :
                df = pd.DataFrame(np.stack((stat1_list[i], stat2_list[i]), axis=1), columns=[stat1_name_list[i], stat2_name_list[i]])
                df.to_csv(save_path + save_name_list[i])
        i += 1

    if verbose :
        print("Calculations completed ")

    return (stat1_list, stat2_list, stat1_name_list, stat2_name_list)


def get_mean_max (global_folder_path : str, channel : str, folder_list : list[str] = [], HV_list : list[int] = [], save : bool = False, save_path : str = "saves_test", invert : bool = False, verbose : bool = False, n_seg : int = -1, len_seg : int = -1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] :
    '''
    Get information about the maxima of each pulse, including the mean value of all the maxima.

    Arguments :
    global_folder_path - str : path of the folder containing the data folders for each HV value
    channel - str : desired channel of the oscilloscope (ex : "C1" or "C3") (only one channel accepted)
    folder_list - (optional) list of str : the names of the folders for each HV value
    HV_list - (optional) list of int : the HV values (needs to be in the same order as folder_list)
    save - (optional) bool : whether you want to save the results in a new .csv file (defaut is False)
    save_path - (optional) str : where you want to save (default is "saves_test")
    invert - (optional) bool : whether you want the signal to be inverted (default is False)
    verbose - (optional) bool : whether you want the function to print its status (default is False)
    n_seg - (optional) int : number of segments
    len_seg - (optional) int : number of frame per segment

    Returns :
    max_mean - numpy.array of float : mean value of the maxima for each HV value
    max_list_stack - numpy.array of float : maximum of each pulse
    argmax_list - numpy.array of int : index of the maximum of each pulse
    time_max_list - numpy.array of float : time of the maximum of each pulse
    HV_list - numpy.array of int : HV values corresponding to max_mean
    HV_full_list - numpy.array of int : HV values corresponding to max_list_stack, argmax_list, time_max_list and name_list
    name_list - numpy.array of str : name of the file
    '''
    t_start = time.time()
    max_list = []
    argmax_list = []
    time_max_list = []
    max_mean = []
    n_seg_list = []
    name_list = []

    folder_name = get_folder_name(global_folder_path)
    if verbose :
        print("Starting peak calculations on " + folder_name)

    if folder_list == [] :
        for d in os.scandir(global_folder_path) :
            folder_list.append(d.name)
    if HV_list == [] :
        for f in folder_list :
            HV_list.append(get_HV(f))
    nb_folders = len(folder_list)

    if global_folder_path[-1] != "/" :
        global_folder_path += "/"

    if save_path[-1] != "/" :
        save_path += "/"

    sort_multiple_lists(HV_list, [folder_list])

    for i in range (nb_folders) :
        if verbose :
            completion = int((i/nb_folders)*100)
            print("Peak calculations completed by : " + str(completion) + "%")
            if completion != 0 :
                remaining = (100 - completion) * (time.time() - t_start) / completion
                print("Estimated remaining time : " + str(round(remaining, 2)) + " seconds")
            print("Current folder for peak calculations : " + folder_list[i])
        n_seg_, len_seg_, data_signal_, name_ = fast_folder_reader(global_folder_path + folder_list[i], channel, n_seg=n_seg, len_seg=len_seg)
        nb_channels = len(n_seg_)
        if invert :
            for j in range(nb_channels) :
                for df in data_signal_[j] :
                    df["Ampl"] = - df["Ampl"]
        n_seg_list.append([])
        max_list.append([])
        n_seg_list[i] += n_seg_[0]
        for j in range(len(n_seg_[0])) :
            df = data_signal_[0][j]
            for k in range (n_seg_[0][j]) :
                curve = get_pulse(df, len_seg_[0][j], k)
                max_list[i].append(max(curve[1]))
                argmax_list.append(np.argmax(curve[1]))
                time_max_list.append(curve[0][argmax_list[-1]])
                name_list.append(name_[0][j])
        max_mean.append(np.mean(max_list[i]))

    HV_full_list = np.array([])
    for i in range (nb_folders) :
        for j in range (len(n_seg_list[i])) :
            HV_full_list = np.hstack((HV_full_list, np.array([HV_list[i]]*n_seg_list[i][j])))

    max_list_stack = np.array(max_list[0])
    for i in range (1,len(max_list)) :
        max_list_stack = np.hstack((max_list_stack, max_list[i]))

    if save :
        df_max_mean = pd.DataFrame(np.stack((HV_list, max_mean), axis=1), columns=["HV", "max_mean"])
        df_max_mean.to_csv(save_path + "max_mean_" + folder_name + "_" + channel)
        df_max_list = pd.DataFrame(np.stack((HV_full_list, max_list_stack, name_list), axis=1), columns=["HV", "max_list", "name"])
        df_max_list.to_csv(save_path + "max_list_" + folder_name + "_" + channel)
        df_argmax_list = pd.DataFrame(np.stack((HV_full_list, argmax_list, name_list), axis=1), columns=["HV", "argmax_list", "name"])
        df_argmax_list.to_csv(save_path + "argmax_list_" + folder_name + "_" + channel)
        df_time_max_list = pd.DataFrame(np.stack((HV_full_list, time_max_list, name_list), axis=1), columns=["HV", "time_max_list", "name"])
        df_time_max_list.to_csv(save_path + "time_max_list_" + folder_name + "_" + channel)

    if verbose :
        print("Peak calculations completed ")

    return (np.array(max_mean), max_list_stack, np.array(argmax_list), np.array(time_max_list), np.array(HV_list), HV_full_list, np.array(name_list))

def get_mean_max_from_save (save_path : str, folder_name : str, channel : str, tbeam : bool = False) :
    max_mean = []
    max_list = np.array(pd.read_csv(save_path + "max_" + folder_name + "_" + channel)["max"])

    if tbeam :
        print("Mean value of the maxima : " + str(np.mean(max_list)) + " V")

    else :
        HV_list = []
        HV_full_list = np.array(pd.read_csv(save_path + "max_" + folder_name + "_" + channel)["HV"])
        for hv in HV_full_list :
            if not(hv in HV_list) :
                HV_list.append(hv)
                max_mean.append(np.mean(max_list[HV_full_list == hv]))
        df_max_mean = pd.DataFrame(np.stack((HV_list, max_mean), axis=1), columns=["HV", "max_mean"])
        df_max_mean.to_csv(save_path + "max_mean_" + folder_name + "_" + channel)

def plot_difference (global_folder_path : str, save_mean_pulse_path : str, channel : str, normalized : bool = False, invert : bool = False, save_path : str = "saves_test", verbose : bool = False, n_seg : int = -1, len_seg : int = -1) -> tuple[np.ndarray, np.ndarray, bool] :
    '''
    Plots the difference between the mean value of all maxima and the maximum of the mean pulse, from a save file of the mean pulse.

    Arguments :
    global_folder_path - str : path of the folder containing the data folders for each HV value
    save_mean_pulse_path - str : path of the folder containing the save files of mean pulses
    channel - str : desired channel of the oscilloscope (ex : "C1" or "C3") (only one channel accepted)
    normalized - (optional) bool : whether you want the results to be normalized (divided by the maximum of the mean pulse) (default is False)
    invert - (optional) bool : whether you want the signal to be inverted (default is False)
    save_path - (optional) str : path of the folder containing the save files from get_mean_max (default is "./saves_test")
    verbose - (optional) bool : whether you want the function to print its status (default is False)
    n_seg - (optional) int : number of segments
    len_seg - (optional) int : number of frame per segment
    
    Returns :
    HV_list - numpy.array of int : the HV values
    dif - numpy.array of float : the difference between the mean of max and the max of mean for each HV value (positive if mean(max) > max(mean))
    normalized - bool : whether the results are normalized
    '''
    folder_name = get_folder_name(global_folder_path)

    max_mean = np.array(pd.read_csv(save_path + "/max_mean_" + folder_name + "_" + channel)["max_mean"])
    HV_list = np.array(pd.read_csv(save_path + "/max_mean_" + folder_name + "_" + channel)["HV"])

    maxs = plot_function_HV_from_save(save_mean_pulse_path, channel, function_max, save_list=[], HV_list=[], plot=False)

    dif = np.array(max_mean) - np.array(maxs[1][0])
    if normalized :
        dif = dif / maxs[1][0]
    #print(dif)
    #print(np.abs(dif).sum())

    plt.plot(HV_list, dif)
    plt.xlabel("HV (V)")
    if normalized :
        plt.ylabel("Normalized difference ('%' of max of mean)")
    else :
        plt.ylabel("Difference (V)")
    plt.title("Difference between max of mean and mean of max for " + folder_name + " " + channel)
    plt.grid()
    plt.show()

    return (HV_list, dif, normalized)

def plot_scatter_max (global_folder_path : str, save_mean_pulse_path : str, channel : str, HV : int, save_path : str = "saves_test", invert : bool = False, verbose : bool = False, n_seg : int = -1, len_seg : int = -1) -> None :
    '''
    Plots the mean pulse for a given HV value, as well as all the maximum points for the same HV value.

    Arguments :
    global_folder_path - str : path of the folder containing the data folders for each HV value
    save_mean_pulse_path - str : path of the folder containing the save files of mean pulses
    channel - str : desired channel of the oscilloscope (ex : "C1" or "C3") (only one channel accepted)
    HV - int : the desired HV value
    save_path - (optional) str : path of the folder containing the save files from get_mean_max (default is "./saves_test")
    invert - (optional) bool : whether you want the signal to be inverted (default is False)
    verbose - (optional) bool : whether you want the function to print its status (default is False)
    n_seg - (optional) int : number of segments
    len_seg - (optional) int : number of frame per segment

    Return : None
    '''
    folder_name = get_folder_name(global_folder_path)
    
    if save_path[-1] != "/" :
        save_path += "/"
    if save_mean_pulse_path[-1] != "/" :
        save_mean_pulse_path += "/"

    HV_full_list = np.array(pd.read_csv(save_path + "max_" + folder_name + "_" + channel)["HV"])
    max_list = np.array(pd.read_csv(save_path + "max_" + folder_name + "_" + channel)["max"])
    time_max_list = np.array(pd.read_csv(save_path + "time_max_" + folder_name + "_" + channel)["time_max"])

    plt.scatter(time_max_list[HV_full_list==HV], max_list[HV_full_list==HV], marker=".")
    for f in os.scandir(save_mean_pulse_path) :
        file_name = f.name
        if (channel in file_name) and (str(HV) in file_name) :
            df_mean = pd.read_csv(save_mean_pulse_path + file_name)
    t_mean = np.array(df_mean["Time"])
    signal_mean = np.array(df_mean["Ampl"])
    plt.plot(t_mean, signal_mean, "red")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (V)")
    plt.title(folder_name + " " + channel + " HV" + str(HV))
    plt.grid()
    plt.show()

def plot_extrem_max (folder_path : str, save_mean_pulse_path : str, channel : str, HV : int = 0, invert : bool = False, n_seg : int = -1, len_seg : int = -1) -> None :
    '''
    Plots the mean pulse for a given HV value, as well as all the maximum points for the same HV value.

    Arguments :
    folder_path - str : path of the folder containing the data files
    save_mean_pulse_path - str : path of the folder containing the save files of mean pulses
    channel - str : desired channel of the oscilloscope (ex : "C1" or "C3") (only one channel accepted)
    HV - (optional) int : HV value
    invert - (optional) bool : whether you want the signal to be inverted (default is False)
    verbose - (optional) bool : whether you want the function to print its status (default is False)
    n_seg - (optional) int : number of segments
    len_seg - (optional) int : number of frame per segment

    Return : None
    '''
    folder_name = get_folder_name(folder_path)
    
    if HV == 0 :
        HV = get_HV(folder_name)

    df_mean = pd.read_csv(save_mean_pulse_path)
    t_mean = np.array(df_mean["Time"])
    signal_mean = np.array(df_mean["Ampl"])
    plt.plot(t_mean, signal_mean, "red")

    max_mean = max(signal_mean)

    n_seg_list, len_seg_list, data_signal_list, name_list = fast_folder_reader(folder_path, channel, n_seg=n_seg, len_seg=len_seg)
    nb_channels = len(n_seg_list)
    if invert :
        for j in range(nb_channels) :
            for df in data_signal_list[j] :
                df["Ampl"] = - df["Ampl"]
    for j in range(len(n_seg_list[0])) :
        df = data_signal_list[0][j]
        for k in range (n_seg_list[0][j]) :
            curve = get_pulse(df, len_seg_list[0][j], k)
            if max(curve[1]) > 1.2*max_mean :
                plt.plot(curve[0], curve[1])

    
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (V)")
    plt.title(folder_name + " " + channel + " HV" + str(HV))
    plt.grid()
    plt.show()

def plot_max_std (global_folder_path : str, save_mean_pulse_path : str, channel : str, normalized : bool = False, invert : bool = False, save_path : str = "saves_test", verbose : bool = False, n_seg : int = -1, len_seg : int = -1) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool] :
    '''
    Plots the standard deviation of the maxima for a given HV value.

    Arguments :
    global_folder_path - str : path of the folder containing the data folders for each HV value
    save_mean_pulse_path - str : path of the folder containing the save files of mean pulses
    channel - str : desired channel of the oscilloscope (ex : "C1" or "C3") (only one channel accepted)
    normalized - (optional) bool : whether you want the results to be normalized (divided by the maximum of the mean pulse) (default is False)
    invert - (optional) bool : whether you want the signal to be inverted (default is False)
    save_path - (optional) str : path of the folder containing the save files from get_mean_max (default is "./saves_test")
    verbose - (optional) bool : whether you want the function to print its status (default is False)
    n_seg - (optional) int : number of segments
    len_seg - (optional) int : number of frame per segment

    Return :
    HV_list - numpy.array of int : the HV values
    max_std- numpy.array of float : standard deviation of the value of the maxima (in V) for each HV value
    time_max_std - numpy.array of float : standard deviation of the time of the maxima (in seconds) for each HV value
    normalized - bool : whether the results are normalized
    '''
    folder_name = get_folder_name(global_folder_path)

    HV_list = np.array(pd.read_csv(save_path + "/max_mean_" + folder_name + "_" + channel)["HV"])
    HV_full_list = np.array(pd.read_csv(save_path + "/max_" + folder_name + "_" + channel)["HV"])
    max_list = np.array(pd.read_csv(save_path + "/max_" + folder_name + "_" + channel)["max"])
    time_max_list = np.array(pd.read_csv(save_path + "/time_max_" + folder_name + "_" + channel)["time_max"])
    
    nb_HV = len(HV_list)
    max_std = np.zeros(nb_HV)
    time_max_std = np.zeros(nb_HV)
    for i in range(nb_HV) :
        max_std[i] = np.std(max_list[HV_full_list==HV_list[i]])
        time_max_std[i] = np.std(time_max_list[HV_full_list==HV_list[i]])
    if normalized :
        maxs = plot_function_HV_from_save(save_mean_pulse_path, channel, function_max, save_list=[], HV_list=[], plot=False)
        max_std = max_std / maxs[1][0]
    
    #print("Standard deviation")
    #print(max_std)

    #print("Standard deviation in time")
    #print(time_max_std)

    plt.plot(HV_list, max_std)
    plt.xlabel("HV (V)")
    if normalized :
        plt.ylabel("Normalized standard deviation ('%' of max of mean)")
    else :
        plt.ylabel("Standard deviation (V)")
    plt.title(folder_name + " " + channel)
    plt.grid()
    plt.show()

    plt.clf()
    plt.plot(HV_list, time_max_std)
    plt.xlabel("HV (V)")
    plt.ylabel("Standard deviation in time (s)")
    plt.title(folder_name + " " + channel)
    plt.grid()
    plt.show()

    return (HV_list, max_std, time_max_std, normalized)