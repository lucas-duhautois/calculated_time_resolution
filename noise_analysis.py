import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import zipfile

from HV_analysis import plot_function_HV_from_save, function_max
from data_read import fast_folder_reader, fast_file_reader
from peak_analysis import get_pulse
from mean_pulse import get_folder_name, plot_pulse

def get_limit_noise_once (global_folder_path : str, channel : str | tuple[str, ...], save_path : str, invert : bool | tuple[bool, ...] = False, verbose : bool = False, folder_list : list = [], n_seg=-1, len_seg=-1) -> tuple :
    # return(100,300) # coinc santa cruz PMT1 PMT3
    
    if folder_list == [] :
        for d in os.scandir(global_folder_path) :
            folder_list.append(d.name)

    if type(channel) == str :
        channel = (channel,)
    nb_channel = len(channel)
    t_limit = [0] * nb_channel

    for i in range(nb_channel) :
        test_limit = False
        while not(test_limit) :
            plt.clf()
            for folder_name in folder_list :
                for f in os.scandir (os.path.join(global_folder_path, folder_name[0])) :
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
                        plot_pulse(pulse4[0], pulse4[1], title="Test " + channel[i], show = False)
                        break
            plt.axvline(float(pulse0[0][t_limit[i]]), color = "red")
            plt.grid()
            plt.show(block = False)
            print("Current t_limit : " + str(t_limit[i]))
            test = input("What value do you want t_limit to be ? ")
            t_limit[i] = int(test)
            test = input("Do you want to check t_limit again ? [y/n] ")
            if test == "n" :
                test_limit = True
    return tuple(t_limit)

def get_limit_multiple_noise_once (global_folder_path : str, channel : str | tuple[str, ...], save_path : str, invert : bool | tuple[bool, ...] = False, verbose : bool = False, folder_list : list = [], n_seg = -1, len_seg = -1) -> tuple :
    # return((1000,1000,1000,1000),5)

    if folder_list == [] :
        for d in os.scandir(global_folder_path) :
            folder_list.append(d.name)

    if type(channel) == str :
        channel = (channel,)
    nb_channel = len(channel)
    t_limit = [0] * nb_channel

    for i in range(nb_channel) :
        test_limit = False
        while not(test_limit) :
            plt.clf()
            for folder_name in folder_list :
                if type(folder_name) == tuple :
                    folder_name = folder_name[0]

                if folder_name[-4:] == ".zip" :
                    with zipfile.ZipFile(global_folder_path + folder_name) as z:
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
                                    plot_pulse(pulse4[0], pulse4[1], title="Test " + channel[i], show = False)
                                    break
                else :
                    for f in os.scandir (os.path.join(global_folder_path, folder_name[0])) :
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
                            plot_pulse(pulse4[0], pulse4[1], title="Test " + channel[i], show = False)
                            break
            plt.axvline(float(pulse0[0][t_limit[i]]), color = "red")
            plt.grid()
            plt.show(block = False)
            print("Current t_limit : " + str(t_limit[i]))
            test = input("What value do you want t_limit to be ? ")
            t_limit[i] = int(test)
            test = input("Do you want to check t_limit again ? [y/n] ")
            if test == "n" :
                test_limit = True
    return (tuple(t_limit),5)

def get_limit_noise (folder_path : str | tuple[str, ...], channel : str | tuple[str, ...]) -> tuple :
    if type(channel) == str :
        channel = (channel,)
    nb_channel = len(channel)
    t_limit = [0] * nb_channel

    for i in range(nb_channel) :
        for f in os.scandir (folder_path) :
            if channel[i] in f.name :
                test_limit = False
                n_seg, len_seg, data_signal = fast_file_reader(f.path)
                pulse0 = get_pulse(data_signal, len_seg, 0)
                pulse1 = get_pulse(data_signal, len_seg, 1)
                pulse2 = get_pulse(data_signal, len_seg, 2)
                pulse3 = get_pulse(data_signal, len_seg, 3)
                pulse4 = get_pulse(data_signal, len_seg, 4)
                pulse5 = get_pulse(data_signal, len_seg, 5)
                pulse6 = get_pulse(data_signal, len_seg, 6)
                pulse7 = get_pulse(data_signal, len_seg, 7)
                pulse8 = get_pulse(data_signal, len_seg, 8)
                pulse9 = get_pulse(data_signal, len_seg, 9)
                while not(test_limit) :
                    plt.clf()
                    plot_pulse(pulse0[0], pulse0[1], title="Test " + channel[i], show = False)
                    plot_pulse(pulse1[0], pulse1[1], title="Test " + channel[i], show = False)
                    plot_pulse(pulse2[0], pulse2[1], title="Test " + channel[i], show = False)
                    plot_pulse(pulse3[0], pulse3[1], title="Test " + channel[i], show = False)
                    plot_pulse(pulse4[0], pulse4[1], title="Test " + channel[i], show = False)
                    plot_pulse(pulse5[0], pulse5[1], title="Test " + channel[i], show = False)
                    plot_pulse(pulse6[0], pulse6[1], title="Test " + channel[i], show = False)
                    plot_pulse(pulse7[0], pulse7[1], title="Test " + channel[i], show = False)
                    plot_pulse(pulse8[0], pulse8[1], title="Test " + channel[i], show = False)
                    plot_pulse(pulse9[0], pulse9[1], title="Test " + channel[i], show = False)
                    plt.axvline(float(pulse0[0][t_limit[i]]), color = "red")
                    plt.grid()
                    plt.show(block = False)
                    print("Current t_limit : " + str(t_limit[i]))
                    test = input("What value do you want t_limit to be ? ")
                    t_limit[i] = int(test)
                    test = input("Do you want to check t_limit again ? [y/n] ")
                    if test == "n" :
                        test_limit = True
                break
    return tuple(t_limit)

def noise_std (folder_path : str | tuple[str, ...], channel : str | tuple[str, ...], parameters : tuple, global_name : str = "", HV_value : int = 0, verbose : bool = False, n_seg : int = -1, len_seg : int = -1, reader_result : tuple = ()) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[str], list[str]] :
    '''
    Calculate the noise standard deviation from a folder containing data files. Can be used with get_stats function.

    Arguments :
    folder_path - str or tuple of str : path of the folder(s)
    channel - str or tuple of str : desired channel of the oscilloscope (ex : "C1" or "C3") (1 or more channels accepted)
    parameters - tuple : index of the limit of the time zone before there is any signal, for each channel
    global_name - (optional) str : name of the global working folder
    HV_value - (optional) int : HV value corresponding to the working folder
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
    if len(parameters) == 0 :
        t_limit = get_limit_noise(folder_path, channel)
    else :
        t_limit = parameters

    n_seg_list, len_seg_list, data_signal_list, name_list = fast_folder_reader(folder_path, channel, n_seg=n_seg, len_seg=len_seg)
    nb_channels = len(n_seg_list)
    noise_list = [[] for i in range (nb_channels)]

    # print(nb_channels)

    for i in range (nb_channels) :
        # print(len(n_seg_list[i]))
        for j in range(len(n_seg_list[i])) :
            df = data_signal_list[i][j]
            for k in range (int(len(df)/len_seg_list[i][j])) :
                curve = get_pulse(df, len_seg_list[i][j], k)
                noise_list[i].append(np.std(curve[1][:t_limit[i]]))
                # print(noise_list)
    
    stat1_list = []
    stat2_list = []
    stat1_name_list = ["HV"] * nb_channels
    stat2_name_list = ["noise_std"] * nb_channels
    save_name_list = []
    for i in range (nb_channels) :
        stat1_list += [np.array([HV_value] * len(noise_list[i]))]
        stat2_list += [np.array(noise_list[i])]
        save_name_list += ["noise_std_" + global_name + "_" + channel[i]]
    return (stat1_list, stat2_list, stat1_name_list, stat2_name_list, save_name_list)

def multiple_noise_std (folder_path : str | tuple[str, ...], channel : str | tuple[str, ...], parameters : tuple, global_name : str = "", HV_value : int = 0, verbose : bool = False, n_seg : int = -1, len_seg : int = -1, reader_result : tuple = ()) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[str], list[str]] :
    '''
    Calculate the noise standard deviation from a folder containing data files. Can be used with get_stats function.

    Arguments :
    folder_path - str or tuple of str : path of the folder(s)
    channel - str or tuple of str : desired channel of the oscilloscope (ex : "C1" or "C3") (1 or more channels accepted)
    parameters - tuple : index of the limit of the time zone before there is any signal, for each channel
    global_name - (optional) str : name of the global working folder
    HV_value - (optional) int : HV value corresponding to the working folder
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
    if len(parameters) == 0 :
        t_limit = get_limit_noise(folder_path, channel)
    else :
        t_limit = parameters[0]
    if type(channel) == str :
        channel = (channel,)
    nb_channels = len(channel)

    if len(reader_result) == 0 :
        n_seg_list, len_seg_list, data_signal_list, name_list = fast_folder_reader(folder_path, channel, n_seg=n_seg, len_seg=len_seg)
    else :
        n_seg_list, len_seg_list, data_signal_list, name_list = reader_result
    noise_list = [[] for i in range (nb_channels)]

    for i in range (nb_channels) :
        for j in range(len(n_seg_list[i])) :
            df = data_signal_list[i][j]
            for k in range (n_seg_list[i][j]) :
                curve = get_pulse(df, len_seg_list[i][j], k)
                signal = curve[1]

                n = len(signal)

                for m in range (parameters[1]) :
                    signal_crop = signal[int(n * m / parameters[1]) : int(n * (m+1) / parameters[1])]
                    noise_list[i].append(np.std(signal_crop[:t_limit[i]]))
    
    stat1_list = []
    stat2_list = []
    stat1_name_list = ["HV"] * nb_channels
    stat2_name_list = ["noise_std"] * nb_channels
    save_name_list = []
    for i in range (nb_channels) :
        stat1_list += [np.array([HV_value] * len(noise_list[i]))]
        stat2_list += [np.array(noise_list[i])]
        save_name_list += ["noise_std_" + global_name + "_" + channel[i]]
    return (stat1_list, stat2_list, stat1_name_list, stat2_name_list, save_name_list)

def get_noise_mean_from_save (save_path : str, folder_name : str, channel : str, tbeam : bool = False) :
    noise_mean = []
    noise_list = np.array(pd.read_csv(save_path + "noise_std_" + folder_name + "_" + channel)["noise_std"])

    if tbeam :
        print("Mean value of the noise standard deviation : " + str(np.mean(noise_list)) + " V")

    else :
        HV_list = []
        HV_full_list = np.array(pd.read_csv(save_path + "noise_std_" + folder_name + "_" + channel)["HV"])
        for hv in HV_full_list :
            if not(hv in HV_list) :
                HV_list.append(hv)
                noise_mean.append(np.mean(noise_list[HV_full_list == hv]))
        df_max_mean = pd.DataFrame(np.stack((HV_list, noise_mean), axis=1), columns=["HV", "noise_mean"])
        df_max_mean.to_csv(save_path + "noise_mean_" + folder_name + "_" + channel)

def plot_noise_std (global_folder_path : str, save_mean_pulse_path : str, channel : str, t_limit : int = 200, normalized : bool = False, plot : bool = True, save_path : str = "saves_test", verbose : bool = False, n_seg : int = -1, len_seg : int = -1) -> tuple[np.ndarray, np.ndarray, bool] :
    '''
    Plots the mean value of the standard deviation of noise for each HV value.

    Arguments :
    global_folder_path - str : path of the folder containing the data folders for each HV value
    save_mean_pulse_path - str : path of the folder containing the save files of mean pulses
    channel - str : desired channel of the oscilloscope (ex : "C1" or "C3") (only one channel accepted)
    t_limit - (optional) int : index of the limit of the time zone before there is any signal
    normalized - (optional) bool : whether you want the results to be normalized (divided by the maximum of the mean pulse) (default is False)
    save_path - (optional) str : path of the folder containing the save files from get_mean_max (default is "./saves_test")
    verbose - (optional) bool : whether you want the function to print its status (default is False)
    n_seg - (optional) int : number of segments
    len_seg - (optional) int : number of frame per segment
    
    Returns :
    HV_list - numpy.array of int : HV values corresponding to max_mean
    noise_mean - numpy.array of float : mean value of the maxima for each HV value
    normalized - bool : whether the results are normalized
    '''
    folder_name = get_folder_name(global_folder_path)

    noise_mean = np.array(pd.read_csv(save_path + "/noise_mean_" + folder_name + "_" + channel)["noise_mean"])
    HV_list = np.array(pd.read_csv(save_path + "/noise_mean_" + folder_name + "_" + channel)["HV"])
        
    if normalized :
        maxs = plot_function_HV_from_save(save_mean_pulse_path, channel, function_max, save_list=[], HV_list=[], plot=False)
        noise_mean = noise_mean / maxs[1][0]
    #print(noise_mean)

    if plot :
        print(HV_list)
        print([int(x*1e5)/100 for x in noise_mean])
        plt.plot(HV_list, noise_mean)
        plt.xlabel("HV (V)")
        if normalized :
            plt.ylabel("Normalized standard deviation of noise ('%' of max of mean)")
        else :
            plt.ylabel("Standard deviation of noise (V)")
        plt.title(folder_name + " " + channel)
        plt.grid()
        plt.show()

    return (HV_list, noise_mean, normalized)

def combine_noise_std (paths : list, save_path : str) :
    total_noise_std = np.array([])
    for path in paths :
        noise_std = np.array(pd.read_csv(path)["noise_std"])
        if len(total_noise_std) == 0 :
            total_noise_std = noise_std
        else :
            total_noise_std = np.hstack((total_noise_std, noise_std))
    df = pd.DataFrame(total_noise_std, columns=["noise_std"])
    df.to_csv(save_path + "combined_noise_std")