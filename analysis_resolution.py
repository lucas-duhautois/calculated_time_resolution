import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model

from noise_analysis import get_noise_mean_from_save, plot_noise_std
from HV_analysis import plot_function_HV_from_save, function_max, function_time_up

color = {
    "px12 2D":          "yellow",
    "px22 2D":          "magenta",
    "HRHG":             "blue",
    "epi saclay":       "orange",
    "px12 ampl":        "red",
    "px12 HV":          "green",
    "px22 ampl":        "pink",
    "px22 HV":          "cyan",
    "santa cruz laser": "black",

    "HRHG Sr":          "#56B4E9",  # bleu ciel
    "px12 Sr":          "#868686",  # gris moyen
    "px22 Sr":          "#4A4A4A",  # gris très foncé
    "epi saclay Sr":    "#999900",  # olive 
    "santa cruz":       "#7B1FA2",  # violet foncé
}

save_folder = "new_saves/Miniv2_ledIR/px12_2D/"
HV_values = [50,200,400]
ampl_values = [1800,1900,2000,2100,2200,2300,2400,2500]
save_folder2 = "new_saves/Miniv2_ledIR/px22_2D/"
HV_values2 = [50,200,400]
ampl_values2 = [2200,2400,2600,2800,3000,3200]
thresholds = [0.5,0.5,0.5]

res_1c_C1 = np.zeros((len(HV_values), len(ampl_values)))
res_1c_C2 = np.zeros((len(HV_values), len(ampl_values)))
res_1c_C3 = np.zeros((len(HV_values), len(ampl_values)))

res_1c_C1_fit = np.zeros((len(HV_values), len(ampl_values)))
res_1c_C2_fit = np.zeros((len(HV_values), len(ampl_values)))
res_1c_C3_fit = np.zeros((len(HV_values), len(ampl_values)))

res_2c_C2 = np.zeros((len(HV_values), len(ampl_values)))
res_2c_C3 = np.zeros((len(HV_values), len(ampl_values)))

res_2c_C2_fit = np.zeros((len(HV_values), len(ampl_values)))
res_2c_C3_fit = np.zeros((len(HV_values), len(ampl_values)))

res_calc_C1 = np.zeros((len(HV_values), len(ampl_values)))
res_calc_C2 = np.zeros((len(HV_values), len(ampl_values)))
res_calc_C3 = np.zeros((len(HV_values), len(ampl_values)))

ampl_analog = np.zeros((len(HV_values), len(ampl_values)))

res_2c_C2_px22 = np.zeros((len(HV_values2), len(ampl_values2)))
res_calc_C2_px22 = np.zeros((len(HV_values2), len(ampl_values2)))
ampl_analog_px22 = np.zeros((len(HV_values2), len(ampl_values2)))

for i,HV in enumerate(HV_values) :
    for j,ampl in enumerate(ampl_values):
        # dif_data_C1 = pd.read_csv(save_folder + "ampl" + str(ampl) + "_HV" + str(HV) + "/dif_t_arrival_px12-2D_C1_" + str(thresholds[0]))
        # dif_data_C2 = pd.read_csv(save_folder + "ampl" + str(ampl) + "_HV" + str(HV) + "/dif_t_arrival_px12-2D_C2_" + str(thresholds[0]))
        # dif_data_C3 = pd.read_csv(save_folder + "ampl" + str(ampl) + "_HV" + str(HV) + "/dif_t_arrival_px12-2D_C3_" + str(thresholds[0]))

        # res_1c_C1[i][j] = np.std(np.array(dif_data_C1["dif_t_arr"]))/np.sqrt(2)
        # res_1c_C2[i][j] = np.std(np.array(dif_data_C2["dif_t_arr"]))/np.sqrt(2)
        # res_1c_C3[i][j] = np.std(np.array(dif_data_C3["dif_t_arr"]))/np.sqrt(2)

        data_C1 = pd.read_csv(save_folder + "ampl" + str(ampl) + "_HV" + str(HV) + "/t_arrival_px12-2D_C1_" + str(thresholds[0]))
        data_C2 = pd.read_csv(save_folder + "ampl" + str(ampl) + "_HV" + str(HV) + "/t_arrival_px12-2D_C2_" + str(thresholds[0]))
        # data_C3 = pd.read_csv(save_folder + "ampl" + str(ampl) + "_HV" + str(HV) + "/t_arrival_px12-2D_C3_" + str(thresholds[0]))
        #         
        res_2c_C2[i][j] = np.std(np.array(data_C2["t_arr"]) - np.array(data_C1["t_arr"])[:len(data_C2["t_arr"])])
        # res_2c_C3[i][j] = np.std(np.array(data_C3["t_arr"]) - np.array(data_C1["t_arr"])[:len(data_C3["t_arr"])])

        # get_noise_mean_from_save(save_folder + "ampl" + str(ampl) + "_HV" + str(HV) + "/", "px12-2D", "C1", tbeam=False)
        get_noise_mean_from_save(save_folder + "ampl" + str(ampl) + "_HV" + str(HV) + "/", "px12-2D", "C2", tbeam=False)
        # get_noise_mean_from_save(save_folder + "ampl" + str(ampl) + "_HV" + str(HV) + "/", "px12-2D", "C3", tbeam=False)

        # _, noise_C1, _ = plot_noise_std("../Data/Miniv2_ledIR/px12-2D/", save_folder + "ampl" + str(ampl) + "_HV" + str(HV) + "/mean_pulse/", "C1", save_path=save_folder + "ampl" + str(ampl) + "_HV" + str(HV) + "/", plot=False)
        _, noise_C2, _ = plot_noise_std("../Data/Miniv2_ledIR/px12-2D/", save_folder + "ampl" + str(ampl) + "_HV" + str(HV) + "/mean_pulse/", "C2", save_path=save_folder + "ampl" + str(ampl) + "_HV" + str(HV), plot=False)
        # _, noise_C3, _ = plot_noise_std("../Data/Miniv2_ledIR/px12-2D/", save_folder + "ampl" + str(ampl) + "_HV" + str(HV) + "/mean_pulse/", "C3", save_path=save_folder + "ampl" + str(ampl) + "_HV" + str(HV) + "/", plot=False)

        _, amplitude = plot_function_HV_from_save(save_folder + "ampl" + str(ampl) + "_HV" + str(HV) + "/mean_pulse/", ("C1","C2","C3"), function_max, save_list=[], HV_list=[], plot=False, n_pulse = 5)
 
        _, time_up = plot_function_HV_from_save(save_folder + "ampl" + str(ampl) + "_HV" + str(HV) + "/mean_pulse/", ("C1","C2","C3"), function_time_up, save_list=[], HV_list=[], plot=False, n_pulse=5)

        # res_calc_C1[i][j] = noise_C1[0]*time_up[0][0]/amplitude[0][0]
        res_calc_C2[i][j] = noise_C2[0]*time_up[1][0]/amplitude[1][0]
        # res_calc_C3[i][j] = noise_C3[0]*time_up[2][0]/amplitude[2][0]

        ampl_analog[i][j] = amplitude[1][0]


for i,HV in enumerate(HV_values2) :
    for j,ampl in enumerate(ampl_values2):
        data_C1_px22 = pd.read_csv(save_folder2 + "ampl" + str(ampl) + "_HV" + str(HV) + "/t_arrival_px22-2D_C1_" + str(thresholds[0]))
        data_C2_px22 = pd.read_csv(save_folder2 + "ampl" + str(ampl) + "_HV" + str(HV) + "/t_arrival_px22-2D_C2_" + str(thresholds[0]))
        
        res_2c_C2_px22[i][j] = np.std(np.array(data_C2_px22["t_arr"]) - np.array(data_C1_px22["t_arr"])[:len(data_C2_px22["t_arr"])])

        get_noise_mean_from_save(save_folder2 + "ampl" + str(ampl) + "_HV" + str(HV) + "/", "px22-2D", "C2", tbeam=False)

        _, noise_C2_px22, _ = plot_noise_std("../Data/Miniv2_ledIR/px22-2D/", save_folder2 + "ampl" + str(ampl) + "_HV" + str(HV) + "/mean_pulse/", "C2", save_path=save_folder2 + "ampl" + str(ampl) + "_HV" + str(HV), plot=False)

        _, amplitude_px22 = plot_function_HV_from_save(save_folder2 + "ampl" + str(ampl) + "_HV" + str(HV) + "/mean_pulse/", ("C1","C2","C3"), function_max, save_list=[], HV_list=[], plot=False, n_pulse = 4)
        
        _, time_up_px22 = plot_function_HV_from_save(save_folder2 + "ampl" + str(ampl) + "_HV" + str(HV) + "/mean_pulse/", ("C1","C2","C3"), function_time_up, save_list=[], HV_list=[], plot=False, n_pulse=4)

        res_calc_C2_px22[i][j] = noise_C2_px22[0]*time_up_px22[1][0]/amplitude_px22[1][0]

        ampl_analog_px22[i][j] = amplitude_px22[1][0]

# Plot res 2D

# channel = "C2"

# res_1c = res_2c_C3 if channel=="C3" else res_2c_C2

# x = np.array(ampl_values)
# y = np.array(HV_values)

# X,Y = np.meshgrid(x,y)
# X = np.array(ampl_analog)*1000

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# print(res_1c)
# ax.plot_surface(X, Y, res_1c, rstride=1, cstride=1,cmap='viridis_r', edgecolor = 'none')
# plt.title("Time resolution (s) - Miniv2 px12 " + channel)
# plt.xlabel("Analog amplitude (mV)")
# plt.ylabel("HV value (V)")
# plt.show()



# Plot correlation resolution 1c/2c

# res_1c = res_1c_C2
# res_2c = res_2c_C2

# plt.scatter(res_1c.reshape((len(HV_values)*len(ampl_values),)), res_2c.reshape((len(HV_values)*len(ampl_values),)))

# lm = linear_model.LinearRegression()
# lm.fit(res_1c.reshape((-1,1)), res_2c.reshape((len(HV_values)*len(ampl_values),)))
# if lm.intercept_ > 0 :
#     print("y="+ str(int(lm.coef_[0]*1e6)/1e6) +"x+" +str(int(lm.intercept_*1e6)/1e6))
# else :
#     print("y="+ str(int(lm.coef_[0]*1e6)/1e6) +"x-" +str(int(-lm.intercept_*1e6)/1e6))

# x = np.arange(min(np.min(res_1c), np.min(res_2c)),max(np.max(res_1c), np.max(res_2c)), 1e-12)
# plt.plot(x, lm.coef_[0]*x+lm.intercept_, label="linear model")
# plt.legend()

# plt.title("Correlation between 1-channel and 2-channel measured time resolutions - Miniv2 px12")
# plt.xlabel("1-channel resolution (s)")
# plt.ylabel("2-channel resolution (s)")

# plt.show()



# Plot correlation resolution 2c/calc

# px12

channel = "C2"

res_2c = res_2c_C3 if channel=="C3" else res_2c_C2
res_calc = res_calc_C2

plt.scatter(res_2c.reshape((len(HV_values)*len(ampl_values),)), res_calc.reshape((len(HV_values)*len(ampl_values),)), label="px12 (2D scan)", color=color["px12 2D"])
# for i in range (len(HV_values)):
#     plt.scatter(res_2c[i]*1e12, res_calc[i]*1e12, label="px12 bis")

lm = linear_model.LinearRegression()
lm.fit(res_2c.reshape((-1,1)), res_calc.reshape((len(HV_values)*len(ampl_values),)))
if lm.intercept_ >=0 :
    model = "y="+ str(int(lm.coef_[0]*1e2)/1e2) +"x+" +str(int(lm.intercept_*1e12)/1e12)
else :
    model = "y="+ str(int(lm.coef_[0]*1e2)/1e2) +"x-" +str(-int(lm.intercept_*1e12)/1e12)

x = np.arange(np.min(res_2c),np.max(res_2c), 1e-12)
plt.plot(x, (lm.coef_[0]*x+lm.intercept_), label="linear model : " + model, color=color["px12 2D"])

# px22

plt.scatter(res_2c_C2_px22.reshape((len(HV_values2)*len(ampl_values2),)), res_calc_C2_px22.reshape((len(HV_values2)*len(ampl_values2),)), label="px22 (2D scan)", color=color["px22 2D"])
# for i in range (len(HV_values)):
#     plt.scatter(res_2c_C2_px22[i]*1e12, res_calc_C2_px22[i]*1e12, label="px12 bis")

lm = linear_model.LinearRegression()
lm.fit(res_2c_C2_px22.reshape((-1,1)), res_calc_C2_px22.reshape((len(HV_values2)*len(ampl_values2),)))
if lm.intercept_ >=0 :
    model = "y="+ str(int(lm.coef_[0]*1e2)/1e2) +"x+" +str(int(lm.intercept_*1e12)/1e12)
else :
    model = "y="+ str(int(lm.coef_[0]*1e2)/1e2) +"x-" +str(-int(lm.intercept_*1e12)/1e12)

x = np.arange(np.min(res_2c_C2_px22),np.max(res_2c_C2_px22), 1e-12)
plt.plot(x, (lm.coef_[0]*x+lm.intercept_), label="linear model : " + model, color=color["px22 2D"])

# plt.title("Correlation between measured and calculated time resolutions - Miniv2 px12 " + channel)
# plt.xlabel("Measured resolution (s)")
# plt.ylabel("Calculated resolution (s)")


res_HRHG = np.array([656,385,303,254,227,185,174,155,147,141,135])/1e12
calc_HRHG = np.array([265.5054645,173.2910156,140.1144726,121.9512195,106.9896641,93.4414832,87.1329418,79.16875313,73.47826087,68.82826476,64.94057725])/1e12

res_HRHG_Sr = np.array([807,763,733,585,462])/1e12 # fit gaussian (noisy distribution), PM threshold = 0.8V, t arrival threshold = 90% (50% for HV100, HV125 and HV150)
ampl_HRHG_Sr = np.array([28,47,77,87,102]) # mV (most probable value), start from HV60
noise_HRHG_Sr = np.array([1.43,2.17,1.81,1.45,1.81]) # mV, start from HV60
time_up_HRHG_Sr = np.array([11.4,10.7,8.6,5.7,4.9])/1e9 # ns, start from HV60

res_epi_saclay = np.array([317,275,248,214,193,161,142,128,164])/1e12
calc_epi_saclay = np.array([212.2238164,185.0632911,163.7873754,143.6053131,122.9206349,98.65979381,76.01098352,54.83711048,32.32576351])/1e12

res_epi_saclay_Sr = np.array([528, 419, 280, 249, 234])/1e12 # fit gaussian (noisy distribution), PM threshold = 0.8V, t arrival threshold = 50%
ampl_epi_saclay_Sr = np.array([22,22,28,45,62]) # mV (most probable value)
noise_epi_saclay_Sr = np.array([1.82,1.83,1.84,1.85,1.84]) # mV (most probable value)
time_up_epi_saclay_Sr = np.array([4.2,4.1,4.1,4.2,4.3])/1e9 # ns, start from HV60

res_px12_ampl = np.array([669,541,440,374,318,273,326,300,327])/1e12
calc_px12_ampl = np.array([648.7822878,456.6514378,344.4868628,256.0578662,209.8042124,184.3523189,204.741336,234.7521965,231.9716076])/1e12

res_px12_HV = np.array([268,185,179,182,178,180,180,178,177,178,180,177,178,177,175,175,174,174,175,178])/1e12
calc_px12_HV = np.array([348.6115912,295.2596867,290.3185895,286.1735009,286.5868758,281.3531353,281.6436093,280.3532693,282.0221234,282.8376704,282.8084667,280.2085484,279.4871795,279.8343685,279.0526534,280.0372941,279.7776161,279.6323224,280.0395257,279.254322])/1e12

res_px22_ampl = np.array([1042,833,750,596,475,398,299,274,243,220,203,184])/1e12
calc_px22_ampl = np.array([912.8508124,718.6440678,523.9201115,427.0793037,382.9039813,318.1109378,258.0415852,216.4111812,176.8010988,142.7707199,121.5763716,106.6113867])/1e12

res_px22_HV = np.array([222,214,206,203,201,201,198,200,199,196,199,197,197,194,196,196,195,196])/1e12
calc_px22_HV = np.array([184.6570185,183.7489755,178.7114846,180.0419678,179.8742138,178.2537835,178.2002093,175.7056216,176.3949,178.5531221,179.2841268,177.7725784,176.2917933,176.0752688,174.9970736,177.4611702,174.8538012,176.7049449])/1e12

res_santa_cruz = np.array([1159,850,828,724,664,660,683,800,944,500])/1e12
calc_santa_cruz = np.array([1642.857143,1451.851852,1295.238095,1000,813.8297872,637.5,388.3495146,282.3529412,312.5,423.7837838])/1e12

res_santa_cruz_laser = np.array([84,69,55,119])/1e12
calc_santa_cruz_laser = np.array([44.3809524,23.3659615,11.3521926,58.2803298])/1e12

res_santa_cruz_Sr = np.array([191,200])/1e12 # coincPMnew (PM<0.8) / coinc PMT1 PMT3 (PM<0.7)
ampl_santa_cruz_Sr = np.array([2.2, 2.5]) # mV coincPMnew / coinc PMT1 PMT3
noise_santa_cruz_Sr = np.array([0.25,0.15]) # mV coincPMnew / coinc PMT1 PMT3
time_up_santa_cruz_Sr = np.array([2.00,1.3])/1e9 # ns coincPMnew / coinc PMT1 PMT3

res_new_HRHG_2ch = np.array([340.66])/1e12 # ps
res_new_HRHG_1ch = np.array([480.8])/1e12 # ps to divide by sqrt(2)
ampl_new_HRHG = np.array([129.672]) # mV
noise_new_HRHG = np.array([4.5]) # mV
time_up_new_HRHG = np.array([9.3])/1e9 # ns

res_new_epiHG_2ch = np.array([281.12])/1e12 # ps
res_new_epiHG_1ch = np.array([400.7])/1e12 # ps to divide by sqrt(2)
ampl_new_epiHG = np.array([123.142]) # mV
noise_new_epiHG = np.array([4.75]) # mV
time_up_new_epiHG = np.array([7.4])/1e9 # ns

res_new_epiHG_laser_2ch = np.array([121.7])/1e12 # ps
res_new_epiHG_laser_1ch = np.array([165.25])/1e12 # ps to divide by sqrt(2)
ampl_new_epiHG_laser = np.array([212.841]) # mV
noise_new_epiHG_laser = np.array([3.35]) # mV
time_up_new_epiHG_laser = np.array([4.899])/1e9 # ns

plt.scatter(res_new_HRHG_2ch, time_up_new_HRHG*noise_new_HRHG/ampl_new_HRHG, label="HRHG", color=color["HRHG"], marker="^")

plt.scatter(res_HRHG, calc_HRHG, label="HRHG", color=color["HRHG"])

lm = linear_model.LinearRegression()
lm.fit(res_HRHG.reshape((-1,1)), calc_HRHG)
model = "y="+ str(int(lm.coef_[0]*1e2)/1e2) +"x+" +str(int(lm.intercept_*1e12)/1e12)
x = np.arange(np.min(res_HRHG),np.max(res_HRHG), 1e-12)
plt.plot(x, (lm.coef_[0]*x+lm.intercept_), label="linear model : " + model, color=color["HRHG"])

plt.scatter(res_new_epiHG_2ch, time_up_new_epiHG*noise_new_epiHG/ampl_new_epiHG, label="epi_saclay", color=color["epi saclay"], marker="^")

plt.scatter(res_new_epiHG_laser_2ch, time_up_new_epiHG_laser*noise_new_epiHG_laser/ampl_new_epiHG_laser, label="epi_saclay (laser)", color=color["santa cruz laser"], marker="^")

plt.scatter(res_epi_saclay, calc_epi_saclay, label="epi_saclay", color=color["epi saclay"])

lm = linear_model.LinearRegression()
lm.fit(res_epi_saclay.reshape((-1,1)), calc_epi_saclay)
if lm.intercept_ >=0 :
    model = "y="+ str(int(lm.coef_[0]*1e2)/1e2) +"x+" +str(int(lm.intercept_*1e12)/1e12)
else :
    model = "y="+ str(int(lm.coef_[0]*1e2)/1e2) +"x-" +str(-int(lm.intercept_*1e12)/1e12)
x = np.arange(np.min(res_epi_saclay),np.max(res_epi_saclay), 1e-12)
plt.plot(x, (lm.coef_[0]*x+lm.intercept_), label="linear model : " + model, color=color["epi saclay"])

plt.scatter(res_px12_ampl, calc_px12_ampl, label = "px12 ampl", color=color["px12 ampl"])

lm = linear_model.LinearRegression()
lm.fit(res_px12_ampl.reshape((-1,1)), calc_px12_ampl)
if lm.intercept_ >=0 :
    model = "y="+ str(int(lm.coef_[0]*1e2)/1e2) +"x+" +str(int(lm.intercept_*1e12)/1e12)
else :
    model = "y="+ str(int(lm.coef_[0]*1e2)/1e2) +"x-" +str(-int(lm.intercept_*1e12)/1e12)
x = np.arange(np.min(res_px12_ampl),np.max(res_px12_ampl), 1e-12)
plt.plot(x, (lm.coef_[0]*x+lm.intercept_), label="linear model : " + model, color=color["px12 ampl"])

plt.scatter(res_px12_HV, calc_px12_HV, label="px12 HV", color=color["px12 HV"])

lm = linear_model.LinearRegression()
lm.fit(res_px12_HV.reshape((-1,1)), calc_px12_HV)
if lm.intercept_ >=0 :
    model = "y="+ str(int(lm.coef_[0]*1e2)/1e2) +"x+" +str(int(lm.intercept_*1e12)/1e12)
else :
    model = "y="+ str(int(lm.coef_[0]*1e2)/1e2) +"x-" +str(-int(lm.intercept_*1e12)/1e12)
x = np.arange(np.min(res_px12_HV),np.max(res_px12_HV), 1e-12)
plt.plot(x, (lm.coef_[0]*x+lm.intercept_), label="linear model : " + model, color=color["px12 HV"])

plt.scatter(res_px22_ampl, calc_px22_ampl, label = "px22 ampl", color=color["px22 ampl"])

lm = linear_model.LinearRegression()
lm.fit(res_px22_ampl.reshape((-1,1)), calc_px22_ampl)
if lm.intercept_ >=0 :
    model = "y="+ str(int(lm.coef_[0]*1e2)/1e2) +"x+" +str(int(lm.intercept_*1e12)/1e12)
else :
    model = "y="+ str(int(lm.coef_[0]*1e2)/1e2) +"x-" +str(-int(lm.intercept_*1e12)/1e12)
x = np.arange(np.min(res_px22_ampl),np.max(res_px22_ampl), 1e-12)
plt.plot(x, (lm.coef_[0]*x+lm.intercept_), label="linear model : " + model, color=color["px22 ampl"])

plt.scatter(res_px22_HV, calc_px22_HV, label="px22 HV", color=color["px22 HV"])

lm = linear_model.LinearRegression()
lm.fit(res_px22_HV.reshape((-1,1)), calc_px22_HV)
if lm.intercept_ >=0 :
    model = "y="+ str(int(lm.coef_[0]*1e2)/1e2) +"x+" +str(int(lm.intercept_*1e12)/1e12)
else :
    model = "y="+ str(int(lm.coef_[0]*1e2)/1e2) +"x-" +str(-int(lm.intercept_*1e12)/1e12)
x = np.arange(np.min(res_px22_HV),np.max(res_px22_HV), 1e-12)
plt.plot(x, (lm.coef_[0]*x+lm.intercept_), label="linear model : " + model, color=color["px22 HV"])

# plt.scatter(res_santa_cruz, calc_santa_cruz, label = "santa cruz", color = "black")

# lm = linear_model.LinearRegression()
# lm.fit(res_santa_cruz.reshape((-1,1)), calc_santa_cruz)
# if lm.intercept_ >=0 :
#     model = "y="+ str(int(lm.coef_[0]*1e2)/1e2) +"x+" +str(int(lm.intercept_*1e12)/1e12)
# else :
#     model = "y="+ str(int(lm.coef_[0]*1e2)/1e2) +"x-" +str(-int(lm.intercept_*1e12)/1e12)
# x = np.arange(np.min(res_santa_cruz),np.max(res_santa_cruz), 1e-12)
# plt.plot(x, (lm.coef_[0]*x+lm.intercept_), label="linear model : " + model, color=color["santa cruz"])

plt.scatter(res_santa_cruz_laser, calc_santa_cruz_laser, label = "santa cruz (laser)", color = color["santa cruz laser"])

lm = linear_model.LinearRegression()
lm.fit(res_santa_cruz_laser.reshape((-1,1)), calc_santa_cruz_laser)
if lm.intercept_ >=0 :
    model = "y="+ str(int(lm.coef_[0]*1e2)/1e2) +"x+" +str(int(lm.intercept_*1e12)/1e12)
else :
    model = "y="+ str(int(lm.coef_[0]*1e2)/1e2) +"x-" +str(-int(lm.intercept_*1e12)/1e12)
x = np.arange(np.min(res_santa_cruz_laser),np.max(res_santa_cruz_laser), 1e-12)
plt.plot(x, (lm.coef_[0]*x+lm.intercept_), label="linear model : " + model, color=color["santa cruz laser"])

plt.scatter(res_santa_cruz_Sr, time_up_santa_cruz_Sr*noise_santa_cruz_Sr/ampl_santa_cruz_Sr, label="santa cruz Sr", color=color["santa cruz"], marker="+")

plt.scatter(res_HRHG_Sr, time_up_HRHG_Sr*noise_HRHG_Sr/ampl_HRHG_Sr, label = "HRHG Sr", color = color["HRHG Sr"], marker="+")

lm = linear_model.LinearRegression()
lm.fit(res_HRHG_Sr.reshape((-1,1)), time_up_HRHG_Sr*noise_HRHG_Sr/ampl_HRHG_Sr)
if lm.intercept_ >=0 :
    model = "y="+ str(int(lm.coef_[0]*1e2)/1e2) +"x+" +str(int(lm.intercept_*1e12)/1e12)
else :
    model = "y="+ str(int(lm.coef_[0]*1e2)/1e2) +"x-" +str(-int(lm.intercept_*1e12)/1e12)
x = np.arange(np.min(res_HRHG_Sr),np.max(res_HRHG_Sr), 1e-12)
plt.plot(x, (lm.coef_[0]*x+lm.intercept_), label="linear model : " + model, color=color["HRHG Sr"])

plt.scatter(res_epi_saclay_Sr, time_up_epi_saclay_Sr*noise_epi_saclay_Sr/ampl_epi_saclay_Sr, label = "epi_saclay Sr", color = color["epi saclay Sr"], marker="+")

lm = linear_model.LinearRegression()
lm.fit(res_epi_saclay_Sr.reshape((-1,1)), time_up_epi_saclay_Sr*noise_epi_saclay_Sr/ampl_epi_saclay_Sr)
if lm.intercept_ >=0 :
    model = "y="+ str(int(lm.coef_[0]*1e2)/1e2) +"x+" +str(int(lm.intercept_*1e12)/1e12)
else :
    model = "y="+ str(int(lm.coef_[0]*1e2)/1e2) +"x-" +str(-int(lm.intercept_*1e12)/1e12)
x = np.arange(np.min(res_epi_saclay_Sr),np.max(res_epi_saclay_Sr), 1e-12)
plt.plot(x, (lm.coef_[0]*x+lm.intercept_), label="linear model : " + model, color=color["epi saclay Sr"])

plt.legend()

plt.title("Comparaison résolutions mesurées et calculées")
plt.xlabel("Résolution mesurée")
plt.ylabel("Résolution calculée")
plt.legend()
plt.grid()

plt.show()
