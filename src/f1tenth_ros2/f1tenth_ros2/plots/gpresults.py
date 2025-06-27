'''
No longer useful. Redirect to read_CSV.py
'''

import numpy as np
import matplotlib.pyplot as plt



def collect_data(line_type, ORIGINAL=None, DOWNSAMPLED=None, sample_size=None):
    sepang_bestgp    = {}
    shanghai_bestgp  = {}
    YasMarina_bestgp = {}
    if not ORIGINAL:
        # print("Speed changing")
        if line_type == 'raceline':
            # print('raceline')
            if DOWNSAMPLED:
                # print('downsampled')
                if sample_size == 'half':
                    # print('half-sampled')
                    sepang_bestgp = {
                        'Kernels': 'RQ + Linear',
                        'Time': 33.73,
                        'RMSE': 0.10060893,
                        'R2': 0.95584347,
                        'Explain Variance': 0.9860452
                    }   
                    shanghai_bestgp = {
                        'Kernels': 'RQ + Linear',
                        'Time': 49.36,
                        'RMSE': 0.07900171,
                        'R2': 0.97511311,
                        'Explain Variance': 0.97970688
                    }
                    YasMarina_bestgp = {
                        'Kernels': 'RQ + Linear',
                        'Time': 44.5,
                        'RMSE': 0.13193171,
                        'R2': 0.95617065,
                        'Explain Variance': 0.96305785
                    }
                elif sample_size == 'one-third':
                    # print('one-third')
                    sepang_bestgp = {
                        'Kernels': 'RQ + Linear',
                        'Time': 27.31,
                        'RMSE': 0.10907591,
                        'R2': 0.94801277,
                        'Explain Variance': 0.97895568
                    }   
                    shanghai_bestgp = {
                        'Kernels': 'RQ + Linear',
                        'Time': 23.64,
                        'RMSE': 0.09694863,
                        'R2': 0.96224655,
                        'Explain Variance': 0.96689729
                    }
                    YasMarina_bestgp = {
                        'Kernels': 'RQ + Linear',
                        'Time': 7.94,
                        'RMSE': 0.15519895,
                        'R2': 0.93727732,
                        'Explain Variance': 0.94576344
                    }
            else: # Full sample data
                # print("Full sample")
                sepang_bestgp = {
                    'Kernels': 'RQ + Linear',
                    'Time': 276.02,
                    'RMSE': 0.09223654,
                    'R2': 0.96268554,
                    'Explain Variance': 0.99217944
                }   
                shanghai_bestgp = {
                    'Kernels': 'RQ + Linear',
                    'Time': 247.97,
                    'RMSE': 0.05972167,
                    'R2': 0.9857226,
                    'Explain Variance': 0.99031996
                }
                YasMarina_bestgp = {
                    'Kernels': 'Matern + Linear',
                    'Time': 70.76,
                    'RMSE': 0.11462006,
                    'R2': 0.96778908,
                    'Explain Variance': 0.97445889
                }
        if line_type == 'centerline':
            # print('centerline')
            if DOWNSAMPLED:
                if sample_size == 'half':
                    # print('half-sampled')
                    sepang_bestgp = {
                        'Kernels': 'RQ + Linear',
                        'Time': 34.71,
                        'RMSE': 0.07122412,
                        'R2': 0.9796913,
                        'Explain Variance': 0.9843631
                    }   
                    shanghai_bestgp = {
                        'Kernels': 'RQ * Linear',
                        'Time': 32.07,
                        'RMSE': 0.06899632,
                        'R2': 0.98181465,
                        'Explain Variance': 0.98242876
                    }
                    YasMarina_bestgp = {
                        'Kernels': 'RQ + Linear',
                        'Time': 32.07,
                        'RMSE': 0.1702553,
                        'R2': 0.94431356,
                        'Explain Variance': 0.95063994
                    }
                elif sample_size == 'one-third':
                    # print('one-third')
                    sepang_bestgp = {
                        'Kernels': 'RQ + Linear',
                        'Time': 35.34,
                        'RMSE': 0.08259735,
                        'R2': 0.97284384,
                        'Explain Variance': 0.97779047
                    }   
                    shanghai_bestgp = {
                        'Kernels': 'RQ +/* Linear',
                        'Time': 30.57,
                        'RMSE': 0.08379016,
                        'R2': 0.97334211,
                        'Explain Variance': 0.97414399
                    }
                    YasMarina_bestgp = {
                        'Kernels': 'Matern * Linear',
                        'Time': 19.96,
                        'RMSE': 0.18226916,
                        'R2': 0.93480148,
                        'Explain Variance': 0.94154867
                    }
            else: # Full sample data
                # print('full-sampled')
                sepang_bestgp = {
                    'Kernels': 'RQ + Linear',
                    'Time': 445.01,
                    'RMSE': 0.05686949,
                    'R2': 0.98700603,
                    'Explain Variance': 0.99149292
                }   
                shanghai_bestgp = {
                    'Kernels': 'RQ +/* Linear',
                    'Time': 265.49,
                    'RMSE': 0.04621721,
                    'R2': 0.99180978,
                    'Explain Variance': 0.99237526
                }
                YasMarina_bestgp = {
                    'Kernels': 'Matern + PERIODIC',
                    'Time': 112.5,
                    'RMSE': 0.13507984,
                    'R2': 0.96588359,
                    'Explain Variance': 0.97305015
                }
    else: # Speed capped
        # print('Speed capped')
        if line_type == 'raceline':
            # print('raceline')
            if DOWNSAMPLED:
                # print('downsampled')
                if sample_size == 'half':
                    # print('half-sampled')
                    sepang_bestgp = {
                        'Kernels': 'RQ + Linear',
                        'Time': 25.34,
                        'RMSE': 0.06124809,
                        'R2': 0.97903807,
                        'Explain Variance': 0.98041744
                    }   
                    shanghai_bestgp = {
                        'Kernels': 'RBF * RQ',
                        'Time': 18.86,
                        'RMSE': 0.07828026,
                        'R2': 0.97315234,
                        'Explain Variance': 0.97355551
                    }
                    YasMarina_bestgp = {
                        'Kernels': 'RQ + Linear',
                        'Time': 19.5,
                        'RMSE': 0.12366307,
                        'R2': 0.96017501,
                        'Explain Variance': 0.9616797
                    }
                elif sample_size == 'one-third':
                    # print('one-third')
                    sepang_bestgp = {
                        'Kernels': 'RQ +/* Linear',
                        'Time': 13.34,
                        'RMSE': 0.07642474,
                        'R2': 0.96741712,
                        'Explain Variance': 0.97486864
                    }   
                    shanghai_bestgp = {
                        'Kernels': 'RQ + Linear',
                        'Time': 13.19,
                        'RMSE': 0.10282238,
                        'R2': 0.952474,
                        'Explain Variance': 0.95250596
                    }
                    YasMarina_bestgp = {
                        'Kernels': 'RQ + Linear',
                        'Time': 4.15,
                        'RMSE': 0.15882839,
                        'R2': 0.93463373,
                        'Explain Variance': 0.93853474
                    }
            else: # Full sample data
                # print('full-sampled')
                sepang_bestgp = {
                    'Kernels': 'RQ + Linear',
                    'Time': 134.49,
                    'RMSE': 0.05626349,
                    'R2': 0.98225866,
                    'Explain Variance': 0.98761259
                }   
                shanghai_bestgp = {
                    'Kernels': 'Matern + Linear',
                    'Time': 239.94,
                    'RMSE': 0.0579899,
                    'R2': 0.98115319,
                    'Explain Variance': 0.98529214
                }
                YasMarina_bestgp = {
                    'Kernels': 'RBF * RQ',
                    'Time': 50.92,
                    'RMSE': 0.09077648,
                    'R2': 0.97895764,
                    'Explain Variance': 0.98188128
                }
        if line_type == 'centerline':
            # print('centerline')
            if DOWNSAMPLED:
                if sample_size == 'half':
                    sepang_bestgp = {
                        'Kernels': 'RQ * Linear',
                        'Time': 19.96,
                        'RMSE': 0.07082653,
                        'R2': 0.97295479,
                        'Explain Variance': 0.97528897
                    }   
                    shanghai_bestgp = {
                        'Kernels': 'RQ + Linear',
                        'Time': 7.67,
                        'RMSE': 0.0727436,
                        'R2': 0.97147091,
                        'Explain Variance': 0.9722872
                    }
                    YasMarina_bestgp = {
                        'Kernels': 'Matern + PERIODIC',
                        'Time': 35.01,
                        'RMSE': 0.18938751,
                        'R2': 0.92827124,
                        'Explain Variance': 0.95529766
                    }
                elif sample_size == 'one-third':
                    sepang_bestgp = {
                        'Kernels': 'RBF * RQ',
                        'Time': 7.12,
                        'RMSE': 0.07701028,
                        'R2': 0.96834438,
                        'Explain Variance': 0.9689003
                    }   
                    shanghai_bestgp = {
                        'Kernels': 'RQ + Linear',
                        'Time': 6.62,
                        'RMSE': 0.07894515,
                        'R2': 0.96673372,
                        'Explain Variance': 0.9672206
                    }
                    YasMarina_bestgp = {
                        'Kernels': 'RBF * RQ',
                        'Time': 3.94,
                        'RMSE': 0.18626971,
                        'R2': 0.93190804,
                        'Explain Variance': 0.93901742
                    }
            else: # Full sample data
                sepang_bestgp = {
                    'Kernels': 'Matern + PERIODIC',
                    'Time': 234.89,
                    'RMSE': 0.05496632,
                    'R2': 0.98375078,
                    'Explain Variance': 0.98496603
                }   
                shanghai_bestgp = {
                    'Kernels': 'RBF + Linear',
                    'Time': 28.46,
                    'RMSE': 0.06282533,
                    'R2': 0.97877202,
                    'Explain Variance': 0.97951023
                }
                YasMarina_bestgp = {
                    'Kernels': 'Matern + PERIODIC',
                    'Time': 97.24,
                    'RMSE': 0.1656858,
                    'R2': 0.94547174,
                    'Explain Variance': 0.97257808
                }
    return [sepang_bestgp, shanghai_bestgp, YasMarina_bestgp]

#######################################################
# Plots
# plot time difference of Sepang
def plot_results(dataX, dataY, type):
    if type == 'Time':
        X = dataX
        full = []
        half = []
        thir = []
        for i in range(len(dataY)):
            
            full.append(dataY[i][0])
            half.append(dataY[i][1])
            thir.append(dataY[i][2])
        X_axis = np.arange(len(X))
        plt.figure()
        bar1 = plt.bar(X_axis - 0.2, full, 0.2, label = 'full')
        bar2 = plt.bar(X_axis + 0.0, half, 0.2, label = 'half')
        bar3 = plt.bar(X_axis + 0.2, thir, 0.2, label = 'one-third')
        plt.xticks(X_axis, X)
        # bar = plt.bar(dataX, dataY, width=0.4)
        for rect in bar1+bar2+bar3:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, height,  f'{height:.2f}', ha='center', va='bottom')
        plt.xlabel('Scenarios')
        plt.ylabel('Time [seconds]')
        plt.grid(True)
        
    elif type == 'EV':
        X = ['FUll', 'HALF', 'ONE-THIRD'] 
        t1 = dataY[0]
        t2 = dataY[1]
        t3 = dataY[2]
        t4 = dataY[3]
        plt.plot(X, t1, 'o-', label='ra_non-cap') # 'ra_acc', 'ra_cap', 'cen_acc', 'cen_cap'
        plt.plot(X, t2, 'o-', label='ra_cap')
        plt.plot(X, t3, 'o-', label='cen_non-cap') 
        plt.plot(X, t4, 'o-', label='cen_cap')
        plt.xlabel('Scenarios')
        plt.ylabel('Explain Variance')
    elif type == 'RA':
        X = ['FUll', 'HALF', 'ONE-THIRD'] 
        t1 = dataY[0]
        t2 = dataY[1]
        t3 = dataY[2]
        # t4 = dataY[3]
        plt.plot(X, t1, 'o-', label='Sepang') # 'ra_acc', 'ra_cap', 'cen_acc', 'cen_cap'
        plt.plot(X, t2, 'o-', label='Shanghai')
        plt.plot(X, t3, 'o-', label='YasMarina') 
        # plt.plot(X, t4, 'o-', label='cen_cap')
        plt.xlabel('Scenarios')
        plt.ylabel('Explain Variance')
    plt.legend()
    plt.show()

    # # plot RMSE difference of Sepang
    # dataX = ['ra_acc_full', 'ra_cap_full', 'cen_acc_full', 'cen_cap_full']
    # plt.figure()
    # line = plt.plot(dataX, dataY[1],'o-')
    # for x,y in zip(dataX, dataY[1]):
    #     label = "{:.5f}".format(y)
    #     plt.annotate(label, # this is the text
    #                 (x,y), # these are the coordinates to position the label
    #                 textcoords="offset points", # how to position the text
    #                 xytext=(20,10), # distance from text to points (x,y)
    #                 ha='center') # horizontal alignment can be left, right or center
    # plt.xlabel('Scenarios')
    # plt.ylabel('RMSE')
    # plt.grid(True)

    # # plot Explain Variances difference of Sepang
    # dataX = ['ra_acc_full', 'ra_cap_full', 'cen_acc_full', 'cen_cap_full']
    # plt.figure()
    # line = plt.plot(dataX, dataY[2],'o-')
    # for x,y in zip(dataX, dataY[2]):
    #     label = "{:.5f}".format(y)
    #     plt.annotate(label, # this is the text
    #                 (x,y), # these are the coordinates to position the label
    #                 textcoords="offset points", # how to position the text
    #                 xytext=(20,10), # distance from text to points (x,y)
    #                 ha='center') # horizontal alignment can be left, right or center
    # plt.xlabel('Scenarios')
    # plt.ylabel('Explain Variance')
    # plt.grid(True)
    

if __name__ == '__main__':
    cen_cap_half = collect_data('centerline', ORIGINAL=True, DOWNSAMPLED=True,sample_size='half')
    cen_acc_half = collect_data('centerline', ORIGINAL=False, DOWNSAMPLED=True,sample_size='half')
    ra_cap_half  = collect_data('raceline', ORIGINAL=True, DOWNSAMPLED=True,sample_size='half')
    ra_acc_half  = collect_data('raceline', ORIGINAL=False, DOWNSAMPLED=True,sample_size='half')

    cen_cap_thir = collect_data('centerline', ORIGINAL=True, DOWNSAMPLED=True,sample_size='one-third')
    cen_acc_thir = collect_data('centerline', ORIGINAL=False, DOWNSAMPLED=True,sample_size='one-third')
    ra_cap_thir  = collect_data('raceline', ORIGINAL=True, DOWNSAMPLED=True,sample_size='one-third')
    ra_acc_thir  = collect_data('raceline', ORIGINAL=False, DOWNSAMPLED=True,sample_size='one-third')

    cen_cap_full = collect_data('centerline', ORIGINAL=True, DOWNSAMPLED=False)
    cen_acc_full = collect_data('centerline', ORIGINAL=False, DOWNSAMPLED=False)
    ra_cap_full  = collect_data('raceline', ORIGINAL=True, DOWNSAMPLED=False)
    ra_acc_full  = collect_data('raceline', ORIGINAL=False, DOWNSAMPLED=False)

    map_index    = 2
    map_list     = ['Sepang', 'Shanghai', 'YasMarina']
    map_name     = map_list[map_index]
    # Raceline & changing speed
    TIME_SP_RA = [ra_acc_full[map_index]['Time'], ra_acc_half[map_index]['Time'], ra_acc_thir[map_index]['Time']]
    RMSE_SP_RA = [ra_acc_full[map_index]['RMSE'], ra_acc_half[map_index]['RMSE'], ra_acc_thir[map_index]['RMSE']]
    EV_SP_RA   = [ra_acc_full[map_index]['Explain Variance'], ra_acc_half[map_index]['Explain Variance'], ra_acc_thir[map_index]['Explain Variance']]
    # Centerline & changing speed
    TIME_SP_CA = [cen_acc_full[map_index]['Time'], cen_acc_half[map_index]['Time'], cen_acc_thir[map_index]['Time']]
    RMSE_SP_CA = [cen_acc_full[map_index]['RMSE'], cen_acc_half[map_index]['RMSE'], cen_acc_thir[map_index]['RMSE']]
    EV_SP_CA   = [cen_acc_full[map_index]['Explain Variance'], cen_acc_half[map_index]['Explain Variance'], cen_acc_thir[map_index]['Explain Variance']]
    # Raceline & capped speed
    TIME_SP_RC = [ra_cap_full[map_index]['Time'], ra_cap_half[map_index]['Time'], ra_cap_thir[map_index]['Time']]
    RMSE_SP_RC = [ra_cap_full[map_index]['RMSE'], ra_cap_half[map_index]['RMSE'], ra_cap_thir[map_index]['RMSE']]
    EV_SP_RC   = [ra_cap_full[map_index]['Explain Variance'], ra_cap_half[map_index]['Explain Variance'], ra_cap_thir[map_index]['Explain Variance']]
    # Centerline & capped speed
    TIME_SP_CC = [cen_cap_full[map_index]['Time'], cen_cap_half[map_index]['Time'], cen_cap_thir[map_index]['Time']]
    RMSE_SP_CC = [cen_cap_full[map_index]['RMSE'], cen_cap_half[map_index]['RMSE'], cen_cap_thir[map_index]['RMSE']]
    EV_SP_CC   = [cen_cap_full[map_index]['Explain Variance'], cen_cap_half[map_index]['Explain Variance'], cen_cap_thir[map_index]['Explain Variance']]
    # Best kernel across maps
    EV_RA_MAPF = [ra_acc_full[0]['Explain Variance'], ra_acc_full[1]['Explain Variance'], ra_acc_full[2]['Explain Variance']]
    EV_RA_MAPH = [ra_acc_half[0]['Explain Variance'], ra_acc_half[1]['Explain Variance'], ra_acc_half[2]['Explain Variance']]
    EV_RA_MAPT = [ra_acc_thir[0]['Explain Variance'], ra_acc_thir[1]['Explain Variance'], ra_acc_thir[2]['Explain Variance']]

    dataX      = ['ra_non-cap', 'cen_non-cap', 'ra_cap',  'cen_cap']
    Time_data  = [TIME_SP_RA, TIME_SP_CA, TIME_SP_RC, TIME_SP_CC]
    EV_data    = [EV_SP_RA, EV_SP_CA, EV_SP_RC, EV_SP_CC]
    EV_RA      = [EV_RA_MAPF, EV_RA_MAPH, EV_RA_MAPT]
    # print(EV_SP_RA)
    plot_results(dataX, EV_RA, 'RA')