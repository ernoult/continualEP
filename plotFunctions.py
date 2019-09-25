import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pickle
import torch

fontsize = 12

def plot_T(NT, DT, *arg): 

    args = arg[0]
    toymodel = args.toymodel
    learning_rule = args.learning_rule

        if args.learning_rule == 'vf':
            #****************FIGURE SIZE****************#           
            fig = plt.figure(figsize = (4, 2*len(NT)))
            plt.rcParams.update({'font.size': fontsize})
            #*******************************************#	
            #plt.subplots_adjust(hspace = 1)
            plt.rcParams.update({'font.size': fontsize})
            N = int((len(NT) - 1)/2)
            for i in range(N):
                plt.subplot(len(NT), 1, 2*i +1)
                for j in range(10):
                    ind_temp0, ind_temp1 = np.random.randint(NT[2*i][0, :, :].size(0)), np.random.randint(NT[2*i][0, :, :].size(1))
                    plt.plot(NT[2*i][:, ind_temp0, ind_temp1].cpu().numpy(), label='NT'+str(2*i) + str(2*i+1)+'['+str(ind_temp0)+str(ind_temp1)+']',color='C'+str(j))
                    plt.plot(DT[2*i][:, ind_temp0, ind_temp1].cpu().numpy(), label='DT'+ str(2*i) + str(2*i+1)+'['+str(ind_temp0)+str(ind_temp1)+']',color='C'+str(j),linestyle='--')
                plt.xlabel('t')
                plt.title(r'$\Delta_{W_{' + str(2*i) + str(2*i + 1)+r'}}^{\rm EP}$, $-\nabla_{W_{' + str(2*i) + str(2*i + 1)+r'}}^{\rm BPTT}$')
                plt.grid()
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

                plt.subplot(len(NT), 1, 2*i + 2)
                for j in range(10):
                    ind_temp0, ind_temp1 = np.random.randint(NT[2*i + 1][0, :, :].size(0)), np.random.randint(NT[2*i + 1][0, :, :].size(1))
                    plt.plot(NT[2*i + 1][:, ind_temp0, ind_temp1].cpu().numpy(), label='NT'+str(2*i + 1) + str(2*i)+'['+str(ind_temp0)+str(ind_temp1)+']',color='C'+str(j))
                    plt.plot(DT[2*i + 1][:, ind_temp0, ind_temp1].cpu().numpy(), label='DT'+ str(2*i + 1) + str(2*i)+'['+str(ind_temp0)+str(ind_temp1)+']',color='C'+str(j),linestyle='--')
                plt.xlabel('t')
                plt.title(r'$\Delta_{W_{' + str(2*i + 1) + str(2*i)+r'}}^{\rm EP}$, $-\nabla_{W_{' + str(2*i + 1) + str(2*i)+r'}}^{\rm BPTT}$')      
                plt.grid()
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

            plt.subplot(len(NT), 1, len(NT))
            for i in range(10):
                ind_tempx, ind_temp1 = np.random.randint(NT[-1][0, :, :].size(0)), np.random.randint(NT[-1][0, :, :].size(1))
                plt.plot(NT[-1][:, ind_tempx, ind_temp1].cpu().numpy(), label='NT' + str(len(NT))+'['+str(ind_tempx)+str(ind_temp1)+']',color='C'+str(i))
                plt.plot(DT[-1][:, ind_tempx, ind_temp1].cpu().numpy(), label='DT'+ str(len(NT))+'['+str(ind_tempx)+str(ind_temp1)+']',color='C'+str(i),linestyle='--')
            plt.xlabel('t')
            plt.title(r'$\Delta_{W_{'+ str(N - 1) +r'x}}^{\rm EP}$, $-\nabla_{W_{'+ str(N - 1) +r'x}}^{\rm BPTT}$')
            plt.grid()
            plt.subplots_adjust(hspace = 0.5)
            fig.tight_layout()
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        else:
            N = int((len(NT) - 1)/2) + 1 
            #****************FIGURE SIZE****************#           
            fig = plt.figure(figsize = (4, 2*N))
            plt.rcParams.update({'font.size': fontsize})
            #*******************************************#	
            plt.rcParams.update({'font.size': fontsize})
            for i in range(N - 1):
                plt.subplot(N, 1, 1 + i)
                for j in range(10):
                    ind_temp0, ind_temp1 = np.random.randint(NT[2*i][0, :, :].size(0)), np.random.randint(NT[2*i][0, :, :].size(1))
                    plt.plot(NT[2*i][:, ind_temp0, ind_temp1].cpu().numpy(), 
                            label='NT'+str(2*i) + str(2*i+1)+'['+str(ind_temp0)+str(ind_temp1)+']',color='C'+str(j))
                    plt.plot(DT[2*i][:, ind_temp0, ind_temp1].cpu().numpy(), 
                            label='DT'+ str(2*i) + str(2*i+1)+'['+str(ind_temp0)+str(ind_temp1)+']',color='C'+str(j),linestyle='--')
                plt.xlabel('t')
                plt.title(r'$\Delta_{W_{' + str(i) + str(i + 1)+r'}}^{\rm EP}$, $-\nabla_{W_{' + str(i) + str(i + 1)+r'}}^{\rm BPTT}$')
                plt.grid()
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


            plt.subplot(N, 1, N)
            for i in range(10):
                ind_tempx, ind_temp1 = np.random.randint(NT[-1][0, :, :].size(0)), np.random.randint(NT[-1][0, :, :].size(1))
                plt.plot(NT[-1][:, ind_tempx, ind_temp1].cpu().numpy(), 
                        label='NT' + str(len(NT))+'['+str(ind_tempx)+str(ind_temp1)+']',color='C'+str(i))
                plt.plot(DT[-1][:, ind_tempx, ind_temp1].cpu().numpy(), 
                        label='DT'+ str(len(NT))+'['+str(ind_tempx)+str(ind_temp1)+']',color='C'+str(i),linestyle='--')
            plt.xlabel('t')
            plt.title(r'$\Delta_{W_{'+ str(N - 1) +r'x}}^{\rm EP}$, $-\nabla_{W_{'+ str(N - 1) +r'x}}^{\rm BPTT}$')
            plt.grid()
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

            plt.subplots_adjust(hspace = 0.5)
            fig.tight_layout()
        


def plot_S(nS, dS):
    #****************FIGURE SIZE****************#
    fig = plt.figure(figsize = (4, 2*len(nS)))
    plt.rcParams.update({'font.size': fontsize})  
    #*******************************************#
    plt.rcParams.update({'font.size': fontsize})  
    for i in range(len(nS)):       
        plt.subplot(len(nS), 1, 1 + i)
        for j in range(5):
            if (i > 0):
                n = np.random.randint(nS[i].size(2))
                plt.plot(nS[i][:, 0, n].cpu().numpy(),label='nS'+str(i)+'['+str(n)+']',color='C'+str(j))
                plt.plot(dS[i][:, 0, n].cpu().numpy(), label='dS'+str(i)+'['+str(n)+']',color='C'+str(j),linestyle='--')
            else:
                plt.plot(nS[i][:, 0, j].cpu().numpy(),label='nS'+str(i)+'['+str(j)+']',color='C'+str(j))
                plt.plot(dS[i][:, 0, j].cpu().numpy(), label='dS'+str(i)+'['+str(j)+']',color='C'+str(j),linestyle='--')
  
        plt.xlabel('t')
        plt.title(r'$\Delta_{s_{' + str(i) +r'}}^{\rm EP}$, $-\nabla_{s_{' + str(i) +r'}}^{\rm BPTT}$')   
        plt.grid()    
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.subplots_adjust(hspace = 0.5)
        fig.tight_layout()

        

def plot_results(what, *arg):
    if what == 'error':     
        error_train_tab = arg[0]
        error_test_tab = arg[1]
        epochs = len(error_train_tab)
        plt.figure(figsize=(5, 5))
        plt.plot(np.linspace(1, epochs, epochs), error_train_tab, label = 'training error')
        plt.plot(np.linspace(1, epochs, epochs), error_test_tab, label = 'test error')
        plt.legend(loc = 'best')
        plt.xlabel('Epochs')
        plt.ylabel('Error rate (%)')
        plt.grid()


if __name__ == '__main__':
    BASE_PATH = os.getcwd() + '/results' 
    infile = open(BASE_PATH,'rb')
    results_dict = pickle.load(infile)
    infile.close()
	
    if 'nS' in results_dict:
        nS = results_dict['nS']
        dS = results_dict['dS']
        nT = results_dict['nT']
        dT = results_dict['dT']
        args = results_dict['args'] 
        hist_S, hist_T = compute_Hist(nS, dS, nT, dT)
        plot_Hist(hist_S, hist_T, nT, args)   
        plt.show()
        plot_S(nS, dS)
                  
        plot_T(nT, dT, args)                                 
        plt.show()

    if 'error_train_tab' in results_dict:
        plot_results('error', results_dict['error_train_tab'], results_dict['error_test_tab'])
        print('Elapsed time: {}'.format(results_dict['elapsed_time']))
        plt.title('EP')

    
        



    

