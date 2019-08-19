from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.optim as optim
import os, sys
import datetime
from shutil import copyfile
import copy

def train(net, train_loader, epoch, learning_rule): 

    net.train()
    loss_tot = 0
    correct = 0
    criterion = nn.MSELoss(reduction = 'sum')
    
    if learning_rule == 'BPTT':
        for i in range(len(net.w)):
            if net.w[i] is not None:
                net.w[i].weight.requires_grad = True
                if net.w[i].bias is not None:
                    net.w[i].bias.requires_grad = True                                
       
    #****************************DEBUG*****************************#

    if net.debug:
        dicts_syn = {'sign': np.zeros(len(train_loader)), 'zero': np.zeros(len(train_loader)),
                 'mean_w': np.zeros(len(train_loader)), 'std_w': np.zeros(len(train_loader)),
                 'mean_bias': np.zeros(len(train_loader)), 'std_bias': np.zeros(len(train_loader)),
                'align_1': np.zeros(len(train_loader)), 'align_2': np.zeros(len(train_loader))}
        hyperdict_syn = []
        for i in range(len(net.w)):
            hyperdict_syn.append(copy.deepcopy(dicts_syn))

        dicts_neu = {'satmin': np.zeros(len(train_loader)), 'satmax': np.zeros(len(train_loader))}
        hyperdict_neu = []
        for i in range(net.ns):
            hyperdict_neu.append(copy.deepcopy(dicts_neu))
    #**************************************************************# 
    
    for batch_idx, (data, targets) in enumerate(train_loader):            
        s = net.initHidden(data.size(0))
        
        if net.cuda:
            data, targets = data.to(net.device), targets.to(net.device)
            for i in range(net.ns):
                s[i] = s[i].to(net.device)
            
        if learning_rule == 'BPTT':    
            net.zero_grad()
            s = net.forward(data, s, method = 'withgrad')
            pred = s[0].data.max(1, keepdim=True)[1]          
            loss = (1/(2*s[0].size(0)))*criterion(s[0], targets)
            ###############################* BPTT *###################################              
            loss.backward()
            with torch.no_grad():                      
                for i in range(len(net.w)):
                    if net.w[i] is not None:			
                        w_temp = net.w[i].weight
                        w_temp -= net.lr_tab[int(np.floor(i/2))]*w_temp.grad
                        if net.w[i].bias is not None:
                            w_temp = net.w[i].bias
                            w_temp -= net.lr_tab[int(np.floor(i/2))]*w_temp.grad                                   
            ##########################################################################
            
        elif learning_rule == 'ep':
            with torch.no_grad():
                s = net.forward(data, s)
                pred = s[0].data.max(1, keepdim=True)[1]
                loss = (1/(2*s[0].size(0)))*criterion(s[0], targets)
                ###################################* EQPROP *############################################
                seq = [i.clone for i in s]
                #for i in range(len(s)):
                    #seq.append(s[i].clone())

                #***************************************debug_cep***************************************#
                if not net.debug_cep:
                    if net.randbeta > 0:
                        signbeta = 2*np.random.binomial(1, net.randbeta, 1).item() - 1
                        beta = signbeta*net.beta
                    else:
                        beta = net.beta
                    #print(beta)
                    s = net.forward(data, s, target = targets, beta = beta, method = 'nograd')
                    if not net.cep:   
                        Dw = net.computeGradients(data, s, seq, beta)
                        net.updateWeights(Dw)
                else:
                    s, Dw = net.forward(data, s, target = targets, beta = net.beta, method = 'nograd')
                    with torch.no_grad():
                        for ind, w_temp in enumerate(net.w):
                            if w_temp is not None:
                                w_temp.weight -= net.lr_tab_debug[int(np.floor(ind/2))]*Dw[0][ind]
                                w_temp.bias -= net.lr_tab_debug[int(np.floor(ind/2))]*Dw[1][ind]  
                        
                    net.updateWeights(Dw)                            
                #***********************************************************************************#

                #########################################################################################

        elif learning_rule == 'vf':
            with torch.no_grad():
                s = net.forward(data, s)
                pred = s[0].data.max(1, keepdim=True)[1]
                loss = (1/(2*s[0].size(0)))*criterion(s[0], targets)
                ###################################* VF-EQPROP *#########################################
                seq = [i.clone() for i in s]
                    #for i in range(len(s)):
                        #seq.append(s[i].clone())

                #******************************************FORMER C-VF******************************************#
                if net.randbeta > 0:
                    signbeta = 2*np.random.binomial(1, net.randbeta, 1).item() - 1
                    beta = signbeta*net.beta
                else:
                    beta = net.beta

                if not net.former:	
                    s, Dw = net.forward(data, s, target = targets, beta = beta, method = 'nograd')
                else:
                    s, Dw = net.forward(data, s, target = targets, beta = beta, method = 'nograd', seq = seq)
                #***********************************************************************************************#

                if not net.cep:
                    if not net.former:                   			
                        net.updateWeights(Dw)
                    else:
                        Dw_former = net.computeGradients(data, s, seq, beta)
                        net.updateWeights(Dw_former)
                #########################################################################################                
	

                   
        loss_tot += loss                     
        targets_temp = targets.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets_temp.data.view_as(pred)).cpu().sum()
                                    
        if (batch_idx + 1)% 100 == 0:
           print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
               epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
               100. * (batch_idx + 1) / len(train_loader), loss.data))



        #***********************DEBUG***********************#
        if net.debug:
            counter_sign_T, counter_zero_T, counter_sat, counter_w, counter_bias, counter_align = receipe_mb(net, data, targets, batch_idx)

            for ind in range(len(counter_sign_T)):#fc layers
                hyperdict_syn[ind]['sign'][batch_idx] = counter_sign_T[ind]
                hyperdict_syn[ind]['zero'][batch_idx] = counter_zero_T[ind]

                #*********************WEIGHT TRACKING*********************#
                hyperdict_syn[ind]['mean_w'][batch_idx] = counter_w[0][ind]
                hyperdict_syn[ind]['std_w'][batch_idx] = counter_w[1][ind]
                hyperdict_syn[ind]['mean_bias'][batch_idx] = counter_bias[0][ind]
                hyperdict_syn[ind]['std_bias'][batch_idx] = counter_bias[1][ind]
                #*********************************************************#

                #*******************ALIGNMENT TRACKING********************#
                hyperdict_syn[ind]['align_1'][batch_idx] = counter_align[0][ind]
                hyperdict_syn[ind]['align_2'][batch_idx] = counter_align[1][ind]
                #*********************************************************#

            for ind in range(net.ns):
                hyperdict_neu[ind]['satmin'][batch_idx] = counter_sat[0][ind]
                hyperdict_neu[ind]['satmax'][batch_idx] = counter_sat[1][ind]

        #***************************************************#  

       
        
    loss_tot /= len(train_loader.dataset)
    
    
    print('\nAverage Training loss: {:.4f}, Training Error Rate: {:.2f}% ({}/{})\n'.format(
       loss_tot,100*(len(train_loader.dataset)- correct.item() )/ len(train_loader.dataset), len(train_loader.dataset)-correct.item(), len(train_loader.dataset),
       ))

    if not net.debug:
        return 100*(len(train_loader.dataset)- correct.item())/ len(train_loader.dataset)

    else:
        return 100*(len(train_loader.dataset)- correct.item())/ len(train_loader.dataset), [hyperdict_neu, hyperdict_syn]
       
    

def receipe_mb(net, data, targets, batch_idx):
    
    counter_sign_T = np.zeros(len(net.w))
    counter_zero_T = np.zeros(len(net.w))

    counter_satmax = np.zeros(len(net.w))
    counter_satmin = np.zeros(len(net.w))

    #********************WEIGHT TRACKING********************#
    counter_mean_w = np.zeros(len(net.w))
    counter_std_w = np.zeros(len(net.w))
    counter_mean_bias = np.zeros(len(net.w))
    counter_std_bias = np.zeros(len(net.w))
    #*******************************************************#

    #*************COUNTER ALIGN*************#
    counter_align_1 = np.zeros(len(net.w))
    counter_align_2 = np.zeros(len(net.w))
    #***************************************#

    batch_size = data.size(0)                                  
    s = net.initHidden(batch_size)
    if net.cuda:
        data, targets = data.to(net.device), targets.to(net.device)
        for i in range(len(s)):
            s[i] = s[i].to(net.device)
    
    #Check dS, nS, DT computation
    nS, dS, DT, seq = compute_nSdSdT(net, data, targets)

    #Check NT computation		       
    NT = compute_nT(net, data, targets, wholeProcess = False, diff = False)

      
    for i in range(len(NT)):
        size_temp = DT[i][-1, :].view(-1,).size()[0]
        counter_temp = ((torch.sign(NT[i][-1, :]) == torch.sign(DT[i][-1, :])) & (torch.abs(NT[i][-1, :]) > 0) & (torch.abs(DT[i][-1, :]) > 0)).sum().item()*100/size_temp

        counter_temp_2 = ((NT[i][-1, :] == DT[i][-1, :]) & (NT[i][-1, :] == torch.zeros_like(NT[i][-1, :]))).sum().item()*100/size_temp		
        counter_sign_T[i] = counter_temp
        counter_zero_T[i] = counter_temp_2
        counter_mean_w[i] = net.w[i].weight.data.mean()
        counter_std_w[i] = net.w[i].weight.data.std()
        if net.w[i].bias is not None:
            counter_mean_bias[i] = net.w[i].bias.data.mean()
            counter_std_bias[i] = net.w[i].bias.data.std()

        if (batch_idx + 1)% 100 == 0:
            print('fc layer {}: {:.1f}% (same sign, total), i.e. {:.1f}% (stricly non zero), {:.1f}% (both zero)'.format(i, counter_temp + counter_temp_2, counter_temp, counter_temp_2))
        del counter_temp, counter_temp_2

    #******************ALIGNMENT***********************#
    for i in range(int(np.floor((len(NT) - 1)/2))):
        size_temp = DT[2*i][-1, :].view(-1,).size()[0]
        counter_align_1[2*i] = (torch.sign(net.w[2*i].weight.data) == torch.sign(torch.transpose(net.w[2*i + 1].weight.data, 0, 1))).sum().item()*100/size_temp
	
        counter_align_2[2*i] = np.arccos((net.w[2*i].weight.data*torch.transpose(net.w[2*i + 1].weight.data, 0 ,1)).sum().item()/np.sqrt((net.w[2*i].weight.data**2).sum().item()*(net.w[2*i + 1].weight.data**2).sum().item()))
    #**************************************************#

    for i in range(len(seq)):
        size_temp = seq[i].view(-1,).size()[0]
        counter_temp = (seq[i] == 0).sum().item()*100/size_temp
        counter_temp_2 = (seq[i] == 1).sum().item()*100/size_temp
        counter_satmin[i] = counter_temp
        counter_satmax[i] = counter_temp_2 
        if (batch_idx + 1)% 100 == 0:
            print('saturation in layer {}: {:.1f}% (min), {:.1f}% (max)'.format(i, counter_temp, counter_temp_2))  
        del counter_temp, counter_temp_2  
   

    
    return counter_sign_T, counter_zero_T, [counter_satmin, counter_satmax], [counter_mean_w, counter_std_w], [counter_mean_bias, counter_std_bias], [counter_align_1, counter_align_2]



def computeInitialAngle(net):
    counter_align_1 = np.zeros(len(net.w))
    counter_align_2 = np.zeros(len(net.w))
    dict = {'align_1': [], 'align_2': []}
    hyperdict = []
    for i in range(len(net.w)):
        hyperdict.append(copy.deepcopy(dict))

    for i in range(int(np.floor((len(net.w) - 1)/2))):
        size_temp = net.w[2*i].weight.data.view(-1,).size()[0]
        counter_align_1[2*i] = (torch.sign(net.w[2*i].weight.data) == torch.sign(torch.transpose(net.w[2*i + 1].weight.data, 0, 1))).sum().item()*100/size_temp
        counter_align_2[2*i] = (180/np.pi)*np.arccos((net.w[2*i].weight.data*torch.transpose(net.w[2*i + 1].weight.data, 0 ,1)).sum().item()/np.sqrt((net.w[2*i].weight.data**2).sum().item()*(net.w[2*i + 1].weight.data**2).sum().item()))
        hyperdict[2*i]['align_1'] = counter_align_1[2*i]
        hyperdict[2*i]['align_2'] = counter_align_2[2*i]

    return hyperdict

def evaluate(net, test_loader): 

    net.eval()
    loss_tot_test = 0
    correct_test = 0
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader): 
            s = net.initHidden(data.size(0))             
            if net.cuda:
                data, targets = data.to(net.device), targets.to(net.device)
                for i in range(net.ns):
                    s[i] = s[i].to(net.device)
                                  
            s = net.forward(data, s, method = 'nograd')
             
            loss_tot_test += (1/2)*((s[0]-targets)**2).sum()                
            pred = s[0].data.max(1, keepdim = True)[1]
            targets_temp = targets.data.max(1, keepdim = True)[1]
            correct_test += pred.eq(targets_temp.data.view_as(pred)).cpu().sum()
            
    loss_tot_test = loss_tot_test / len(test_loader.dataset)
    accuracy = correct_test.item() / len(test_loader.dataset)
    print('\nAverage Test loss: {:.4f}, Test Error Rate: {:.2f}% ({}/{})\n'.format(
        loss_tot_test,100. *(len(test_loader.dataset)- correct_test.item() )/ len(test_loader.dataset), len(test_loader.dataset)-correct_test.item(), len(test_loader.dataset)))        
    return 100 *(len(test_loader.dataset)- correct_test.item() )/ len(test_loader.dataset) 
    
    

def compute_nSdSdT(net, data, target): 

    beta = net.beta
    batch_size_temp = data.size(0)
    s = net.initHidden(batch_size_temp)    
    if net.cuda: 
        for i in range(net.ns):
            s[i] = s[i].to(net.device)
        
    net.zero_grad()
    s, nS = net.forward(data, s, target = target, method = 'nS')
    
    
    seq = []
    for i in range(len(s)):         
        seq.append(s[i].clone())
    with torch.no_grad():
        s, dS, dT = net.forward(data, s, seq, target = target, beta = beta, method = 'dSdT')

    return nS, dS, dT, seq


def compute_nT(net, data, target, wholeProcess = True, diff = True):

    batch_size_temp = data.size(0)
    
    NT = []
    for i in range(len(net.w)):
        if net.w[i] is not None:
            NT.append(torch.zeros(net.Kmax, net.w[i].weight.size(0), net.w[i].weight.size(1)))
        else:
            NT.append(None)
        
    criterion = nn.MSELoss(reduction = 'sum')
    if wholeProcess:
        for K in range(net.Kmax):
            print(K)
            s = net.initHidden(batch_size_temp)
            if net.cuda: 
                for i in range(net.ns):
                    s[i] = s[i].to(net.device)     
            net.zero_grad()
            s = net.forward(data, s, method = 'withgrad', K = K)    
            loss = (1/(2.0*s[0].size(0)))*criterion(s[0], target)
            loss.backward()
            
            for i in range(len(NT)):
                if net.w[i] is not None:
                    NT[i][K, :, :] = net.w[i].weight.grad.clone()
    else:
            s = net.initHidden(batch_size_temp)
            if net.cuda: 
                for i in range(net.ns):
                    s[i] = s[i].to(net.device)     
            net.zero_grad()
            s = net.forward(data, s, method = 'withgrad', K = net.Kmax)    
            loss = (1/(2.0*s[0].size(0)))*criterion(s[0], target)
            loss.backward()
            
            for i in range(len(NT)):
                if net.w[i] is not None:
                    NT[i][-1, :, :] = net.w[i].weight.grad.clone()

    if diff:
        NT = compute_diffT(NT)
                     
    return NT
  
        
def compute_diffT(NT):

    nT = []

    for i in range(len(NT)):
        if NT[i] is not None:
            nT.append(torch.zeros_like(NT[i]))
        else:
            nT.append(None)          

    for i in range(len(NT)):
        if NT[i] is not None:
            for t in range(NT[i].size(0) - 1):
                nT[i][t + 1, :, :] = NT[i][t + 1, :, :] - NT[i][t, :, :]

    return nT


def receipe(net, train_loader, N_trials):

    counter_sign_Theta = np.zeros((N_trials, len(net.w)))
    counter_zero_Theta = np.zeros((N_trials, len(net.w)))
  
    for n in range(N_trials):
        print('mini-batch {}/{}'.format(n + 1, N_trials))
        batch_idx, (data, targets) = next(enumerate(train_loader))
        batch_size = data.size(0)                                  
        s = net.initHidden(batch_size)
        if net.cuda:
            data, targets = data.to(net.device), targets.to(net.device)
            for i in range(len(s)):
                s[i] = s[i].to(net.device)
        
        #Check dS, nS, DT computation
        nS, dS, DT, _ = compute_nSdSDT(net, data, targets)

        #Check NT computation		       
        NT = computeNT(net, data, targets, wholeProcess = False)

        #***************************COMPUTE PROPORTION OF SYNAPSES WHICH HAVE THE GOOD SIGN******************************#
        
        for i in range(len(NT)):
            if NT[i] is not None:
                size_temp = DT[i][-1, :].view(-1,).size()[0]
                counter_temp = ((torch.sign(NT[i][-1, :]) == torch.sign(DT[i][-1, :])) & (torch.abs(NT[i][-1, :]) > 0) & (torch.abs(DT[i][-1, :]) > 0)).sum().item()*100/size_temp

                counter_temp_2 = ((NT[i][-1, :] == DT[i][-1, :]) & (NT[i][-1, :] == torch.zeros_like(NT[i][-1, :]))).sum().item()*100/size_temp
                                 
                counter_sign_Theta[n, i] = counter_temp
                counter_zero_Theta[n, i] = counter_temp_2

                print('layer {}: {:.1f}% (same sign, total), i.e. {:.1f}% (stricly non zero), {:.1f}% (both zero)'.format(int(i/2), counter_temp + counter_temp_2, counter_temp, counter_temp_2))

    
    #***************************************************************************************************************#

    print('************Statistics on {} trials************'.format(N_trials))

    for i in range(len(NT)):
        if NT[i] is not None:
            print('average layer {}: {:.1f} +- {:.1f}%  (same sign, total), i.e. {:.1f} +- {:.1f}%  (stricly non zero), {:.1f} +- {:.1f}%  (both zero)'.format(int(i/2), 
                    counter_sign_Theta[:, i].mean() + counter_zero_Theta[:, i].mean(), 
                    counter_sign_Theta[:, i].std() + counter_zero_Theta[:, i].std(), 
                    counter_sign_Theta[:, i].mean(), 
                    counter_sign_Theta[:, i].std(), 
                    counter_zero_Theta[:, i].mean(),
                    counter_zero_Theta[:, i].std()))

    print('***********************************************')
    print('done')

        

def createPath(args):

    if args.action == 'train':
        BASE_PATH = os.getcwd() + '/' 

        if args.cep:
            name = 'c-' + args.learning_rule
        else:
            name = args.learning_rule

        if args.discrete:
            name = name + '_disc'
        else:
            name = name + '_cont'
        name = name + '_' + str(len(args.size_tab) - 2) + 'hidden'
                    
        BASE_PATH = BASE_PATH + name

        if not os.path.exists(BASE_PATH):
            os.mkdir(BASE_PATH)

        BASE_PATH = BASE_PATH + '/' + datetime.datetime.now().strftime("cuda" + str(args.device_label)+"-%Y-%m-%d")

        if not os.path.exists(BASE_PATH):
            os.mkdir(BASE_PATH)

        files = os.listdir(BASE_PATH)

        if not files:
            BASE_PATH = BASE_PATH + '/' + 'Trial-1'
        else:
            tab = []
            for names in files:
                if not names[-2] == '-':
                    tab.append(int(names[-2] + names[-1]))
                else:    
                    tab.append(int(names[-1]))
            print(tab)    			
            BASE_PATH = BASE_PATH + '/' + 'Trial-' + str(max(tab)+1)                                
        
        os.mkdir(BASE_PATH) 
        filename = 'results'   
        
        #************************************#
        copyfile('plotFunctions.py', BASE_PATH + '/plotFunctions.py')
        #************************************#

        return BASE_PATH, name
    
    elif args.action == 'plotcurves':
        BASE_PATH = os.getcwd() + '/' 

        if args.cep:
            name = 'c-' + args.learning_rule
        else:
            name = args.learning_rule
                
        if args.discrete:
            name = name + '_disc'
        else:
            name = name + '_cont'
        name = name + '_' + str(len(args.size_tab) - 2) + 'hidden'
                    
        BASE_PATH = BASE_PATH + name

        if not os.path.exists(BASE_PATH):
            os.mkdir(BASE_PATH)

        files = os.listdir(BASE_PATH)

        if not files:
            BASE_PATH = BASE_PATH + '/' + 'Trial-1'
        else:
            tab = []
            for names in files:
                tab.append(int(names[-1]))
            BASE_PATH = BASE_PATH + '/' + 'Trial-' + str(max(tab)+1)                                
        
        os.mkdir(BASE_PATH) 
        filename = 'results'   
        
        #********************************************************#
        copyfile('plotFunctions.py', BASE_PATH + '/plotFunctions.py')
        #********************************************************#

        return BASE_PATH, name


def createHyperparameterfile(BASE_PATH, name, args):    

    if args.action == 'train':
        learning_rule = args.learning_rule
        if args.cep:
            learning_rule = 'c-' + learning_rule
        hyperparameters = open(BASE_PATH + r"/hyperparameters.txt","w+") 
        L = [" TRAINING: list of hyperparameters " + "(" + name + ", " + datetime.datetime.now().strftime("cuda" + str(args.device_label)+"-%Y-%m-%d") + ") \n",
			"- Learning rule: " + learning_rule + "\n",
            "- Weight initialization: " + args.weight_initialization + "\n",
            "- T: {}".format(args.T) + "\n",
            "- Kmax: {}".format(args.Kmax) + "\n",
            "- beta: {:.2f}".format(args.beta) + "\n", 
            "- batch size: {}".format(args.batch_size) + "\n",
            "- activation function: " + args.activation_function + "\n",
            "- number of epochs: {}".format(args.epochs) + "\n",
            "- learning rates: {}".format(args.lr_tab) + "\n"]

        if not args.discrete:
            L.append("- dt: {:.3f}".format(args.dt) + "\n")   

        if args.randbeta > 0:
            L.append("- Probability of beta sign switching: {}".format(args.randbeta) + "\n")

        if args.angle > 0:
		    L.append("- Initial angle between forward and backward weights: {}".format(args.angle) + "\n")

        L.append("- layer sizes: {}".format(args.size_tab) + "\n")

        hyperparameters.writelines(L) 
        hyperparameters.close()
    
    elif args.action == 'plotcurves':        
        hyperparameters = open(BASE_PATH + r"/hyperparameters.txt","w+") 
        L = ["NABLA-DELTA CURVES: list of hyperparameters " + "(" + name + ", " + datetime.datetime.now().strftime("cuda" + str(args.device_label)+"-%Y-%m-%d") + ") \n",
            "- Learning rule: " + args.learning_rule + "\n",
            "- T: {}".format(args.T) + "\n",
            "- Kmax: {}".format(args.Kmax) + "\n",
            "- beta: {:.2f}".format(args.beta) + "\n", 
            "- batch size: {}".format(args.batch_size) + "\n",
            "- activation function: " + args.activation_function + "\n"]

        if not args.discrete:
            L.append("- dt: {:.3f}".format(args.dt) + "\n")   


        L.append("- layer sizes: {}".format(args.size_tab) + "\n")

        hyperparameters.writelines(L) 
        hyperparameters.close()        

        
