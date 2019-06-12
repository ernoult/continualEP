from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.optim as optim
import torch.nn.functional as F

from main import rho, rhop, rhop2


#*****************************VF, continuous *********************************#

class VFcont(nn.Module):
    def __init__(self, device_label, size_tab, lr_tab, T, Kmax, beta, dt = 1, weight_initialization = 'tied', no_clamp = False):
        super(VFcont, self).__init__()
        self.T = T
        self.Kmax = Kmax        
        self.dt = dt

        self.size_tab = size_tab
        self.lr_tab = lr_tab
        self.ns = len(size_tab) - 1
        self.nsyn = 2*(self.ns - 1) + 1

        if device_label >= 0:    
            device = torch.device("cuda:"+str(device_label)+")")
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False   
        self.device = device
        self.no_clamp = no_clamp
        self.beta = beta

        w = nn.ModuleList([])                            
        for i in range(self.ns - 1):
            w.append(nn.Linear(size_tab[i + 1], size_tab[i], bias = True))
            w.append(nn.Linear(size_tab[i], size_tab[i + 1], bias = False))
            
        w.append(nn.Linear(size_tab[-1], size_tab[-2]))
               
        if weight_initialization == 'tied':
            for i in range(self.ns - 1):
                w[2*i + 1].weight.data = torch.transpose(w[2*i].weight.data.clone(), 0, 1)
                	
        self.w = w
        self = self.to(device)        

    def stepper(self, data, s, target = None, beta = 0, return_derivatives = False):
        dsdt = []
        dsdt.append(-s[0] + self.w[0](rho(s[1])))     
        if beta > 0:
            dsdt[0] = dsdt[0] + beta*(target-s[0])

        for i in range(1, self.ns - 1):
            dsdt.append(-s[i] + self.w[2*i](rho(s[i + 1])) + self.w[2*i - 1](rho(s[i - 1])))

        dsdt.append(-s[-1] + self.w[-1](rho(data)) + self.w[-2](rho(s[-2])))
        
        for i in range(self.ns):
            s[i] = s[i] + self.dt*dsdt[i]
            
        if not self.no_clamp:
            for i in range(self.ns):
                s[i] = s[i].clamp(min = 0).clamp(max = 1)
                dsdt[i] = torch.where((s[i] == 0)|(s[i] ==1), torch.zeros_like(dsdt[i], device = self.device), dsdt[i])
                                     
        if return_derivatives:
           return s, dsdt
        else:
            return s
    
    def forward(self, data, s, seq = None, method = 'nograd',  beta = 0, target = None, **kwargs):
        T = self.T
        Kmax = self.Kmax
        if len(kwargs) > 0:
            K = kwargs['K']
        else:
            K = Kmax
        if (method == 'withgrad'):
            for t in range(T):             
                if t == T - 1 - K:
                    for i in range(self.ns):
                        s[i] = s[i].detach()
                        s[i].requires_grad = True    
                    data = data.detach()
                    data.requires_grad = True             
                s = self.stepper(data, s)
            return s
                
        elif (method == 'nograd'):
            if beta == 0:
                for t in range(T):                      
                    s = self.stepper(data, s)
            else:
                for t in range(Kmax):                      
                    s = self.stepper(data, s, target, beta)
            return s                   
                    
        elif (method == 'nS'):
            s_tab = []
            for i in range(self.ns):
                s_tab.append([])
            
            criterion = nn.MSELoss(reduction = 'sum')
            for t in range(T):
                for i in range(self.ns):                 
                    s_tab[i].append(s[i])                    
                    s_tab[i][t].retain_grad()                      
                s = self.stepper(data, s)

            for i in range(self.ns):                 
                s_tab[i].append(s[i])                    
                s_tab[i][-1].retain_grad()                
            loss = (1/(2.0*s[0].size(0)))*criterion(s[0], target)
            loss.backward()
            
            
            nS = []
            for i in range(self.ns):
                nS.append(torch.zeros(Kmax, 1, s[i].size(1), device = self.device))
            
            for t in range(Kmax):
                ###############################nS COMPUTATION#####################################
                for i in range(self.ns):
                    if (t < i):
                        nS[i][t, :, :] = torch.zeros_like(nS[i][t, :, :])
                    else:    
                        nS[i][t, :, :] = (s_tab[i][T - t].grad).sum(0).unsqueeze(0)
                ####################################################################################      
        
               
            return s, nS     
            
        elif (method == 'dSDT'):

                DT = []

                for i in range(len(self.w)):
                    if self.w[i] is not None:
                        DT.append(torch.zeros(Kmax, self.w[i].weight.size(0), self.w[i].weight.size(1)))
                    else:
                        DT.append(None)        
                

                dS = []
                for i in range(self.ns):
                    dS.append(torch.zeros(Kmax, 1, self.size_tab[i], device = self.device))              
                
                    
                for t in range(Kmax):
                    s, dsdt = self.stepper(data, s, target, beta, return_derivatives = True)
                    ###############################dS COMPUTATION#####################################
                    for i in range(self.ns):
                        if (t < i):
                            dS[i][t, :, :] = torch.zeros_like(dS[i][t, :, :])
                        else:
                            dS[i][t, :, :] = -(1/(beta*s[i].size(0)))*dsdt[i].sum(0).unsqueeze(0)                        
                    ######################################################################################


                    #############DT COMPUTATION##################
                    gradw, _ = self.computeGradients(beta, data, s, seq)
                    for i in range(len(gradw)):
                        if gradw[i] is not None:
                            DT[i][t, :, :] = - gradw[i]
                    #####################################################
                                                       
                                                                             
        return s, dS, DT
        
        
    def initHidden(self, batch_size):
        s = []
        for i in range(self.ns):
            s.append(torch.zeros(batch_size, self.size_tab[i], requires_grad = True))            
        return s
        
              
    def computeGradients(self, beta, data, s, seq):
        gradw = []
        gradw_bias = []
        batch_size = s[0].size(0)
               
        for i in range(self.ns - 1):                
            gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(s[i] - seq[i], 0, 1), rho(seq[i + 1]))) 
            gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(s[i + 1] - seq[i + 1], 0, 1), rho(seq[i])))                
            gradw_bias.append((1/(beta*batch_size))*(s[i] - seq[i]).sum(0))
            gradw_bias.append(None)                                                                                  
                                                                
        gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(s[-1] - seq[-1], 0, 1), rho(data)))
        gradw_bias.append((1/(beta*batch_size))*(s[-1] - seq[-1]).sum(0))


        return  gradw, gradw_bias

  
    def updateWeights(self, beta, data, s, seq):
        gradw, gradw_bias = self.computeGradients(beta, data, s, seq)
        lr_tab = self.lr_tab
        for i in range(len(self.w)):
            if self.w[i] is not None:
                self.w[i].weight += lr_tab[int(np.floor(i/2))]*gradw[i]
            if gradw_bias[i] is not None:
                self.w[i].bias += lr_tab[int(np.floor(i/2))]*gradw_bias[i] 


#*****************************VF, discrete *********************************#

class VFdisc(nn.Module):
    def __init__(self, device_label, size_tab, lr_tab, T, Kmax, beta, dt = 1, weight_initialization = 'tied'):
        super(VFdisc, self).__init__()   
        self.T = T
        self.Kmax = Kmax     
        self.dt = dt
        self.size_tab = size_tab
        self.lr_tab = lr_tab
        self.ns = len(size_tab) - 1
        self.nsyn = 2*(self.ns - 1) + 1
        if device_label >= 0:    
            device = torch.device("cuda:"+str(device_label)+")")
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False   
        self.device = device
        self.beta = beta

        w = nn.ModuleList([])                         
        for i in range(self.ns - 1):
            w.append(nn.Linear(size_tab[i + 1], size_tab[i], bias = True))
            w.append(nn.Linear(size_tab[i], size_tab[i + 1], bias = False))
            
        w.append(nn.Linear(size_tab[-1], size_tab[-2]))
               
        if weight_initialization == 'tied':
            for i in range(self.ns - 1):
                w[2*i + 1].weight.data = torch.transpose(w[2*i].weight.data.clone(), 0, 1)
                                 
        self.w = w
        self = self.to(device)

    def stepper(self, data, s, target = None, beta = 0, return_derivatives = False):
        dsdt = []
        dsdt.append(-s[0] + rho(self.w[0](s[1])))
        if beta > 0:
            dsdt[0] = dsdt[0] + beta*(target-s[0])

        for i in range(1, self.ns - 1):
            dsdt.append(-s[i] + rho(self.w[2*i](s[i + 1]) + self.w[2*i - 1](s[i - 1])))

        dsdt.append(-s[-1] + rho(self.w[-1](data) + self.w[-2](s[-2])))

        for i in range(self.ns):
            s[i] = s[i] + self.dt*dsdt[i]

        if return_derivatives:
           return s, dsdt
        else:
            return s
    
    def forward(self, data, s, seq = None, method = 'nograd',  beta = 0, target = None, **kwargs):
        T = self.T
        Kmax = self.Kmax
        if len(kwargs) > 0:
            K = kwargs['K']
        else:
            K = Kmax
        if (method == 'withgrad'):
            for t in range(T):             
                if t == T - 1 - K:
                    for i in range(self.ns):
                        s[i] = s[i].detach()
                        s[i].requires_grad = True    
                    data = data.detach()
                    data.requires_grad = True             
                s = self.stepper(data, s)
            return s
                
        elif (method == 'nograd'):
            if beta == 0:
                for t in range(T):                      
                    s = self.stepper(data, s)
            else:
                for t in range(Kmax):                      
                    s = self.stepper(data, s, target, beta)
            return s                   
                    
        elif (method == 'nS'):
            s_tab = []
            for i in range(self.ns):
                s_tab.append([])
            
            criterion = nn.MSELoss(reduction = 'sum')
            for t in range(T):
                for i in range(self.ns):                 
                    s_tab[i].append(s[i])                    
                    s_tab[i][t].retain_grad()                      
                s = self.stepper(data, s)

            for i in range(self.ns):                 
                s_tab[i].append(s[i])                    
                s_tab[i][-1].retain_grad()                
            loss = (1/(2.0*s[0].size(0)))*criterion(s[0], target)
            loss.backward()
            
            
            nS = []
            for i in range(self.ns):
                nS.append(torch.zeros(Kmax, 1, s[i].size(1), device = self.device))
             
            for t in range(Kmax):
                ###############################nS COMPUTATION#####################################
                for i in range(self.ns):
                    if (t < i):
                        nS[i][t, :, :] = torch.zeros_like(nS[i][t, :, :])
                    else:    
                        nS[i][t, :, :] = (s_tab[i][T - t].grad).sum(0).unsqueeze(0)
                ####################################################################################               
               
            return s, nS     
            
        elif (method == 'dSDT'):

                DT = []

                for i in range(len(self.w)):
                    if self.w[i] is not None:
                        DT.append(torch.zeros(Kmax, self.w[i].weight.size(0), self.w[i].weight.size(1)))
                    else:
                        DT.append(None)        
                

                dS = []
                for i in range(self.ns):
                    dS.append(torch.zeros(Kmax, 1, self.size_tab[i], device = self.device))              
                
                    
                for t in range(Kmax):
                    s, dsdt = self.stepper(data, s, target, beta, return_derivatives = True)
                    ###############################dS COMPUTATION#####################################:
                    for i in range(self.ns):
                        if (t < i):
                            dS[i][t, :, :] = torch.zeros_like(dS[i][t, :, :])
                        else:
                            dS[i][t, :, :] = -(1/(beta*s[i].size(0)))*dsdt[i].sum(0).unsqueeze(0)                        
                    ######################################################################################


                    #############DT COMPUTATION##################
                    gradw, _ = self.computeGradients(beta, data, s, seq)
                    for i in range(len(gradw)):
                        if gradw[i] is not None:
                            DT[i][t, :, :] = - gradw[i]
                    #####################################################
                                                       
                                                                             
        return s, dS, DT
        
        
    def initHidden(self, batch_size):
        s = []
        for i in range(self.ns):
            s.append(torch.zeros(batch_size, self.size_tab[i], requires_grad = True))            
        return s
        
              
    def computeGradients(self, beta, data, s, seq):
        gradw = []
        gradw_bias = []
        batch_size = s[0].size(0)
                   
        for i in range(self.ns - 1):                
            gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(torch.mul(rhop2(seq[i]), s[i] - seq[i]), 0, 1), seq[i + 1]))
            gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(torch.mul(rhop2(seq[i + 1]), s[i + 1] - seq[i + 1]), 0, 1), seq[i]))       
   
            gradw_bias.append((1/(beta*batch_size))*torch.mul(rhop2(seq[i]), s[i] - seq[i]).sum(0))   
            gradw_bias.append(None)                                                                                  
                                                                
        gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(torch.mul(rhop2(seq[-1]), s[-1] - seq[-1]), 0, 1), data))
        gradw_bias.append(None)
                                                                                                                                                                       
        return  gradw, gradw_bias

  
    def updateWeights(self, beta, data, s, seq):
        gradw, gradw_bias = self.computeGradients(beta, data, s, seq)
        lr_tab = self.lr_tab
        for i in range(len(self.w)):
            if self.w[i] is not None:
                self.w[i].weight += lr_tab[int(np.floor(i/2))]*gradw[i]
            if gradw_bias[i] is not None:
                self.w[i].bias += lr_tab[int(np.floor(i/2))]*gradw_bias[i] 
   


#*****************************toy model, VF, continuous *********************************#

class toyVFcont(nn.Module):
    def __init__(self, device_label, size_tab, lr_tab,T, Kmax, beta, dt = 1, weight_initialization = 'tied', no_clamp = False):
        super(toyVFcont, self).__init__()
        self.T = T
        self.Kmax = Kmax        
        self.dt = dt

        self.size_tab = size_tab
        self.lr_tab = lr_tab
        self.ns = len(size_tab) - 1
        self.nsyn = 2*(self.ns - 1) + 1
        if device_label >= 0:    
            device = torch.device("cuda:"+str(device_label)+")")
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False   
        self.device = device
        self.no_clamp = no_clamp
        self.beta = beta

        w = nn.ModuleList([])

        #fully connected architecture
        w.append(nn.Linear(size_tab[0], size_tab[0], bias = False))
        w.append(nn.Linear(size_tab[1], size_tab[0], bias = False))
        w.append(nn.Linear(size_tab[2], size_tab[0], bias = False))
        w.append(nn.Linear(size_tab[1], size_tab[1], bias = False))          
        w.append(nn.Linear(size_tab[2], size_tab[1], bias = False))
        w.append(nn.Linear(size_tab[0], size_tab[1], bias = False))

        #no self connection
        '''
        with torch.no_grad():
            w[0].weight[np.arange(size_tab[0]), np.arange(size_tab[0])] = 0
            w[3].weight[np.arange(size_tab[1]), np.arange(size_tab[1])] = 0 
        '''                      

        w[0].weight.data = 10**(-3)*w[0].weight.data.clone()
        w[1].weight.data = 10**(-3)*w[1].weight.data.clone()
        w[2].weight.data = 10**(-3)*w[2].weight.data.clone() 
        w[3].weight.data = 10**(-3)*w[3].weight.data.clone()
        w[4].weight.data = 10**(-3)*w[4].weight.data.clone()
        w[5].weight.data = 10**(-3)*w[5].weight.data.clone()                 
        
        
        if weight_initialization == 'tied':
            w[0].weight.data = 0.5*(w[0].weight.data.clone() + torch.transpose(w[0].weight.data.clone(), 0, 1))
            w[3].weight.data = 0.5*(w[3].weight.data.clone() + torch.transpose(w[3].weight.data.clone(), 0, 1))                     
            w[1].weight.data = torch.transpose(w[5].weight.data.clone(), 0, 1)              

        self.w = w
        self = self.to(device)

    def stepper(self, data, s, target = None, beta = 0, return_derivatives = False):        
        dsdt = []
        dsdt.append(-s[0] + self.w[0](rho(s[0])) + self.w[1](rho(s[1])) + self.w[2](rho(data)))
        if beta > 0:
            dsdt[0] = dsdt[0] + beta*(target-s[0])
        dsdt.append(-s[1] + self.w[3](rho(s[1])) + self.w[4](rho(data)) + self.w[5](rho(s[0])))
        for i in range(self.ns):
            s[i] = s[i] + self.dt*dsdt[i]
            
        if not self.no_clamp:
            for i in range(self.ns):
                s[i] = s[i].clamp(min = 0).clamp(max = 1)
                dsdt[i] = torch.where((s[i] == 0)|(s[i] ==1), torch.zeros_like(dsdt[i], device = self.device), dsdt[i])
                           
        if return_derivatives:
           return s, dsdt
        else:
            return s
    
    def forward(self, data, s, seq = None, method = 'nograd',  beta = 0, target = None, **kwargs):
        T = self.T
        Kmax = self.Kmax
        if len(kwargs) > 0:
            K = kwargs['K']
        else:
            K = Kmax
        if (method == 'withgrad'):
            for t in range(T):             
                if t == T - 1 - K:
                    for i in range(self.ns):
                        s[i] = s[i].detach()
                        s[i].requires_grad = True    
                    data = data.detach()
                    data.requires_grad = True             
                s = self.stepper(data, s)
            return s
                
        elif (method == 'nograd'):
            if beta == 0:
                for t in range(T):                      
                    s = self.stepper(data, s)
            else:
                for t in range(Kmax):                      
                    s = self.stepper(data, s, target, beta)
            return s                   
                    
        elif (method == 'nS'):
            s_tab = []
            for i in range(self.ns):
                s_tab.append([])
            
            criterion = nn.MSELoss(reduction = 'sum')
            for t in range(T):
                for i in range(self.ns):                 
                    s_tab[i].append(s[i])                    
                    s_tab[i][t].retain_grad()                      
                s = self.stepper(data, s)

            for i in range(self.ns):                 
                s_tab[i].append(s[i])                    
                s_tab[i][-1].retain_grad()                
            loss = (1/(2.0*s[0].size(0)))*criterion(s[0], target)
            loss.backward()
            
            
            nS = []
            for i in range(self.ns):
                nS.append(torch.zeros(Kmax, 1, s[i].size(1), device = self.device))
            
            for t in range(Kmax):
                ###############################nS COMPUTATION#####################################
                for i in range(self.ns):
                     if ((i > 0) & (t == 0)):
                        nS[i][t, :, :] = torch.zeros_like(nS[i][t, :, :])
                     else:      
                        nS[i][t, :, :] = (s_tab[i][T - t].grad).sum(0).unsqueeze(0)                
                ####################################################################################
            return s, nS     
            
        elif (method == 'dSDT'):

                DT = []

                for i in range(len(self.w)):
                    if self.w[i] is not None:
                        DT.append(torch.zeros(Kmax, self.w[i].weight.size(0), self.w[i].weight.size(1)))
                    else:
                        DT.append(None)        
                

                dS = []
                for i in range(self.ns):
                    dS.append(torch.zeros(Kmax, 1, self.size_tab[i], device = self.device))              
                
                    
                for t in range(Kmax):
                    s, dsdt = self.stepper(data, s, target, beta, return_derivatives = True)
                    ###############################dS COMPUTATION#####################################
                    for i in range(self.ns):
                        dS[i][t, :, :] = -(1/(beta*s[i].size(0)))*dsdt[i].sum(0).unsqueeze(0)
                    ######################################################################################


                    #############DT COMPUTATION##################
                    gradw, _ = self.computeGradients(beta, data, s, seq)
                    for i in range(len(gradw)):
                        if gradw[i] is not None:
                            DT[i][t, :, :] = - gradw[i]
                    #####################################################
                                                       
                                                                             
        return s, dS, DT
        
        
    def initHidden(self, batch_size):
        s = []
        for i in range(self.ns):
            s.append(torch.zeros(batch_size, self.size_tab[i], requires_grad = True))            
        return s
        
              
    def computeGradients(self, beta, data, s, seq):
        gradw = []
        gradw_bias = []
        batch_size = s[0].size(0)

        gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(s[0] - seq[0], 0, 1), rho(seq[0])))
        gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(s[0] - seq[0], 0, 1), rho(seq[1])))
        gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(s[0] - seq[0], 0, 1), rho(data)))
        gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(s[1] - seq[1], 0, 1), rho(seq[1])))
        gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(s[1] - seq[1], 0, 1), rho(data)))
        gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(s[1] - seq[1], 0, 1), rho(seq[0])))


        return  gradw, gradw_bias

  
    def updateWeights(self, beta, data, s, seq):
        gradw, gradw_bias = self.computeGradients(beta, data, s, seq)
        lr_tab = self.lr_tab
        for i in range(len(self.w)):
            if self.w[i] is not None:
                self.w[i].weight += lr_tab[int(np.floor(i/2))]*gradw[i]
            if gradw_bias[i] is not None:
                self.w[i].bias += lr_tab[int(np.floor(i/2))]*gradw_bias[i]     


#*****************************toy model, VF, discrete *********************************#

class toyVFdisc(nn.Module):
    def __init__(self, device_label, size_tab, lr_tab, T, Kmax, beta, dt = 1, weight_initialization = 'tied'):
        super(toyVFdisc, self).__init__()
        self.T = T
        self.Kmax = Kmax
        self.dt = dt
        self.size_tab = size_tab
        self.lr_tab = lr_tab
        self.ns = len(size_tab) - 1
        self.nsyn = 2*(self.ns - 1) + 1
        if device_label >= 0:    
            device = torch.device("cuda:"+str(device_label)+")")
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False   
        self.device = device
        self.beta = beta

        w = nn.ModuleList([])
 
        #fully connected architecture
        w.append(nn.Linear(size_tab[0], size_tab[0], bias = False))
        w.append(nn.Linear(size_tab[1], size_tab[0], bias = False))
        w.append(nn.Linear(size_tab[2], size_tab[0], bias = False))
        w.append(nn.Linear(size_tab[1], size_tab[1], bias = False))          
        w.append(nn.Linear(size_tab[2], size_tab[1], bias = False))
        w.append(nn.Linear(size_tab[0], size_tab[1], bias = False))

        #no self connection
        '''
        with torch.no_grad():
            w[0].weight[np.arange(size_tab[0]), np.arange(size_tab[0])] = 0
            w[3].weight[np.arange(size_tab[1]), np.arange(size_tab[1])] = 0 
        '''                      

        w[0].weight.data = 10**(-3)*w[0].weight.data.clone()
        w[1].weight.data = 10**(-3)*w[1].weight.data.clone()
        w[2].weight.data = 10**(-3)*w[2].weight.data.clone() 
        w[3].weight.data = 10**(-3)*w[3].weight.data.clone()
        w[4].weight.data = 10**(-3)*w[4].weight.data.clone()
        w[5].weight.data = 10**(-3)*w[5].weight.data.clone()                 
        
        
        if weight_initialization == 'tied':
            w[0].weight.data = 0.5*(w[0].weight.data.clone() + torch.transpose(w[0].weight.data.clone(), 0, 1))
            w[3].weight.data = 0.5*(w[3].weight.data.clone() + torch.transpose(w[3].weight.data.clone(), 0, 1))                     
            w[1].weight.data = torch.transpose(w[5].weight.data.clone(), 0, 1)              

             
        self.w = w
        self = self.to(device)

    def stepper(self, data, s, target = None, beta = 0, return_derivatives = False):

        dsdt = []
        dsdt.append(-s[0] + rho(self.w[0](s[0]) + self.w[1](s[1]) + self.w[2](data)))
        if beta > 0:
            dsdt[0] = dsdt[0] + beta*(target-s[0])

        dsdt.append(-s[1] + rho(self.w[3](s[1]) + self.w[4](data) + self.w[5](s[0])))

        for i in range(self.ns):
            s[i] = s[i] + self.dt*dsdt[i]
       
        if return_derivatives:
           return s, dsdt
        else:
            return s
    
    def forward(self, data, s, seq = None, method = 'nograd',  beta = 0, target = None, **kwargs):
        T = self.T
        Kmax = self.Kmax
        if len(kwargs) > 0:
            K = kwargs['K']
        else:
            K = Kmax
        if (method == 'withgrad'):
            for t in range(T):             
                if t == T - 1 - K:
                    for i in range(self.ns):
                        s[i] = s[i].detach()
                        s[i].requires_grad = True    
                    data = data.detach()
                    data.requires_grad = True             
                s = self.stepper(data, s)
            return s
                
        elif (method == 'nograd'):
            if beta == 0:
                for t in range(T):                      
                    s = self.stepper(data, s)
            else:
                for t in range(Kmax):                      
                    s = self.stepper(data, s, target, beta)
            return s                   
                    
        elif (method == 'nS'):
            s_tab = []
            for i in range(self.ns):
                s_tab.append([])
            
            criterion = nn.MSELoss(reduction = 'sum')
            for t in range(T):
                for i in range(self.ns):                 
                    s_tab[i].append(s[i])                    
                    s_tab[i][t].retain_grad()                      
                s = self.stepper(data, s)

            for i in range(self.ns):                 
                s_tab[i].append(s[i])                    
                s_tab[i][-1].retain_grad()                
            loss = (1/(2.0*s[0].size(0)))*criterion(s[0], target)
            loss.backward()
            
            
            nS = []
            for i in range(self.ns):
                nS.append(torch.zeros(Kmax, 1, s[i].size(1), device = self.device))
            
            for t in range(Kmax):
                ###############################nS COMPUTATION#####################################
                for i in range(self.ns):
                     if ((i > 0) & (t == 0)):
                        nS[i][t, :, :] = torch.zeros_like(nS[i][t, :, :])
                     else:      
                        nS[i][t, :, :] = (s_tab[i][T - t].grad).sum(0).unsqueeze(0)                
                ####################################################################################
            return s, nS     
            
        elif (method == 'dSDT'):

                DT = []

                for i in range(len(self.w)):
                    if self.w[i] is not None:
                        DT.append(torch.zeros(Kmax, self.w[i].weight.size(0), self.w[i].weight.size(1)))
                    else:
                        DT.append(None)        
                

                dS = []
                for i in range(self.ns):
                    dS.append(torch.zeros(Kmax, 1, self.size_tab[i], device = self.device))              
                
                    
                for t in range(Kmax):
                    s, dsdt = self.stepper(data, s, target, beta, return_derivatives = True)
                    ###############################dS COMPUTATION#####################################
                    for i in range(self.ns):
                        dS[i][t, :, :] = -(1/(beta*s[i].size(0)))*dsdt[i].sum(0).unsqueeze(0)
                    ######################################################################################


                    #############DT COMPUTATION##################
                    gradw, _ = self.computeGradients(beta, data, s, seq)
                    for i in range(len(gradw)):
                        if gradw[i] is not None:
                            DT[i][t, :, :] = - gradw[i]
                    #####################################################
                                                       
                                                                             
        return s, dS, DT
        
        
    def initHidden(self, batch_size):
        s = []
        for i in range(self.ns):
            s.append(torch.zeros(batch_size, self.size_tab[i], requires_grad = True))            
        return s
        
              
    def computeGradients(self, beta, data, s, seq):
        gradw = []
        gradw_bias = []
        batch_size = s[0].size(0)

        gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(torch.mul(1 - seq[0]**2, s[0] - seq[0]), 0, 1), seq[0]))
        gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(torch.mul(1 - seq[0]**2, s[0] - seq[0]), 0, 1), seq[1]))
        gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(torch.mul(1 - seq[0]**2, s[0] - seq[0]), 0, 1), data))
        gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(torch.mul(1 - seq[1]**2, s[1] - seq[1]), 0, 1), seq[1]))
        gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(torch.mul(1 - seq[1]**2, s[1] - seq[1]), 0, 1), data))
        gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(torch.mul(1 - seq[1]**2, s[1] - seq[1]), 0, 1), seq[0]))                    
                            
        return  gradw, gradw_bias

  
    def updateWeights(self, beta, data, s, seq):
        gradw, gradw_bias = self.computeGradients(beta, data, s, seq)
        lr_tab = self.lr_tab
        for i in range(len(self.w)):
            if self.w[i] is not None:
                self.w[i].weight += lr_tab[int(np.floor(i/2))]*gradw[i]
            if gradw_bias[i] is not None:
                self.w[i].bias += lr_tab[int(np.floor(i/2))]*gradw_bias[i]

