from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.optim as optim
import torch.nn.functional as F


from main import rho, rhop


#*****************************VF, real-time setting *********************************#

class VFcont(nn.Module):

    def __init__(self, args):
        super(VFcont, self).__init__()        
        self.T = args.T
        self.Kmax = args.Kmax        
        self.dt = args.dt
        self.size_tab = args.size_tab
        self.lr_tab = args.lr_tab
        self.ns = len(args.size_tab) - 1
        self.nsyn = 2*(self.ns - 1) + 1
        self.cep = args.cep

        if args.device_label >= 0:    
            device = torch.device("cuda:"+str(args.device_label)+")")
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False   

        self.device = device
        self.no_clamp = args.no_clamp
        self.beta = args.beta

        w = nn.ModuleList([])    
                        
        for i in range(self.ns - 1):
            w.append(nn.Linear(args.size_tab[i + 1], args.size_tab[i], bias = True))
            w.append(nn.Linear(args.size_tab[i], args.size_tab[i + 1], bias = False))
            
        w.append(nn.Linear(args.size_tab[-1], args.size_tab[-2]))
               
        #By default, reciprocal weights have the same initial values
        for i in range(self.ns - 1):
            w[2*i + 1].weight.data = torch.transpose(w[2*i].weight.data.clone(), 0, 1)

        #****************************TUNE INITIAL ANGLE****************************#        
        if args.angle > 0:
            p_switch = 0.5*(1 - np.cos(np.pi*args.angle/180))
            for i in range(self.ns - 1):
                mask = 2*torch.bernoulli((1 - p_switch)*torch.ones_like(w[2*i + 1].weight.data)) - 1
                w[2*i + 1].weight.data = w[2*i + 1].weight.data*mask
                angle = (180/np.pi)*np.arccos((w[2*i + 1].weight.data*torch.transpose(w[2*i].weight.data, 0 ,1)).sum().item()/np.sqrt((w[2*i + 1].weight.data**2).sum().item()*(w[2*i].weight.data**2).sum().item()))
                print('Angle between forward and backward weights: {:.2f} degrees'.format(angle))
                del angle, mask
        #**************************************************************************#
                	
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
       

        s_old = []
        for ind, s_temp in enumerate(s):
            s_old.append(s_temp.clone())

        if self.no_clamp:
            for i in range(self.ns):
                s[i] = s[i] + self.dt*dsdt[i]
        else:
            for i in range(self.ns):
                s[i] = (s[i] + self.dt*dsdt[i]).clamp(min = 0).clamp(max = 1)
                dsdt[i] = torch.where((s[i] == 0)|(s[i] ==1), torch.zeros_like(dsdt[i], device = self.device), dsdt[i])

        #*****************************C-EP*****************************#
        if (beta > 0):
            dw = self.computeGradients(data, s, s_old)
            if self.cep:
                with torch.no_grad(): 
                    self.updateWeights(dw)
                                                 
            if return_derivatives:
                return s, dsdt, dw
            else:
                return s, dw
        else:
            return s 
        #**************************************************************#
    
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

            #*************ADD GRADIENT ACCUMULATION HERE*************#
            if beta == 0:
                for t in range(T):                      
                    s = self.stepper(data, s)
                return s
            else:
                Dw = self.initGrad()                                          
                for t in range(Kmax):
                    s, dw = self.stepper(data, s, target, beta)

                    
                    with torch.no_grad():
                        for ind_type, dw_temp in enumerate(dw):
                            for ind, dw_temp_layer in enumerate(dw_temp):
                                if dw_temp_layer is not None:
                                    Dw[ind_type][ind] += dw_temp_layer

                return s, Dw     
            #********************************************************#              
                    
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
                #**********************************nS COMPUTATION**********************************#
                for i in range(self.ns):
                    if (t < i):
                        nS[i][t, :, :] = torch.zeros_like(nS[i][t, :, :])
                    else:    
                        nS[i][t, :, :] = (s_tab[i][T - t].grad).sum(0).unsqueeze(0)
                #**********************************************************************************#      
        
               
            return s, nS     
            
        elif (method == 'dSdT'):

                DT = []

                for i in range(len(self.w)):
                    if self.w[i] is not None:
                        DT.append(torch.zeros(Kmax, self.w[i].weight.size(0), self.w[i].weight.size(1)))
                    else:
                        DT.append(None)        
                
                dS = []
                for i in range(self.ns):
                    dS.append(torch.zeros(Kmax, 1, self.size_tab[i], device = self.device))              
                
                #*******************************************C-EP*******************************************#
               
                for t in range(Kmax):
                    s, dsdt, dw = self.stepper(data, s, target, beta, return_derivatives = True)
                    #***********************************dS COMPUTATION***********************************#
                    for i in range(self.ns):
                        if (t < i):
                            dS[i][t, :, :] = torch.zeros_like(dS[i][t, :, :])
                        else:
                            dS[i][t, :, :] = -(1/(beta*s[i].size(0)))*dsdt[i].sum(0).unsqueeze(0)                        
                    #************************************************************************************#

                    #*******************dT COMPUTATION******************#
                    for ind, dw_temp in enumerate(dw[0]):
                        if (dw_temp is not None) & (t > 0):
                            DT[ind][t, :, :] = - dw_temp
                    #***************************************************#       

                #******************************************************************************************#             
                                                                                                                                   
        return s, dS, DT
                
    def initHidden(self, batch_size):
        s = []
        for i in range(self.ns):
            s.append(torch.zeros(batch_size, self.size_tab[i], requires_grad = True))            
        return s

    def initGrad(self):
        gradw = []
        gradw_bias =[]
        for ind, w_temp in enumerate(self.w):
            gradw.append(torch.zeros_like(w_temp.weight))
            if w_temp.bias is not None:
                gradw_bias.append(torch.zeros_like(w_temp.bias))
            else:
                gradw_bias.append(None)

        return gradw, gradw_bias

              
    def computeGradients(self, data, s, seq):
        gradw = []
        gradw_bias = []
        batch_size = s[0].size(0)
        beta = self.beta
               
        for i in range(self.ns - 1):                
            gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(s[i] - seq[i], 0, 1), rho(seq[i + 1]))) 
            gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(s[i + 1] - seq[i + 1], 0, 1), rho(seq[i])))                
            gradw_bias.append((1/(beta*batch_size))*(s[i] - seq[i]).sum(0))
            gradw_bias.append(None)                                                                                  
                                                                
        gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(s[-1] - seq[-1], 0, 1), rho(data)))
        gradw_bias.append((1/(beta*batch_size))*(s[-1] - seq[-1]).sum(0))


        return  gradw, gradw_bias

    #**************************NEW**************************# 
    def updateWeights(self, gradw):
        lr_tab = self.lr_tab
        for i in range(len(self.w)):
            if self.w[i] is not None:
                self.w[i].weight += lr_tab[int(np.floor(i/2))]*gradw[0][i]
            if gradw[1][i] is not None:
                self.w[i].bias += lr_tab[int(np.floor(i/2))]*gradw[1][i]
    #*******************************************************#
       


#*****************************VF, discrete-time setting *********************************#

class VFdisc(nn.Module):
    
    def __init__(self, args):
        super(VFdisc, self).__init__()   
        self.T = args.T
        self.Kmax = args.Kmax     
        self.dt = 1
        self.size_tab = args.size_tab
        self.lr_tab = args.lr_tab
        self.ns = len(args.size_tab) - 1
        self.nsyn = 2*(self.ns - 1) + 1
        self.cep = args.cep
        if args.device_label >= 0:    
            device = torch.device("cuda:"+str(args.device_label)+")")
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False   
        self.device = device
        self.beta = args.beta
    	

        #*********RANDOM BETA*********#
        self.randbeta = args.randbeta
        #*****************************#

        w = nn.ModuleList([])                         
        for i in range(self.ns - 1):
            w.append(nn.Linear(args.size_tab[i + 1], args.size_tab[i], bias = True))
            w.append(nn.Linear(args.size_tab[i], args.size_tab[i + 1], bias = False))
            
        w.append(nn.Linear(args.size_tab[-1], args.size_tab[-2]))
        
        #By default, reciprocal weights have the same initial values
        for i in range(self.ns - 1):
            w[2*i + 1].weight.data = torch.transpose(w[2*i].weight.data.clone(), 0, 1)

        #****************************TUNE INITIAL ANGLE****************************#        
        if args.angle > 0:
            p_switch = 0.5*(1 - np.cos(np.pi*args.angle/180))
            for i in range(self.ns - 1):
                mask = 2*torch.bernoulli((1 - p_switch)*torch.ones_like(w[2*i + 1].weight.data)) - 1
                w[2*i + 1].weight.data = w[2*i + 1].weight.data*mask
                angle = (180/np.pi)*np.arccos((w[2*i + 1].weight.data*torch.transpose(w[2*i].weight.data, 0 ,1)).sum().item()/np.sqrt((w[2*i + 1].weight.data**2).sum().item()*(w[2*i].weight.data**2).sum().item()))
                print('Angle between forward and backward weights: {:.2f} degrees'.format(angle))
                del angle, mask
        #**************************************************************************#


        self.w = w
        self = self.to(device)
    
              

    def stepper(self, data, s, target = None, beta = 0, return_derivatives = False, seq = None):
        dsdt = []
        dsdt.append(-s[0] + rho(self.w[0](s[1])))
        if np.abs(beta) > 0:
            dsdt[0] = dsdt[0] + beta*(target-s[0])

        for i in range(1, self.ns - 1):
            dsdt.append(-s[i] + rho(self.w[2*i](s[i + 1]) + self.w[2*i - 1](s[i - 1])))

        dsdt.append(-s[-1] + rho(self.w[-1](data) + self.w[-2](s[-2])))


        if seq is None:	
            s_old = []
            for ind, s_temp in enumerate(s): s_old.append(s_temp.clone())

        else:
            s_old = [i.clone() for i in seq]

        for i in range(self.ns):
            s[i] = s[i] + self.dt*dsdt[i]

        #*****************************C-VF*****************************#
        if (np.abs(beta) > 0):
            dw = self.computeGradients(data, s, s_old, beta)
            if self.cep:
                with torch.no_grad(): 
                    self.updateWeights(dw)
                                                 
            if return_derivatives:
                return s, dsdt, dw
            else:
                return s, dw
        else:
            return s 
        #**************************************************************#
    
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
            #*************ADD GRADIENT ACCUMULATION HERE*************#
            if beta == 0:
                for t in range(T):                      
                    s = self.stepper(data, s)
                return s
            else:
                Dw = self.initGrad()                                  
                for t in range(Kmax):
                    s, dw = self.stepper(data, s, target, beta)
                    with torch.no_grad():
                        for ind_type, dw_temp in enumerate(dw):
                            for ind, dw_temp_layer in enumerate(dw_temp):
                                if dw_temp_layer is not None:
                                    Dw[ind_type][ind] += dw_temp_layer

                return s, Dw     
            #********************************************************#                 
                    
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
                #**********************************nS COMPUTATION**********************************#
                for i in range(self.ns):
                    if (t < i):
                        nS[i][t, :, :] = torch.zeros_like(nS[i][t, :, :])
                    else:    
                        nS[i][t, :, :] = (s_tab[i][T - t].grad).sum(0).unsqueeze(0)
                #**********************************************************************************#               
               
            return s, nS     
            
        elif (method == 'dSdT'):

                DT = []

                for i in range(len(self.w)):
                    if self.w[i] is not None:
                        DT.append(torch.zeros(Kmax, self.w[i].weight.size(0), self.w[i].weight.size(1)))
                    else:
                        DT.append(None)        
                

                dS = []
                for i in range(self.ns):
                    dS.append(torch.zeros(Kmax, 1, self.size_tab[i], device = self.device))              
                
                    
                #*******************************************C-EP*******************************************#
               
                for t in range(Kmax):
                    s, dsdt, dw = self.stepper(data, s, target, beta, return_derivatives = True)
                    #*********************************dS COMPUTATION*************************************#
                    for i in range(self.ns):
                        if (t < i):
                            dS[i][t, :, :] = torch.zeros_like(dS[i][t, :, :])
                        else:
                            dS[i][t, :, :] = -(1/(beta*s[i].size(0)))*dsdt[i].sum(0).unsqueeze(0)                        
                    #************************************************************************************#

                    #*******************dT COMPUTATION******************#
                    for ind, dw_temp in enumerate(dw[0]):
                        if (dw_temp is not None) & (t > 0):
                            DT[ind][t, :, :] = - dw_temp
                    #***************************************************#       

                #******************************************************************************************#   
                                                       
                                                                             
        return s, dS, DT
        
        
    def initHidden(self, batch_size):
        s = []
        for i in range(self.ns):
            s.append(torch.zeros(batch_size, self.size_tab[i], requires_grad = True))            
        return s
    
    def initGrad(self):
        gradw = []
        gradw_bias =[]
        for ind, w_temp in enumerate(self.w):
            gradw.append(torch.zeros_like(w_temp.weight))
            if w_temp.bias is not None:
                gradw_bias.append(torch.zeros_like(w_temp.bias))
            else:
                gradw_bias.append(None)

        return gradw, gradw_bias     
              
    def computeGradients(self, data, s, seq, beta):
        gradw = []
        gradw_bias = []
        batch_size = s[0].size(0)

        for i in range(self.ns - 1):                

            gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(s[i] - seq[i], 0, 1), seq[i + 1]))
            gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(s[i + 1] - seq[i + 1], 0, 1), seq[i]))
            gradw_bias.append((1/(beta*batch_size))*(s[i] - seq[i]).sum(0))
             
            gradw_bias.append(None)                                                                                  
                                                                
        gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(s[-1] - seq[-1], 0, 1), data))
        gradw_bias.append(None)
                                                                                                                                                                       
        return  gradw, gradw_bias

    def updateWeights(self, gradw):
        lr_tab = self.lr_tab
        for i in range(len(self.w)):
            if self.w[i] is not None:
                self.w[i].weight += lr_tab[int(np.floor(i/2))]*gradw[0][i]
            if gradw[1][i] is not None:
                self.w[i].bias += lr_tab[int(np.floor(i/2))]*gradw[1][i]

#*****************************EP, real-time setting *********************************#

class EPcont(nn.Module):
    def __init__(self, args):
        super(EPcont, self).__init__()
        
        self.T = args.T
        self.Kmax = args.Kmax
        self.dt = args.dt
        self.size_tab = args.size_tab
        self.lr_tab = args.lr_tab
        self.ns = len(args.size_tab) - 1
        self.nsyn = 2*(self.ns - 1) + 1
        self.cep = args.cep
        if args.device_label >= 0:    
            device = torch.device("cuda:"+str(args.device_label)+")")
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False    
        self.device = device
        self.no_clamp = args.no_clamp
        self.beta = args.beta
                    
        w = nn.ModuleList([])           
        for i in range(self.ns - 1):

            w.append(nn.Linear(args.size_tab[i + 1], args.size_tab[i], bias = True))
            w.append(None)

        w.append(nn.Linear(args.size_tab[-1], args.size_tab[-2]))                                         
        self.w = w
        self = self.to(device)

    def stepper(self, data, s, target = None, beta = 0, return_derivatives = False):
        dsdt = []
        dsdt.append(-s[0] + self.w[0](rho(s[1])))     
        if beta > 0:
            dsdt[0] = dsdt[0] + beta*(target-s[0])

        for i in range(1, self.ns - 1):
            dsdt.append(-s[i] + torch.mul(rhop(s[i]), self.w[2*i](rho(s[i + 1])) + torch.mm(rho(s[i - 1]), self.w[2*(i-1)].weight)))

        dsdt.append(-s[-1] + torch.mul(rhop(s[-1]), self.w[-1](rho(data)) + torch.mm(rho(s[-2]), self.w[-3].weight)))

        s_old = []
        for ind, s_temp in enumerate(s):
            s_old.append(s_temp.clone())

        if self.no_clamp:
            for i in range(self.ns):
                s[i] = s[i] + self.dt*dsdt[i]
        else:
            for i in range(self.ns):
                s[i] = (s[i] + self.dt*dsdt[i]).clamp(min = 0).clamp(max = 1)
                dsdt[i] = torch.where((s[i] == 0)|(s[i] == 1), torch.zeros_like(dsdt[i], device = self.device), dsdt[i])

        #*****************************C-EP*****************************#
        if (self.cep) & (beta > 0):
            dw = self.computeGradients(data, s, s_old)
            if self.cep:
                with torch.no_grad(): 
                    self.updateWeights(dw)
                                             
        if return_derivatives:
            dw = self.computeGradients(data, s, s_old)
            return s, dsdt, dw
        else:
            return s
        #**************************************************************#
    
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
                for i in range(self.ns):
                #***********************************nS COMPUTATION*********************************#
                    if (t < i):
                        nS[i][t, :, :] = torch.zeros_like(nS[i][t, :, :])
                    else:    
                        nS[i][t, :, :] = (s_tab[i][T - t].grad).sum(0).unsqueeze(0)
                #**********************************************************************************#
          

            return s, nS     
            
        elif (method == 'dSdT'):               
                DT = []
                for i in range(len(self.w)):
                    if self.w[i] is not None:
                        DT.append(torch.zeros(Kmax, self.w[i].weight.size(0), self.w[i].weight.size(1)))
                    else:
                        DT.append(None)                        
                dS = []
                for i in range(self.ns):
                    dS.append(torch.zeros(Kmax, 1, self.size_tab[i], device = self.device))              
                
                    
                #*******************************************C-EP*******************************************#
               
                for t in range(Kmax):
                    s, dsdt, dw = self.stepper(data, s, target, beta, return_derivatives = True)
                    #***********************************dS COMPUTATION***********************************#
                    for i in range(self.ns):
                        if (t < i):
                            dS[i][t, :, :] = torch.zeros_like(dS[i][t, :, :])
                        else:
                            dS[i][t, :, :] = -(1/(beta*s[i].size(0)))*dsdt[i].sum(0).unsqueeze(0)                        
                    #************************************************************************************#

                    #********************dT COMPUTATION*****************#
                    for ind, dw_temp in enumerate(dw[0]):
                        if (dw_temp is not None) & (t > 0):
                            DT[ind][t, :, :] = - dw_temp
                    #***************************************************#       

                #******************************************************************************************#
                                                       
                                                                             
        return s, dS, DT
        
        
    def initHidden(self, batch_size):
        s = []
        for i in range(self.ns):
            s.append(torch.zeros(batch_size, self.size_tab[i], requires_grad = True))            
        return s
        
              
    def computeGradients(self, data, s, seq):
        gradw = []
        gradw_bias = []
        batch_size = s[0].size(0)
        beta = self.beta
             
        for i in range(self.ns - 1):
            gradw.append((1/(beta*batch_size))*(torch.mm(torch.transpose(rho(s[i]), 0, 1), rho(s[i + 1])) - torch.mm(torch.transpose(rho(seq[i]), 0, 1), rho(seq[i + 1])))) 
            gradw.append(None)            
            gradw_bias.append((1/(beta*batch_size))*(rho(s[i]) - rho(seq[i])).sum(0))
            gradw_bias.append(None)                                                                                  
                                                                
        gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(rho(s[-1]) - rho(seq[-1]), 0, 1), rho(data)))
        gradw_bias.append((1/(beta*batch_size))*(rho(s[-1]) - rho(seq[-1])).sum(0))
               
        return  gradw, gradw_bias

    def updateWeights(self, gradw):

        lr_tab = self.lr_tab
        for i in range(len(self.w)):
            if self.w[i] is not None:
                self.w[i].weight += lr_tab[int(np.floor(i/2))]*gradw[0][i]
            if gradw[1][i] is not None:
                self.w[i].bias += lr_tab[int(np.floor(i/2))]*gradw[1][i]

#*****************************EP, prototypical *********************************#

class EPdisc(nn.Module):
    def __init__(self, args):
        super(EPdisc, self).__init__()

                
        self.T = args.T
        self.Kmax = args.Kmax        
        self.dt = 1
        self.size_tab = args.size_tab
        self.lr_tab = args.lr_tab
        self.ns = len(args.size_tab) - 1
        self.nsyn = 2*(self.ns - 1) + 1
        self.cep = args.cep
        if args.device_label >= 0:    
            device = torch.device("cuda:"+str(args.device_label)+")")
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False   
        self.device = device
        self.beta = args.beta

        #**************debug_cep C-EP**************#
        self.debug_cep = args.debug_cep
        if args.debug_cep:
            lr_tab_debug = []
            for lr in self.lr_tab:
                lr_tab_debug.append(10**(-5)*lr)
            self.lr_tab_debug = lr_tab_debug
        #**************************************#

        
        #*********RANDOM BETA*********#
        self.randbeta = args.randbeta
        #*****************************#

        w = nn.ModuleList([])
                           
        for i in range(self.ns - 1):
            w.append(nn.Linear(args.size_tab[i + 1], args.size_tab[i], bias = True))
            w.append(None)
            
        w.append(nn.Linear(args.size_tab[-1], args.size_tab[-2]))                             
        self.w = w
        self = self.to(device)

    def stepper(self, data, s, target = None, beta = 0, return_derivatives = False):
        dsdt = []
        dsdt.append(-s[0] + rho(self.w[0](s[1])))
        if np.abs(beta) > 0:
            dsdt[0] = dsdt[0] + beta*(target-s[0])

        for i in range(1, self.ns - 1):
            dsdt.append(-s[i] + rho(self.w[2*i](s[i + 1]) + torch.mm(s[i - 1], self.w[2*(i-1)].weight)))

        dsdt.append(-s[-1] + rho(self.w[-1](data) + torch.mm(s[-2], self.w[-3].weight)))

        s_old = []
        for ind, s_temp in enumerate(s):
            s_old.append(s_temp.clone())

        for i in range(self.ns):
            s[i] = s[i] + self.dt*dsdt[i]

        #*****************************C-EP*****************************#
        if (self.cep) & (np.abs(beta) > 0):
            dw = self.computeGradients(data, s, s_old, beta)
            if (self.cep) & (not self.debug_cep):
                with torch.no_grad(): 
                    self.updateWeights(dw)

            elif (self.cep) & (self.debug_cep):
                with torch.no_grad(): 
                    self.updateWeights(dw, debug_cep = True)  
                         
        if return_derivatives:
            dw = self.computeGradients(data, s, s_old, self.beta)
            return s, dsdt, dw
        else:
            return s
        #**************************************************************#
    
    def forward(self, data, s, seq = None, method = 'nograd', beta = 0, target = None, **kwargs):
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
                return s

            elif (np.abs(beta) > 0) & (not self.debug_cep):
                for t in range(Kmax):                      
                    s = self.stepper(data, s, target, beta)
                return s
             
            elif (np.abs(beta) > 0) & (self.debug_cep):
                Dw = self.initGrad()                                          
                for t in range(Kmax):
                    s, _, dw = self.stepper(data, s, target, beta, return_derivatives = True)
                    
                    with torch.no_grad():
                        for ind_type, dw_temp in enumerate(dw):
                            for ind, dw_temp_layer in enumerate(dw_temp):
                                if dw_temp_layer is not None:
                                    Dw[ind_type][ind] += dw_temp_layer

                return s, Dw                     
                    
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
                for i in range(self.ns):
                    #**********************************nS COMPUTATION**********************************#
                    if (t < i):
                        nS[i][t, :, :] = torch.zeros_like(nS[i][t, :, :])
                    else:    
                        nS[i][t, :, :] = (s_tab[i][T - t].grad).sum(0).unsqueeze(0)
                    #**********************************************************************************#

                                      
               
            return s, nS     
            
        elif (method == 'dSdT'):           
            DT = []

            for i in range(len(self.w)):
                if self.w[i] is not None:
                    DT.append(torch.zeros(Kmax, self.w[i].weight.size(0), self.w[i].weight.size(1)))
                else:
                    DT.append(None)        


            dS = []
            for i in range(self.ns):
                dS.append(torch.zeros(Kmax, 1, self.size_tab[i], device = self.device))              

                
            #*******************************************C-EP*******************************************#

            for t in range(Kmax):
                s, dsdt, dw = self.stepper(data, s, target, beta, return_derivatives = True)
                #***********************************dS COMPUTATION***********************************#
                for i in range(self.ns):
                    if (t < i):
                        dS[i][t, :, :] = torch.zeros_like(dS[i][t, :, :])
                    else:
                        dS[i][t, :, :] = -(1/(beta*s[i].size(0)))*dsdt[i].sum(0).unsqueeze(0)                        
                ##***********************************************************************************#

                #********************dT COMPUTATION*****************#
                for ind, dw_temp in enumerate(dw[0]):
                    if (dw_temp is not None) & (t > 0):
                        DT[ind][t, :, :] = - dw_temp
                #***************************************************#       

            #******************************************************************************************#
                                                   
                                                                         
            return s, dS, DT
        
        
    def initHidden(self, batch_size):
        s = []
        for i in range(self.ns):
            s.append(torch.zeros(batch_size, self.size_tab[i], requires_grad = True))            
        return s

    def initGrad(self):
        gradw = []
        gradw_bias =[]
        for w_temp in self.w:
            if w_temp is not None:
                gradw.append(torch.zeros_like(w_temp.weight))
                gradw_bias.append(torch.zeros_like(w_temp.bias))
            else:
                gradw.append(None)
                gradw_bias.append(None)

        return gradw, gradw_bias
              
    def computeGradients(self, data, s, seq, beta):
        gradw = []
        gradw_bias = []
        batch_size = s[0].size(0)

                
        for i in range(self.ns - 1):
            gradw.append((1/(beta*batch_size))*(torch.mm(torch.transpose(s[i], 0, 1), s[i + 1]) - torch.mm(torch.transpose(seq[i], 0, 1), seq[i + 1]))) 
            gradw.append(None)            
            gradw_bias.append((1/(beta*batch_size))*(s[i] - seq[i]).sum(0))
            gradw_bias.append(None)                                                                                  
                                                                
        gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(s[-1] - seq[-1], 0, 1), data))
        gradw_bias.append((1/(beta*batch_size))*(s[-1] - seq[-1]).sum(0))
                                                                                                                                                                
        return  gradw, gradw_bias

    def updateWeights(self, gradw, debug_cep = False):
        if not debug_cep:
            lr_tab = self.lr_tab
        else:
            lr_tab = self.lr_tab_debug

        for i in range(len(self.w)):
            if self.w[i] is not None:
                self.w[i].weight += lr_tab[int(np.floor(i/2))]*gradw[0][i]
            if gradw[1][i] is not None:
                self.w[i].bias += lr_tab[int(np.floor(i/2))]*gradw[1][i]



