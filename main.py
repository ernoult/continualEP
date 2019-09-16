import argparse
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import pickle
import datetime

from netClasses import *
from netFunctions import * 
from plotFunctions import *


#***************C- EP VERSION***************#

# Training settings
parser = argparse.ArgumentParser(description='VF & C-EP')
parser.add_argument(
    '--batch-size',
    type=int,
    default=20,
    metavar='N',
    help='input batch size for training (default: 20)')
parser.add_argument(
    '--test-batch-size',
    type=int,
    default=1000,
    metavar='N',
    help='input batch size for testing (default: 1000)') 
parser.add_argument(
    '--epochs',
    type=int,
    default=1,
    metavar='N',
help='number of epochs to train (default: 1)')    
parser.add_argument(
    '--lr_tab',
    nargs = '+',
    type=float,
    default=[0.05, 0.1],
    metavar='LR',
    help='learning rate (default: [0.05, 0.1])')
parser.add_argument(
    '--size_tab',
    nargs = '+',
    type=int,
    default=[10],
    metavar='ST',
    help='tab of layer sizes (default: [10])')    
parser.add_argument(
    '--dt',
    type=float,
    default=0.2,
    metavar='DT',
    help='time discretization (default: 0.2)') 
parser.add_argument(
    '--T',
    type=int,
    default=100,
    metavar='T',
    help='number of time steps in the forward pass (default: 100)')
parser.add_argument(
    '--Kmax',
    type=int,
    default=25,
    metavar='Kmax',
    help='number of time steps in the backward pass (default: 25)')
parser.add_argument(
    '--beta',
    type=float,
    default=1,
    metavar='BETA',
    help='nudging parameter (default: 1)') 
parser.add_argument(
    '--weight-initialization',
    type=str,
    default='tied',
    metavar='WINIT',
    help='weight initialization (default: tied)')
parser.add_argument(
    '--action',
    type=str,
    default='train',
    help='action to execute (default: train)')    
parser.add_argument(
    '--activation-function',
    type=str,
    default='sigm',
    metavar='ACTFUN',
    help='activation function (default: sigmoid)')
parser.add_argument(
    '--no-clamp',
    action='store_true',
    default=False,
    help='clamp neurons between 0 and 1 (default: True)')
parser.add_argument(
    '--discrete',
    action='store_true',
    default=False, 
    help='discrete-time dynamics (default: False)')
parser.add_argument(
    '--toymodel',
    action='store_true',
    default=False, 
    help='Implement fully connected toy model (default: False)')                                                    
parser.add_argument(
    '--device-label',
    type=int,
    default=0,
    help='selects cuda device (default 0, -1 to select )')

#***********************************************#
parser.add_argument(
    '--benchmark',
    action='store_true',
    default=False, 
    help='benchmark wrt BPTT (default: False)')
parser.add_argument(
    '--learning-rule',
    type=str,
    default='ep',
    metavar='LR',
    help='learning rule (ep/vf, default: ep)')
parser.add_argument(
    '--cep',
    action='store_true',
    default=False, 
    help='continual ep/vf (default: False)')
#***********************************************#

#******************debug C-EP******************#
parser.add_argument(
    '--debug-cep',
    action='store_true',
    default=False, 
    help='debug cep (default: False)')
#**********************************************#

#***************FIX SEED***************#
parser.add_argument(
    '--seed',
    nargs = '+',
    type=int,
    default=[],
    metavar='SEED',
    help='seed (default: None')
#**************************************#

#**************************FORMER-VF-LR**************************#
parser.add_argument(
    '--former',
    action='store_true',
    default=False, 
    help='uses former version of VF learning rule (default: False)')
#****************************************************************#


#*************************SPARSITY*************************#
parser.add_argument(
    '--sparsity',
    type=float,
    default=0,
    help='weight sparsity (defaut: 0)')
#**********************************************************#

#*************************COMPRESSION*************************#
parser.add_argument(
    '--compression',
    type=float,
    default=1,
    help='compression (defaut: 1)')
#**********************************************************#
#******************debug C-EP******************#
parser.add_argument(
    '--debug',
    action='store_true',
    default=False, 
    help='debug (default: False)')
#**********************************************#

#**********************rand beta**********************#
parser.add_argument(
    '--randbeta',
    type=float,
    default=0,
    help='probability of switching beta (defaut: 0)')
#*****************************************************#

#**********************initial angle**********************#
parser.add_argument(
    '--angle',
    type=float,
    default=0,
    help='initial angle between forward and backward weights(defaut: 0)')
#*********************************************************#


args = parser.parse_args()


if not not args.seed:
    torch.manual_seed(args.seed[0])	


batch_size = args.batch_size
batch_size_test = args.test_batch_size

class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)
        
        
class ReshapeTransformTarget:
    def __init__(self, number_classes):
        self.number_classes = number_classes
    
    def __call__(self, target):
        target=torch.tensor(target).unsqueeze(0).unsqueeze(1)
        target_onehot=torch.zeros((1,self.number_classes))    
        return target_onehot.scatter_(1, target, 1).squeeze(0)

        

mnist_transforms=[torchvision.transforms.ToTensor(),ReshapeTransform((-1,))]

train_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST(root='./data', train=True, download=True,
                     transform=torchvision.transforms.Compose(mnist_transforms),
                     target_transform=ReshapeTransformTarget(10)),
batch_size = args.batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST(root='./data', train=False, download=True,
                     transform=torchvision.transforms.Compose(mnist_transforms),
                     target_transform=ReshapeTransformTarget(10)),
batch_size = args.test_batch_size, shuffle=True)


if  args.activation_function == 'sigm':
    def rho(x):
        return 1/(1+torch.exp(-(4*(x-0.5))))
    def rhop(x):
        return 4*torch.mul(rho(x), 1 -rho(x))
    def rhop2(x):
        return 4*torch.mul(x, 1 - x)

elif args.activation_function == 'hardsigm':
    def rho(x):
        return x.clamp(min = 0).clamp(max = 1)

    def rhop(x):
        return ((x >= 0) & (x <= 1)).float()

elif args.activation_function == 'tanh':
    def rho(x):
        return torch.tanh(x)
    def rhop(x):
        return 1 - torch.tanh(x)**2
    def rhop2(x):
        return 1 - x**2   
     
                    
if __name__ == '__main__':
    
    input_size = 28

    #Build the net 
    if (not args.toymodel) & (not args.discrete) & (args.learning_rule == 'vf') :

        net = VFcont(args)

        if args.benchmark:
            net_bptt = VFcont(args)

            net_bptt.load_state_dict(net.state_dict())

    if (not args.toymodel) & (not args.discrete) & (args.learning_rule == 'ep') :

        net = EPcont(args)

        if args.benchmark:
            net_bptt = EPcont(args)

            net_bptt.load_state_dict(net.state_dict())


    elif (not args.toymodel) & (args.discrete) & (args.learning_rule == 'vf'):

        net = VFdisc(args)        

        if args.benchmark:
            net_bptt = VFdisc(args)

            net_bptt.load_state_dict(net.state_dict())

    elif (not args.toymodel) & (args.discrete) & (args.learning_rule == 'ep'):

        net = EPdisc(args)

        if args.benchmark:
            net_bptt = EPdisc(args)
            net_bptt.load_state_dict(net.state_dict())         


    elif (args.toymodel) & (not args.discrete):
        net = toyVFcont(args)

    elif (args.toymodel) & (args.discrete):

        net = toyVFdisc(args)   
                                  

    if args.action == 'plotcurves':
        if (not args.toymodel):
            batch_idx, (example_data, example_targets) = next(enumerate(train_loader))    
        else:
            example_data = torch.rand((args.batch_size, net.size_tab[-1]))
            example_targets = torch.zeros((args.batch_size, net.size_tab[0]))
            example_targets[np.arange(args.batch_size), np.random.randint(net.size_tab[0], size = (1,))] = 1
                   
        if net.cuda: 
            example_data, example_targets = example_data.to(net.device), example_targets.to(net.device)    
	    
        x = example_data
        target = example_targets 
                    
        nS, dS, dT, _ = compute_nSdSdT(net, x, target)
        plot_S(nS, dS)
        plt.show()
        nT = compute_nT(net, x, target)
	 
        plot_T(nT, dT, args)
        plt.show()
                        		
        #create path              
        BASE_PATH, name = createPath(args)

        #save hyperparameters
        createHyperparameterfile(BASE_PATH, name, args)
        
        results_dict = {'nS' : nS, 'dS' : dS, 'nT': nT, 'dT': dT, 'args': args}
                          
        outfile = open(os.path.join(BASE_PATH, 'results'), 'wb')
        pickle.dump(results_dict, outfile)
        outfile.close()


    if args.action == 'RMSE':

        batch_idx, (example_data, example_targets) = next(enumerate(train_loader))    
                   
        if net.cuda: 
            example_data, example_targets = example_data.to(net.device), example_targets.to(net.device)    
	    
        x = example_data
        target = example_targets 
                    
        nS, dS, dT, _ = compute_nSdSdT(net, x, target)
        nT = compute_nT(net, x, target)
                        		
        #create path              
        BASE_PATH, name = createPath(args)

        #save hyperparameters
        createHyperparameterfile(BASE_PATH, name, args)
        
        #*******WATCH OUT: compute and save *ONLY* RelMSE*******#
        RMSE_S, RMSE_T = compute_RMSE(nS, dS, nT, dT)
        results_dict = {'RMSE_S': RMSE_S , 'RMSE_T': RMSE_T}
                          
        outfile = open(os.path.join(BASE_PATH, 'results'), 'wb')
        pickle.dump(results_dict, outfile)
        outfile.close()


    if args.action == 'cosRMSE':

        batch_idx, (example_data, example_targets) = next(enumerate(train_loader))    
                   
        if net.cuda: 
            example_data, example_targets = example_data.to(net.device), example_targets.to(net.device)    
	    
        x = example_data
        target = example_targets 
                    
        nS, dS, dT, _ = compute_nSdSdT(net, x, target)
        nT = compute_nT(net, x, target)
                        		
        #create path              
        #BASE_PATH, name = createPath(args)

        #save hyperparameters
        #createHyperparameterfile(BASE_PATH, name, args)
        
        #*******WATCH OUT: compute and save *ONLY* RelMSE*******#
        theta_S, theta_T = compute_cosRMSE(nS, dS, nT, dT)
        print(theta_T)
        #results_dict = {'theta_S': theta_S , 'theta_T': theta_T}
                          
        #outfile = open(os.path.join(BASE_PATH, 'results'), 'wb')
        #pickle.dump(results_dict, outfile)
        #outfile.close()
     
                                                            
                                    
    elif args.action == 'train':


        #create path              
        BASE_PATH, name = createPath(args)

        #save hyperparameters
        createHyperparameterfile(BASE_PATH, name, args)

        #benchmark wrt BPTT
        if args.benchmark:
            error_train_bptt_tab = []
            error_test_bptt_tab = []  

            for epoch in range(1, args.epochs + 1):
                error_train_bptt = train(net_bptt, train_loader, epoch, 'BPTT')
                error_test_bptt = evaluate(net_bptt, test_loader)
                error_train_bptt_tab.append(error_train_bptt)
                error_test_bptt_tab.append(error_test_bptt)  
                results_dict = {'error_train_bptt_tab' : error_train_bptt_tab, 'error_test_bptt_tab' : error_test_bptt_tab}
                  
                outfile = open(os.path.join(BASE_PATH, 'results'), 'wb')
                pickle.dump(results_dict, outfile)
                outfile.close()  
        
            results_dict_bptt = results_dict
        
        #train with EP
        error_train_tab = []
        error_test_tab = []  

        if args.debug:
            dicts_syn = { 'sign':[], 'zero':[], 'mean_w': [], 'std_w': [],
                        'mean_bias': [], 'std_bias': [],
                        'align_1': [], 'align_2': []}

            dicts_neu = {'satmin':[], 'satmax': []}

            hyperdict_syn = []
            for _ in range(len(net.w)):
                hyperdict_syn.append(copy.deepcopy(dicts_syn))

            hyperdict_neu = []
            for _ in range(net.ns):
                hyperdict_neu.append(copy.deepcopy(dicts_neu))

        #******RECORD INITIAL WEIGHT ANGLE******#
        if args.weight_initialization == 'any':
            angle = computeInitialAngle(net)	
            results_angle = {'angle': angle}
        #***************************************#

        #*****MEASURE ELAPSED TIME*****#
        start_time = datetime.datetime.now()
        #******************************#

        for epoch in range(1, args.epochs + 1):
            if not args.debug:
                error_train = train(net, train_loader, epoch, args.learning_rule)
                error_train_tab.append(error_train)
            else:
                error_train, hyperdict_mb = train(net, train_loader, epoch, args.learning_rule)
                error_train_tab.append(error_train)

                for ind, dicts_temp in enumerate(hyperdict_mb[0]):
                    for indkey, (key, value) in enumerate(dicts_temp.items()):
                        hyperdict_neu[ind][key] = np.concatenate((hyperdict_neu[ind][key], value))  

                for ind, dicts_temp in enumerate(hyperdict_mb[1]):
                    for indkey, (key, value) in enumerate(dicts_temp.items()):
                        hyperdict_syn[ind][key] = np.concatenate((hyperdict_syn[ind][key], value))      
                                     
                results_debug = {'hyperdict_neu': hyperdict_neu, 'hyperdict_syn': hyperdict_syn}

            error_test = evaluate(net, test_loader)         
            error_test_tab.append(error_test) ;
            results_dict = {'error_train_tab' : error_train_tab, 'error_test_tab' : error_test_tab,  'elapsed_time': datetime.datetime.now() - start_time}

            #******RECORD INITIAL WEIGHT ANGLE******#
            if args.weight_initialization == 'any':
                results_dict.update(results_angle)    
            #***************************************#

            if args.debug:
                results_dict.update(results_debug)

            if args.benchmark:
                results_dict_bptt.update(results_dict)    
                outfile = open(os.path.join(BASE_PATH, 'results'), 'wb')
                pickle.dump(results_dict_bptt, outfile)
                outfile.close()
            else:   
                outfile = open(os.path.join(BASE_PATH, 'results'), 'wb')
                pickle.dump(results_dict, outfile)
                outfile.close()                  
       

    elif args.action == 'prop':
        prop = receipe(net, train_loader, 20)
        print(prop)
        #create path              
        BASE_PATH, name = createPath(args)

        #save hyperparameters
        createHyperparameterfile(BASE_PATH, name, args)
        
        results_dict = {'prop': prop}
                          
        outfile = open(os.path.join(BASE_PATH, 'results'), 'wb')
        pickle.dump(results_dict, outfile)
        outfile.close()     
