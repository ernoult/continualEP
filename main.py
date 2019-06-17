import argparse
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import pickle

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
    '--training-method',
    type=str,
    default='eqprop',
    metavar='TMETHOD',
    help='training method (default: eqprop)')
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

args = parser.parse_args()

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

elif args.activation_function == 'tanh':
    def rho(x):
        return torch.tanh(x)
    def rhop(x):
        return 1 - torch.tanh(x)**2
    def rhop2(x):
        return 1 - x**2   

elif args.activation_function == 'relu':
    def rho(x):
        return x.clamp(min = 0)
    def rhop(x):
        return x > 0
    def rhop2(x):
        return 0
     
                    
if __name__ == '__main__':
    
    input_size = 28

    #Build the net 
    if (not args.toymodel) & (not args.discrete) & (args.learning_rule == 'vf') :

        net = VFcont(args.device_label, args.size_tab, args.lr_tab, 
                    args.T, args.Kmax, args.beta, 
                    dt = args.dt, no_clamp = args.no_clamp, 
                    weight_initialization = args.weight_initialization,
                    cep = args.cep)

        if args.benchmark:
            net_bptt = VFcont(args.device_label, args.size_tab, args.lr_tab, 
                        args.T, args.Kmax, args.beta, 
                        dt = args.dt, no_clamp = args.no_clamp, 
                        weight_initialization = args.weight_initialization)

            net_bptt.load_state_dict(net.state_dict())

    if (not args.toymodel) & (not args.discrete) & (args.learning_rule == 'ep') :

        net = EPcont(args.device_label, args.size_tab, args.lr_tab, 
                    args.T, args.Kmax, args.beta, 
                    dt = args.dt, no_clamp = args.no_clamp,
                    cep = args.cep)

        if args.benchmark:
            net_bptt = EPcont(args.device_label, args.size_tab, args.lr_tab, 
                        args.T, args.Kmax, args.beta, 
                        dt = args.dt, no_clamp = args.no_clamp)

            net_bptt.load_state_dict(net.state_dict())


    elif (not args.toymodel) & (args.discrete) & (args.learning_rule == 'vf'):

        net = VFdisc(args.device_label, args.size_tab, args.lr_tab, 
                    args.T, args.Kmax, args.beta, 
                    weight_initialization = args.weight_initialization,
                    cep = args.cep)

        if args.benchmark:
            net_bptt = VFdisc(args.device_label, args.size_tab, args.lr_tab, 
                        args.T, args.Kmax, args.beta, 
                        weight_initialization = args.weight_initialization)

            net_bptt.load_state_dict(net.state_dict())

    elif (not args.toymodel) & (args.discrete) & (args.learning_rule == 'ep'):

        net = EPdisc(args.device_label, args.size_tab, args.lr_tab, 
			        args.T, args.Kmax, args.beta, cep = args.cep)

        if args.benchmark:
            net_bptt = EPdisc(args.device_label, args.size_tab, args.lr_tab, 
			            args.T, args.Kmax, args.beta)

            net_bptt.load_state_dict(net.state_dict())         


    elif (args.toymodel) & (not args.discrete):
        net = toyVFcont(args.device_label, args.size_tab, args.lr_tab, 
                        args.T, args.Kmax, args.beta, 
                        dt = args.dt, no_clamp = args.no_clamp,
                        weight_initialization = args.weight_initialization,
                        cep = args.cep)

    elif (args.toymodel) & (args.discrete):

        net = toyVFdisc(args.device_label, args.size_tab, args.lr_tab, 
                        args.T, args.Kmax, args.beta, 
                        weight_initialization = args.weight_initialization,
                        cep = args.cep)   
                                  

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
        
            
        nS, dS, DT, _ = compute_nSdSDT(net, x, target)
        plot_S(nS, dS)
        plt.show()
        NT = compute_NT(net, x, target)
        #instNT, instDT = computeInstantaneousTheta(NT, DT)
        plot_T(NT, DT, args)
        plt.show()
        
        #create path              
        BASE_PATH, name = createPath(args)

        #save hyperparameters
        createHyperparameterfile(BASE_PATH, name, args)
        
        results_dict = {'nS' : nS, 'dS' : dS, 'NT': NT, 'DT': DT, 'toymodel': args.toymodel}
                          
        outfile = open(os.path.join(BASE_PATH, 'results'), 'wb')
        pickle.dump(results_dict, outfile)
        outfile.close()     
                                              
                                    
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

        for epoch in range(1, args.epochs + 1):
            error_train = train(net, train_loader, epoch, args.training_method)
            error_test = evaluate(net, test_loader)
            error_train_tab.append(error_train)
            error_test_tab.append(error_test) ;
            results_dict = {'error_train_tab' : error_train_tab, 'error_test_tab' : error_test_tab}  
            if args.benchmark:
                results_dict_bptt.update(results_dict)    
                outfile = open(os.path.join(BASE_PATH, 'results'), 'wb')
                pickle.dump(results_dict_bptt, outfile)
                outfile.close()
            else:   
                outfile = open(os.path.join(BASE_PATH, 'results'), 'wb')
                pickle.dump(results_dict, outfile)
                outfile.close()                  
       

    elif args.action == 'receipe':
        receipe(net, train_loader, args.N_trials)        
