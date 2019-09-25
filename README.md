*******************************************************

# "Equilibrium Propagation with Continual Weight Updates" 

*******************************************************

The following document provides details about the code provided, alongside the commands to be run to reproduce
the results appearing in the draft. 


I - Package requirements

* Our code is compatible with Python 2.7 or 3.

* Our virtual environment contains the following packages (after executing a pip freeze command):

    #Basic packages:
    absl-py==0.7.1
    astor==0.7.1
    backports.functools-lru-cache==1.5
    backports.weakref==1.0.post1
    cycler==0.10.0
    enum34==1.1.6
    funcsigs==1.0.2
    futures==3.2.0
    gast==0.2.2
    grpcio==1.19.0
    h5py==2.9.0
    kiwisolver==1.0.1
    Markdown==3.1
    matplotlib==2.2.4
    mock==2.0.0
    numpy==1.16.2
    pbr==5.1.3
    Pillow==6.0.0
    protobuf==3.7.1
    pyparsing==2.3.1
    python-dateutil==2.8.0
    pytz==2018.9
    PyYAML==5.1
    scipy==1.2.1
    six==1.12.0
    subprocess32==3.5.3
    termcolor==1.1.0
    Werkzeug==0.15.2

    #Relevant packages for our project:
    Keras==2.2.4
    Keras-Applications==1.0.6
    Keras-Preprocessing==1.0.5
    torch==1.0.1.post2
    torchvision==0.2.2.post3
    tensorboard==1.12.2
    tensorflow==1.12.0

* To create an environment to run our code:

  i) Install Python 2.7 or 3.
  ii) Install pip.
  iii) Run pip install virtualenv.
  ii) Run mkdir myproject.
  iii) Run cd myproject.
  iv) Run virtualenv myenv.
  v) Create a requirements.txt file containing the package requirements of the previous bullet.
  vi) source myenv/bin/activate.
  vii) Run pip install -r requirements.txt.

II - Files

* The project contains the following files:

  i) main.py: executes the code, with arguments specified in a parser.

  ii) netClasses.py: contains the network classes.

  iii) netFunctions: contains the functions to run on the networks.

  iv) plotFunctions: contains the functions to plot the results. 

III - Details about main.py

* main.py proceeds in the following way:

  i) It first parses arguments typed in the terminal to build a network and get optimization parameters
  (i.e. learning rates, mini-batch size, network topology, etc.)

  ii) It loads the MNIST data set with torchvision.

  iii) It builds the nets using netClasses.py.

  iv) It takes one of the three actions that can be fed into the parser.


* The parser takes the following arguments:

  i) Optimization arguments:  

    --batch-size: training batch size.

    --test-batch-size: test batch size used to compute the test error.

    --epochs: number of epochs.

    --lr_tab: learning rates tab, to be provided from the output layer towards the first layer.
              Example: --lr_tab 0.01 0.04 will apply a learning rate of 0.01 to W_{01} and 0.04 to W_{12}.

    --randbeta: probability that the sign of beta switches accross mini-batches (see Appendix F.1 for details). Default: 0. 	


  ii) Network arguments: 

    --size_tab: specify the topology of the network, backward from the output layer.
                Example: --size_tab 10 512 784.
                It is also used alongside --C_tab to define the fully connected part of a
                convolutional architecture -- see below. 

    --discrete: specifies if we are in the discrete-time (discrete = True) or real-time (discrete = False) setting. 

    --dt: time increment in the real-time setting (denoted \epsilon in the draft). 

    --T: number of steps in the first phase. 

    --Kmax: number of steps in the second phase. 

    --beta: nudging parameter. 
 
    --activation-function: selects the activation function used: either 'tanh', 'sigm' (for sigmoid) or 'hardsigm' (for hard-sigmoid).

    --no-clamp: specifies whether we clamp the updates (no-clamp = False) or not when training in the real-time setting.

    --learning-rule: specifies the learning rule used: either 'ep' (Equilibrium Propagation) or 'vf' (Vector Field Equilibrium Propagation).

    --c-ep: specifies whether the updates are continual (True) or not (False). Default: False. 

    --angle: specify the weight angle between forward and backward weights in degrees (default: 0 degree). See Appendix F.1 for the angle definition.
                            

  iii) Others:

    --action: specifies the action to take in main.py (see next bullet).

    --device-label: selects the cuda device to run the simulation on. 

    --debug-cep: activates the debugging procedure of C-EP (see Appendix ... for the pseudo-algorithm). Default: False. See Appendix F.2 for details about the debugging procedure.

    --seed: selects a specific seed (default: None). 

    --angle-grad: specifies whether we compute the angle between EP updates and BPTT gradients before learning.

* main.py can take two different actions:

  i) 'train': the network is trained with the arguments provided in the parser. It is trained by default with EP, C-EP,
      or C-VF depending on the arguments provided in the parser. Results are automatically saved in a folder sorted by
      date, GPU ID and trial number, along with a .txt file containing all hyperparameters. 

  ii) 'plotcurves': we demonstrate the GDD property on the network with the arguments provided in the parser. Results are 
       automatically saved in a folder sorted by trial number, along with a .txt file containing all hyperparameters. 


IV-  Details about netClasses.py


* There are four network classes (see Appendix E for the precise model definitions):

  i) EPcont: builds fully connected layered architectures with tied weights in the real-time setting.

  ii) EPdisc: builds fully connected layered architectures in the discrete-time setting. 

  iii) VFcont: builds fully connected layered architectures with untied weights (i.e. "vector field") in the real-time setting.

  iv) VFdisc: builds fully connected layered architectures with untied weights (i.e. "vector field") in the discrete-time setting.


* Each neural network class contains the following features and methods:

  i) Each class is a subclass of torch.nn.Module.

  ii) stepper: runs the network dynamics between two consecutive steps.

  iii) forward: runs the network dynamics over T steps, with many options depending on the context
      forward is being used. 

  iv) initHidden: initializes hidden units to zero.

  v) initGrad: initializes weight updates to zero.

  vi) computeGradients: compute gradients parameters given the state of the neural network. 

  vii) updateWeights: updates the parameters given the gradient parameters.


V - Details about netFunctions.py

* netFunctions.py contains the following functions:
  
  i) train: trains the model on the training set. 

  ii) evaluate: evaluates the model on the test set. 

  iii) compute_nSdSdT: computes \nabla^{BPTT}_{s} (nS), \Delta^{C-EP}_{s} (dS) and \Delta^{C-EP}_{\theta} (dT).
                    

  iv) compute_nT: computes \nabla^{BPTT}_{\theta} (nT). 

  v) compute_diffT: computes \nabla^{BPTT}_{\theta} (resp. \Delta^{C-EP}_{\theta}) from \sum(\nabla^{BPTT}_{\theta}) 
                   (resp. \sum(\Delta^{C-EP}_{\theta})).
 

  viii)	compute_angleGrad: computes the angle between the total C-EP update and the total BPTT gradient. 


  vii) createPath: creates a path to a directory depending on the date, the gpu device used, the model simulated and on the trial number
                where the results will be saved. 

  ix) createHyperparameterfile: creates a .txt file saved along with the results with all the hyperparameters. 


VI - Details about plotFunctions.py

* plotFunctions.py contains the following functions:

  i) plot_T: plots \nabla^{BPTT}_{\theta} and \Delta^{C-EP}_{\theta} processes.

  ii) plot_S: plots \nabla^{BPTT}_{s} and \Delta^{C-EP}_{s} processes.

  iii) plot_results: plots the test and train accuracy as a function of epochs. 


******************************************************************************

VII - Commands to be run in the terminal to reproduce the results of the paper

******************************************************************************

* Everytime a simulation is run, a result folder is created and contains plotFunctions.py. To visualize the results,
  plotFunctions.py has to be run within the result folder. 

* Section 4
  
  i) Subsection 4.1, Table of Fig. 4 [C-EP and C-VF results (with symmetric weights initially) on MNIST]:

    - EP in the discrete-time setting, 1 hidden layer (EP-1h):

      python main.py --action 'train' --discrete --size_tab 10 512 784 --lr_tab 0.04 0.08 --epochs 30 --T 30 --Kmax 10 --beta 0.1

    - EP in the discrete-time setting, 2 hidden layers (EP-2h):
  
      python main.py --action 'train' --discrete --size_tab 10 512 512 784 --lr_tab 0.005 0.05 0.2 --epochs 50 --T 100 --Kmax 20 --beta 0.5

    - C-EP in the discrete-time setting, 1 hidden layer (C-EP-1h):

      python main.py --action 'train' --discrete --size_tab 10 512 784 --lr_tab 0.0028 0.0056 --epochs 100 --T 40 --Kmax 15 --beta 0.2 --cep

    - C-EP in the discrete-time setting, 2 hidden layers (C-EP-2h):

      python main.py --action 'train' --discrete --size_tab 10 512 512 784 --lr_tab 0.00018 0.0018 0.01 --epochs 150 --T 100 --Kmax 20 --beta 0.5 --cep
    
    - C-VF in the discrete-time setting, 1 hidden layer (C-VF-1h)
      [the initial angle between forward and backward weights is zero, i.e. \Psi(\theta_f, \theta_b) = 0]:

      python main.py --action 'train' --discrete --size_tab 10 512 784 --lr_tab 0.0038 0.0076 --epochs 100 --T 40 --Kmax 15 --beta 0.20 --cep --learning-rule 'vf' 
      --randbeta 0.5

    - C-VF in the discrete-time setting, 2 hidden layers (C-VF-2h)
      [the initial angle between forward and backward weights is zero, i.e. \Psi(\theta_f, \theta_b) = 0]:

      python main.py --action 'train' --discrete --size_tab 10 512 512 784 --lr_tab 0.00016 0.0016 0.009 --epochs 150 --T 100 --Kmax 20 --beta 0.35
      --cep --learning-rule 'vf' --randbeta 0.5


  ii) Subsection 4.1, plot of Fig. 4 [C-VF MNIST results as a function of the initial weight angle between forward and backward weights]:

    -  Curve 'C-VF-1h': run the following command with angle_value \in {0, 22.5, 45, 67.5, 90, 112.5, 135, 167.5, 180}

     python main.py --action 'train' --discrete --size_tab 10 512 784 --lr_tab 0.0038 0.0076 --epochs 100 --T 40 --Kmax 15 --beta 0.20 --cep --learning-rule 'vf' --randbeta 0.5 --angle angle_value

    -  Curve 'C-VF-2h': run the following command with angle_value \in {0, 22.5, 45, 67.5, 90, 112.5, 135, 167.5, 180}

     python main.py --action 'train' --discrete --size_tab 10 512 512 784 --lr_tab 0.00016 0.0016 0.009 --epochs 150 --T 100 --Kmax 20 --beta 0.35 --cep --learning-rule 'vf' --randbeta 0.5 --angle angle_value


  iii) Subsection 4.3, Fig. 5 (a) [EP, C-EP and C-VF results as a function of the initial angle between the total EP update and the total BPTT gradient]:

    - EP in the discrete-time setting, 1 hidden layer (EP-1h):

      python main.py --action 'train' --discrete --size_tab 10 512 784 --lr_tab 0.04 0.08 --epochs 30 --T 30 --Kmax 10 --beta 0.1 --angle-grad

    - EP in the discrete-time setting, 2 hidden layers (EP-2h):
  
      python main.py --action 'train' --discrete --size_tab 10 512 512 784 --lr_tab 0.005 0.05 0.2 --epochs 50 --T 100 --Kmax 20 --beta 0.5 --angle-grad 

    - C-EP in the discrete-time setting, 1 hidden layer (C-EP-1h):

      python main.py --action 'train' --discrete --size_tab 10 512 784 --lr_tab 0.0028 0.0056 --epochs 100 --T 40 --Kmax 15 --beta 0.2 --cep --angle-grad

    - C-EP in the discrete-time setting, 2 hidden layers (C-EP-2h):

      python main.py --action 'train' --discrete --size_tab 10 512 512 784 --lr_tab 0.00018 0.0018 0.01 --epochs 150 --T 100 --Kmax 20 --beta 0.5 --cep --angle-grad

    - C-VF in the discrete-time setting, 1 hidden layer (C-VF-1h). Run the following command with angle_value \in {0, 22.5, 45, 67.5, 90, 112.5, 135, 167.5, 180}:

      python main.py --action 'train' --discrete --size_tab 10 512 784 --lr_tab 0.0038 0.0076 --epochs 100 --T 40 --Kmax 15 --beta 0.20 --cep --learning-rule 'vf' 
      --randbeta 0.5 --angle angle_value --angle-grad

    - C-VF in the discrete-time setting, 2 hidden layers (C-VF-2h). Run the following command with angle_value \in {0, 22.5, 45, 67.5, 90, 112.5, 135, 167.5, 180}:

      python main.py --action 'train' --discrete --size_tab 10 512 512 784 --lr_tab 0.00016 0.0016 0.009 --epochs 150 --T 100 --Kmax 20 --beta 0.35
      --cep --learning-rule 'vf' --randbeta 0.5 --angle angle_value --angle-grad


  iv) Subsection 4.3, Fig. 5 (b) [select the same seed for each plot to draw a direct comparison]

    - EP in the continuous-time setting, 1 hidden layer (EP-1h):

      python main.py --action 'plotcurves' --no-clamp --batch-size 1 --size_tab 10 512 784 --activation-function 'tanh' --dt 0.08 --beta 0.005 --T 800 --Kmax 80 --seed 0

    - C-EP in the continuous-time setting, 1 hidden layer (C-EP-1h):


      python main.py --action 'plotcurves' --no-clamp --batch-size 1 --size_tab 10 512 784 --activation-function 'tanh' --dt 0.08 --beta 0.005 --T 800 --Kmax 80 --cep --lr_tab 0.00002 0.00002


    - C-VF in the continuous-time setting, 1 hidden layer (C-VF-1h) with an angle of 45 degrees between forward and backward weights:

      python main.py --action 'plotcurves' --no-clamp --batch-size 1 --size_tab 10 512 784 --activation-function 'tanh' --dt 0.08 --beta 0.005 --T 800 --Kmax 80 --learning-rule 'vf' --lr_tab 0.00002 0.00002 --cep
        


* Appendix E.6:

   i) Fig. 8-9, left (C-EP in the discrete-time setting with \eta = 0):
 
     python main.py --action 'plotcurves' --discrete --batch-size 1 --size_tab 10 512 784 --activation-function 'tanh' --beta 0.01 --T 150 --Kmax 10 --learning-rule 'ep'

   ii) Fig. 8-9, right (C-EP in the discrete-time setting with \eta > 0):
 
     python main.py --action 'plotcurves' --discrete --batch-size 1 --size_tab 10 512 784 --activation-function 'tanh' --beta 0.01 --T 150 --Kmax 10 --learning-rule 'ep' --lr_tab 0.00002 0.00002 --cep

   iii) Fig. 10-11, left (C-EP in the real-time setting with \eta = 0):

     python main.py --action 'plotcurves' --no-clamp --batch-size 1 --size_tab 10 512 784 --activation-function 'tanh' --dt 0.08 --beta 0.005 --T 800 --Kmax 80

   iv) Fig. 10-11, right (C-EP in the real-time setting with \eta > 0):

     python main.py --action 'plotcurves' --no-clamp --batch-size 1 --size_tab 10 512 784 --activation-function 'tanh' --dt 0.08 --beta 0.005 --T 800 --Kmax 80 --cep --lr_tab 0.00002 0.00002

   v) Fig. 12-13, left (C-VF in the discrete-time setting with \eta = 0 and a weight angle of 0 degree):
 
     python main.py --action 'plotcurves' --discrete --batch-size 1 --size_tab 10 512 784 --activation-function 'tanh' --beta 0.01 --T 150 --Kmax 10 --learning-rule 'vf'

   vi) Fig. 12-13, right (C-VF in the discrete-time setting with \eta > 0 and a weight angle of 0 degree):
 
     python main.py --action 'plotcurves' --discrete --batch-size 1 --size_tab 10 512 784 --activation-function 'tanh' --beta 0.01 --T 150 --Kmax 10 --learning-rule 'vf' --lr_tab 0.00002 0.00002 --cep


   vii) Fig. 14-15, left (C-VF in the real-time setting with \eta = 0 and a weight angle of 0 degree):
 
     python main.py --action 'plotcurves' --no-clamp --batch-size 1 --size_tab 10 512 784 --activation-function 'tanh' --dt 0.08 --beta 0.005 --T 800 --Kmax 80 --learning-rule 'vf'

   viii) Fig. 14-15, right (C-VF in the real-time setting with \eta > 0 and a weight angle of 0 degree):
 
     python main.py --action 'plotcurves' --no-clamp --batch-size 1 --size_tab 10 512 784 --activation-function 'tanh' --dt 0.08 --beta 0.005 --T 800 --Kmax 80 --learning-rule 'vf' --lr_tab 0.00002 0.00002 --cep


* Appendix F.2:
 
   i) Debugging procedure of C-EP for 1 hidden layer:

     python main.py --action 'train' --discrete --size_tab 10 512 784 --lr_tab 0.04 0.08 --epochs 30 --T 30 --Kmax 10 --beta 0.1 --cep --debug-cep

   ii) Debugging procedure of C-EP for 2 hidden layers:

     python main.py --action 'train' --discrete --size_tab 10 512 512 784 --lr_tab 0.005 0.05 0.2 --epochs 50 --T 100 --Kmax 20 --beta 0.5 --cep --debug-cep
