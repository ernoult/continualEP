# EP-VF
Discrete Equilibrium Propagation with Vector Field Dynamics (VF) and Continual dynamics (C-EP)

* For training: 

- EP, 1 hidden layer:

python main.py --action 'train' --discrete --size_tab 10 512 784 --lr_tab 0.04 0.08 --epochs 30 --T 30 --Kmax 10 --beta 0.1

- EP, 2 hidden layers:

python main.py --action 'train' --discrete --size_tab 10 512 512 784 --lr_tab 0.005 0.05 0.2 --epochs 50 --T 100 --Kmax 20 --beta 0.5


- C-EP, 1 hidden layer:

python main.py --action 'train' --discrete --size_tab 10 512 784 --lr_tab 0.0028 0.0056 --epochs 100 --T 40 --Kmax 15 --beta 0.2 --cep

- C-EP, 2 hidden layers:

python main.py --action 'train' --discrete --size_tab 10 512 512 784 --lr_tab 0.00018 0.0018 0.01 --epochs 150 --T 100 --Kmax 20 --beta 0.5 --cep


- VF, 1 hidden layer, with initial arbitrary theta (e.g. theta = 22.5, default 0)

python main.py --action 'train' --discrete --size_tab 10 512 784 --lr_tab 0.075 0.15 --epochs 30 --T 30 --Kmax 10 --beta 0.1 --learning-rule 'vf' --former --angle 22.5


- VF, 2 hidden, with initial arbitrary theta (e.g. theta = 22.5, default 0):

python main.py --action 'train' --discrete --size_tab 10 512 512 784 --lr_tab 0.009 0.09 0.3 --epochs 50 --T 100 --Kmax 20 --beta 0.5 --learning-rule 'vf' --former --angle 22.5

- C-VF, 1 hidden layer, with initial arbitrary theta (e.g. theta = 22.5, default 0):

python main.py --action 'train' --discrete --size_tab 10 512 784 --lr_tab 0.0038 0.0076 --epochs 100 --T 40 --Kmax 15 --beta 0.20 --cep --learning-rule 'vf' --randbeta 0.5 --angle 22.5

- C-VF, 2 hidden layers, with initial arbitrary theta (e.g. theta = 22.5, default 0): 

python main.py --action 'train' --discrete --size_tab 10 512 512 784 --lr_tab 0.00021 0.0021 0.0125 --epochs 150 --T 100 --Kmax 20 --beta 0.4 --cep --learning-rule 'vf' --randbeta 0.5 --angle 22.5

* For RelMSE computation *in training conditions* : same commands as before, changing the 'action' parser argument into 'plotcurves'

