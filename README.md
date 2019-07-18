# EP-VF
Discrete Equilibrium Propagation with Vector Field Dynamics (VF) and Continual dynamics (C-EP)

- C-EP, 1 hidden layer:

python main.py --action 'train' --discrete --size_tab 10 512 784 --lr_tab 0.003 0.006 --epochs 100 --T 40 --Kmax 15 --beta 0.2 --cep

- C-EP, 2 hidden layers:

python main.py --action 'train' --discrete --size_tab 10 512 512 784 --lr_tab 0.00018 0.0018 0.01 --epochs 150 --T 100 --Kmax 20 --beta 0.5 --cep

- VF, 1 hidden layer, symmetric weight initialization:

python main.py --action 'train' --discrete --size_tab 10 512 784 --lr_tab 0.075 0.15 --epochs 30 --T 30 --Kmax 10 --beta 0.1 --learning-rule 'vf' --former

- VF, 1 hidden layer, random weight initialization:

python main.py --action 'train' --discrete --size_tab 10 512 784 --lr_tab 0.075 0.15 --epochs 30 --T 30 --Kmax 10 --beta 0.1 --learning-rule 'vf' --former --weight-initialization 'any'

- VF, 2 hidden, symmetric weight initialization:

python main.py --action 'train' --discrete --size_tab 10 512 512 784 --lr_tab 0.009 0.09 0.3 --epochs 50 --T 100 --Kmax 20 --beta 0.5 --learning-rule 'vf' --former

- VF, 2 hidden, random weight initialization:

python main.py --action 'train' --discrete --size_tab 10 512 512 784 --lr_tab 0.009 0.09 0.3 --epochs 50 --T 100 --Kmax 20 --beta 0.5 --learning-rule 'vf' --former --weight-initialization 'any'
