#!/bin/sh
python experiments.py --screen_print --plot_interval 1 --marksize 5 --epoch 10 --hidden_act tanh --last_act softmax \
--lr 0.01 --gamma 0.97 --layers_dim '[10, 10, 10, 10, 10, 10, 2]' --loss ce --bin_min -1 --bin_max 1 --project_name debug
# python experiments.py --screen_print --plot_interval 1 --marksize 5 --epoch 100 --hidden_act relu --last_act softmax \
# --lr 0.1 --gamma 0.97 --project_name E100_Hrelu_Lsoftmax_LR0.1_GA0.97

# python experiments.py --screen_print --plot_interval 1 --marksize 5 --epoch 100 --hidden_act relu --last_act sigmoid \
# --lr 0.1 --gamma 0.97 --project_name E100_Hrelu_Lsigmoid_LR0.1_GA0.97
# python experiments.py --screen_print --plot_interval 1 --marksize 5 --epoch 100 --hidden_act relu --last_act softmax \
# --lr 0.1 --gamma 0.9 --project_name E100_Hrelu_Lsoftmax_LR0.1_GA0.9

# python experiments.py --screen_print --plot_interval 1 --marksize 5 --epoch 100 --hidden_act tanh --last_act iden --project_name E100_Htanh_Liden

# python experiments.py --screen_print --plot_interval 1 --marksize 5 --epoch 100 --hidden_act tanh --last_act sigmoid --project_name E100_Htanh_Lsigmoid

# python experiments.py --screen_print --plot_interval 1 --marksize 5 --epoch 100 --hidden_act tanh --last_act softmax --project_name E100_Htanh_Lsoftmax

# python experiments.py --screen_print --plot_interval 1 --marksize 5 --epoch 100 --hidden_act relu --last_act iden --project_name E100_Hrelu_Liden

# python experiments.py --screen_print --plot_interval 1 --marksize 5 --epoch 100 --hidden_act relu --last_act sigmoid --project_name E100_Hrelu_Lsigmoid

# python experiments.py --screen_print --plot_interval 1 --marksize 5 --epoch 100 --hidden_act relu --last_act softmax --project_name E100_Hrelu_Lsoftmax

# python experiments.py --screen_print --plot_interval 1 --marksize 5 --epoch 1000 --hidden_act relu --last_act iden --project_name E1000_Hrelu_Liden

# python experiments.py --screen_print --plot_interval 1 --marksize 5 --epoch 1000 --hidden_act relu --last_act sigmoid --project_name E1000_Hrelu_Lsigmoid

# python experiments.py --screen_print --plot_interval 1 --marksize 5 --epoch 1000 --hidden_act relu --last_act softmax --project_name E1000_Hrelu_Lsoftmax