#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   neurips2019_disentanglement_challenge_starter_kit-master 

File Name:  loop_tf.py

"""

import os
import re

if __name__ == '__main__':
    lr = ['2e-4', '5e-4']
    batch_size = ['256']
    train_step = ['20000']
    # beta_tc = [2, 3, 3.5, 4.5, 6, 8, 10]
    dip_vae_lam = [2]
    for i in range(len(lr)):
        for j in range(len(batch_size)):
            for _dip_vae_lam in dip_vae_lam:
                with open('tensorflow/model.gin', 'r') as fr:
                    lines = fr.readlines()
                with open('tensorflow/model.gin', 'w+') as fw:
                    for line in lines:
                        if line.strip():
                            tmp = re.split("\s+", line.strip())
                            if tmp[0] == "model.batch_size":
                                fw.writelines("model.batch_size = " + batch_size[j] + '\n')
                                continue
                            elif tmp[0] == "model.training_steps":
                                fw.writelines("model.training_steps = " + train_step[j] + '\n')
                                continue
                            elif tmp[0] == "vae_optimizer.learning_rate":
                                fw.writelines("vae_optimizer.learning_rate = " + lr[i] + '\n')
                                continue
                        fw.writelines(line)

                os.system("python tensorflow/train_tensorflow_for_loop.py --dip-vae-lam " + str(_dip_vae_lam) +
                          ";python local_evaluation_for_loop.py")

