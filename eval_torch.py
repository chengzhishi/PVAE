#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import local_evaluation_for_loop
from pytorch import train_pytorch_for_loop
from datetime import datetime

if __name__ == '__main__':
    elapsed_time, args = train_pytorch_for_loop.train_pytorch_main()

    result_path = './results/final_score_pytorch.txt'
    with open(result_path, 'a+') as fw:
        fw.write('\n##############################################################################################\n\n')
        fw.write('date time: ' + datetime.now().strftime("%Y%m%d_%H%M%S") + '\n\n')

        fw.write('elapsed time: ' + str(elapsed_time) + '\n\n')
        fw.write('########## pytorch args: ##########\n\n')
        fw.write('batch size: ' + str(args.batch_size) + '\n')
        fw.write('epochs: ' + str(args.epochs) + '\n')

        fw.write('\n')
    
    final_scores = local_evaluation_for_loop.eval_main(eval_pytorch = True)

    with open(result_path, 'a+') as fw:
        fw.write('\n##################################################\n\n')
        fw.write('\n########## Final Score: ##########\n\n')
        fw.write('dci: ' + str(final_scores['dci']) + '\n')
        fw.write('factor_vae_metric: ' + str(final_scores['factor_vae_metric']) + '\n')
        fw.write('sap_score: ' + str(final_scores['sap_score']) + '\n')
        fw.write('mig: ' + str(final_scores['mig']) + '\n')
        fw.write('irs: ' + str(final_scores['irs']) + '\n\n')

