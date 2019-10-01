#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import local_evaluation_for_loop
from datetime import datetime
from tf import train_tensorflow_for_loop


if __name__ == '__main__':
    elapsed_time, gin_bindings = train_tensorflow_for_loop.train_tf_main()
    final_scores = local_evaluation_for_loop.eval_main()

    # gin_bindings = [
    #     "dataset.name = '{}'".format('aaa'),
    #     "model.model = @beta_tc_vae()",
    #     "beta_tc_vae.beta = 4"
    # ]
    # final_scores = {'dci': 0.123797058589091, 'factor_vae_metric': 0.2824, 'sap_score': 0.002599999999999942,
    #                 'mig': 0.012375536130820422, 'irs': 0.623302371088628}

    result_path = './results/final_score_tf_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.txt'
    with open(result_path, 'w+') as fw:
        fw.write('Final Score:\n\n')
        fw.write('dci: ' + str(final_scores['dci']) + '\n')
        fw.write('factor_vae_metric: ' + str(final_scores['factor_vae_metric']) + '\n')
        fw.write('sap_score: ' + str(final_scores['sap_score']) + '\n')
        fw.write('mig: ' + str(final_scores['mig']) + '\n')
        fw.write('irs: ' + str(final_scores['irs']) + '\n\n')

        fw.write('elapsed time: ' + str(elapsed_time) + '\n\n')
        fw.write('gin_bindings: \n\n')
        for gin_b in gin_bindings:
            fw.write(gin_b + '\n')
        fw.write('\n')
        fw.write('\n##################################################\n\n')
        with open('./tf/model.gin', 'r') as fr:
            for line in fr.readlines():
                fw.write(line)

