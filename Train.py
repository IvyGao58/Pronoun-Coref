#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import time

import tensorflow as tf
import util
import model

if __name__ == "__main__":
    config = util.initialize_from_env()

    for k in range(1, config["k_fold"] + 1):
        print('It is {}th fold running...'.format(str(k)))

        config["train_path"] = config["basic_dir"] + '/train_' + str(k) + '.jsonlines'
        config["eval_path"] = config["basic_dir"] + '/dev_' + str(k) + '.jsonlines'

        report_frequency = config["report_frequency"]
        eval_frequency = config["eval_frequency"]

        tf.reset_default_graph()

        coref_model = model.KnowledgePronounCorefModel(config)
        saver = tf.train.Saver()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config['GPU'])

        log_dir = config["log_dir"]

        max_f1 = 0
        max_eval_matrix = None

        with tf.Session() as session:
            writer = tf.summary.FileWriter(log_dir, session.graph, flush_secs=20)
            print('begin to initialize global variables...')
            session.run(tf.global_variables_initializer())
            coref_model.start_enqueue_thread(session)
            accumulated_loss = 0.0

            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("Restoring from: {}".format(ckpt.model_checkpoint_path))
                saver.restore(session, ckpt.model_checkpoint_path)

            initial_time = time.time()
            patience = 0

            while True:
                tf_loss, tf_global_step, _ = session.run([coref_model.loss, coref_model.global_step, coref_model.train_op])
                accumulated_loss += tf_loss

                # if tf_global_step >= 250000:
                if patience > config["max_patience"]:
                    print('We have finished our training')
                    break

                if tf_global_step % report_frequency == 0:
                    total_time = time.time() - initial_time
                    steps_per_second = tf_global_step / total_time

                    average_loss = accumulated_loss / report_frequency
                    print("[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second))
                    writer.add_summary(util.make_summary({"loss": average_loss}), tf_global_step)
                    accumulated_loss = 0.0

                if tf_global_step % eval_frequency == 0:
                    saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
                    eval_summary, eval_f1, eval_result = coref_model.evaluate(session)

                    if eval_f1 > max_f1:
                        max_f1 = eval_f1
                        max_eval_matrix = eval_result
                        util.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)),
                                             os.path.join(log_dir, "model.max.ckpt"))
                        patience = 0
                    else:
                        patience += 1

                    print('start to write')
                    writer.add_summary(eval_summary, tf_global_step)
                    writer.add_summary(util.make_summary({"max_eval_f1": max_f1}), tf_global_step)
                    writer.flush()

                    print("[{}] evaL_f1={:.2f}, max_f1={:.2f}".format(tf_global_step, eval_f1, max_f1))

            writer.close()

            with open(config["eval_matrix_path"], 'a+', encoding='utf-8') as file:
                if max_eval_matrix is not None:
                    file.write('{}th fold.{}'.format(k, '\n'))
                    for key, val in max_eval_matrix.items():
                        file.write("{}: {:.2f}%{}".format(key, val * 100, '\n'))
                else:
                    file.write('{}th fold. empty result. {}'.format(k, '\n'))
            file.close()

        # clean saved files
        for root, dirs, files in os.walk(config["log_dir"], topdown=False):
            for file_name in files:
                delete_file_name = os.path.join(root, file_name)
                os.remove(delete_file_name)
                print(f'{delete_file_name} done...')
        os.rmdir(config["log_dir"])
        os.rmdir(config["log_root"])

    with open(config["eval_matrix_path"], 'r', encoding='utf-8') as file:
        k_fold_result = file.readlines()
    file.close()

    f1 = []
    precision = []
    recall = []
    pattern = re.compile(r"(\d{1,2}[\.]\d{1,2})")
    for line in k_fold_result:
        match = pattern.search(line)
        if match:
            value = float(match.group(1))
            if 'F1' in line:
                f1.append(value)
            elif 'precision' in line:
                precision.append(value)
            elif 'recall' in line:
                recall.append(value)

    avg_f1 = sum(f1)/len(f1)
    avg_precision = sum(precision)/len(precision)
    avg_recall = sum(recall)/len(recall)

    with open(config["eval_matrix_path"], 'a+', encoding='utf-8') as file:
        file.write('avg_k_fold_f1: {}'.format(avg_f1))
        file.write('avg_k_fold_precision: {}'.format(avg_precision))
        file.write('avg_k_fold_recall: {}'.format(avg_recall))
    file.close()

