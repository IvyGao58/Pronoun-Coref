#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf
from tensorflow.python import debug as tf_debug

import GCNModel as model
# import p2sModel as model
# import BiAffineModel as model
import util

if __name__ == "__main__":
    config = util.initialize_from_env()

    report_frequency = config["report_frequency"]
    eval_frequency = config["eval_frequency"]

    model = model.KnowledgePronounCorefModel(config)
    saver = tf.train.Saver()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['GPU'])

    log_dir = config["log_dir"]

    max_f1 = 0

    default_config = tf.ConfigProto()
    default_config.gpu_options.allow_growth = True

    with tf.Session(config=default_config) as session:
        # session = tf_debug.LocalCLIDebugWrapperSession(session, ui_type="readline")
        writer = tf.summary.FileWriter(log_dir, session.graph, flush_secs=20)
        print('begin to initialize global variables...')
        session.run(tf.global_variables_initializer())
        model.start_enqueue_thread(session)
        accumulated_loss = 0.0

        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring from: {}".format(ckpt.model_checkpoint_path))
            saver.restore(session, ckpt.model_checkpoint_path)

        initial_time = time.time()
        print('begin to run the model...')

        patience = 0
        while True:
            tf_loss, tf_global_step, summaries, _ = session.run([model.loss, model.global_step, model.summaries, model.train_op])

            accumulated_loss += tf_loss

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
                writer.add_summary(summaries, tf_global_step)

            if tf_global_step % eval_frequency == 0:
                saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
                eval_summary, eval_f1 = model.evaluate(session)

                if eval_f1 > max_f1:
                    max_f1 = eval_f1
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
