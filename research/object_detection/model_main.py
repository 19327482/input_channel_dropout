# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Binary to run train and evaluation on object detection model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from absl import flags

from object_detection import model_hparams
from object_detection import model_lib

flags.DEFINE_string(
    'model_dir', None, 'Path to output model directory '
                       'where event and checkpoint files will be written.')
flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
                                                  'file.')
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
flags.DEFINE_boolean('checkpoint_was_trained_on_rgb_only', False,
                     'Wether the pretrained checkpoint was rgb only checkpoint (for instance COCO) or a checkpoint trained with additional channels')
flags.DEFINE_boolean('eval_training_data', False,
                     'If training data should be evaluated for this job. Note '
                     'that one call only use this in eval-only mode, and '
                     '`checkpoint_dir` must be supplied.')
flags.DEFINE_integer('sample_1_of_n_eval_examples', 1, 'Will sample one of '
                                                       'every n eval input examples, where n is provided.')
flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample '
                                                                'one of every n train input examples for evaluation, '
                                                                'where n is provided. This is only used if '
                                                                '`eval_training_data` is True.')
flags.DEFINE_string(
    'hparams_overrides', None, 'Hyperparameter overrides, '
                               'represented as a string containing comma-separated '
                               'hparam_name=value pairs.')
flags.DEFINE_string(
    'checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
                            '`checkpoint_dir` is provided, this binary operates in eval-only mode, '
                            'writing resulting metrics to `model_dir`.')
flags.DEFINE_boolean(
    'run_once', False, 'If running in eval-only mode, whether to run just '
                       'one round of eval vs running continuously (default).'
)
FLAGS = flags.FLAGS


def main(unused_argv):
    flags.mark_flag_as_required('model_dir')
    flags.mark_flag_as_required('pipeline_config_path')
    config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir)

    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
        pipeline_config_path=FLAGS.pipeline_config_path,
        train_steps=FLAGS.num_train_steps,
        sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
        sample_1_of_n_eval_on_train_examples=(
            FLAGS.sample_1_of_n_eval_on_train_examples),
        checkpoint_was_trained_on_rgb_only=FLAGS.checkpoint_was_trained_on_rgb_only)
    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']

    if FLAGS.checkpoint_dir:
        if FLAGS.eval_training_data:
            name = 'training_data'
            input_fn = eval_on_train_input_fn
        else:
            name = 'validation_data'
            # The first eval input will be evaluated.
            input_fn = eval_input_fns[0]
        if FLAGS.run_once:
            estimator.evaluate(input_fn,
                               steps=None,
                               checkpoint_path=tf.train.latest_checkpoint(
                                   FLAGS.checkpoint_dir))
        else:
            model_lib.continuous_eval(estimator, FLAGS.checkpoint_dir, input_fn,
                                      train_steps, name)
    else:

        class VariableUpdaterHook(tf.train.SessionRunHook):
            def __init__(self):
                super(VariableUpdaterHook, self).__init__()
                # variable name should be like: parent/scope/some/path/variable_name:0
                # self._global_step_tensor = None
                # self.variable = None
                # self.frequency = frequency
                # self.variable_name = variable_name

            def after_create_session(self, session, coord):
                print("after" * 100)
                all_variables = tf.global_variables()

                # original_first_layer_weights_var_name = "FirstStageFeatureExtractor/resnet_v1_50/conv1/weights_original:0"
                # original_first_layer_weights = \
                #     [variable for variable in all_variables if variable.name == original_first_layer_weights_var_name]
                original_first_layer_weights = \
                    [variable for variable in all_variables if "_original" in variable.name]
                original_first_layer_weights_found = len(original_first_layer_weights) == 1
                if not original_first_layer_weights_found:
                    return
                original_first_layer_weights = original_first_layer_weights[0]
                print(original_first_layer_weights)

                # first_layer_weights_var_name = "FirstStageFeatureExtractor/resnet_v1_50/conv1/weights:0"
                # print(first_layer_weights)
                # first_layer_weights = \
                #     [variable for variable in all_variables if variable.name == first_layer_weights_var_name][0]
                # for variable in all_variables:
                #     print(variable.name)
                first_layer_weights = \
                    [variable for variable in all_variables if variable.name.endswith("resnet_v1_50/conv1/weights:0")][0]

                rebuild_first_layer_op = tf.get_default_graph().get_operation_by_name("rebuild_first_layer")
                print(rebuild_first_layer_op)

                session.run(original_first_layer_weights.initializer)
                print(original_first_layer_weights)
                a = session.run(first_layer_weights.initializer)
                print(a)
                print(first_layer_weights)
                new_weights = session.run(rebuild_first_layer_op)
                new_weights = session.run(first_layer_weights)
                print(type(new_weights))

            # def begin(self):
            #     self._global_step_tensor = tf.train.get_global_step()

            # def after_run(self, run_context, run_values):
            #     global_step = run_context.session.run(self._global_step_tensor)
            #     if global_step % self.frequency == 0:
            #         new_variable_value = complicated_algorithm(...)
            #         assign_op = self.variable.assign(new_variable_value)
            #         run_context.session.run(assign_op)

        train_spec, eval_specs = model_lib.create_train_and_eval_specs(
            train_input_fn,
            eval_input_fns,
            eval_on_train_input_fn,
            predict_input_fn,
            train_steps,
            eval_on_train_data=False,
            train_hooks=[VariableUpdaterHook()])

        # Currently only a single Eval Spec is allowed.
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])


if __name__ == '__main__':
    tf.app.run()
