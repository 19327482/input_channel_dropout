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

"""Helper functions for manipulating collections of variables during training.
"""
import logging
import math
import os
import re

import tensorflow as tf
from tensorflow.python.ops import variables as tf_variables

slim = tf.contrib.slim


# TODO(derekjchow): Consider replacing with tf.contrib.filter_variables in
# tensorflow/contrib/framework/python/ops/variables.py
def filter_variables(variables, filter_regex_list, invert=False):
    """Filters out the variables matching the filter_regex.

    Filter out the variables whose name matches the any of the regular
    expressions in filter_regex_list and returns the remaining variables.
    Optionally, if invert=True, the complement set is returned.

    Args:
      variables: a list of tensorflow variables.
      filter_regex_list: a list of string regular expressions.
      invert: (boolean).  If True, returns the complement of the filter set; that
        is, all variables matching filter_regex are kept and all others discarded.

    Returns:
      a list of filtered variables.
    """
    kept_vars = []
    variables_to_ignore_patterns = list(filter(None, filter_regex_list))
    for var in variables:
        add = True
        for pattern in variables_to_ignore_patterns:
            if re.match(pattern, var.op.name):
                add = False
                break
        if add != invert:
            kept_vars.append(var)
    return kept_vars


def multiply_gradients_matching_regex(grads_and_vars, regex_list, multiplier):
    """Multiply gradients whose variable names match a regular expression.

    Args:
      grads_and_vars: A list of gradient to variable pairs (tuples).
      regex_list: A list of string regular expressions.
      multiplier: A (float) multiplier to apply to each gradient matching the
        regular expression.

    Returns:
      grads_and_vars: A list of gradient to variable pairs (tuples).
    """
    variables = [pair[1] for pair in grads_and_vars]
    matching_vars = filter_variables(variables, regex_list, invert=True)
    for var in matching_vars:
        logging.info('Applying multiplier %f to variable [%s]',
                     multiplier, var.op.name)
    grad_multipliers = {var: float(multiplier) for var in matching_vars}
    return slim.learning.multiply_gradients(grads_and_vars,
                                            grad_multipliers)


def freeze_gradients_matching_regex(grads_and_vars, regex_list):
    """Freeze gradients whose variable names match a regular expression.

    Args:
      grads_and_vars: A list of gradient to variable pairs (tuples).
      regex_list: A list of string regular expressions.

    Returns:
      grads_and_vars: A list of gradient to variable pairs (tuples) that do not
        contain the variables and gradients matching the regex.
    """
    variables = [pair[1] for pair in grads_and_vars]
    matching_vars = filter_variables(variables, regex_list, invert=True)
    kept_grads_and_vars = [pair for pair in grads_and_vars
                           if pair[1] not in matching_vars]
    for var in matching_vars:
        logging.info('Freezing variable [%s]', var.op.name)
    return kept_grads_and_vars


def get_variables_available_in_checkpoint(variables,
                                          checkpoint_path,
                                          include_global_step=True,
                                          checkpoint_was_trained_on_rgb_only=False):
    """Returns the subset of variables available in the checkpoint.

    Inspects given checkpoint and returns the subset of variables that are
    available in it.

    TODO(rathodv): force input and output to be a dictionary.

    Args:
      variables: a list or dictionary of variables to find in checkpoint.
      checkpoint_path: path to the checkpoint to restore variables from.
      include_global_step: whether to include `global_step` variable, if it
        exists. Default True.

    Returns:
      A list or dictionary of variables.
    Raises:
      ValueError: if `variables` is not a list or dict.
    """
    if isinstance(variables, list):
        variable_names_map = {}
        for variable in variables:
            if isinstance(variable, tf_variables.PartitionedVariable):
                name = variable.name
            else:
                name = variable.op.name
            variable_names_map[name] = variable
    elif isinstance(variables, dict):
        variable_names_map = variables
    else:
        raise ValueError('`variables` is expected to be a list or dict.')
    ckpt_reader = tf.train.NewCheckpointReader(checkpoint_path)
    ckpt_vars_to_shape_map = ckpt_reader.get_variable_to_shape_map()
    if not include_global_step:
        ckpt_vars_to_shape_map.pop(tf.GraphKeys.GLOBAL_STEP, None)
    vars_in_ckpt = {}
    for variable_name, variable in sorted(variable_names_map.items()):
        print(variable_name)
        if "resnet_v1_50/conv1/weights" in variable_name:
            print("-" * 100)
            print(variable_name)
        if variable_name in ckpt_vars_to_shape_map:
            print("LINE 148")
            # elif variable_name == "FirstStageFeatureExtractor/resnet_v1_50/conv1/weights_original":
            if "_original" in variable_name:
                pass
                # print("OOO" * 100)
                # print("OOO" * 100)
                # print("OOO" * 100)
                # print("OOO" * 100)
                # print("OOO" * 100)
            # if variable_name == "FirstStageFeatureExtractor/resnet_v1_50/conv1/weights":
            elif variable_name.endswith("resnet_v1_50/conv1/weights"):
                # print("*" * (2**4))
                # print(variable_name)
                # print(variable)
                if checkpoint_was_trained_on_rgb_only:
                    # original_first_layer_weights = tf.Variable(tf.zeros((7, 7, 3, 64)), name="FirstStageFeatureExtractor/resnet_v1_50/conv1/weights_original")
                    variable_name_original = variable_name + "_original"
                    original_first_layer_weights = tf.Variable(tf.zeros((7, 7, 3, 64)), name=variable_name_original)
                    # print(original_first_layer_weights)
                    vars_in_ckpt[variable_name] = original_first_layer_weights
                    all_variables = tf.global_variables()
                    original_first_layer_weights_var_name = variable_name_original + ":0"
                    original_first_layer_weights = \
                        [variable for variable in all_variables if
                         variable.name == original_first_layer_weights_var_name][0]
                    # print(original_first_layer_weights)
                    first_layer_weights_var_name = variable_name + ":0"
                    first_layer_weights = \
                        [variable for variable in all_variables if variable.name == first_layer_weights_var_name][0]
                    # print(first_layer_weights)
                    channels_mean = tf.reduce_mean(original_first_layer_weights, axis=2)
                    n_additionnal_channels = int(os.environ["N_ADDITIONAL"])
                    n_total_channels = n_additionnal_channels + 3
                    channels_mean = tf.expand_dims(channels_mean, axis=2) * 3. / float(n_total_channels)
                    # new_first_layer_weights = tf.concat([original_first_layer_weights * 3. / float(n_total_channels)] + [channels_mean] * n_additionnal_channels,
                    #                                     axis=2)
                    rgb_channels_weights = [original_first_layer_weights * 3. / float(n_total_channels)]
                    additional_channels_weights = [channels_mean] * n_additionnal_channels
                    # additional_channels_weights = tf.random.normal((7, 7, n_additionnal_channels, 64), mean=0, stddev=math.sqrt(2/(7*7 + 3*3*128)))
                    # additional_channels_weights = [additional_channels_weights]
                    new_first_layer_weights = tf.concat(rgb_channels_weights + additional_channels_weights, axis=2)
                    first_layer_weights.assign(new_first_layer_weights, name="rebuild_first_layer")
                    print("REBUILDING FIRST LAYER WEIGHTS")
                    print("REBUILDING FIRST LAYER WEIGHTS")
                    print("REBUILDING FIRST LAYER WEIGHTS")
                else:
                    print("RESTORING FIRST LAYER WEIGHTS FROM CKPT")
                    print("RESTORING FIRST LAYER WEIGHTS FROM CKPT")
                    print("RESTORING FIRST LAYER WEIGHTS FROM CKPT")
                    vars_in_ckpt[variable_name] = variable
            elif ckpt_vars_to_shape_map[variable_name] == variable.shape.as_list():
                vars_in_ckpt[variable_name] = variable
            else:
                logging.warning('Variable [%s] is available in checkpoint, but has an '
                                'incompatible shape with model variable. Checkpoint '
                                'shape: [%s], model variable shape: [%s]. This '
                                'variable will not be initialized from the checkpoint.',
                                variable_name, ckpt_vars_to_shape_map[variable_name],
                                variable.shape.as_list())
        else:
            logging.warning('Variable [%s] is not available in checkpoint',
                            variable_name)
    if isinstance(variables, list):
        return vars_in_ckpt.values()
    return vars_in_ckpt
