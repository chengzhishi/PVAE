# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
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
#! /rds/general/user/cs618/home/anaconda3/envs/NIPS/bin/python3
#PBS -lselect=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=P100 -lwalltime=15:00:00
###PBS -lselect=1:ncpus=4:mem=96gb:ngpus=1:gpu_type=P1000 -lwalltime=15:00:00

# We group all the imports at the top.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.methods.unsupervised import train
#import train ##if pruning , use customized train
from disentanglement_lib.methods.unsupervised import vae
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results
import tensorflow as tf
from numba import cuda
import gin.tf
import aicrowd_helpers
import math
import time
import L0layer
from disentanglement_lib.methods.shared import architectures  # pylint: disable=unused-import
from disentanglement_lib.methods.shared import losses  # pylint: disable=unused-import
from disentanglement_lib.methods.shared import optimizers  # pylint: disable=unused-import
from disentanglement_lib.methods.unsupervised import gaussian_encoder_model
from six.moves import range
from six.moves import zip
from hvae import *

# 0. Settings
# ------------------------------------------------------------------------------
# By default, we save all the results in subdirectories of the following path.
base_path = os.getenv("AICROWD_OUTPUT_PATH", "../scratch/shared")
experiment_name = os.getenv("AICROWD_EVALUATION_NAME", "experiment_name")
#experiment_name = str(time.time())
DATASET_NAME = os.getenv("AICROWD_DATASET_NAME", "mpi3d_toy")
ROOT = os.getenv("NDC_ROOT", "..")
overwrite = True

# 0.1 Helpers
# ------------------------------------------------------------------------------


def get_full_path(filename):
    return os.path.join(ROOT, "tensorflow", filename)
#########BetaTCVAE
def gaussian_log_density(samples, mean, log_var):
  pi = tf.constant(math.pi)
  normalization = tf.log(2. * pi)
  inv_sigma = tf.exp(-log_var)
  tmp = (samples - mean)
  return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)


def total_correlation(z, z_mean, z_logvar):
  """Estimate of total correlation on a batch.
  We need to compute the expectation over a batch of: E_j [log(q(z(x_j))) -
  log(prod_l q(z(x_j)_l))]. We ignore the constants as they do not matter
  for the minimization. The constant should be equal to (num_latents - 1) *
  log(batch_size * dataset_size)
  Args:
    z: [batch_size, num_latents]-tensor with sampled representation.
    z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
    z_logvar: [batch_size, num_latents]-tensor with log variance of the encoder.
  Returns:
    Total correlation estimated on a batch.
  """
  # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
  # tensor of size [batch_size, batch_size, num_latents]. In the following
  # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
  log_qz_prob = gaussian_log_density(
      tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0),
      tf.expand_dims(z_logvar, 0))
  # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
  # + constant) for each sample in the batch, which is a vector of size
  # [batch_size,].
  log_qz_product = tf.reduce_sum(
      tf.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False),
      axis=1,
      keepdims=False)
  # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
  # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
  log_qz = tf.reduce_logsumexp(
      tf.reduce_sum(log_qz_prob, axis=2, keepdims=False),
      axis=1,
      keepdims=False)
  return tf.reduce_mean(log_qz - log_qz_product)

#########DIPVAE
def compute_covariance_z_mean(z_mean):
  """Computes the covariance of z_mean.
  Uses cov(z_mean) = E[z_mean*z_mean^T] - E[z_mean]E[z_mean]^T.
  Args:
    z_mean: Encoder mean, tensor of size [batch_size, num_latent].
  Returns:
    cov_z_mean: Covariance of encoder mean, tensor of size [num_latent,
      num_latent].
  """
  expectation_z_mean_z_mean_t = tf.reduce_mean(
      tf.expand_dims(z_mean, 2) * tf.expand_dims(z_mean, 1), axis=0)
  expectation_z_mean = tf.reduce_mean(z_mean, axis=0)
  cov_z_mean = tf.subtract(
      expectation_z_mean_z_mean_t,
      tf.expand_dims(expectation_z_mean, 1) * tf.expand_dims(
          expectation_z_mean, 0))
  return cov_z_mean

def make_metric_fn(*names):
  """Utility function to report tf.metrics in model functions."""

  def metric_fn(*args):
    return {name: tf.metrics.mean(vec) for name, vec in zip(names, args)}

  return metric_fn

def regularize_diag_off_diag_dip(covariance_matrix, lambda_od, lambda_d):
  """Compute on and off diagonal regularizers for DIP-VAE models.
  Penalize deviations of covariance_matrix from the identity matrix. Uses
  different weights for the deviations of the diagonal and off diagonal entries.
  Args:
    covariance_matrix: Tensor of size [num_latent, num_latent] to regularize.
    lambda_od: Weight of penalty for off diagonal elements.
    lambda_d: Weight of penalty for diagonal elements.
  Returns:
    dip_regularizer: Regularized deviation from diagonal of covariance_matrix.
  """
  covariance_matrix_diagonal = tf.diag_part(covariance_matrix)
  covariance_matrix_off_diagonal = covariance_matrix - tf.diag(
      covariance_matrix_diagonal)
  dip_regularizer = tf.add(
      lambda_od * tf.reduce_sum(covariance_matrix_off_diagonal**2),
      lambda_d * tf.reduce_sum((covariance_matrix_diagonal - 1)**2))
  return dip_regularizer
########################################################################
# Register Execution Start
########################################################################
aicrowd_helpers.execution_start()



# Train a custom VAE model.
@gin.configurable("L0BetaTCVAE")
class L0BetaTCVAE(vae.BaseVAE):
  """BetaTCVAE model."""

  def __init__(self, beta=gin.REQUIRED):
    """Creates a beta-TC-VAE model.
    Based on Equation 4 with alpha = gamma = 1 of "Isolating Sources of
    Disentanglement in Variational Autoencoders"
    (https://arxiv.org/pdf/1802.04942).
    If alpha = gamma = 1, Eq. 4 can be written as ELBO + (1 - beta) * TC.
    Args:
      beta: Hyperparameter total correlation.
    """
    self.beta = beta

  def compute_gaussian_kl(self, z_mean, z_logvar, mask):
      """Compute KL divergence between input Gaussian and Standard Normal."""
      return tf.reduce_mean(
          0.5 * tf.reduce_sum(
              (tf.square(z_mean) + tf.exp(z_logvar) - z_logvar - 1) * mask, [1]),
          name="kl_loss")

  def model_fn(self, features, labels, mode, params):
    """TPUEstimator compatible model function."""
    del labels
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    data_shape = features.get_shape().as_list()[1:]
    output = self.gaussian_encoder(features, is_training=is_training)
    if len(output) == 2:
        z_mean, z_logvar = output
    else:
        z_mean, z_logvar, L0_reg, mask = output
    z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)
    reconstructions = self.decode(z_sampled, data_shape, is_training)
    per_sample_loss = losses.make_reconstruction_loss(features, reconstructions)
    reconstruction_loss = tf.reduce_mean(per_sample_loss)
    kl_loss = compute_gaussian_kl(z_mean, z_logvar, mask)
    # regularizer = self.regularizer(kl_loss, z_mean, z_logvar, z_sampled, mask, L0_reg)
    regularizer = self.regularizer(kl_loss, z_mean, z_logvar, z_sampled)
    if len(output) == 2:
        loss = tf.add(reconstruction_loss, regularizer, name="loss")
    else:
        loss = tf.add(reconstruction_loss, regularizer+L0_reg/500000., name="loss")

    loss = tf.add(reconstruction_loss, regularizer, name="loss")
    elbo = tf.add(reconstruction_loss, kl_loss, name="elbo")
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = optimizers.make_vae_optimizer()
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      train_op = optimizer.minimize(
          loss=loss, global_step=tf.train.get_global_step())
      train_op = tf.group([train_op, update_ops])
      # tf.summary.scalar("L0_reg", L0_reg)
      # tf.summary.scalar("mask_sum",tf.reduce_sum(mask))
      tf.summary.scalar("reconstruction_loss", reconstruction_loss)
      tf.summary.scalar("elbo", elbo)

      logging_hook = tf.train.LoggingTensorHook({
          "loss": loss,
          "reconstruction_loss": reconstruction_loss,
          "elbo": -elbo
      },
                                                every_n_iter=100)
      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          train_op=train_op,
          training_hooks=[logging_hook])
    elif mode == tf.estimator.ModeKeys.EVAL:
      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metrics=(make_metric_fn("reconstruction_loss", "elbo",
                                       "regularizer", "kl_loss"),
                        [reconstruction_loss, -elbo, regularizer, kl_loss]))
    else:
      raise NotImplementedError("Eval mode not supported.")

  def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    tc = (self.beta - 1.) * total_correlation(z_sampled, z_mean, z_logvar)
    return tc + kl_loss

# gin_bindings = [
#     "dataset.name = '{}'".format(DATASET_NAME),
#     "model.model = @BottleneckVAE()",
#     "BottleneckVAE.gamma = 4",
#     "BottleneckVAE.target = 10."
# ]
@gin.configurable("DIPVAE")
class DIPVAE(vae.BaseVAE):
  """DIPVAE model."""

  def __init__(self,
               lambda_od=gin.REQUIRED,
               lambda_d_factor=gin.REQUIRED,
               dip_type="i"):
    """Creates a DIP-VAE model.
    Based on Equation 6 and 7 of "Variational Inference of Disentangled Latent
    Concepts from Unlabeled Observations"
    (https://openreview.net/pdf?id=H1kG7GZAW).
    Args:
      lambda_od: Hyperparameter for off diagonal values of covariance matrix.
      lambda_d_factor: Hyperparameter for diagonal values of covariance matrix
        lambda_d = lambda_d_factor*lambda_od.
      dip_type: "i" or "ii".
      d = 10 od
    """
    self.lambda_od = lambda_od
    self.lambda_d_factor = lambda_d_factor
    self.dip_type = dip_type

  def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    cov_z_mean = compute_covariance_z_mean(z_mean)
    lambda_d = self.lambda_d_factor * self.lambda_od
    if self.dip_type == "i":  # Eq 6 page 4
      # mu = z_mean is [batch_size, num_latent]
      # Compute cov_p(x) [mu(x)] = E[mu*mu^T] - E[mu]E[mu]^T]
      cov_dip_regularizer = regularize_diag_off_diag_dip(
          cov_z_mean, self.lambda_od, lambda_d)
    elif self.dip_type == "ii":
      cov_enc = tf.matrix_diag(tf.exp(z_logvar))
      expectation_cov_enc = tf.reduce_mean(cov_enc, axis=0)
      cov_z = expectation_cov_enc + cov_z_mean
      cov_dip_regularizer = regularize_diag_off_diag_dip(
          cov_z, self.lambda_od, lambda_d)
    else:
      raise NotImplementedError("DIP variant not supported.")
    return kl_loss + cov_dip_regularizer
# gin_bindings = [
#     "dataset.name = '{}'".format(DATASET_NAME),
#     "model.model = @BetaTCVAE()",
#     "BetaTCVAE.beta = 6."
# ]
# gin_bindings = [
#     "dataset.name = '{}'".format(DATASET_NAME),
#     "model.model = @DIPVAE()",
#     "DIPVAE.lambda_od = 2.",
#     "DIPVAE.lambda_d_factor = 20."
# ]


@gin.configurable("lconv_encoder", whitelist=[])
def lconv_encoder(input_tensor, num_latent, is_training=True):
  """Convolutional encoder used in beta-VAE paper for the chairs data.

  Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
  Concepts with a Constrained Variational Framework"
  (https://openreview.net/forum?id=Sy2fzU9gl)

  Args:
    input_tensor: Input tensor of shape (batch_size, 64, 64, num_channels) to
      build encoder on.
    num_latent: Number of latent variables to output.
    is_training: Whether or not the graph is built for training (UNUSED).

  Returns:
    means: Output tensor of shape (batch_size, num_latent) with latent variable
      means.
    log_var: Output tensor of shape (batch_size, num_latent) with latent
      variable log variances.
  """
  del is_training

  e1 = tf.layers.conv2d(
      inputs=input_tensor,
      filters=32,
      kernel_size=4,
      strides=2,
      activation=tf.nn.relu,
      padding="same",
      name="e1",
  )
  e2 = tf.layers.conv2d(
      inputs=e1,
      filters=32,
      kernel_size=4,
      strides=2,
      activation=tf.nn.relu,
      padding="same",
      name="e2",
  )
  e3 = tf.layers.conv2d(
      inputs=e2,
      filters=64,
      kernel_size=2,
      strides=2,
      activation=tf.nn.relu,
      padding="same",
      name="e3",
  )
  e4 = tf.layers.conv2d(
      inputs=e3,
      filters=64,
      kernel_size=2,
      strides=2,
      activation=tf.nn.relu,
      padding="same",
      name="e4",
  )
  flat_e4 = tf.layers.flatten(e4)
  e5 = tf.layers.dense(flat_e4, 256, activation=tf.nn.relu, name="e5")
  l0  = L0layer.L0Pair(256, num_latent, droprate_init=0.2, weight_decay=0.001, lambda_l0=0.1)
  means, log_var, regularization, mask = l0(e5)
  # regularization = self.lambda_l0 * self.fc_latent[0].regularization().cuda() / 50000.
  # mask = self.fc_latent[0].sample_mask()
  # means = tf.layers.dense(e5, num_latent, activation=None, name="means")
  # log_var = tf.layers.dense(e5, num_latent, activation=None, name="log_var")
  return means, log_var, regularization, mask


def gaussian_log_density(samples, mean, log_var):
  pi = tf.constant(math.pi)
  normalization = tf.log(2. * pi)
  inv_sigma = tf.exp(-log_var)
  tmp = (samples - mean)
  return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)


def total_correlation(z, z_mean, z_logvar):
  """Estimate of total correlation on a batch.
  We need to compute the expectation over a batch of: E_j [log(q(z(x_j))) -
  log(prod_l q(z(x_j)_l))]. We ignore the constants as they do not matter
  for the minimization. The constant should be equal to (num_latents - 1) *
  log(batch_size * dataset_size)
  Args:
    z: [batch_size, num_latents]-tensor with sampled representation.
    z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
    z_logvar: [batch_size, num_latents]-tensor with log variance of the encoder.
  Returns:
    Total correlation estimated on a batch.
  """
  # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
  # tensor of size [batch_size, batch_size, num_latents]. In the following
  # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
  log_qz_prob = gaussian_log_density(
      tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0),
      tf.expand_dims(z_logvar, 0))
  # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
  # + constant) for each sample in the batch, which is a vector of size
  # [batch_size,].
  log_qz_product = tf.reduce_sum(
      tf.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False),
      axis=1,
      keepdims=False)
  # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
  # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
  log_qz = tf.reduce_logsumexp(
      tf.reduce_sum(log_qz_prob, axis=2, keepdims=False),
      axis=1,
      keepdims=False)
  return tf.reduce_mean(log_qz - log_qz_product)

@gin.configurable("h_encoder", whitelist=[])
def h_encoder(input_tensor, num_latent, is_training=False):
  """Convolutional encoder used in beta-VAE paper for the chairs data.
  Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
  Concepts with a Constrained Variational Framework"
  (https://openreview.net/forum?id=Sy2fzU9gl)
  Args:
    input_tensor: Input tensor of shape (batch_size, 64, 64, num_channels) to
      build encoder on.
    num_latent: Number of latent variables to output.
    is_training: Whether or not the graph is built for training (UNUSED).
  Returns:
    means: Output tensor of shape (batch_size, num_latent) with latent variable
      means.
    log_var: Output tensor of shape (batch_size, num_latent) with latent
      variable log variances.
  """
  cs = [1, 64, 128, 256]
  i0 = inference0(input_tensor, cs, is_training)
  l0 = ladder0(input_tensor, cs, 4,is_training)
  i1 = inference1(i0, cs, is_training)
  l1 = ladder1(i0, cs, 3, is_training)
  l2 = ladder2(i1, cs, 3, is_training)
  means = tf.concat([l0[0], l1[0], l2[0]], 1)
  log_var = tf.concat([l0[1], l1[1], l2[1]], 1)

  return means, log_var

@gin.configurable("h_decoder", whitelist=[])
def h_decoder(latent_tensor,output_shape, is_training=False):
  """Convolutional decoder used in beta-VAE paper for the chairs data.
  Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
  Concepts with a Constrained Variational Framework"
  (https://openreview.net/forum?id=Sy2fzU9gl)
  Args:
    latent_tensor: Input tensor of shape (batch_size,) to connect decoder to.
    output_shape: Shape of the data.
    is_training: Whether or not the graph is built for training (UNUSED).
  Returns:
    Output tensor of shape (batch_size, 64, 64, num_channels) with the [0,1]
      pixel intensities.
  """
  #print("##############################################")
  #print(latent_tensor.shape, latent_tensor[:,7:10].shape)
  data_dims = 64
  fs = [data_dims, data_dims // 2, data_dims // 4, data_dims // 8,
        data_dims // 16]
  cs = [1, 64, 128, 256]
  l0, l1, l2 = latent_tensor[:, :4], latent_tensor[:, 4:7], latent_tensor[:, 7:10]
  d1 = generative2(l2, cs, is_training=is_training)
  d2 = generative1(d1, cs, l1, is_training=is_training)
  d3 = generative0(d2, cs, fs, l0, is_training=is_training)
  return d3
"""

@gin.configurable("h_beta_tc_vae")
class HBetaTCVAE(vae.BaseVAE):
  HBetaTCVAE model
  def __init__(self, beta=gin.REQUIRED, ladder0=gin.REQUIRED, ladder1=gin.REQUIRED, ladder2=gin.REQUIRED):
    self.h_latent = [ladder0, ladder1, ladder2]
    self.beta = beta
    self.cs = [1, 64, 128, 256]
  
  def compute_gaussian_kl(self, z_mean, z_logvar):
  
    return tf.reduce_mean(0.5 * tf.reduce_sum(
        tf.square(z_mean) + tf.exp(z_logvar) - z_logvar - 1, [1]),name="kl_loss")

  def model_fn(self, features, labels, mode, params):
      TPUEstimator compatible model function.
      del labels
      is_training = (mode == tf.estimator.ModeKeys.TRAIN)
      data_shape = features.get_shape().as_list()[1:]
      data_dims = data_shape[1]
      fs = [data_dims, data_dims // 2, data_dims // 4, data_dims // 8,
             data_dims // 16]
      z_mean, z_logvar = h_encoder(features, self.h_latent, self.cs, is_training=is_training)
      z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)
      reconstructions = h_decoder(z_sampled, self.cs, fs, self.h_latent,is_training=is_training)
      per_sample_loss = losses.make_reconstruction_loss(features, reconstructions)
      reconstruction_loss = tf.reduce_mean(per_sample_loss)
      kl_loss = self.compute_gaussian_kl(z_mean, z_logvar)
      regularizer = self.regularizer(kl_loss, z_mean, z_logvar, z_sampled)
      loss = tf.add(reconstruction_loss, regularizer, name="loss")
      elbo = tf.add(reconstruction_loss, kl_loss, name="elbo")
      if mode == tf.estimator.ModeKeys.TRAIN:
          optimizer = optimizers.make_vae_optimizer()
          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
          train_op = optimizer.minimize(
              loss=loss, global_step=tf.train.get_global_step())
          train_op = tf.group([train_op, update_ops])
          tf.summary.scalar("reconstruction_loss", reconstruction_loss)
          tf.summary.scalar("elbo", -elbo)

          logging_hook = tf.train.LoggingTensorHook({
              "loss": loss,
              "reconstruction_loss": reconstruction_loss,
              "elbo": -elbo
          },
              every_n_iter=100)
          return tf.contrib.tpu.TPUEstimatorSpec(
              mode=mode,
              loss=loss,
              train_op=train_op,
              training_hooks=[logging_hook])
      elif mode == tf.estimator.ModeKeys.EVAL:
          return tf.contrib.tpu.TPUEstimatorSpec(
              mode=mode,
              loss=loss,
              eval_metrics=(make_metric_fn("reconstruction_loss", "elbo",
                                           "regularizer", "kl_loss"),
                            [reconstruction_loss, -elbo, regularizer, kl_loss]))
      else:
          raise NotImplementedError("Eval mode not supported.")

  def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    tc = (self.beta - 1.) * total_correlation(z_sampled, z_mean, z_logvar)
    return tc + kl_loss

#hbetatcvae
gin_bindings = [
    "dataset.name = '{}'".format(DATASET_NAME),
    "model.model = @h_beta_tc_vae()",
    "h_beta_tc_vae.beta = 4.8",
    "h_beta_tc_vae.ladder0 = 4",
    "h_beta_tc_vae.ladder1 = 3",
    "h_beta_tc_vae.ladder2 = 3"

]
"""
@gin.configurable("DIPTCVAE")
class DIPTCVAE(vae.BaseVAE):
  #BetaTCVAE model.

  def __init__(self, beta=gin.REQUIRED,
               lambda_od=gin.REQUIRED,
               lambda_d_factor=gin.REQUIRED,
               dip_type=gin.REQUIRED):
    self.beta = beta
    self.lambda_od = lambda_od
    self.lambda_d_factor = lambda_d_factor
    self.dip_type = dip_type
  def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    tc = (self.beta - 1.) * total_correlation(z_sampled, z_mean, z_logvar)
    cov_z_mean = compute_covariance_z_mean(z_mean)
    lambda_d = self.lambda_d_factor * self.lambda_od
    if self.dip_type == "i":
        cov_dip_regularizer = regularize_diag_off_diag_dip(
            cov_z_mean, self.lambda_od, lambda_d)
    elif self.dip_type == "ii":
        cov_enc = tf.matrix_diag(tf.exp(z_logvar))
        expectation_cov_enc = tf.reduce_mean(cov_enc, axis=0)
        cov_z = expectation_cov_enc + cov_z_mean
        cov_dip_regularizer = regularize_diag_off_diag_dip(
            cov_z, self.lambda_od, lambda_d)
    else:
        raise NotImplementedError("DIP variant not supported.")
    return kl_loss + (tc  + cov_dip_regularizer)/2
#diptc i
# gin_bindings = [
#     "dataset.name = '{}'".format(DATASET_NAME),
#     "model.model = @DIPTCVAE()",
#     "DIPTCVAE.beta = 6.",
#     "DIPTCVAE.lambda_od = 2.",
#     "DIPTCVAE.lambda_d_factor = 20.",
#     "DIPTCVAE.dip_type='i'"
# ]
#diptc ii
# gin_bindings = [
#     "dataset.name = '{}'".format(DATASET_NAME),
#     "model.model = @DIPTCVAE()",
#     "DIPTCVAE.beta = 6.",
#     "DIPTCVAE.lambda_od = 2.",
#     "DIPTCVAE.lambda_d_factor = 20.",
#     "DIPTCVAE.dip_type='ii'"
# ]
# #factor vae
gin_bindings = [
     "dataset.name = '{}'".format(DATASET_NAME),
     "model.model = @factor_vae()",
     "factor_vae.gamma = 6.4"
]
#tcvae
# gin_bindings = [
#     "dataset.name = '{}'".format(DATASET_NAME),
#     "model.model = @beta_tc_vae()",
#     "beta_tc_vae.beta = 4.8"
# ]
#L0BetaTCVAE
# gin_bindings = [
#     "dataset.name = '{}'".format(DATASET_NAME),
#     "model.model = @L0BetaTCVAE()",
#     "L0BetaTCVAE.beta = 4.0"
# ]

# #dipvae i
'''
gin_bindings = [
     "dataset.name = '{}'".format(DATASET_NAME),
     "model.model = @dip_vae()",
     "dip_vae.lambda_od = 2",
     "dip_vae.lambda_d_factor = 20",
     "dip_vae.dip_type = 'i'"
] '''

#dipvae ii
# gin_bindings = [
#     "dataset.name = '{}'".format(DATASET_NAME),
#     "model.model = @dip_vae()",
#     "dip_vae.lambda_od = 2",
#     "dip_vae.lambda_d_factor = 20",
#     "dip_vae.dip_type = 'ii'"
# ]

# Call training module to train the custom model.
experiment_output_path = os.path.join(base_path, experiment_name)

########################################################################
# Register Progress (start of training)
# @gin.configurable("export_as_tf_hub", whitelist=[])
# def export_as_tf_hub(gaussian_encoder_model,
#                      observation_shape,
#                      checkpoint_path,
#                      export_path,
#                      drop_collections=None):
#   Exports the provided GaussianEncoderModel as a TFHub module.
#
#   Args:
#     gaussian_encoder_model: GaussianEncoderModel to be exported.
#     observation_shape: Tuple with the observations shape.
#     checkpoint_path: String with path where to load weights from.
#     export_path: String with path where to save the TFHub module to.
#     drop_collections: List of collections to drop from the graph.
#   
#
#   def module_fn(is_training):
#     """Module function used for TFHub export."""
#     with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
#       # Add a signature for the Gaussian encoder.
#       image_placeholder = tf.placeholder(
#           dtype=tf.float32, shape=[None] + observation_shape)
#       mean, logvar, reg, mask = gaussian_encoder_model.gaussian_encoder(
#           image_placeholder, is_training)
#       hub.add_signature(
#           name="gaussian_encoder",
#           inputs={"images": image_placeholder},
#           outputs={
#               "mean": mean,
#               "logvar": logvar
#           })
#
#       # Add a signature for reconstructions.
#       latent_vector = gaussian_encoder_model.sample_from_latent_distribution(
#           mean, logvar)
#       reconstructed_images = gaussian_encoder_model.decode(
#           latent_vector, observation_shape, is_training)
#       hub.add_signature(
#           name="reconstructions",
#           inputs={"images": image_placeholder},
#           outputs={"images": reconstructed_images})
#
#       # Add a signature for the decoder.
#       latent_placeholder = tf.placeholder(
#           dtype=tf.float32, shape=[None, mean.get_shape()[1]])
#       decoded_images = gaussian_encoder_model.decode(latent_placeholder,
#                                                      observation_shape,
#                                                      is_training)
#
#       hub.add_signature(
#           name="decoder",
#           inputs={"latent_vectors": latent_placeholder},
#           outputs={"images": decoded_images})
#
#   # Export the module.
#   # Two versions of the model are exported:
#   #   - one for "test" mode (the default tag)
#   #   - one for "training" mode ("is_training" tag)
#   # In the case that the encoder/decoder have dropout, or BN layers, these two
#   # graphs are different.
#   tags_and_args = [
#       ({"train"}, {"is_training": True}),
#       (set(), {"is_training": False}),
#   ]
#   spec = hub.create_module_spec(module_fn, tags_and_args=tags_and_args,
#                                 drop_collections=drop_collections)
#   spec.export(export_path, checkpoint_path=checkpoint_path)
########################################################################
aicrowd_helpers.register_progress(0.0)
start_time = time.time()
train.train_with_gin(
    os.path.join(experiment_output_path, "model"), overwrite,
    [get_full_path("model.gin")], gin_bindings)
# path=os.path.join(experiment_output_path, str(time.time()))
# train.train_with_gin(
#     path, overwrite,
#     [get_full_path("model.gin")], gin_bindings)
elapsed_time = time.time() - start_time
print("##################################Elapsed TIME##############################")
print(elapsed_time)
print("##################################Elapsed TIME##############################")
########################################################################
# Register Progress (end of training, start of representation extraction)
########################################################################
aicrowd_helpers.register_progress(0.90)

# Extract the mean representation for both of these models.
representation_path = os.path.join(experiment_output_path, "representation")
model_path = os.path.join(experiment_output_path, "model")
# model_path =path
# representation_path=path
# This contains the settings:
postprocess_gin = [get_full_path("postprocess.gin")]
postprocess.postprocess_with_gin(model_path, representation_path, overwrite,
                                 postprocess_gin)

print("Written output to : ", experiment_output_path)

########################################################################
# Register Progress (of representation extraction)
########################################################################
aicrowd_helpers.register_progress(1.0)

########################################################################
# Submit Results for evaluation
########################################################################
cuda.close() 
aicrowd_helpers.submit()

