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


# We group all the imports at the top.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.methods.unsupervised import vae
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results
import tensorflow as tf
from numba import cuda
import gin.tf
import aicrowd_helpers
import math

# 0. Settings
# ------------------------------------------------------------------------------
# By default, we save all the results in subdirectories of the following path.
base_path = os.getenv("AICROWD_OUTPUT_PATH", "../scratch/shared")
experiment_name = os.getenv("AICROWD_EVALUATION_NAME", "experiment_name")
DATASET_NAME = os.getenv("AICROWD_DATASET_NAME", "cars3d")
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
@gin.configurable("BetaTCVAE")
class BetaTCVAE(vae.BaseVAE):
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
#     "BetaTCVAE.beta = 15."
# ]
gin_bindings = [
    "dataset.name = '{}'".format(DATASET_NAME),
    "model.model = @DIPVAE()",
    "DIPVAE.lambda_od = 2.",
    "DIPVAE.lambda_d_factor = 20."
]
# Call training module to train the custom model.
experiment_output_path = os.path.join(base_path, experiment_name)

########################################################################
# Register Progress (start of training)
########################################################################
aicrowd_helpers.register_progress(0.0)

train.train_with_gin(
    os.path.join(experiment_output_path, "model"), overwrite,
    [get_full_path("model.gin")], gin_bindings)

########################################################################
# Register Progress (end of training, start of representation extraction)
########################################################################
aicrowd_helpers.register_progress(0.90)

# Extract the mean representation for both of these models.
representation_path = os.path.join(experiment_output_path, "representation")
model_path = os.path.join(experiment_output_path, "model")
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
