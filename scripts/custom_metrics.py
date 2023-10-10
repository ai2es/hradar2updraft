import tensorflow as tf 
import tensorflow.keras.backend as K
import numpy as np
import tensorflow_probability as tfp

##############################################################################################################
########################################### Classification ###################################################
##############################################################################################################

class MaxCriticalSuccessIndex(tf.keras.metrics.Metric):
    """ 
    Calcualte the element-wise [e.g., pixel-wise],
    maximum Critical Success Index out of 20 thresholds.
    If update_state is called more than once, it will add give you the new max 
    csi over all calls. In other words, it accumulates the TruePositives, 
    FalsePositives and FalseNegatives. 
    
    If you want to use it on a fresh y_true,y_pred, make sure you run:

    metric.reset_state()

    then 

    metric(y_true,y_pred) #<-- this will also store the counts of TP etc.

    The shapes of y_true and y_pred should match 
    
    """ 

    def __init__(self, name="max_csi",
                scope=None,
                thresholds=np.arange(0.05,1.05,0.05).tolist(),
                 **kwargs):
        super(MaxCriticalSuccessIndex, self).__init__(name=name, **kwargs)

        #initialize csi value, if no data given, it will be 0 
        self.csi = self.add_weight(name="csi", initializer="zeros")

        if scope is None:
            tp = tf.keras.metrics.TruePositives(thresholds=thresholds)
            fp = tf.keras.metrics.FalsePositives(thresholds=thresholds)
            fn = tf.keras.metrics.FalseNegatives(thresholds=thresholds)
        else:
            with scope:
                tp = tf.keras.metrics.TruePositives(thresholds=thresholds)
                fp = tf.keras.metrics.FalsePositives(thresholds=thresholds)
                fn = tf.keras.metrics.FalseNegatives(thresholds=thresholds)

        #store defined metric functions
        self.tp = tp 
        self.fp = fp 
        self.fn = fn
        #flush functions just in case 
        self.tp.reset_state()
        self.fp.reset_state()
        self.fn.reset_state()

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        if (len(y_true.shape[1:]) > 2 ) and (y_true.shape[-1] == 2):
            #convert back to 1 map 
            y_true = tf.where(y_true[:,:,:,1]>0,1,0)
            #ypred[:,:,:,0] = 1 - y_pred[:,:,:,1]
            y_pred = y_pred[:,:,:,1]
            #ravel for pixelwise comparison
            y_true = tf.experimental.numpy.ravel(y_true)
            y_pred = tf.experimental.numpy.ravel(y_pred)
        #if the output is a map (batch,nx,ny,nl) ravel it
        elif (len(y_true.shape[1:]) > 2):
            y_true = tf.experimental.numpy.ravel(y_true)
            y_pred = tf.experimental.numpy.ravel(y_pred)
        

        #call vectorized stats metrics, add them to running amount of each
        self.tp.update_state(y_true,y_pred)
        self.fp.update_state(y_true,y_pred)
        self.fn.update_state(y_true,y_pred)

        #calc current max csi (so we can get updates per batch)
        self.csi_val = tf.reduce_max(self.tp.result()/(self.tp.result() + self.fn.result() + self.fp.result()))

        #assign the value to the csi 'weight'
        self.csi.assign(self.csi_val)
      
    def result(self):
        return self.csi

    def reset_state(self):
        # Reset the counts 
        self.csi.assign(0.0)
        self.tp.reset_state()
        self.fp.reset_state()
        self.fn.reset_state()

##############################################################################################################
##############################################################################################################
##############################################################################################################


##############################################################################################################
############################################# Regression #####################################################
##############################################################################################################

class MeanError(tf.keras.metrics.Metric):
    """ 
    Calcualte the element-wise [e.g., pixel-wise] bias (i.e., error)
    
    For alot of meteorology problems, there is often alot of 0 pixels. This biases
    the normal metrics, and makes the values in-coherent. One way around that is to 
    calcualte the metric on only the non-zero truth pixels. 
    
    If you want to use it on a fresh y_true,y_pred, make sure you run:

    metric.reset_state()

    then 

    metric(y_true,y_pred)

    The shapes of y_true and y_pred should match!
    
    """ 

    def __init__(self, name="me",
                 **kwargs):
        super(MeanError, self).__init__(name=name, **kwargs)

        #initialize cme value, if no data given, it will be 0 
        self.me = self.add_weight(name="me", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        #ravel to 1-d tensor, makes grabing just the non-zeros easier
        y_true_flat = tf.experimental.numpy.ravel(y_true)
        y_pred_flat = tf.experimental.numpy.ravel(y_pred)

        #assign the value to the conditional mean 
        self.me.assign(tf.math.reduce_mean(tf.math.subtract(y_true_flat,y_pred_flat)))
      
    def result(self):
        return self.me

    def reset_state(self):
        # Reset the counts 
        self.me.assign(0.0)

class ConditionalMeanError(tf.keras.metrics.Metric):
    """ 
    Calcualte the element-wise [e.g., pixel-wise] conditional bias (i.e., error)
    
    For alot of meteorology problems, there is often alot of 0 pixels. This biases
    the normal metrics, and makes the values in-coherent. One way around that is to 
    calcualte the metric on only the non-zero truth pixels. 
    
    If you want to use it on a fresh y_true,y_pred, make sure you run:

    metric.reset_state()

    then 

    metric(y_true,y_pred)

    The shapes of y_true and y_pred should match!
    
    """ 

    def __init__(self, name="cme",
                 **kwargs):
        super(ConditionalMeanError, self).__init__(name=name, **kwargs)

        #initialize cme value, if no data given, it will be 0 
        self.cme = self.add_weight(name="cme", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        #ravel to 1-d tensor, makes grabing just the non-zeros easier
        y_true_flat = tf.experimental.numpy.ravel(y_true)
        y_pred_flat = tf.experimental.numpy.ravel(y_pred)

        #find non-zero flash locations 
        mask = tf.where(y_true_flat>0.0)
        #check to make sure there is at least 1 flash
        if tf.equal(tf.size(mask), 0):
            pass
        else:
            y_true_flat = tf.gather(y_true_flat,indices=mask)
            y_pred_flat = tf.gather(y_pred_flat,indices=mask)

            #assign the value to the conditional mean 
            self.cme.assign(tf.math.reduce_mean(tf.math.subtract(y_true_flat,y_pred_flat)))
      
    def result(self):
        return self.cme

    def reset_state(self):
        # Reset the counts 
        self.cme.assign(0.0)

class ConditionalMeanAbsoluteError(tf.keras.metrics.Metric):
    """ 
    Calcualte the element-wise [e.g., pixel-wise] conditional absolute mean error.
    
    For alot of meteorology problems, there is often alot of 0 pixels. This biases
    the normal metrics, and makes the values in-coherent. One way around that is to 
    calcualte the metric on only the non-zero truth pixels. 
    
    If you want to use it on a fresh y_true,y_pred, make sure you run:

    metric.reset_state()

    then 

    metric(y_true,y_pred)

    The shapes of y_true and y_pred should match!
    
    """ 

    def __init__(self, name="cmae",
                 mae = tf.keras.metrics.MeanAbsoluteError(),
                 **kwargs):
        super(ConditionalMeanAbsoluteError, self).__init__(name=name, **kwargs)

        #initialize csi value, if no data given, it will be 0 
        self.cmae = self.add_weight(name="cmae", initializer="zeros")
        #store defined metric functions
        self.mae = mae 
        #flush metric
        self.mae.reset_state()

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        #ravel to 1-d tensor, makes grabing just the non-zeros easier
        y_true_flat = tf.experimental.numpy.ravel(y_true)
        y_pred_flat = tf.experimental.numpy.ravel(y_pred)

        #find non-zero flash locations 
        mask = tf.where(y_true_flat>0.0)
        #check to make sure there is at least 1 flash
        if tf.equal(tf.size(mask), 0):
            pass
        else:
            y_true_flat = tf.gather(y_true_flat,indices=mask)
            y_pred_flat = tf.gather(y_pred_flat,indices=mask)

            #calc mae on that new vector 
            self.mae.update_state(y_true_flat,y_pred_flat)

        #assign the value to the conditional mean 
        self.cmae.assign(self.mae.result())
      
    def result(self):
        return self.cmae

    def reset_state(self):
        # Reset the counts 
        self.cmae.assign(0.0)
        self.mae.reset_state()
        
class ConditionalRootMeanSquaredError(tf.keras.metrics.Metric):
    """ 
    Calcualte the element-wise [e.g., pixel-wise] conditional root mean square error.
    
    For alot of meteorology problems, there is often alot of 0 pixels. This biases
    the normal metrics, and makes the values in-coherent. One way around that is to 
    calcualte the metric on only the non-zero truth pixels. 
    
    If you want to use it on a fresh y_true,y_pred, make sure you run:

    metric.reset_state()

    then 

    metric(y_true,y_pred)

    The shapes of y_true and y_pred should match!
    
    """ 

    def __init__(self, name="crmse",
                 rmse = tf.keras.metrics.RootMeanSquaredError(),
                 **kwargs):
        super(ConditionalRootMeanSquaredError, self).__init__(name=name, **kwargs)

        #initialize csi value, if no data given, it will be 0 
        self.crmse = self.add_weight(name="crmse", initializer="zeros")

        #store defined metric functions
        self.rmse = rmse 
        self.rmse.reset_state()

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        #ravel to 1-d tensor 
        y_true_flat = tf.experimental.numpy.ravel(y_true)
        y_pred_flat = tf.experimental.numpy.ravel(y_pred)
          
        #find non-zero flash locations 
        mask = tf.where(y_true_flat>0.0)
        #check to make sure there is at least 1 flash
        if tf.equal(tf.size(mask), 0):
            pass
        else:
            y_true_flat = tf.gather(y_true_flat,indices=mask)
            y_pred_flat = tf.gather(y_pred_flat,indices=mask)

            #calc rmse on that new vector 
            self.rmse.update_state(y_true_flat,y_pred_flat)
        
        #assign the value to the conditional mean 
        self.crmse.assign(self.rmse.result())
      
    def result(self):
        return self.crmse

    def reset_state(self):
        # Reset the counts 
        self.crmse.assign(0.0)
        self.rmse.reset_state()

class ImageRootMeanSquaredError(tf.keras.metrics.Metric):
    """ 
    Calcualte the image total root mean square error. 

    Does the predicted image produce the same number of flashes?

    This function expects a 2d image prediction. [need to add exit if not the case]
    """ 

    def __init__(self, name="irmse",
                 rmse = tf.keras.metrics.RootMeanSquaredError(),
                 **kwargs):
        super(ImageRootMeanSquaredError, self).__init__(name=name, **kwargs)

        #initialize csi value, if no data given, it will be 0 
        self.irmse = self.add_weight(name="irmse", initializer="zeros")
        #store defined metric functions
        self.rmse = rmse 
        self.rmse.reset_state()

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        #Get sum across image 
        y_true_count = tf.math.reduce_sum(y_true,(1,2,3))
        y_pred_count = tf.math.reduce_sum(y_pred,(1,2,3))

        self.y_true_count = y_true_count
        self.y_pred_count = y_pred_count
        #calc mae on that new vector 
        self.rmse.update_state(y_true_count,y_pred_count)
        
        #assign the value to the conditional mean 
        self.irmse.assign(self.rmse.result())
      
    def result(self):
        return self.irmse

    def reset_state(self):
        # Reset the counts 
        self.irmse.assign(0.0)
        self.rmse.reset_state()

class ParaRootMeanSquaredError(tf.keras.metrics.Metric):
    """ 
    Calcualte the element-wise [e.g., pixel-wise] conditional root mean square error.
    
    For alot of meteorology problems, there is often alot of 0 pixels. This biases
    the normal metrics, and makes the values in-coherent. One way around that is to 
    calcualte the metric on only the non-zero truth pixels. 
    
    If you want to use it on a fresh y_true,y_pred, make sure you run:

    metric.reset_state()

    then 

    metric(y_true,y_pred)

    The shapes of y_true and y_pred should match!
    
    """ 

    def __init__(self, name="prmse",
                 scope=None,
                 **kwargs):
        super(ParaRootMeanSquaredError, self).__init__(name=name, **kwargs)

        #initialize csi value, if no data given, it will be 0 
        self.prmse = self.add_weight(name="prmse", initializer="zeros")

        if scope is None:
            rmse = tf.keras.metrics.RootMeanSquaredError()
        else:
            with scope:
                rmse = tf.keras.metrics.RootMeanSquaredError()

        #store defined metric functions
        self.rmse = rmse 
        self.rmse.reset_state()

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_pred = tf.cast(y_pred[:, :,0], tf.float64)
        y_true = tf.cast(y_true[:, :,0], tf.float64)
        
          
        #calc rmse on that new vector 
        self.rmse.update_state(y_true,y_pred)
        
        #assign the value to the conditional mean 
        self.prmse.assign(self.rmse.result())
      
    def result(self):
        return self.prmse

    def reset_state(self):
        # Reset the counts 
        self.prmse.assign(0.0)
        self.rmse.reset_state()


class ParaRootMeanSquaredError2(tf.keras.metrics.Metric):
    """ 
    Calcualte the element-wise [e.g., pixel-wise] conditional root mean square error for the 4-parameter dist.
    
    If you want to use it on a fresh y_true,y_pred, make sure you run:

    metric.reset_state()

    then 

    metric(y_true,y_pred)

    The shapes of y_true and y_pred should match!
    
    """ 

    def __init__(self, name="prmse",
                 scope=None,
                 **kwargs):
        super(ParaRootMeanSquaredError2, self).__init__(name=name, **kwargs)

        #initialize csi value, if no data given, it will be 0 
        self.prmse = self.add_weight(name="prmse", initializer="zeros")

        #store defined metric functions
        if scope is None:
            rmse = tf.keras.metrics.RootMeanSquaredError()
        else:
            with scope:
                rmse = tf.keras.metrics.RootMeanSquaredError()
                
        self.rmse = rmse 
        self.rmse.reset_state()

    def update_state(self, y_true, y_pred, sample_weight=None):

        #Chase adaptation to get 50th percentile 
        y_pred = tf.cast(y_pred, tf.float64)
        y_true = tf.cast(y_true[...,0], tf.float64)

        root_power = tf.constant(1.,tf.float64)/tf.math.multiply(tf.constant(10.,tf.float64),tf.cast(tf.math.exp(1.),tf.float64))
        mu = y_pred[...,0] 
        sigma = tf.math.pow(tf.math.exp(y_pred[...,1]),root_power)
        gamma = y_pred[..., 2] 
        tau = tf.math.pow(tf.math.exp(y_pred[...,3]),root_power)
        cond_dist = tfp.distributions.SinhArcsinh(mu, sigma,skewness=gamma,tailweight=tau)
        #use if normal, should make this a flag?
        # cond_dist = tfp.distributions.TruncatedNormal(loc=mu,scale=sigma,low=0,high=75)

        #over write y_pred with 50th percentile 
        y_pred = cond_dist.quantile(0.5)
        
        #calc rmse on that new vector 
        self.rmse.update_state(y_true,y_pred)
        
        #assign the value to the conditional mean 
        self.prmse.assign(self.rmse.result())
      
    def result(self):
        return self.prmse

    def reset_state(self):
        # Reset the counts 
        self.prmse.assign(0.0)
        self.rmse.reset_state()

        
##############################################################################################################
##############################################################################################################
##############################################################################################################

#TENSORFLOW ADDON ADAPTATION 

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Implements R^2 scores."""
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric
from tensorflow.python.ops import weights_broadcast_ops

from typeguard import typechecked
from tensorflow_addons.utils.types import AcceptableDTypes


_VALID_MULTIOUTPUT = {"raw_values", "uniform_average", "variance_weighted"}


def _reduce_average(
    input_tensor: tf.Tensor, axis=None, keepdims=False, weights=None
) -> tf.Tensor:
    """Computes the (weighted) mean of elements across dimensions of a tensor."""
    if weights is None:
        return tf.reduce_mean(input_tensor, axis=axis, keepdims=keepdims)

    weighted_sum = tf.reduce_sum(weights * input_tensor, axis=axis, keepdims=keepdims)
    sum_of_weights = tf.reduce_sum(weights, axis=axis, keepdims=keepdims)
    average = weighted_sum / sum_of_weights
    return average


# @tf.keras.utils.register_keras_serializable(package="Addons")
class RSquare_Adapted(Metric):
    """Compute R^2 score.

    This is also called the [coefficient of determination
    ](https://en.wikipedia.org/wiki/Coefficient_of_determination).
    It tells how close are data to the fitted regression line.

    - Highest score can be 1.0 and it indicates that the predictors
        perfectly accounts for variation in the target.
    - Score 0.0 indicates that the predictors do not
        account for variation in the target.
    - It can also be negative if the model is worse.

    The sample weighting for this metric implementation mimics the
    behaviour of the [scikit-learn implementation
    ](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)
    of the same metric.

    Can also calculate the Adjusted R2 Score.

    Args:
        multioutput: `string`, the reduce method for scores.
            Should be one of `["raw_values", "uniform_average", "variance_weighted"]`.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        num_regressors: (Optional) Number of indepedent regressors used (Adjusted R2).
            Defaults to zero(standard R2 score).

    Usage:

    >>> y_true = np.array([1, 4, 3], dtype=np.float32)
    >>> y_pred = np.array([2, 4, 4], dtype=np.float32)
    >>> metric = tfa.metrics.r_square.RSquare()
    >>> metric.update_state(y_true, y_pred)
    >>> result = metric.result()
    >>> result.numpy()
    0.57142854
    """

    @typechecked
    def __init__(
        self,
        name: str = "r_square",
        dtype: AcceptableDTypes = None,
        multioutput: str = "uniform_average",
        num_regressors: tf.int32 = 0,
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)

        if "y_shape" in kwargs:
            warnings.warn(
                "y_shape has been removed, because it's automatically derived,"
                "and will be deprecated in Addons 0.18.",
                DeprecationWarning,
            )

        if multioutput not in _VALID_MULTIOUTPUT:
            raise ValueError(
                "The multioutput argument must be one of {}, but was: {}".format(
                    _VALID_MULTIOUTPUT, multioutput
                )
            )
        self.multioutput = multioutput
        self.num_regressors = num_regressors
        self.num_samples = self.add_weight(name="num_samples", dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        
        #Chase adaptation to get 50th percentile 
        y_pred = tf.cast(y_pred, tf.float64)
        y_true = tf.cast(y_true[...,0], tf.float64)
        root_power = tf.constant(1.,tf.float64)/tf.math.multiply(tf.constant(10.,tf.float64),tf.cast(tf.math.exp(1.),tf.float64))
        mu = y_pred[...,0] 
        sigma = tf.math.pow(tf.math.exp(y_pred[...,1]),root_power)
        gamma = y_pred[..., 2] 
        tau = tf.math.pow(tf.math.exp(y_pred[...,3]),root_power)
        cond_dist = tfp.distributions.SinhArcsinh(mu, sigma,skewness=gamma,tailweight=tau)
        
        # cond_dist = tfp.distributions.TruncatedNormal(loc=mu,scale=sigma,low=0,high=75)

        #over write y_pred with 50th percentile 
        y_pred =cond_dist.quantile(0.5)
        
        y_pred = tf.experimental.numpy.ravel(y_pred)
        y_true = tf.experimental.numpy.ravel(y_true)

        if not hasattr(self, "squared_sum"):
            self.squared_sum = self.add_weight(
                name="squared_sum",
                shape=y_true.shape[1:],
                initializer="zeros",
                dtype=self._dtype,
            )
        if not hasattr(self, "sum"):
            self.sum = self.add_weight(
                name="sum",
                shape=y_true.shape[1:],
                initializer="zeros",
                dtype=self._dtype,
            )
        if not hasattr(self, "res"):
            self.res = self.add_weight(
                name="residual",
                shape=y_true.shape[1:],
                initializer="zeros",
                dtype=self._dtype,
            )
        if not hasattr(self, "count"):
            self.count = self.add_weight(
                name="count",
                shape=y_true.shape[1:],
                initializer="zeros",
                dtype=self._dtype,
            )

        y_true = tf.cast(y_true, dtype=self._dtype)
        y_pred = tf.cast(y_pred, dtype=self._dtype)
        if sample_weight is None:
            sample_weight = 1
        sample_weight = tf.cast(sample_weight, dtype=self._dtype)
        sample_weight = weights_broadcast_ops.broadcast_weights(
            weights=sample_weight, values=y_true
        )

        weighted_y_true = y_true * sample_weight
        self.sum.assign_add(tf.reduce_sum(weighted_y_true, axis=0))
        self.squared_sum.assign_add(tf.reduce_sum(y_true * weighted_y_true, axis=0))
        self.res.assign_add(
            tf.reduce_sum((y_true - y_pred) ** 2 * sample_weight, axis=0)
        )
        self.count.assign_add(tf.reduce_sum(sample_weight, axis=0))
        self.num_samples.assign_add(tf.size(y_true))

    def result(self) -> tf.Tensor:
        mean = self.sum / self.count
        total = self.squared_sum - self.sum * mean
        raw_scores = 1 - (self.res / total)
        raw_scores = tf.where(tf.math.is_inf(raw_scores), 0.0, raw_scores)

        if self.multioutput == "raw_values":
            r2_score = raw_scores
        elif self.multioutput == "uniform_average":
            r2_score = tf.reduce_mean(raw_scores)
        elif self.multioutput == "variance_weighted":
            r2_score = _reduce_average(raw_scores, weights=total)
        else:
            raise RuntimeError(
                "The multioutput attribute must be one of {}, but was: {}".format(
                    _VALID_MULTIOUTPUT, self.multioutput
                )
            )

        if self.num_regressors < 0:
            raise ValueError(
                "num_regressors parameter should be greater than or equal to zero"
            )

        if self.num_regressors != 0:
            if self.num_regressors > self.num_samples - 1:
                UserWarning(
                    "More independent predictors than datapoints in adjusted r2 score. Falls back to standard r2 "
                    "score."
                )
            elif self.num_regressors == self.num_samples - 1:
                UserWarning(
                    "Division by zero in adjusted r2 score. Falls back to standard r2 score."
                )
            else:
                n = tf.cast(self.num_samples, dtype=tf.float32)
                p = tf.cast(self.num_regressors, dtype=tf.float32)

                num = tf.multiply(tf.subtract(1.0, r2_score), tf.subtract(n, 1.0))
                den = tf.subtract(tf.subtract(n, p), 1.0)
                r2_score = tf.subtract(1.0, tf.divide(num, den))

        return r2_score

    def reset_state(self) -> None:
        # The state of the metric will be reset at the start of each epoch.
        K.batch_set_value([(v, np.zeros(v.shape)) for v in self.variables])

    def reset_states(self):
        # Backwards compatibility alias of `reset_state`. New classes should
        # only implement `reset_state`.
        # Required in Tensorflow < 2.5.0
        return self.reset_state()

    def get_config(self):
        config = {
            "multioutput": self.multioutput,
        }
        base_config = super().get_config()
        return {**base_config, **config}