# -*- coding: utf-8 -*-
"""Convolutional MoE layers. author: Emin Orhan
based on the the standard convolutional layers in Keras.
https://raw.githubusercontent.com/eminorhan/mixture-of-experts/master/ConvolutionalMoE.py
"""
from keras.engine.topology import InputSpec
from ._conv import _ConvMoE


class Conv3DMoE(_ConvMoE):
    """3D convolution layer (e.g. spatial convolution over volumes).

    # Input shape
        5D tensor with shape:
        `(samples, channels, conv_dim1, conv_dim2, conv_dim3)`
        if `data_format` is `"channels_first"`
        or 5D tensor with shape:
        `(samples, conv_dim1, conv_dim2, conv_dim3, channels)`
        if `data_format` is `"channels_last"`.

    # Output shape
        5D tensor with shape:
        `(samples, n_filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)`
        if `data_format` is `"channels_first"`
        or 5D tensor with shape:
        `(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, n_filters)`
        if `data_format` is `"channels_last"`.
        `new_conv_dim1`, `new_conv_dim2` and `new_conv_dim3` values might have changed due to padding.
    """

    def __init__(self,
                 n_filters,
                 n_experts_per_filter,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=(1,1,1),
                 expert_activation=None,
                 gating_activation=None,
                 use_expert_bias=True,
                 use_gating_bias=True,
                 expert_kernel_initializer_scale=1.0,
                 gating_kernel_initializer_scale=1.0,
                 expert_bias_initializer='zeros',
                 gating_bias_initializer='zeros',
                 expert_kernel_regularizer=None,
                 gating_kernel_regularizer=None,
                 expert_bias_regularizer=None,
                 gating_bias_regularizer=None,
                 expert_kernel_constraint=None,
                 gating_kernel_constraint=None,
                 expert_bias_constraint=None,
                 gating_bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        super(Conv3DMoE, self).__init__(
            rank=3,
            n_filters=n_filters,
            n_experts_per_filter=n_experts_per_filter,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            expert_activation=expert_activation,
            gating_activation=gating_activation,
            use_expert_bias=use_expert_bias,
            use_gating_bias=use_gating_bias,
            expert_kernel_initializer_scale=expert_kernel_initializer_scale,
            gating_kernel_initializer_scale=gating_kernel_initializer_scale,
            expert_bias_initializer=expert_bias_initializer,
            gating_bias_initializer=gating_bias_initializer,
            expert_kernel_regularizer=expert_kernel_regularizer,
            gating_kernel_regularizer=gating_kernel_regularizer,
            expert_bias_regularizer=expert_bias_regularizer,
            gating_bias_regularizer=gating_bias_regularizer,
            expert_kernel_constraint=expert_kernel_constraint,
            gating_kernel_constraint=gating_kernel_constraint,
            expert_bias_constraint=expert_bias_constraint,
            gating_bias_constraint=gating_bias_constraint,
            activity_regularizer=activity_regularizer,
            **kwargs)
        self.input_spec = InputSpec(ndim=5)

    def get_config(self):
        config = super(Conv3DMoE, self).get_config()
        config.pop('rank')
        return config


# Aliases
Convolution3DMoE = Conv3DMoE
