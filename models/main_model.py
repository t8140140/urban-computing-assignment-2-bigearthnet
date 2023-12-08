# -*- coding: utf-8 -*-
#
# Main model to define initial placeholders, feed dictionary, and image normalization. 
#
# Author: Gencer Sumbul, http://www.user.tu-berlin.de/gencersumbul/
# Email: gencer.suembuel@tu-berlin.de
# Date: 27 Feb 2021
# Version: 1.1.1

import numpy as np
import tensorflow as tf
from nets.resnet_utils import resnet_arg_scope
from nets.resnet_v1 import resnet_v1_50
from BigEarthNet import BAND_STATS
from utils import sparse_to_dense

class Model:
    def __init__(self, label_type, modality):
        self.label_type = label_type
        self.modality = modality
        self.prediction_threshold = 0.5
        self.is_training = tf.placeholder(tf.bool, [])
        self.nb_class = 19 if label_type == 'BigEarthNet-19' else 43
        self.B01 = tf.placeholder(tf.float32, [None, 20, 20], name='B01')
        self.B02 = tf.placeholder(tf.float32, [None, 120, 120], name='B02')
        self.B03 = tf.placeholder(tf.float32, [None, 120, 120], name='B03')
        self.B04 = tf.placeholder(tf.float32, [None, 120, 120], name='B04')
        self.B05 = tf.placeholder(tf.float32, [None, 60, 60], name='B05')
        self.B06 = tf.placeholder(tf.float32, [None, 60, 60], name='B06')
        self.B07 = tf.placeholder(tf.float32, [None, 60, 60], name='B07')
        self.B08 = tf.placeholder(tf.float32, [None, 120, 120], name='B08')
        self.B8A = tf.placeholder(tf.float32, [None, 60, 60], name='B8A')
        self.B09 = tf.placeholder(tf.float32, [None, 20, 20], name='B09')
        self.B11 = tf.placeholder(tf.float32, [None, 60, 60], name='B11')
        self.B12 = tf.placeholder(tf.float32, [None, 60, 60], name='B12')
        self.VV = tf.placeholder(tf.float32, [None, 120, 120], name='VV')
        self.VH = tf.placeholder(tf.float32, [None, 120, 120], name='VH')
        self.bands_10m = tf.stack([self.B04, self.B03, self.B02, self.B08], axis=3)
        self.bands_20m = tf.stack([self.B05, self.B06, self.B07, self.B8A, self.B11, self.B12], axis=3)
        self.bands_60m = tf.stack([self.B01, self.B09], axis=3)
        self.S2_img = tf.concat(
            [
                self.bands_10m , 
                tf.image.resize_bicubic(
                    self.bands_20m, 
                    [120, 120]
                )
            ], 
            axis=3
        )
        self.S1_img = tf.stack([self.VV, self.VH], axis=3)
        self.S12_img =  tf.concat([self.S2_img, self.S1_img], axis=3)
        
        if self.modality == 'S1':
            self.img = self.S1_img
        elif self.modality == 'S2':
            self.img = self.S2_img
        elif self.modality == 'MM':
            self.img = self.S12_img
        
        self.multi_hot_label = tf.placeholder(tf.float32, shape=(None, self.nb_class))
        self.model_path = tf.placeholder(tf.string)

    def feed_dict(self, batch_dict, is_training=False, model_path=''):
        B01  = ((batch_dict['B01'] - BAND_STATS['S2']['mean']['B01']) / BAND_STATS['S2']['std']['B01']).astype(np.float32)
        B02  = ((batch_dict['B02'] - BAND_STATS['S2']['mean']['B02']) / BAND_STATS['S2']['std']['B02']).astype(np.float32)
        B03  = ((batch_dict['B03'] - BAND_STATS['S2']['mean']['B03']) / BAND_STATS['S2']['std']['B03']).astype(np.float32)
        B04  = ((batch_dict['B04'] - BAND_STATS['S2']['mean']['B04']) / BAND_STATS['S2']['std']['B04']).astype(np.float32)
        B05  = ((batch_dict['B05'] - BAND_STATS['S2']['mean']['B05']) / BAND_STATS['S2']['std']['B05']).astype(np.float32)
        B06  = ((batch_dict['B06'] - BAND_STATS['S2']['mean']['B06']) / BAND_STATS['S2']['std']['B06']).astype(np.float32)
        B07  = ((batch_dict['B07'] - BAND_STATS['S2']['mean']['B07']) / BAND_STATS['S2']['std']['B07']).astype(np.float32)
        B08  = ((batch_dict['B08'] - BAND_STATS['S2']['mean']['B08']) / BAND_STATS['S2']['std']['B08']).astype(np.float32)
        B8A  = ((batch_dict['B8A'] - BAND_STATS['S2']['mean']['B8A']) / BAND_STATS['S2']['std']['B8A']).astype(np.float32)
        B09  = ((batch_dict['B09'] - BAND_STATS['S2']['mean']['B09']) / BAND_STATS['S2']['std']['B09']).astype(np.float32)
        B11  = ((batch_dict['B11'] - BAND_STATS['S2']['mean']['B11']) / BAND_STATS['S2']['std']['B11']).astype(np.float32)
        B12  = ((batch_dict['B12'] - BAND_STATS['S2']['mean']['B12']) / BAND_STATS['S2']['std']['B12']).astype(np.float32)
        VV = ((batch_dict['VV'] - BAND_STATS['S1']['mean']['VV']) / BAND_STATS['S1']['std']['VV']).astype(np.float32)
        VH = ((batch_dict['VH'] - BAND_STATS['S1']['mean']['VH']) / BAND_STATS['S1']['std']['VH']).astype(np.float32)
        multi_hot_label = batch_dict[
                'original_labels_multi_hot'
            ].astype(np.float32) if self.label_type == 'original' else batch_dict[
                'BigEarthNet-19_labels_multi_hot'
            ].astype(np.float32)

        # Label and patch names can be read in the following way:
        #
        # original_labels = sparse_to_dense(batch_dict['original_labels'].indices, batch_dict['original_labels'].values)
        # BigEarthNet-19_labels = sparse_to_dense(batch_dict['BigEarthNet-19_labels'].indices, batch_dict['BigEarthNet-19_labels'].values)
        # patch_name_s1 = sparse_to_dense(batch_dict['patch_name_s1'].indices, batch_dict['patch_name_s1'].values)
        # patch_name_s2 = sparse_to_dense(batch_dict['patch_name_s2'].indices, batch_dict['patch_name_s2'].values)
        
        return {
                self.B01: B01,
                self.B02: B02,
                self.B03: B03,
                self.B04: B04,
                self.B05: B05,
                self.B06: B06,
                self.B07: B07,
                self.B08: B08,
                self.B8A: B8A,
                self.B09: B09,
                self.B11: B11,
                self.B12: B12,
                self.VV: VV,
                self.VH: VH,
                self.multi_hot_label: multi_hot_label, 
                self.is_training:is_training,
                self.model_path:model_path
            }

    def define_loss(self):
        self.loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.multi_hot_label, logits=self.logits)
        tf.summary.scalar('sigmoid_cross_entropy_loss', self.loss)
        return self.loss