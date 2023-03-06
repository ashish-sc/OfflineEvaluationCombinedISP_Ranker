import os
import logging
import dataclasses
import pandas as pd
import numpy as np
from typing import cast, Dict, Optional, Sequence, Tuple, Union, List, Text
import math
import sys
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
import tensorflow_recommenders as tfrs
import json
logging.getLogger('tensorflow').propagate = False

tpu_name = "ashish-hfr2"
tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu_name)
strategy = tf.distribute.TPUStrategy(tpu)

cols = "isp_date,model,userId,postId,unified_signal1,combined_score,video_play,mrp,mrpw,unified_signal,hour,dayofweek,is_weekend,is_morning,is_afternoon,is_evening,is_night,tagId,pvplay_0,pvplay_1,pvplay_2,pvplay_3,pvplay_4,pvplay_5,pvplay_6,pvplay_7,pvplay_8,pvplay_9,pvplay_10,pvplay_11,pvplay_12,pvplay_13,pvplay_14,pvplay_15,pvplay_16,pvplay_17,pvplay_18,pvplay_19,pvplay_20,pvplay_21,pvplay_22,pvplay_23,pvplay_24,pvplay_25,pvplay_26,pvplay_27,pvplay_28,pvplay_29,pvplay_30,pvplay_31,pvplay_mask,pfav_0,pfav_1,pfav_2,pfav_3,pfav_4,pfav_5,pfav_6,pfav_7,pfav_8,pfav_9,pfav_10,pfav_11,pfav_12,pfav_13,pfav_14,pfav_15,pfav_16,pfav_17,pfav_18,pfav_19,pfav_20,pfav_21,pfav_22,pfav_23,pfav_24,pfav_25,pfav_26,pfav_27,pfav_28,pfav_29,pfav_30,pfav_31,pfav_mask,plike_0,plike_1,plike_2,plike_3,plike_4,plike_5,plike_6,plike_7,plike_8,plike_9,plike_10,plike_11,plike_12,plike_13,plike_14,plike_15,plike_16,plike_17,plike_18,plike_19,plike_20,plike_21,plike_22,plike_23,plike_24,plike_25,plike_26,plike_27,plike_28,plike_29,plike_30,plike_31,plike_mask,pshare_0,pshare_1,pshare_2,pshare_3,pshare_4,pshare_5,pshare_6,pshare_7,pshare_8,pshare_9,pshare_10,pshare_11,pshare_12,pshare_13,pshare_14,pshare_15,pshare_16,pshare_17,pshare_18,pshare_19,pshare_20,pshare_21,pshare_22,pshare_23,pshare_24,pshare_25,pshare_26,pshare_27,pshare_28,pshare_29,pshare_30,pshare_31,pshare_mask,postLikeRatio2h,postShareRatio2h,postFavRatio2h,postCommentRatio2h,postSVPRatio2h,postLPORatio2h,postLikeRatio1D,postShareRatio1D,postFavRatio1D,postCommentRatio1D,postSVPRatio1D,postLPORatio1D,pcLikeRatio2h,pcShareRatio2h,pcFavRatio2h,pcCommentRatio2h,pcSVPRatio2h,pcLPORatio2h,pcLikeRatio1D,pcShareRatio1D,pcFavRatio1D,pcCommentRatio1D,pcSVPRatio1D,pcLPORatio1D,userDistrict,uvplay_0,uvplay_1,uvplay_2,uvplay_3,uvplay_4,uvplay_5,uvplay_6,uvplay_7,uvplay_8,uvplay_9,uvplay_10,uvplay_11,uvplay_12,uvplay_13,uvplay_14,uvplay_15,uvplay_16,uvplay_17,uvplay_18,uvplay_19,uvplay_20,uvplay_21,uvplay_22,uvplay_23,uvplay_24,uvplay_25,uvplay_26,uvplay_27,uvplay_28,uvplay_29,uvplay_30,uvplay_31,uvplay_mask,ufav_0,ufav_1,ufav_2,ufav_3,ufav_4,ufav_5,ufav_6,ufav_7,ufav_8,ufav_9,ufav_10,ufav_11,ufav_12,ufav_13,ufav_14,ufav_15,ufav_16,ufav_17,ufav_18,ufav_19,ufav_20,ufav_21,ufav_22,ufav_23,ufav_24,ufav_25,ufav_26,ufav_27,ufav_28,ufav_29,ufav_30,ufav_31,ufav_mask,ulike_0,ulike_1,ulike_2,ulike_3,ulike_4,ulike_5,ulike_6,ulike_7,ulike_8,ulike_9,ulike_10,ulike_11,ulike_12,ulike_13,ulike_14,ulike_15,ulike_16,ulike_17,ulike_18,ulike_19,ulike_20,ulike_21,ulike_22,ulike_23,ulike_24,ulike_25,ulike_26,ulike_27,ulike_28,ulike_29,ulike_30,ulike_31,ulike_mask,ushare_0,ushare_1,ushare_2,ushare_3,ushare_4,ushare_5,ushare_6,ushare_7,ushare_8,ushare_9,ushare_10,ushare_11,ushare_12,ushare_13,ushare_14,ushare_15,ushare_16,ushare_17,ushare_18,ushare_19,ushare_20,ushare_21,ushare_22,ushare_23,ushare_24,ushare_25,ushare_26,ushare_27,ushare_28,ushare_29,ushare_30,ushare_31,ushare_mask,video_affinity,userLikeRatio1,userShareRatio1,userFavRatio1,userCommentsRatio1,userSVPRatio1,userLPORatio1,userLikeRatio7,userShareRatio7,userFavRatio7,userCommentsRatio7,userSVPRatio7,userLPORatio7,upcLikeRatio1D,upcShareRatio1D,upcFavRatio1D,upcCommentRatio1D,upcSVPRatio1D,upcLPORatio1D,upcLikeRatio3D,upcShareRatio3D,upcFavRatio3D,upcCommentRatio3D,upcSVPRatio3D,upcLPORatio3D,upcLikeRatio7D,upcShareRatio7D,upcFavRatio7D,upcCommentRatio7D,upcSVPRatio7D,upcLPORatio7D,engtag_0,engtag_1,engtag_2,engtag_3,engtag_4,engtag_5,engtag_6,engtag_7,engtag_8,engtag_9,engtag_10,engtag_11,engtag_12,engtag_13,engtag_14,engtag_15,engtag_16,engtag_17,engtag_18,engtag_19,engtag_20,engtag_21,engtag_22,engtag_23,engtag_24,engtag_mask_0,engtag_mask_1,engtag_mask_2,engtag_mask_3,engtag_mask_4,engtag_mask_5,engtag_mask_6,engtag_mask_7,engtag_mask_8,engtag_mask_9,engtag_mask_10,engtag_mask_11,engtag_mask_12,engtag_mask_13,engtag_mask_14,engtag_mask_15,engtag_mask_16,engtag_mask_17,engtag_mask_18,engtag_mask_19,engtag_mask_20,engtag_mask_21,engtag_mask_22,engtag_mask_23,engtag_mask_24"
col_names = cols.split(",")

num_labels = 1

hour_feat = 1
dayofweek = 1
num_other_features = 5

sparse_features = [
    'userDistrict',
    'tagId'
]

max_sequence_length = 25

vocab_sizes = {
    'userDistrict': 720,
    'tagId': 315000#4000
}

embedding_dims = {
    'userDistrict': 32,
    'tagId': 32,
}

meta = [
    'isp_date','model','userId','postId','unified_signal1','combined_score','video_play'
]

other_feats = [
#     "time_hour","time_dayofweek","time_is_weekend","time_is_morning","time_is_afternoon","time_is_evening","time_is_night"
    'hour', 'dayofweek', 'is_weekend', 'is_morning', 'is_afternoon', 'is_evening', 'is_night'
]

user_sparse_features = [
    'userDistrict'
]

user_dense_features = [
    "uvplay_0","uvplay_1","uvplay_2","uvplay_3","uvplay_4","uvplay_5","uvplay_6","uvplay_7",
    "uvplay_8","uvplay_9","uvplay_10","uvplay_11","uvplay_12","uvplay_13","uvplay_14","uvplay_15",
    "uvplay_16","uvplay_17","uvplay_18","uvplay_19","uvplay_20","uvplay_21","uvplay_22","uvplay_23",
    "uvplay_24","uvplay_25","uvplay_26","uvplay_27","uvplay_28","uvplay_29","uvplay_30","uvplay_31","uvplay_mask",
    "ufav_0","ufav_1","ufav_2","ufav_3","ufav_4","ufav_5","ufav_6","ufav_7",
    "ufav_8","ufav_9","ufav_10","ufav_11","ufav_12","ufav_13","ufav_14","ufav_15",
    "ufav_16","ufav_17","ufav_18","ufav_19","ufav_20","ufav_21","ufav_22","ufav_23",
    "ufav_24","ufav_25","ufav_26","ufav_27","ufav_28","ufav_29","ufav_30","ufav_31","ufav_mask",
    "ulike_0","ulike_1","ulike_2","ulike_3","ulike_4","ulike_5","ulike_6","ulike_7",
    "ulike_8","ulike_9","ulike_10","ulike_11","ulike_12","ulike_13","ulike_14","ulike_15",
    "ulike_16","ulike_17","ulike_18","ulike_19","ulike_20","ulike_21","ulike_22","ulike_23",
    "ulike_24","ulike_25","ulike_26","ulike_27","ulike_28","ulike_29","ulike_30","ulike_31","ulike_mask",
    "ushare_0","ushare_1","ushare_2","ushare_3","ushare_4","ushare_5","ushare_6","ushare_7",
    "ushare_8","ushare_9","ushare_10","ushare_11","ushare_12","ushare_13","ushare_14","ushare_15",
    "ushare_16","ushare_17","ushare_18","ushare_19","ushare_20","ushare_21","ushare_22","ushare_23",
    "ushare_24","ushare_25","ushare_26","ushare_27","ushare_28","ushare_29","ushare_30","ushare_31","ushare_mask",
    "video_affinity",
    "userLikeRatio1","userShareRatio1","userFavRatio1","userCommentsRatio1","userSVPRatio1","userLPORatio1",
    "userLikeRatio7","userShareRatio7","userFavRatio7","userCommentsRatio7","userSVPRatio7","userLPORatio7",
    "upcLikeRatio1D","upcShareRatio1D","upcFavRatio1D","upcCommentRatio1D","upcSVPRatio1D","upcLPORatio1D",
    "upcLikeRatio3D","upcShareRatio3D","upcFavRatio3D","upcCommentRatio3D","upcSVPRatio3D","upcLPORatio3D",
    "upcLikeRatio7D","upcShareRatio7D","upcFavRatio7D","upcCommentRatio7D","upcSVPRatio7D","upcLPORatio7D"
]

user_engaged_tags = [
    "engtag_0","engtag_1","engtag_2","engtag_3","engtag_4",
    "engtag_5","engtag_6","engtag_7","engtag_8","engtag_9",
    "engtag_10","engtag_11","engtag_12","engtag_13","engtag_14",
    "engtag_15","engtag_16","engtag_17","engtag_18","engtag_19",
    "engtag_20","engtag_21","engtag_22","engtag_23","engtag_24",
    
    "engtag_mask_0","engtag_mask_1","engtag_mask_2","engtag_mask_3","engtag_mask_4",
    "engtag_mask_5","engtag_mask_6","engtag_mask_7","engtag_mask_8","engtag_mask_9",
    "engtag_mask_10","engtag_mask_11","engtag_mask_12","engtag_mask_13","engtag_mask_14",
    "engtag_mask_15","engtag_mask_16","engtag_mask_17","engtag_mask_18","engtag_mask_19",
    "engtag_mask_20","engtag_mask_21","engtag_mask_22","engtag_mask_23","engtag_mask_24"
]

post_sparse_features = [
    'tagId'
#     'sparse_features_tagId'
]
post_dense_features = [
    "pvplay_0","pvplay_1","pvplay_2","pvplay_3","pvplay_4","pvplay_5","pvplay_6","pvplay_7",
    "pvplay_8","pvplay_9","pvplay_10","pvplay_11","pvplay_12","pvplay_13","pvplay_14","pvplay_15",
    "pvplay_16","pvplay_17","pvplay_18","pvplay_19","pvplay_20","pvplay_21","pvplay_22","pvplay_23",
    "pvplay_24","pvplay_25","pvplay_26","pvplay_27","pvplay_28","pvplay_29","pvplay_30","pvplay_31","pvplay_mask",
    "pfav_0","pfav_1","pfav_2","pfav_3","pfav_4","pfav_5","pfav_6","pfav_7",
    "pfav_8","pfav_9","pfav_10","pfav_11","pfav_12","pfav_13","pfav_14","pfav_15",
    "pfav_16","pfav_17","pfav_18","pfav_19","pfav_20","pfav_21","pfav_22","pfav_23",
    "pfav_24","pfav_25","pfav_26","pfav_27","pfav_28","pfav_29","pfav_30","pfav_31","pfav_mask",
    "plike_0","plike_1","plike_2","plike_3","plike_4","plike_5","plike_6","plike_7",
    "plike_8","plike_9","plike_10","plike_11","plike_12","plike_13","plike_14","plike_15",
    "plike_16","plike_17","plike_18","plike_19","plike_20","plike_21","plike_22","plike_23",
    "plike_24","plike_25","plike_26","plike_27","plike_28","plike_29","plike_30","plike_31","plike_mask",
    "pshare_0","pshare_1","pshare_2","pshare_3","pshare_4","pshare_5","pshare_6","pshare_7",
    "pshare_8","pshare_9","pshare_10","pshare_11","pshare_12","pshare_13","pshare_14","pshare_15",
    "pshare_16","pshare_17","pshare_18","pshare_19","pshare_20","pshare_21","pshare_22","pshare_23",
    "pshare_24","pshare_25","pshare_26","pshare_27","pshare_28","pshare_29","pshare_30","pshare_31","pshare_mask",
    "postLikeRatio2h","postShareRatio2h","postFavRatio2h","postCommentRatio2h","postSVPRatio2h","postLPORatio2h",
    "postLikeRatio1D","postShareRatio1D","postFavRatio1D","postCommentRatio1D","postSVPRatio1D","postLPORatio1D",
    "pcLikeRatio2h","pcShareRatio2h","pcFavRatio2h","pcCommentRatio2h","pcSVPRatio2h","pcLPORatio2h",
    "pcLikeRatio1D","pcShareRatio1D","pcFavRatio1D","pcCommentRatio1D","pcSVPRatio1D"," pcLPORatio1D"
]

past_post = ["mrp"]
past_post_weights = ["mrpw"]

ignore_features = [

]

DROPOUT = 0.4
L2REG = 1e-4
LR = 0.001

batch_size = 1000#00

#NUM_TRAIN_EXAMPLES = 1199882090 
NUM_TRAIN_EXAMPLES = 43437361#6283392#1384711658 
NUM_EVAL_EXAMPLES = 48829132#216877418 
#NUM_EVAL_EXAMPLES = 2168774
NUM_TEST_EXAMPLES = 46248765#216877418

num_of_validations = 6

data_folder = "isp_ranker_data"#"masknet_hindi_data"#
eval_folder = "isp_ranker_data_valid"#"masknet_hindi_data_valid"#
test_folder = "isp_ranker_data_test"#"masknet_hindi_data_valid"#
model_folder = "unified_signal_Hindi_video_mask_net_serial_sampled"

DATA_DIR = "gs://tpu-cg-us/" + data_folder
EVALDATA_DIR = "gs://tpu-cg-us/" + eval_folder
TESTDATA_DIR = "gs://tpu-cg-us/" + test_folder

MODEL_DIR = "gs://tpu-cg-us/" + model_folder

MODEL_DIR_LOCAL = model_folder

def create_distribute_input_option():
    # Add a try...except block as OSS tensorflow_recommenders is depending on
    # stable TF version, i.e. TF2.4.
    try:
        return tf.distribute.InputOptions(experimental_fetch_to_device=False)
    except TypeError:
        return tf.distribute.InputOptions(experimental_prefetch_to_device=False)

@dataclasses.dataclass
class DataConfig:
    """Dataset config for training and evaluation."""
    input_path: str = ''
    global_batch_size: int = batch_size
    is_training: bool = True
    dtype: str = 'float32'
    shuffle_buffer_size: int = 1000#0
    cycle_length: int = 8
    sharding: bool = True
    num_shards_per_host: int = 8

tag_mapping = pd.read_csv(
    "tagId_mapping.csv",#"sc_ranker_debiasing-sc_ranker_debiasing_tag_index_mapping-000000000000.csv",#"tagId_mapping.csv",#
    dtype={'tagId': 'str'}
)
district_mapping = pd.read_csv(
    "userDistrict_mapping.csv",
    dtype={'userDistrict': 'str'}
)

tag_mapping.sort_values(by='tag_index', axis=0, inplace=True)
tag_mapping.reset_index(drop=True, inplace=True)
district_mapping.fillna("null", inplace=True)

district_mapping.sort_values(by='district_index', axis=0, inplace=True)
district_mapping.reset_index(drop=True, inplace=True)
district_mapping.fillna("null", inplace=True)

tag_index = {
    'keys': list(tag_mapping.tagId),
    'values': list(tag_mapping.tag_index),
}

district_index = {
    'keys': list(district_mapping.userDistrict),
    'values': list(district_mapping.district_index),
}
district_mapping_dict = {}
for i in range(len(district_index["keys"])):
    district_mapping_dict[district_index["keys"][i]] = district_index["values"][i]

tag_mapping_dict = {}
for i in range(len(tag_index["keys"])):
    tag_mapping_dict[tag_index["keys"][i]] = tag_index["values"][i]

with open('tag_mapping_dict.json', 'w') as fp:
    json.dump(tag_mapping_dict, fp)

with open('tag_mapping_dict.json', 'r') as fp:
    data = json.load(fp)

class IspRT(tf.Module):
    # Assume these are populated from embeddings file
    # PostId, PostEmb, PostBias
    def __init__(self, post_embs, post_biases, postIds):
        self.post_embs = tf.constant(post_embs, dtype=tf.float32)
        self.post_biases = tf.constant(post_biases, dtype=tf.float32)
        self.postIds = tf.constant(postIds, dtype='string')
        self.numPosts = len(postIds)
        self.postIdToIndexTable = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(self.postIds, tf.constant(list(range(self.numPosts)), dtype='int64')),
            default_value=-1
        )

    @tf.function
    def getPostEmbsAndBias(self, postIds):
        indexes = self.postIdToIndexTable.lookup(postIds) + 1
        embs_with_sentinel = tf.concat([tf.constant([[0.0]*self.post_embs.shape[1]], dtype="float32"), self.post_embs], 0)
        bias_with_sentinel = tf.concat([tf.constant([0.0], dtype="float32"), self.post_biases], 0)
        return tf.gather(embs_with_sentinel, indexes), tf.gather(bias_with_sentinel, indexes)

df_27 = pd.read_csv("post_embedding_topk_27.csv", converters={"embs": json.loads}, dtype={"postId": "str"})
df_28 = pd.read_csv("post_embedding_topk_28.csv", converters={"embs": json.loads}, dtype={"postId": "str"})
df_01 = pd.read_csv("post_embedding_topk_01.csv", converters={"embs": json.loads}, dtype={"postId": "str"})

model_27 = IspRT(df_27.embs.values.tolist(), df_27.bias.values, df_27.postId.values)
model_28 = IspRT(df_28.embs.values.tolist(), df_28.bias.values, df_28.postId.values)
model_01 = IspRT(df_01.embs.values.tolist(), df_01.bias.values, df_01.postId.values)

class CSVReader27(object):
    def __init__(self, params, model, num_labels, field_delim="&", use_fake_data=False):
        self.params = params
        self.model = model
        self.num_labels = num_labels
        self.field_delim = field_delim
        self._use_fake_data = use_fake_data
        
        self.tag_index = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tag_index['keys'], tag_index['values']),
            default_value=0
        )
        
        self.district_index = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(district_index['keys'], district_index['values']),
            default_value=0
        )
    
    def __call__(self, ctx: tf.distribute.InputContext):
        params = self.params
        batch_size = ctx.get_per_replica_batch_size(
            params.global_batch_size
        ) if ctx else params.global_batch_size
        
        @tf.function
        def _parse_fn(example):
            num_sparse_features = len(vocab_sizes)
            meta_defaults = [''] * len(meta)
            label_defaults = [0.0] * num_labels
            
            other_feat_defaults = [0.0] * (hour_feat+dayofweek+num_other_features)
            
            post_sparse_defaults = ['0'] * len(post_sparse_features)
            post_dense_defaults = [-1.0] * len(post_dense_features)
            
            user_sparse_defaults = ['0'] * len(user_sparse_features)
            user_dense_defaults = [-1.0] * len(user_dense_features)
            user_engaged_tags_defaults = ['0'] * (len(user_engaged_tags)//2) + [0.0] * (len(user_engaged_tags)//2)
            mrp = ['0']
            mrpw = ['0']
            
            record_defaults =   meta_defaults+mrp+mrpw+label_defaults + \
                                other_feat_defaults + \
                                post_sparse_defaults + \
                                post_dense_defaults + \
                                user_sparse_defaults + \
                                user_dense_defaults + \
                                user_engaged_tags_defaults

            fields = tf.io.decode_csv(example, record_defaults,
                                      field_delim=self.field_delim, na_value='')
            
            offset = 0            
            
            meta_feats = {}
            for idx in range(len(meta)):
                if col_names[idx+offset] in ignore_features:
                    continue
                meta_feats[col_names[idx+offset]] = fields[idx+offset]
            offset += len(meta)
            
            features = {'time': {}, 'sparse_features': {}, 'meta': meta_feats}
            d={}
            past_post_emb = []
            for idx in range(len(past_post)):
                if col_names[idx+offset] in ignore_features:
                    continue
                postIds = tf.strings.split(fields[idx + offset], sep=",")
#                 print("postIds ",postIds)
#                 print("model_21.getPostEmbsAndBias(fields[idx + offset]) ",model_21.getPostEmbsAndBias(fields[idx + offset]))
                recent_seq_embs, recent_seq_biases = model_27.getPostEmbsAndBias(postIds)#fields[idx+offset])
                ffm_seq_embs = tf.concat([recent_seq_embs,tf.expand_dims(recent_seq_biases, axis=-1)],axis=-1)
                d['mrp'] = ffm_seq_embs
                past_post_emb.append(ffm_seq_embs)
            print("past_post_emb ",d['mrp'].shape)
            offset += 1#len(past_post)
        
            
            past_post_emb_wt = []
            for idx in range(len(past_post_weights)):
                if col_names[idx+offset] in ignore_features:
                    continue
#                 seq = tf.keras.preprocessing.sequence.pad_sequences(fields[idx+offset],60)
                str_weights = tf.strings.split(fields[idx+offset], sep=",")
                weights = tf.strings.to_number(str_weights, tf.float32)
                d['mrpw'] = tf.expand_dims(weights,axis=-1)#fields[idx+offset], axis=-1)#tf.stack(past_post_emb_wt, axis=1)
            print("shape features['mrp'] ",d['mrp'].shape)
            print("shape features['mrpw'] ",d['mrpw'].shape)
            features['mrp_mrpw'] = tf.reshape(tf.math.divide(tf.math.reduce_sum((d['mrp'] * d['mrpw']).to_tensor(), axis=1), 26),(batch_size,65))
            print("mrp_mrpw shape ",features['mrp_mrpw'].shape)
            offset += 1#len(past_post_weights)
            
            label = tf.cast(fields[offset+0], tf.float32)
            offset += num_labels

            for idx in range(hour_feat+dayofweek+num_other_features):
                if col_names[idx+offset] in ignore_features:
                    continue
                if col_names[idx+offset] in ('hour', 'dayofweek'):
                    features['time'][col_names[idx+offset]] = tf.cast(fields[idx+offset], tf.int32)
                else:
                    features['time'][col_names[idx+offset]] = tf.cast(tf.expand_dims(fields[idx+offset], axis=-1), tf.float32)
            offset += hour_feat+dayofweek+num_other_features
            
            for idx in range(len(post_sparse_features)):
                if col_names[idx+offset] in ignore_features:
                    continue
                    
                if col_names[idx+offset].endswith("tagId"):
                    features['sparse_features'][col_names[idx+offset]] = self.tag_index.lookup(fields[idx+offset])
                else:
                    features['sparse_features'][col_names[idx+offset]] = fields[idx+offset]
            offset += len(post_sparse_features)

            feat = []
            post_embed = []
            for idx in range(len(post_dense_features)):
                if col_names[idx+offset] in ignore_features:
                    continue
                if col_names[idx+offset].startswith('pvplay') or col_names[idx+offset].startswith('pfav') or col_names[idx+offset].startswith('plike') or col_names[idx+offset].startswith('pshare'):
                    post_embed.append(fields[idx + offset])
                feat.append(fields[idx + offset])
            features['post_dense_features'] = tf.stack(feat, axis=1)#fields[offset]
            features['post_embed'] = tf.stack(post_embed, axis=1)#fields[offset+1]
            offset += len(post_dense_features)
            
            
            for idx in range(len(user_sparse_features)):
                if col_names[idx+offset] in ignore_features:
                    continue
                if col_names[idx+offset] == "userDistrict":
                    features['sparse_features'][col_names[idx+offset]] = self.district_index.lookup(fields[idx+offset])
                else:
                    features['sparse_features'][col_names[idx+offset]] = fields[idx+offset]
            offset += len(user_sparse_features)
            
            feat = []
            user_embed = []
            for idx in range(len(user_dense_features)):
                if col_names[idx+offset] in ignore_features:
                    continue
                if col_names[idx+offset].startswith('uvplay') or col_names[idx+offset].startswith('ufav') or col_names[idx+offset].startswith('ulike') or col_names[idx+offset].startswith('ushare'):
                    user_embed.append(fields[idx + offset])
                feat.append(fields[idx + offset])
            features['user_dense_features'] = tf.stack(feat, axis=1)
            features['user_embed'] = tf.stack(user_embed, axis=1)
            offset += len(user_dense_features)
            
            
            eng_tags_mask = []
            eng_tags = []
            for idx in range(len(user_engaged_tags)):
                if col_names[idx+offset] in ignore_features:
                    continue
                if 'mask' in col_names[idx+offset]:
                    eng_tags_mask.append(fields[idx + offset])
                else:
                    eng_tags.append(self.tag_index.lookup(fields[idx + offset]))
            features['sparse_features']['eng_tags'] = tf.stack(eng_tags, axis=1)
            features['eng_tags_mask'] = tf.stack(eng_tags_mask, axis=1)
            print("features['eng_tags_mask'] ",features['eng_tags_mask'].shape)
            offset += len(user_engaged_tags)
            
            return features, label
        
        filenames = tf.data.Dataset.list_files(params.input_path, shuffle=False)
        
        if params.sharding and ctx and ctx.num_input_pipelines > 1:
            filenames = filenames.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
            
        num_shards_per_host = 1
        if params.sharding:
            num_shards_per_host = params.num_shards_per_host

        def make_dataset(shard_index):
            filenames_for_shard = filenames.shard(num_shards_per_host, shard_index)
            dataset = tf.data.TextLineDataset(filenames_for_shard)
            if params.is_training:
                dataset = dataset.shuffle(params.shuffle_buffer_size)
                dataset = dataset.repeat()
            dataset = dataset.batch(batch_size, drop_remainder=True)
            dataset = dataset.map(_parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            
            
            return dataset
        indices = tf.data.Dataset.range(num_shards_per_host)
        dataset = indices.interleave(
            map_func=make_dataset,
            cycle_length=params.cycle_length,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        if self._use_fake_data:
            dataset = dataset.take(1).cache().repeat()
            
        return dataset

class CSVReader28(object):
    def __init__(self, params,model, num_labels, field_delim="&", use_fake_data=False):
        self.params = params
        self.model = model
        self.num_labels = num_labels
        self.field_delim = field_delim
        self._use_fake_data = use_fake_data
        
        self.tag_index = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tag_index['keys'], tag_index['values']),
            default_value=0
        )
        
        self.district_index = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(district_index['keys'], district_index['values']),
            default_value=0
        )
        
    def __call__(self, ctx: tf.distribute.InputContext):
        params = self.params
        batch_size = ctx.get_per_replica_batch_size(
            params.global_batch_size
        ) if ctx else params.global_batch_size
        
        @tf.function
        def _parse_fn(example):
            num_sparse_features = len(vocab_sizes)
            meta_defaults = [''] * len(meta)
            label_defaults = [0.0] * num_labels
            
            other_feat_defaults = [0.0] * (hour_feat+dayofweek+num_other_features)
            
            post_sparse_defaults = ['0'] * len(post_sparse_features)
            post_dense_defaults = [-1.0] * len(post_dense_features)
            
            user_sparse_defaults = ['0'] * len(user_sparse_features)
            user_dense_defaults = [-1.0] * len(user_dense_features)
            user_engaged_tags_defaults = ['0'] * (len(user_engaged_tags)//2) + [0.0] * (len(user_engaged_tags)//2)
            mrp = ['0']
            mrpw = ['0.0']

            record_defaults =   meta_defaults+mrp+mrpw+label_defaults + \
                                other_feat_defaults + \
                                post_sparse_defaults + \
                                post_dense_defaults + \
                                user_sparse_defaults + \
                                user_dense_defaults + \
                                user_engaged_tags_defaults

            fields = tf.io.decode_csv(example, record_defaults,
                                      field_delim=self.field_delim, na_value='')
            return fields
            
            #label = {
            #    col_names[0]: tf.reshape(fields[0], [batch_size, 1])
            #}
            #label = fields[0] #tf.reshape(fields[0], [batch_size, 1])
            offset=0
            meta_feats,d = {},{}
            for idx in range(len(meta)):
                if col_names[idx+offset] in ignore_features:
                    continue
                meta_feats[col_names[idx+offset]] = fields[idx+offset]
            offset += len(meta)
            
            features = {'time': {}, 'sparse_features': {}, 'meta': meta_feats}
            
            past_post_emb = []
            for idx in range(len(past_post)):
                if col_names[idx+offset] in ignore_features:
                    continue
                print("fields[idx + offset] ",fields[idx + offset])
                postIds = tf.strings.split(fields[idx + offset], sep=",")
                print("postIds ",postIds)
#                 seq = tf.keras.preprocessing.sequence.pad_sequences(fields[idx+offset],60)
#                 print("model_21.getPostEmbsAndBias(fields[idx + offset]) ",model_21.getPostEmbsAndBias(fields[idx + offset]))
                recent_seq_embs, recent_seq_biases = self.model.getPostEmbsAndBias(postIds)#fields[idx+offset])
                ffm_seq_embs = tf.concat([recent_seq_embs,tf.expand_dims(recent_seq_biases, axis=-1)],axis=-1)
                d['mrp'] = ffm_seq_embs
                print("ffm_seq_embs ",ffm_seq_embs.shape)
                past_post_emb.append(ffm_seq_embs)
            print("past_post_emb ",past_post_emb[0].shape)
#             d['mrp'] = past_post_emb[0]#tf.stack(past_post_emb, axis=1)
            offset += 1#len(past_post)
        
            
            past_post_emb_wt = []
            
            for idx in range(len(past_post_weights)):
                if col_names[idx+offset] in ignore_features:
                    continue
#                 seq = tf.keras.preprocessing.sequence.pad_sequences(fields[idx+offset],60)
                str_weights = tf.strings.split(tf.convert_to_tensor(fields[idx+offset]), sep=",")
                weights = tf.strings.to_number(str_weights, tf.float64)
#                 seq = tf.keras.preprocessing.sequence.pad_sequences(weights,26)
                d['mrpw'] = tf.expand_dims(weights,axis=-1)#fields[idx+offset], axis=-1)#tf.stack(past_post_emb_wt, axis=1)
            print("shape features['mrp'] ",d['mrp'])
            print("shape features['mrpw'] ",d['mrpw'])
            features['mrp_mrpw'] = tf.reshape(tf.math.divide(tf.math.reduce_sum((d['mrp'] * d['mrpw']).to_tensor(), axis=1), 26),(batch_size,65))
#             tf.reshape((d['mrp'] * d['mrpw']).to_tensor(),(batch_size,65))#tf.stack(d['mrp'] * d['mrpw'], axis=1)#, (-1,1))#tf.tensordot(features['mrpw'],features['mrp'],axes=0)
            print("mrp_mrpw shape ",features['mrp_mrpw'].shape)
            offset += 1#len(past_post_weights)
            
#             isp_label = tf.cast(fields[offset+0],tf.float32)
#             offset+= 1
            
            label = tf.cast(fields[offset+0], tf.float32)
            offset += num_labels

            for idx in range(hour_feat+dayofweek+num_other_features):
                if col_names[idx+offset] in ignore_features:
                    continue
                if col_names[idx+offset] in ('hour', 'dayofweek'):
                    features['time'][col_names[idx+offset]] = tf.cast(fields[idx+offset], tf.int32)
                else:
                    features['time'][col_names[idx+offset]] = tf.cast(tf.expand_dims(fields[idx+offset], axis=-1), tf.float32)
            offset += hour_feat+dayofweek+num_other_features

            
            for idx in range(len(post_sparse_features)):
                if col_names[idx+offset] in ignore_features:
                    continue
                    
                if col_names[idx+offset].endswith("tagId"):
                    features['sparse_features'][col_names[idx+offset]] = self.tag_index.lookup(fields[idx+offset])
                else:
                    features['sparse_features'][col_names[idx+offset]] = fields[idx+offset]
            offset += len(post_sparse_features)

            feat = []
            post_embed = []
            for idx in range(len(post_dense_features)):
                if col_names[idx+offset] in ignore_features:
                    continue
                if col_names[idx+offset].startswith('pvplay') or col_names[idx+offset].startswith('pfav') or col_names[idx+offset].startswith('plike') or col_names[idx+offset].startswith('pshare'):
                    post_embed.append(fields[idx + offset])
                feat.append(fields[idx + offset])
            features['post_dense_features'] = tf.stack(feat, axis=1)#fields[offset]
            features['post_embed'] = tf.stack(post_embed, axis=1)#fields[offset+1]
            offset += len(post_dense_features)
            
            
            for idx in range(len(user_sparse_features)):
                if col_names[idx+offset] in ignore_features:
                    continue
                if col_names[idx+offset] == "userDistrict":
                    features['sparse_features'][col_names[idx+offset]] = self.district_index.lookup(fields[idx+offset])
                else:
                    features['sparse_features'][col_names[idx+offset]] = fields[idx+offset]
            offset += len(user_sparse_features)
            
            feat = []
            user_embed = []
            for idx in range(len(user_dense_features)):
                if col_names[idx+offset] in ignore_features:
                    continue
                if col_names[idx+offset].startswith('uvplay') or col_names[idx+offset].startswith('ufav') or col_names[idx+offset].startswith('ulike') or col_names[idx+offset].startswith('ushare'):
                    user_embed.append(fields[idx + offset])
                feat.append(fields[idx + offset])
            features['user_dense_features'] = tf.stack(feat, axis=1)
            features['user_embed'] = tf.stack(user_embed, axis=1)
            offset += len(user_dense_features)
            
            
            eng_tags_mask = []
            eng_tags = []
            for idx in range(len(user_engaged_tags)):
                if col_names[idx+offset] in ignore_features:
                    continue
                if 'mask' in col_names[idx+offset]:
                    eng_tags_mask.append(fields[idx + offset])
                else:
                    eng_tags.append(self.tag_index.lookup(fields[idx + offset]))
            features['sparse_features']['eng_tags'] = tf.stack(eng_tags, axis=1)
            features['eng_tags_mask'] = tf.stack(eng_tags_mask, axis=1)
            offset += len(user_engaged_tags)
            print("offset is ",offset)
          
            return features, label
        
        filenames = tf.data.Dataset.list_files(params.input_path, shuffle=False)
        
        if params.sharding and ctx and ctx.num_input_pipelines > 1:
            filenames = filenames.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
            
        num_shards_per_host = 1
        if params.sharding:
            num_shards_per_host = params.num_shards_per_host

        def make_dataset(shard_index):
            filenames_for_shard = filenames.shard(num_shards_per_host, shard_index)
            dataset = tf.data.TextLineDataset(filenames_for_shard)
            if params.is_training:
                dataset = dataset.shuffle(params.shuffle_buffer_size)
                dataset = dataset.repeat()
            dataset = dataset.batch(batch_size, drop_remainder=True)
            dataset = dataset.map(_parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            
            return dataset
        indices = tf.data.Dataset.range(num_shards_per_host)
        dataset = indices.interleave(
            map_func=make_dataset,
            cycle_length=params.cycle_length,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        if self._use_fake_data:
            dataset = dataset.take(1).cache().repeat()
            
        return dataset

class CSVReader01(object):
    def __init__(self, params,model, num_labels, field_delim="&", use_fake_data=False):
        self.params = params
        self.model = model
        self.num_labels = num_labels
        self.field_delim = field_delim
        self._use_fake_data = use_fake_data
        
        self.tag_index = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tag_index['keys'], tag_index['values']),
            default_value=0
        )
        
        self.district_index = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(district_index['keys'], district_index['values']),
            default_value=0
        )
        
    def __call__(self, ctx: tf.distribute.InputContext):
        params = self.params
        batch_size = ctx.get_per_replica_batch_size(
            params.global_batch_size
        ) if ctx else params.global_batch_size
        
        @tf.function
        def _parse_fn(example):
            num_sparse_features = len(vocab_sizes)
            meta_defaults = [''] * len(meta)
            label_defaults = [0.0] * num_labels
            
            other_feat_defaults = [0.0] * (hour_feat+dayofweek+num_other_features)
            
            post_sparse_defaults = ['0'] * len(post_sparse_features)
            post_dense_defaults = [-1.0] * len(post_dense_features)
            
            user_sparse_defaults = ['0'] * len(user_sparse_features)
            user_dense_defaults = [-1.0] * len(user_dense_features)
            user_engaged_tags_defaults = ['0'] * (len(user_engaged_tags)//2) + [0.0] * (len(user_engaged_tags)//2)
            mrp = ['0']
            mrpw = ['0.0']

            record_defaults =   meta_defaults+mrp+mrpw+label_defaults + \
                                other_feat_defaults + \
                                post_sparse_defaults + \
                                post_dense_defaults + \
                                user_sparse_defaults + \
                                user_dense_defaults + \
                                user_engaged_tags_defaults

            fields = tf.io.decode_csv(example, record_defaults,
                                      field_delim=self.field_delim, na_value='')
            return fields
            print("fields ",fields)
            
            #label = {
            #    col_names[0]: tf.reshape(fields[0], [batch_size, 1])
            #}
            #label = fields[0] #tf.reshape(fields[0], [batch_size, 1])
            offset=0
            meta_feats = {}
            for idx in range(len(meta)):
                if col_names[idx+offset] in ignore_features:
                    continue
                meta_feats[col_names[idx+offset]] = fields[idx+offset]
            offset += len(meta)
            
            features = {'time': {}, 'sparse_features': {}, 'meta': meta_feats}
            
            past_post_emb = []
            d = {}
            for idx in range(len(past_post)):
                if col_names[idx+offset] in ignore_features:
                    continue
                print("fields[idx + offset] ",fields[idx + offset])
                postIds = tf.strings.split(fields[idx + offset], sep=",")
                print("postIds ",postIds)
#                 seq = tf.keras.preprocessing.sequence.pad_sequences(fields[idx+offset],60)
#                 print("model_21.getPostEmbsAndBias(fields[idx + offset]) ",model_21.getPostEmbsAndBias(fields[idx + offset]))
                recent_seq_embs, recent_seq_biases = self.model.getPostEmbsAndBias(postIds)#fields[idx+offset])
                ffm_seq_embs = tf.concat([recent_seq_embs,tf.expand_dims(recent_seq_biases, axis=-1)],axis=-1)
#                 d['mrp'] = ffm_seq_embs
                print("ffm_seq_embs ",ffm_seq_embs.shape)
                past_post_emb.append(ffm_seq_embs)
            print("past_post_emb ",past_post_emb[0].shape)
            d['mrp'] = past_post_emb[0]#tf.stack(past_post_emb, axis=1)
            offset += len(past_post)
        
            
            past_post_emb_wt = []
            for idx in range(len(past_post_weights)):
                if col_names[idx+offset] in ignore_features:
                    continue
#                 seq = tf.keras.preprocessing.sequence.pad_sequences(fields[idx+offset],60)
                str_weights = tf.strings.split(tf.convert_to_tensor(fields[idx+offset]), sep=",")
                weights = tf.strings.to_number(str_weights, tf.float64)
                d['mrpw'] = tf.expand_dims(weights,axis=-1)#fields[idx+offset], axis=-1)#tf.stack(past_post_emb_wt, axis=1)
            print("shape features['mrp'] ",d['mrp'])
            print("shape features['mrpw'] ",d['mrpw'])
            features['mrp_mrpw'] = tf.reshape(tf.math.divide(tf.math.reduce_sum((d['mrp'] * d['mrpw']).to_tensor(), axis=1), 26),(batch_size,65))
    #             tf.reshape((d['mrp'] * d['mrpw']).to_tensor(),(batch_size,65))#tf.stack(d['mrp'] * d['mrpw'], axis=1)#, (-1,1))#tf.tensordot(features['mrpw'],features['mrp'],axes=0)
            print("mrp_mrpw shape ",features['mrp_mrpw'].shape)
            offset += 1#len(past_post_weights)
            
#             isp_label = tf.cast(fields[offset+0],tf.float32)
#             offset+= 1
            
            label = tf.cast(fields[offset+0], tf.float32)
            offset += num_labels

            for idx in range(hour_feat+dayofweek+num_other_features):
                if col_names[idx+offset] in ignore_features:
                    continue
                if col_names[idx+offset] in ('hour', 'dayofweek'):
                    features['time'][col_names[idx+offset]] = tf.cast(fields[idx+offset], tf.int32)
                else:
                    features['time'][col_names[idx+offset]] = tf.cast(tf.expand_dims(fields[idx+offset], axis=-1), tf.float32)
            offset += hour_feat+dayofweek+num_other_features

            
            for idx in range(len(post_sparse_features)):
                if col_names[idx+offset] in ignore_features:
                    continue
                    
                if col_names[idx+offset].endswith("tagId"):
                    features['sparse_features'][col_names[idx+offset]] = self.tag_index.lookup(fields[idx+offset])
                else:
                    features['sparse_features'][col_names[idx+offset]] = fields[idx+offset]
            offset += len(post_sparse_features)

            feat = []
            post_embed = []
            for idx in range(len(post_dense_features)):
                if col_names[idx+offset] in ignore_features:
                    continue
                if col_names[idx+offset].startswith('pvplay') or col_names[idx+offset].startswith('pfav') or col_names[idx+offset].startswith('plike') or col_names[idx+offset].startswith('pshare'):
                    post_embed.append(fields[idx + offset])
                feat.append(fields[idx + offset])
            features['post_dense_features'] = tf.stack(feat, axis=1)#fields[offset]
            features['post_embed'] = tf.stack(post_embed, axis=1)#fields[offset+1]
            offset += len(post_dense_features)
            
            
            for idx in range(len(user_sparse_features)):
                if col_names[idx+offset] in ignore_features:
                    continue
                if col_names[idx+offset] == "userDistrict":
                    features['sparse_features'][col_names[idx+offset]] = self.district_index.lookup(fields[idx+offset])
                else:
                    features['sparse_features'][col_names[idx+offset]] = fields[idx+offset]
            offset += len(user_sparse_features)
            
            feat = []
            user_embed = []
            for idx in range(len(user_dense_features)):
                if col_names[idx+offset] in ignore_features:
                    continue
                if col_names[idx+offset].startswith('uvplay') or col_names[idx+offset].startswith('ufav') or col_names[idx+offset].startswith('ulike') or col_names[idx+offset].startswith('ushare'):
                    user_embed.append(fields[idx + offset])
                feat.append(fields[idx + offset])
            features['user_dense_features'] = tf.stack(feat, axis=1)
            features['user_embed'] = tf.stack(user_embed, axis=1)
            offset += len(user_dense_features)
            
            
            eng_tags_mask = []
            eng_tags = []
            for idx in range(len(user_engaged_tags)):
                if col_names[idx+offset] in ignore_features:
                    continue
                if 'mask' in col_names[idx+offset]:
                    eng_tags_mask.append(fields[idx + offset])
                else:
                    eng_tags.append(self.tag_index.lookup(fields[idx + offset]))
            features['sparse_features']['eng_tags'] = tf.stack(eng_tags, axis=1)
            features['eng_tags_mask'] = tf.stack(eng_tags_mask, axis=1)
            offset += len(user_engaged_tags)
            print("offset is ",offset)
          
            return features, label
        
        filenames = tf.data.Dataset.list_files(params.input_path, shuffle=False)
        
        if params.sharding and ctx and ctx.num_input_pipelines > 1:
            filenames = filenames.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
            
        num_shards_per_host = 1
        if params.sharding:
            num_shards_per_host = params.num_shards_per_host

        def make_dataset(shard_index):
            filenames_for_shard = filenames.shard(num_shards_per_host, shard_index)
            dataset = tf.data.TextLineDataset(filenames_for_shard)
            if params.is_training:
                dataset = dataset.shuffle(params.shuffle_buffer_size)
                dataset = dataset.repeat()
            dataset = dataset.batch(batch_size, drop_remainder=True)
            dataset = dataset.map(_parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            
            return dataset
        indices = tf.data.Dataset.range(num_shards_per_host)
        dataset = indices.interleave(
            map_func=make_dataset,
            cycle_length=params.cycle_length,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        if self._use_fake_data:
            dataset = dataset.take(1).cache().repeat()
            
        return dataset


train_params = DataConfig(
    input_path = f"{DATA_DIR}/*",
    is_training=True,
    sharding=True
)
train_dataset_callable = CSVReader27(
    params=train_params,
    num_labels=num_labels,
    model = model_27
)
train_dataset = strategy.distribute_datasets_from_function(
    dataset_fn=train_dataset_callable,
    options=create_distribute_input_option()
)


eval_params = DataConfig(
    input_path=f'{EVALDATA_DIR}/*',
    is_training=True,
    sharding=True
)
eval_dataset_callable = CSVReader28(
    params=eval_params,
    num_labels=num_labels,
    model = model_28
)

eval_dataset = strategy.distribute_datasets_from_function(
    dataset_fn=eval_dataset_callable,
    options=create_distribute_input_option()
)


test_params = DataConfig(
    input_path=f'{TESTDATA_DIR}/*',
    is_training=False,
    sharding=False
)
test_dataset_callable = CSVReader01(
    params=test_params,
    num_labels=num_labels,
    model=model_01
)

test_dataset = strategy.distribute_datasets_from_function(
    dataset_fn=test_dataset_callable,
    options=create_distribute_input_option()
)

train_steps = NUM_TRAIN_EXAMPLES // (batch_size)
eval_steps = NUM_EVAL_EXAMPLES // (2*batch_size)
test_steps = NUM_TEST_EXAMPLES // batch_size
validation_interval = train_steps // num_of_validations

print(f"train_steps: {train_steps}, eval_steps: {eval_steps}, test_steps: {test_steps}")

feature_shapes = {
    'time': 36,
    'post_dense_features': 156,
    'user_dense_features': 163,
    'user_embed': 132,
    'post_embed': 132,
    'tagId': embedding_dims['tagId'],
    'userDistrict': embedding_dims['userDistrict'],
    'eng_tags': embedding_dims['tagId'],
    'mrp_mrpw': 65#156
}

class MaskNetModelSerial(tfrs.models.Model):
    def __init__(self):
        super().__init__()
        
        self.rescale_factor = 2.0
        
        self.tag_embedding = tf.keras.layers.Embedding(
                input_dim=vocab_sizes['tagId'],
                output_dim=embedding_dims['tagId'],
                input_length=1
        )
        self.eng_tag_embedding = tf.keras.layers.Embedding(
                input_dim=vocab_sizes['tagId'],
                output_dim=embedding_dims['tagId'],
                input_length=25
        )

        self.district_embedding = tf.keras.layers.Embedding(
                input_dim=vocab_sizes['userDistrict'],
                output_dim=embedding_dims['userDistrict'],
                input_length=1
        )
        
        with tf.compat.v1.variable_scope("MaskBlock_time"):
            self.time_mask = tf.keras.Sequential([
                tf.keras.layers.Dense(
                  units=int(feature_shapes['time']*self.rescale_factor),
                  kernel_initializer=tf.keras.initializers.VarianceScaling(),
                  kernel_regularizer=tf.keras.regularizers.L2(L2REG),
                  activation="relu"
                ),
                tf.keras.layers.Dense(
                  units=feature_shapes['time'],
                  kernel_initializer=tf.keras.initializers.VarianceScaling(),
                  kernel_regularizer=tf.keras.regularizers.L2(L2REG),
                )
            ], name="InstanceGuidedMask_time")
            self.time_norm = tf.keras.layers.LayerNormalization()
            self.time_mask_emb = tf.keras.Sequential([
                tf.keras.layers.Dense(
                  units=feature_shapes['user_embed'] + feature_shapes['userDistrict'],
                  kernel_initializer=tf.keras.initializers.VarianceScaling(),
                  kernel_regularizer=tf.keras.regularizers.L2(L2REG),
                ),
                tf.keras.layers.LayerNormalization()
            ], name="MaskBlock_time")
            
        with tf.compat.v1.variable_scope("MaskBlock_user_sparse"):
            self.user_sparse_mask = tf.keras.Sequential([
                tf.keras.layers.Dense(
                  units=int((feature_shapes['user_embed'] + feature_shapes['userDistrict'])
                            *self.rescale_factor),
                  kernel_initializer=tf.keras.initializers.VarianceScaling(),
                  kernel_regularizer=tf.keras.regularizers.L2(L2REG),
                  activation="relu"
                ),
                tf.keras.layers.Dense(
                  units=(feature_shapes['user_embed'] + feature_shapes['userDistrict']),
                  kernel_initializer=tf.keras.initializers.VarianceScaling(),
                  kernel_regularizer=tf.keras.regularizers.L2(L2REG),
                )
            ], name="InstanceGuidedMask_user_sparse")
            self.user_sparse_mask_emb = tf.keras.Sequential([
                tf.keras.layers.Dense(
                  units=feature_shapes['eng_tags'],
                  kernel_initializer=tf.keras.initializers.VarianceScaling(),
                  kernel_regularizer=tf.keras.regularizers.L2(L2REG),
                ),
                tf.keras.layers.LayerNormalization()
            ], name="MaskBlock_user_sparse")
            
        with tf.compat.v1.variable_scope("MaskBlock_user_tags"):
            self.user_tags_mask = tf.keras.Sequential([
                tf.keras.layers.Dense(
                  units=int(feature_shapes['eng_tags']*self.rescale_factor),
                  kernel_initializer=tf.keras.initializers.VarianceScaling(),
                  kernel_regularizer=tf.keras.regularizers.L2(L2REG),
                  activation="relu"
                ),
                tf.keras.layers.Dense(
                  units=feature_shapes['eng_tags'],
                  kernel_initializer=tf.keras.initializers.VarianceScaling(),
                  kernel_regularizer=tf.keras.regularizers.L2(L2REG),
                )
            ], name="InstanceGuidedMask_user_tags")
            self.user_tags_mask_emb = tf.keras.Sequential([
                tf.keras.layers.Dense(
                  units=feature_shapes['user_dense_features'],
                  kernel_initializer=tf.keras.initializers.VarianceScaling(),
                  kernel_regularizer=tf.keras.regularizers.L2(L2REG),
                ),
                tf.keras.layers.LayerNormalization()
            ], name="MaskBlock_user_tags")
        
        with tf.compat.v1.variable_scope("MaskBlock_user_dense"):
            self.user_dense_mask = tf.keras.Sequential([
                tf.keras.layers.Dense(
                  units=int(feature_shapes['user_dense_features']*self.rescale_factor),
                  kernel_initializer=tf.keras.initializers.VarianceScaling(),
                  kernel_regularizer=tf.keras.regularizers.L2(L2REG),
                  activation="relu"
                ),
                tf.keras.layers.Dense(
                  units=feature_shapes['user_dense_features'],
                  kernel_initializer=tf.keras.initializers.VarianceScaling(),
                  kernel_regularizer=tf.keras.regularizers.L2(L2REG),
                )
            ], name="InstanceGuidedMask_user_dense")
            self.user_dense_mask_emb = tf.keras.Sequential([
                tf.keras.layers.Dense(
                  units=(feature_shapes['post_embed'] + feature_shapes['tagId']),
                  kernel_initializer=tf.keras.initializers.VarianceScaling(),
                  kernel_regularizer=tf.keras.regularizers.L2(L2REG),
                ),
                tf.keras.layers.LayerNormalization()
            ], name="MaskBlock_user_dense")
            
        with tf.compat.v1.variable_scope("MaskBlock_post_sparse"):
            self.post_sparse_mask = tf.keras.Sequential([
                tf.keras.layers.Dense(
                  units=int((feature_shapes['post_embed'] + feature_shapes['tagId'])
                            *self.rescale_factor),
                  kernel_initializer=tf.keras.initializers.VarianceScaling(),
                  kernel_regularizer=tf.keras.regularizers.L2(L2REG),
                  activation="relu"
                ),
                tf.keras.layers.Dense(
                  units=(feature_shapes['post_embed'] + feature_shapes['tagId']),
                  kernel_initializer=tf.keras.initializers.VarianceScaling(),
                  kernel_regularizer=tf.keras.regularizers.L2(L2REG),
                )
            ], name="InstanceGuidedMask_post_sparse")
            self.post_sparse_mask_emb = tf.keras.Sequential([
                tf.keras.layers.Dense(
                  units=feature_shapes['post_dense_features'],
                  kernel_initializer=tf.keras.initializers.VarianceScaling(),
                  kernel_regularizer=tf.keras.regularizers.L2(L2REG),
                ),
                tf.keras.layers.LayerNormalization()
            ], name="MaskBlock_post_sparse")
        
        with tf.compat.v1.variable_scope("MaskBlock_post_dense"):
            self.post_dense_mask = tf.keras.Sequential([
                tf.keras.layers.Dense(
                  units=int(feature_shapes['post_dense_features']*self.rescale_factor),
                  kernel_initializer=tf.keras.initializers.VarianceScaling(),
                  kernel_regularizer=tf.keras.regularizers.L2(L2REG),
                  activation="relu"
                ),
                tf.keras.layers.Dense(
                  units=feature_shapes['post_dense_features'],
                  kernel_initializer=tf.keras.initializers.VarianceScaling(),
                  kernel_regularizer=tf.keras.regularizers.L2(L2REG),
                )
            ], name="InstanceGuidedMask_post_dense")
            self.post_dense_mask_emb = tf.keras.Sequential([
                tf.keras.layers.Dense(
                  units=feature_shapes['post_dense_features'],
                  kernel_initializer=tf.keras.initializers.VarianceScaling(),
                  kernel_regularizer=tf.keras.regularizers.L2(L2REG),
                ),
                tf.keras.layers.LayerNormalization()
            ], name="MaskBlock_post_dense")
        with tf.compat.v1.variable_scope("MaskBlock_recent_post_emb_weights"):
            self.past_post_dense_mask = tf.keras.Sequential([
                tf.keras.layers.Dense(
                  units=int((feature_shapes['mrp_mrpw'])
                            *self.rescale_factor),
                  kernel_initializer=tf.keras.initializers.VarianceScaling(),
                  kernel_regularizer=tf.keras.regularizers.L2(L2REG),
                  activation="relu"
                ),
                tf.keras.layers.Dense(
                  units=(feature_shapes['post_dense_features']),
                  kernel_initializer=tf.keras.initializers.VarianceScaling(),
                  kernel_regularizer=tf.keras.regularizers.L2(L2REG),
                )
            ], name="InstanceGuidedMask_recent_post_dense")
            self.past_post_dense_mask_emb = tf.keras.Sequential([
                tf.keras.layers.Dense(
                  units=feature_shapes['mrp_mrpw'],
                  kernel_initializer=tf.keras.initializers.VarianceScaling(),
                  kernel_regularizer=tf.keras.regularizers.L2(L2REG),
                ),
                tf.keras.layers.LayerNormalization()
            ], name="MaskBlock_recent_post_dense")
        
        
        
        with tf.compat.v1.variable_scope("ClassificationTower"):
            self.classification_tower = tf.keras.Sequential([
              tf.keras.layers.Dense(
                  units=1,
                  kernel_initializer=tf.keras.initializers.VarianceScaling(),
                  kernel_regularizer=tf.keras.regularizers.L2(L2REG),
              )
            ])

        self.final_activation = tf.keras.layers.Activation('relu')
        
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            #loss=tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE),
            loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE),
            metrics=[
#                     tf.keras.metrics.AUC(name="auc"),
#                     tf.keras.metrics.AUC(curve="PR", name="pr-auc"),
                    # tf.keras.metrics.Precision(name="precision"),
                    # tf.keras.metrics.Recall(name="recall"),
                    # tf.keras.metrics.TruePositives(name="TP"),
                    # tf.keras.metrics.FalsePositives(name="FP"),
                    # tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
                    tf.keras.metrics.Accuracy(name='accuracy'),
                    tf.keras.metrics.CosineSimilarity(name='cosine_similarity', axis=-1),
            ]
        )

        
    def compute_loss(self, inputs, training=False) -> tf.Tensor:
        loss = 0
        if len(inputs) == 2:
            features, labels = inputs
            rating_predictions = self(features)
            loss = self.task(labels=labels, predictions=rating_predictions)
        elif len(inputs) == 3:
            features, labels, sample_weight = inputs
            rating_predictions = self(features)
            loss = self.task(labels=labels, predictions=rating_predictions, sample_weight=sample_weight)
        
        loss = tf.reduce_mean(loss)
        return tf.cast(loss,tf.float32) / tf.distribute.get_strategy().num_replicas_in_sync
    
    
    def call(self, inputs):
        sparse_features = inputs["sparse_features"]

        tag_embed = self.tag_embedding(sparse_features['tagId'])
        eng_tag_embed = self.tag_embedding(sparse_features['eng_tags'])
        sequence_length = tf.math.reduce_sum(inputs['eng_tags_mask'], axis=1, keepdims=True) + 0.0001
#         print("tagId shape ",tag_embed.shape," eng_tag_embed shape ",eng_tag_embed.shape)
#         print("sequence_length ",sequence_length)
        
        eng_tag_embed = tf.math.divide(tf.math.reduce_sum(eng_tag_embed, axis=1), sequence_length)
        district_embed = self.district_embedding(sparse_features['userDistrict'])

        hour = tf.one_hot(inputs['time']['hour'], 24)
        dayofweek = tf.one_hot(inputs['time']['dayofweek'], 7)
        time = tf.keras.layers.Concatenate(axis=-1)([
            hour, dayofweek,
            inputs['time']['is_weekend'],
            inputs['time']['is_morning'],
            inputs['time']['is_afternoon'],
            inputs['time']['is_evening'],
            inputs['time']['is_night'],
        ])
        user_sparse = tf.keras.layers.Concatenate()([inputs['user_embed'], district_embed])
        post_sparse = tf.keras.layers.Concatenate()([inputs['post_embed'], tag_embed])
        user_dense = inputs['user_dense_features']
        post_dense = inputs['post_dense_features']
        mrpw_dense = inputs['mrp_mrpw']
#         print("user_sparse shape ",user_sparse.shape," post_sparse shape ",post_sparse.shape," user_dense shape ",user_dense.shape)
#         print("post_dense shape ",post_dense.shape)
#         print("mrpw_dense ",mrpw_dense.shape)
        
        time_norm = self.time_norm(time)
        time_mask = self.time_mask(time)
        time_mask_emb = self.time_mask_emb(tf.keras.layers.Multiply()([time_norm, time_mask]))
        
        
        #user_sparse_norm = self.user_sparse_norm(user_sparse)
        user_sparse_mask = self.user_sparse_mask(user_sparse)
        user_sparse_mask_emb = self.user_sparse_mask_emb(tf.keras.layers.Multiply()([time_mask_emb, user_sparse_mask]))
        
#         user_tags_norm = self.user_tags_norm(eng_tag_embed)
        user_tags_mask = self.user_tags_mask(eng_tag_embed)
        user_tags_mask_emb = self.user_tags_mask_emb(tf.keras.layers.Multiply()([user_sparse_mask_emb, user_tags_mask]))
        
#         user_dense_norm = self.user_dense_norm(user_dense)
        user_dense_mask = self.user_dense_mask(user_dense)
        user_dense_mask_emb = self.user_dense_mask_emb(tf.keras.layers.Multiply()([user_tags_mask_emb, user_dense_mask]))
        
        
#         post_sparse_norm = self.post_sparse_norm(post_sparse)
        post_sparse_mask = self.post_sparse_mask(post_sparse)
        post_sparse_mask_emb = self.post_sparse_mask_emb(tf.keras.layers.Multiply()([user_dense_mask_emb, post_sparse_mask]))
        
#         post_dense_norm = self.post_dense_norm(post_dense)
        post_dense_mask = self.post_dense_mask(post_dense)
        post_dense_mask_emb = self.post_dense_mask_emb(tf.keras.layers.Multiply()([post_sparse_mask_emb, post_dense_mask]))
#         print("user_dense_mask ",user_dense_mask.shape," user_dense_mask_emb ",user_dense_mask_emb.shape)
#         print("post_sparse_mask ",post_sparse_mask.shape," post_sparse_mask_emb ",post_sparse_mask_emb.shape)
#         print("post_dense_mask ",post_dense_mask.shape," post_dense_mask_emb ",post_dense_mask_emb.shape)
        
        past_post_dense_mask = self.past_post_dense_mask(mrpw_dense)
        past_post_dense_mask_emb = self.past_post_dense_mask_emb(tf.keras.layers.Multiply()([post_dense_mask_emb, past_post_dense_mask]))
#         print("past_post_dense_mask ",past_post_dense_mask.shape," past_post_dense_mask_emb ",past_post_dense_mask_emb.shape)
        
        vector = past_post_dense_mask_emb#post_dense_mask_emb#
        
        logits = self.classification_tower(vector)
        
        prediction = self.final_activation(logits)
        
        return tf.reshape(prediction, [-1])

    @property
    def embedding_trainable_variables(self) -> Sequence[tf.Variable]:
        return [] #self.embedding_layer.trainable_variables

    @property
    def deep_trainable_variables(self) -> Sequence[tf.Variable]:
        dense_vars = []
        for layer in self.layers:
            dense_vars.extend(layer.trainable_variables)
        return dense_vars

with strategy.scope():
    embedding_optimizer = tf.keras.optimizers.Adam(lr=0.001)
    deep_optimizer = tf.keras.optimizers.Adagrad(lr=0.1)
    model = MaskNetModelSerial()
    optimizer = tf.keras.optimizers.Adam(lr=0.0005)

    model.compile(optimizer)#=tf.keras.optimizers.Adam())

checkpoints_cb = tf.keras.callbacks.ModelCheckpoint(MODEL_DIR + '/checkpoints/check_iter', save_freq=2000)
tb = tf.keras.callbacks.TensorBoard(MODEL_DIR + '/tb', update_freq=2000, profile_batch=0)

reconstructed_model = tf.keras.models.load_model("gs://tpu-cg-us/combined_model/1/")

history = model.fit(train_dataset, epochs=2, callbacks=[checkpoints_cb, tb],
                    steps_per_epoch=train_steps, validation_data=test_dataset, validation_steps=test_steps//4)

# history = model.fit(eval_dataset, epochs=1, callbacks=[checkpoints_cb, tb],
#                     steps_per_epoch=1)

metrics = model.evaluate(test_dataset, steps=test_steps, return_dict=True)

print("history.history ",history.history)

model.save('gs://tpu-cg-us/combined_model/1/')
tf.keras.models.save_model(model, 'gs://tpu-cg-us/combined_model/2/', save_format='tf', save_traces=False)
