"""One-time CPU ADM reference validation; production evaluation uses GPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
from pathlib import Path
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from vendor import adm_evaluator as adm
from paths import ROOT,WEIGHTS,VALIDATION_DATA
adm.INCEPTION_V3_PATH=str(WEIGHTS/'classify_image_graph_def.pb')
with adm.open_npz_array('/data/checkpoints/VIRTUAL_imagenet256_labeled.npz','arr_0') as reader:
    images=reader.read_batch(32)
np.save(VALIDATION_DATA/'parity_images.npy',images)
with tf.Graph().as_default():
    inp=tf.placeholder(tf.float32,shape=[None,None,None,3])
    pool,_=adm._create_feature_graph(inp)
    config=tf.ConfigProto(device_count={'GPU':0},intra_op_parallelism_threads=8,inter_op_parallelism_threads=2)
    with tf.Session(config=config) as sess:
        feats=np.concatenate([sess.run(pool,{inp:b.astype(np.float32)}).reshape(len(b),-1) for b in np.array_split(images,4)])
np.save(VALIDATION_DATA/'parity_tf_features.npy',feats)
print('ADM CPU reference features saved',feats.shape)
