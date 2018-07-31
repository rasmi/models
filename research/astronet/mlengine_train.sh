#!/bin/sh
# Copyright 2018 Google LLC.
# SPDX-License-Identifier: Apache-2.0
# Start an ML Engine training job.
TFRECORD_DIR="gs://my-bucket/kepler_tfrecords"
JOB_DIR="gs://my-bucket/astronet-cloudml"
NOW=$(date +%Y%m%d%H%M%S)

gcloud ml-engine jobs submit training astronet_train_${NOW} \
  --package-path=${PWD}/astronet \
  --module-name=astronet.train \
  --job-dir=${JOB_DIR} \
  --runtime-version=1.8 \
  -- \
  --model=AstroCNNModel \
  --config_name=local_global \
  --train_files=${TFRECORD_DIR}/train* \
  --eval_files=${TFRECORD_DIR}/val* \
  --model_dir=${JOB_DIR}/model
