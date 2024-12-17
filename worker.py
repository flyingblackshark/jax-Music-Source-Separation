#!/usr/bin/env python
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import rediswq
from infer import start_infer_file
#from model_training import FraudDetectionModelTrainer

# Initialize variables
FILESTORE_PATH = "/data"
OUTPUT_DIR = FILESTORE_PATH + "output/"
QUEUE_NAME = "extract"
HOST = "redis"

def main():
    """
    Workload which:
      1. Claims a filename from a Redis Worker Queue
      2. Reads the dataset from the file
      3. Partially trains the model on the dataset
      4. Saves a model checkpoint and generates a report on
         the performance of the model after the partial training.
      5. Removes the filename from the Redis Worker Queue
      6. Repeats 1 through 5 till the Queue is empty
    """
    q = rediswq.RedisWQ(name="extract", host=HOST)
    print("Worker with sessionID: " + q.sessionID())
    print("Initial queue state: empty=" + str(q.empty()))
    while not q.empty():
        # Claim item in Redis Worker Queue
        item = q.lease(lease_secs=20, block=True, timeout=2)
        if item is not None:
            audio_path = item.decode("utf-8")
            print("Processing audio: " + audio_path)

            start_infer_file(audio_path)

            # Remove item from Redis Worker Queue
            q.complete(item)
        else:
            print("Waiting for work")

    print("Queue empty, exiting")


# Run workload
main()
