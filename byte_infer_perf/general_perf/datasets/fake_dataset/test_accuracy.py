# Copyright 2023 ByteDance and/or its affiliates.
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

import logging
import numpy as np
from array import array
from itertools import chain
from general_perf.datasets import test_accuracy
from tqdm import tqdm

log = logging.getLogger("TestAccuracy")


class AccuracyChecker(test_accuracy.AccuracyChecker):

    def calculate_acc(self, data_percent=10, nvgpu=False):
        log.info("Start to calculate accuracy...")
        num = int((data_percent / 100) * self.dataloader.get_batch_count()
                  ) if data_percent else self.dataloader.get_batch_count()

        diffs = []

        if nvgpu:
            # add warmup
            WARMUP = 10
            print('start of warmup ({})...'.format(WARMUP))
            for i in range(WARMUP):
                test_data = self.dataloader.get_samples(i)
                results = self.runtime_backend.predict(test_data)
            times_range = []
            print('start of benchmark ({})...'.format(num))

        for i in tqdm(range(num)):
            test_data = self.dataloader.get_samples(i)

            if nvgpu:
                import time
                start_time = time.time()

            results = self.runtime_backend.predict(test_data)

            if nvgpu:
                end_time = time.time()
                times_range.append(end_time - start_time)

            if isinstance(results, dict):
                list_key = list(results.keys())
                list_key.sort()
                for key in list_key:
                    diffs.extend(results[key].flatten())
            elif isinstance(results, list):
                for out in results:
                    diffs.extend(out.flatten())
            else:
                diffs.extend(results)

        if nvgpu:
            times_range.sort()
            tail_latency = round(
                times_range[int(len(times_range) * 0.99)] * 1000, 2)
            avg_latency = round(sum(times_range) / num * 1000, 2)
            batch_size = 4
            qps = int(1000.0 * batch_size / avg_latency)
            log.info(
                '[accuracy] Batch size is {}, QPS: {}, Avg Latency:{}, Tail Latency:{}'.
                format(batch_size, qps, avg_latency, tail_latency))

        np.save(self.output_dir + "/{}.npy".format(self.dataloader.name()),
                np.array(diffs),
                allow_pickle=True)
        return {"Fake Dataset Accuracy": 0}
