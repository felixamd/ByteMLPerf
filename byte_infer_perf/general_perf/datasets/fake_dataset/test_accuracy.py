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

    def flatten(self, lst):
        result = []
        for item in lst:
            if isinstance(item, list) or isinstance(item, np.ndarray):
                result.extend(self.flatten(item))
            else:
#                print(f'type of item: {type(item)}')
                result.append(item)
        return result
    

    def calculate_acc(self, data_percent=10):
        log.info("Start to calculate accuracy...")
        num = int((data_percent / 100) * self.dataloader.get_batch_count()
                  ) if data_percent else self.dataloader.get_batch_count()

        diffs = []
        for i in tqdm(range(num)):
            test_data = self.dataloader.get_samples(i)

            results = self.runtime_backend.predict(test_data)
            if isinstance(results, dict):
                list_key = list(results.keys())
                list_key.sort()
                for key in list_key:
                    #log.info('key: {}, type of value: {}'.format(key, type(results[key])))


                    '''
                    flattened = self.flatten(list(results[key]))
                    log.info('key: {}, type of value: {}'.format(key, type(results[key])))
                    if len(flattened) > 1:
                        log.info('flattened 0: {},\n1: {}'.format(flattened[0], flattened[1]))
                    elif len(flattened) > 0:
                        log.info('flattened 0: {}'.format(flattened[0]))
                    else:
                        log.warning('flattened NONE')
                    '''

                    diffs.extend(self.flatten(results[key]))
                    #diffs.extend(array('i', results[key]).flatten())
                    #diffs.extend(self.flatten(results[key]))
                    #diffs.extend(results[key].flatten())
            elif isinstance(results, list):
                for out in results:
                    diffs.extend(out.flatten())
            else:
                diffs.extend(results)

        '''
        log.info('Batch size is {}, Accuracy: {}'.format(
            self.dataloader.cur_bs, 0.0))
        log.info('type of diffs = {}, len of diffs = {}'.format(type(diffs), len(diffs)))
        log.info('diff 0 {},\ndiff 1 {}'.format(diffs[0], diffs[1]))
        '''
        np.save(self.output_dir + "/{}.npy".format(self.dataloader.name()),
                np.array(diffs),
                allow_pickle=True)
        return {"Fake Dataset Accuracy": 0}
