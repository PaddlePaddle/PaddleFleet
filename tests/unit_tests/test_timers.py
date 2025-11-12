# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import time
import unittest

import paddle

from fleet.core.timers import DummyTimer, Timers
from fleet.training.initialize import initialize_fleet


class TestTimers(unittest.TestCase):
    def setUp(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        initialize_fleet(strategy=strategy)

    def test_dummy_timer(self):
        timers = Timers(log_level=1, log_option="max")
        timer = timers("test_timer", log_level=2)
        self.assertIsInstance(timer, DummyTimer)

    def test_max_timers(self):
        timers = Timers(log_level=2, log_option="max")
        timer1 = timers("operation1", log_level=1)
        timer1.start()
        time.sleep(0.1)
        timer1.stop()
        timer2 = timers("operation2", log_level=2)
        timer2.start()
        time.sleep(0.05)
        timer2.stop()
        print(timers.get_all_timers_string())

    def test_all_timers(self):
        timers = Timers(log_level=2, log_option="all")
        timer1 = timers("operation1", log_level=1)
        timer1.start()
        time.sleep(0.1)
        timer1.stop()
        timers.log(["operation1"])

    def test_minmax_timers(self):
        timers = Timers(log_level=2, log_option="minmax")
        timer1 = timers("operation1", log_level=1)
        timer1.start()
        time.sleep(0.1)
        timer1.stop()
        timers.log(["operation1"])


if __name__ == "__main__":
    unittest.main()
