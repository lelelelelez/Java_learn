#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest

class TestFusionSquaredMatSubOp(OpTest):
    def setUp(self):
        self.op_type = 'fusion_squared_mat_sub'
        self.m = 11
        self.n = 12
        self.k = 4
        self.scalar = 0.5
        self.set_conf()
        matx = np.random.random((self.m, self.k)).astype("float32")
        maty = np.random.random((self.k, self.n)).astype("float32")

        self.inputs = {'X': matx, 'Y': maty}
        self.outputs = {
            'Out':
            (np.dot(matx, maty)**2 - np.dot(matx**2, maty**2)) * self.scalar
        }
        self.attrs = {'scalar': self.scalar, }

    def set_conf(self):
        pass

    def test_check_output(self):
        self.check_output()



if __name__ == '__main__':
    unittest.main()


#test1. X's dims size == Y's dims size
maty = np.random.random((self.k)).astype("float32")

#test2. x_dims[1] = y_dims[0]
#InvalidArgumentError: The input tensor X's dims[1] should be equal to Y's dims[0]. But received X's dims[1] = 3, Y's dims[0] = 4.
matx = np.random.random((11, 4)).astype("float32")
maty = np.random.random((4, 12)).astype("float32")
matx1 = np.random.random((11, 3)).astype("float32")
maty1 = np.random.random((4, 12)).astype("float32")
self.inputs = {'X': matx1, 'Y': maty1}
self.outputs = {
    'Out':
    (np.dot(matx, maty)**2 - np.dot(matx**2, maty**2)) * self.scalar
}

#test3. x_dims.size() == 2
#InvalidArgumentError: The input tensor X's dims size should be 2. But received X's dims size = 1.
matx = np.random.random((11, 4)).astype("float32")
maty = np.random.random((4, 12)).astype("float32")
matx1 = np.random.random((3)).astype("float32")
maty1 = np.random.random((3)).astype("float32")
self.inputs = {'X': matx1, 'Y': maty1}
self.outputs = {
    'Out':
    (np.dot(matx, maty)**2 - np.dot(matx**2, maty**2)) * self.scalar
}

#test4. 输入输出不能为空
#暂无法构造出来 python中已经做了检测 无输入时无法跑通case
