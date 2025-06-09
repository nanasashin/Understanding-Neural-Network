import sys
import numpy as np
import matplotlib


test_input = [10.5, 9.6, 3.0]

test_weight = [1.7, 2.1, 3.6]

bias = 3

"""
or

test_output = 0

for i in range(0, test_input.length())
    test_output += test_input[i] * test_weight[i]

test_input += bias
"""
test_output = test_input[0] * test_weight[0] + test_input[1] * test_weight[1] + test_input[2] * test_weight[2] + bias
print(test_output)
