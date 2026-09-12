# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU checks for the bounded actor performance tool; GPU results are separate."""

import pytest

from toolkits.world_model.dreamdojo_actor_perf import accumulation_steps


@pytest.mark.parametrize("micro,expected", [(2, 4), (4, 2), (8, 1)])
def test_seven_gpu_accumulation(micro, expected):
    assert accumulation_steps(56, 7, micro) == expected


@pytest.mark.parametrize("values", [(7, 7, 4), (56, 7, 3), (0, 7, 2), (56, 0, 2)])
def test_reject_uneven_or_empty_batches(values):
    with pytest.raises(ValueError):
        accumulation_steps(*values)
