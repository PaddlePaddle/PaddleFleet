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
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT

nnodes=$PADDLE_TRAINERS_NUM
rank=$PADDLE_TRAINER_ID

for name in `env | grep -E 'PADDLE|ENDPOINT' | awk -F'=' '{print $1}'`; do
  unset ${name}
done


START_RANK=0
END_RANK=1


if [[ $rank -lt $START_RANK ]]; then
   exit 0
fi

if [[ $rank -ge $END_RANK ]]; then
   exit 0
fi

nnodes=$(($END_RANK-$START_RANK))
master=`cat /root/paddlejob/workspace/hostfile | head -n $(($START_RANK+1)) | tail -n 1 | awk '{print $1}'`
port=36677

rank=$(($rank-$START_RANK))

rm -rf ./outputs
rm -rf paddleformers_dist_log

FLAGS_use_stride_compute_kernel=False NNODES=${nnodes} MASTER_ADDR=${master} MASTER_PORT=${port} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 coverage run --source=. --branch paddleformers-cli train glm45.yaml

exit 0

python -m paddle.distributed.launch \
   --log_dir ./outputs/output_$rank/paddle_distributed_logs \
   --master $master:$port \
   --nnodes $nnodes \
   --rank $rank \
   --run_mode=collective \
   run_pretrain.py glm45.json \
   --output_dir ./outputs/checkpoint_$rank
