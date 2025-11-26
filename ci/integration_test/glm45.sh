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

root_dir=$(pwd)

source PaddleFleet/.venv/bin/activate

if [ ! -f $CACHE_DIR/glm45/data.tar ]; then
  mkdir -p $CACHE_DIR/glm45 && cd $CACHE_DIR/glm45
  wget -q --tries=5 --no-proxy https://xly-devops.cdn.bcebos.com/PaddleFleet/glm45_dataset/data.tar --no-check-certificate
  tar -xf data.tar
fi
if [ ! -f $CACHE_DIR/glm45/GLM-4.5-Air.tar ]; then
  mkdir -p $CACHE_DIR/glm45 && cd $CACHE_DIR/glm45
  wget -q --tries=5 --no-proxy https://xly-devops.cdn.bcebos.com/PaddleFleet/zai-org/GLM-4.5-Air.tar --no-check-certificate
  tar -xf GLM-4.5-Air.tar
fi

cd $root_dir/PaddleFormers/examples/experiments/paddlefleet

apt-get update
apt-get install jq -y

jq '.expert_parallel_degree = 8' glm45.json > glm45_single_node.json
jq '.save_steps = 100' glm45_single_node.json > glm45.json
jq --arg dir "1.0 $CACHE_DIR/glm45/data/pre-training/llama_openwebtext_100k" '.input_dir = $dir' glm45.json > glm45_single_node.json
jq --arg dir "$CACHE_DIR/glm45/GLM-4.5-Air" '.model_name_or_path = $dir' glm45_single_node.json > glm45.json
sed -i 's/from paddlefleet\.transformer import LayerSpec/from paddlefleet import LayerSpec/' glm45_provider.py
sed -i 's/from paddlefleet\.transformer import LayerSpec/from paddlefleet import LayerSpec/' gpt_provider.py
sed -i '/if not int(os.getenv("test_ci_no_save_model", 0)):/s/^/# /' run_pretrain.py
sed -i '/trainer.save_model()/s/^/# /' run_pretrain.py
# sed -i 's/num_layers: int = 10/num_layers: int = 5/' glm45_provider.py

python -c "
infile = '$root_dir/PaddleFormers/paddleformers/trainer/training_args.py'
outfile = infile + '.new'
with open(infile) as fin, open(outfile, 'w') as fout:
    for line in fin:
        fout.write(line)
        if line.strip() == '# initialize_fleet(strategy)':
            pad = line[:len(line) - len(line.lstrip())]
            fout.write(pad + 'import paddlefleet\n')
            fout.write(pad + 'paddlefleet.training.initialize.initialize_fleet(strategy)\n')
"
mv $root_dir/PaddleFormers/paddleformers/trainer/training_args.py.new $root_dir/PaddleFormers/paddleformers/trainer/training_args.py

rm -rf checkpoint/
rm -rf outputs/
master=$(hostname -i)
port=36677

unset http_proxy https_proxy
python -m paddle.distributed.launch \
   --log_dir ./log \
   --master $master:$port \
   --nnodes 1 \
   --rank 0 \
   --run_mode=collective \
   run_pretrain.py glm45.json \
   --output_dir ./checkpoint
