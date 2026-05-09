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

if [ -z ${BRANCH} ]; then
    BRANCH="develop"
fi

PADDLE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"
# If you want to add monitoring file modifications, please perform the. github/CODEOWNERS operation

approval_line=`curl -H "Authorization: token ${GITHUB_TOKEN}" https://api.github.com/repos/PaddlePaddle/PaddleFleet/pulls/${PR_ID}/reviews?per_page=10000`
git_files=`git diff --numstat upstream/$BRANCH| wc -l`
git_count=`git diff --numstat upstream/$BRANCH| awk '{sum+=$1}END{print sum}'`
failed_num=1
echo_list=()


function check_approval(){
    person_num=`echo $@|awk '{for (i=2;i<=NF;i++)print $i}'`
    APPROVALS=`echo ${approval_line}|python ${PADDLE_ROOT}/ci/check_pr_approval.py $1 $person_num`
    if [[ "${APPROVALS}" == "FALSE" && "${echo_line}" != "" ]]; then
        add_failed "${failed_num}. ${echo_line}"
    fi
}


function add_failed(){
    failed_num=`expr $failed_num + 1`
    echo_list="${echo_list[@]}$1"
}

function run_tools_test() {
    CUR_PWD=$(pwd)
    cd ${PADDLE_ROOT}/tools
    python $1
    cd ${CUR_PWD}
}


CODESTYLE_APPROVERS="SigureMo risemeup1 swgu98"
CODESTYLE_FILES=(
    "ci/hooks"
    "_typos.toml"
    ".clang-format"
    ".cmakelnitrc"
    ".editorconfig"
    ".pre-commit-config.yaml"
    ".yamlfmt"
    "pyproject.toml"
)
for FILE in "${CODESTYLE_FILES[@]}"; do
    HAS_MODIFIED=$(git diff --name-only upstream/$BRANCH | grep "^${FILE}" || true)
    if [ "${HAS_MODIFIED}" != "" ] && [ "${PR_ID}" != "" ]; then
        echo_line="You must be approved by one of ${CODESTYLE_APPROVERS} for changes in ${FILE}.\n"
        APPROVER_LIST=(${CODESTYLE_APPROVERS})
        check_approval 1 "${APPROVER_LIST[@]}"
    fi
done



MODELCONFIG_APPROVERS="sneaxiy From00 ForFishes Hz188 Waynezee"
MODELCONFIG_FILES=(
    "src/paddlefleet/model_parallel_config.py"
    "src/paddlefleet/transformer/transformer_config.py"
)
for FILE in "${MODELCONFIG_FILES[@]}"; do
    HAS_MODIFIED=$(git diff --name-only upstream/$BRANCH | grep "^${FILE}" || true)
    if [ "${HAS_MODIFIED}" != "" ] && [ "${PR_ID}" != "" ]; then
        echo_line="You must be approved by two of ${MODELCONFIG_APPROVERS} for changes in ${FILE}.\n"
        APPROVER_LIST=(${MODELCONFIG_APPROVERS})
        check_approval 2 "${APPROVER_LIST[@]}"
    fi
done


CUSTOMOP_APPROVERS="risemeup1 From00"
CUSTOMOP_DIR="packages/paddlefleet_ops/src/paddlefleet_ops/_extensions"
HAS_MODIFIED_CUSTOMOP=$(git diff --name-only upstream/$BRANCH | grep "^${CUSTOMOP_DIR}/" || true)
if [ "${HAS_MODIFIED_CUSTOMOP}" != "" ] && [ "${PR_ID}" != "" ]; then
    echo_line="You must be approved by two of ${CUSTOMOP_APPROVERS} for changes in ${CUSTOMOP_DIR}.\n"
    APPROVER_LIST=(${CUSTOMOP_APPROVERS})
    check_approval 2 "${APPROVER_LIST[@]}"
fi



CHECKREQ_APPROVERS="risemeup1 swgu98"
files=$(git diff --name-status upstream/$BRANCH)
while read -r status file; do
    if [[ "$status" == "A" ]] && [[ "$(basename "$file")" == "requirements.txt" ]]; then
        echo_line="You must be approved by one of ${CHECKREQ_APPROVERS} for newly added \"$file\".\n"
        APPROVER_LIST=(${CHECKREQ_APPROVERS})
        check_approval 1 "${APPROVER_LIST[@]}"
    fi
done <<< "$files"


PACKAGING_APPROVERS="risemeup1 SigureMo"
PACKAGING_PATTERNS=(
    "^packages/"
    "^build_backend\.py$"
    "^ops_required_version\.txt$"
    "^pyproject\.toml$"
)
for PATTERN in "${PACKAGING_PATTERNS[@]}"; do
    HAS_MODIFIED=$(git diff --name-only upstream/$BRANCH | grep "${PATTERN}" || true)
    if [ "${HAS_MODIFIED}" != "" ] && [ "${PR_ID}" != "" ]; then
        echo_line="You must be approved by one of ${PACKAGING_APPROVERS} for changes in packaging-related files (${PATTERN}).\n"
        APPROVER_LIST=(${PACKAGING_APPROVERS})
        check_approval 1 "${APPROVER_LIST[@]}"
    fi
done


if [ -n "${echo_list}" ];then
  echo "****************"
  echo -e "${echo_list[@]}"
  echo "There are `expr $failed_num - 1` approved errors."
  echo "****************"
fi

if [ -n "${echo_list}" ]; then
  exit 6
fi
