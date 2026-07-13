#!/bin/bash
# Contact-EC flat FC (원본 데이터셋) 학습 완료 후 평가 스크립트
# 학습: python train.py --model fusion_v2_flatfc --phase 1 --epochs 30 --gpu 0 --fp16 --tag flatfc_orig
# 체크포인트: outputs/checkpoints/flatfc_orig_best.pt

set -e
cd /home/user/Desktop/unlv

CKPT="outputs/checkpoints/flatfc_orig_best.pt"

if [ ! -f "$CKPT" ]; then
    echo "체크포인트 없음: $CKPT"
    echo "학습이 완료될 때까지 대기..."
    until [ -f "$CKPT" ]; do sleep 60; done
fi

echo "=== Contact-EC flat FC 평가 (원본 test 셋) ==="
python evaluate.py \
    --model fusion_v2_flatfc \
    --checkpoint "$CKPT" \
    --split test \
    --hierarchical_inference \
    --gpu 0

echo ""
echo "결과 파일: outputs/results/fusion_v2_flatfc_test_hier_results.json"
echo ""
echo "=== tab:level 업데이트용 수치 ==="
python3 -c "
import json
d = json.load(open('outputs/results/fusion_v2_flatfc_test_hier_results.json'))
for lvl in ['level1','level2','level3','level4']:
    print(f'{lvl}: n={d[lvl][\"n_samples\"]}, micro_f1={d[lvl][\"micro_f1\"]}')
"
