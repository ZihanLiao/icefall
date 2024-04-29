. ./path.sh || exit 1;

stage=1
stop_stage=1

. ./shared/parse_options.sh || exit 1;

set -e pipefail

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CUDA_LAUNCH_BLOCKING=1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ];then
    python zipformer_stateless_cif/train.py \
        --world-size 8 \
        --num-epochs 30 \
        --start-epoch 1 \
        --use-fp16 1 \
        --predictor-left-order 1 \
        --predictor-right-order 1 \
        --max-duration 1000 \
        --exp-dir zipformer_stateless_cif/exp_pinyin_char \
        --base-lr 0.045 
        # --num-encoder-layers 2,2,4,5,4,2 \
        # --feedforward-dim 512,768,1536,2048,1536,768 \
        # --encoder-dim 192,256,512,768,512,256 \
        # --encoder-unmasked-dim 192,192,256,320,256,192 \
        # --max-duration 1000
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ];then
  python zipformer_stateless_cif/decode.py \
        --epoch 28 \
        --avg 5 \
        --exp-dir zipformer_stateless_cif/exp_pinyin_char \
        --lang-dir lang_pinyin_char \

fi