stage=0
stop_stage=0

set -e pipefail

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ];then
    log "Stage 0: Prepare pinyin based lang"
    lang_pinyin_bpe=lang_pinyin_bpe
    mkdir -p $lang_pinyin_bpe

    if [ ! -f $lang_pinyin_bpe/text.raw ];then
        log "Merge all text.raw"

        gunzip -c data/fbank/train/cuts.jsonl.gz \
        | grep -o 'text":\s[^,]*' | sed 's/text": "//g;s/"//g' > $lang_pinyin_bpe/text_train.raw

        gunzip -c data/fbank/dev/cuts.jsonl.gz \
        | grep -o 'text":\s[^,]*' | sed 's/text": "//g;s/"//g' > $lang_pinyin_bpe/text_dev.raw

        for r in text_train.raw text_dev.raw;do
            cat $lang_pinyin_bpe/$r >> $lang_pinyin_bpe/text.raw
        done
    fi

    if [ ! -f $lang_pinyin_bpe/text.seg ];then
        log "Segment text.raw to text.seg"
        python local/seg_words.py \
            -t $lang_pinyin_bpe/text.raw \
            -d data/dict/cn_dict.txt \
            --nj 32 \
            -o $lang_pinyin_bpe/text.seg
        sed -i 's/\///g;s/\s\+/ /g' $lang_pinyin_bpe/text.seg
    fi

    if [ ! -f $lang_pinyin_bpe/words.txt ];then
        (echo '<eps> 0'; echo '!SIL 1'; echo '<SPOKEN_NOISE> 2'; echo '<UNK> 3';) \
        > $lang_pinyin_bpe/words.txt

        cat $lang_pinyin_bpe/text.seg | sed 's/ /\n/g' | sort -u | sed '/^$/d' \
        | awk '{print $1" "NR+3}' >> $lang_pinyin_bpe/words.txt
        
        num_lines=$(< $lang_pinyin_bpe/words.txt wc -l)
        (echo "#0 $num_lines"; echo "<s> $(($num_lines + 1))"; echo "</s> $(($num_lines + 2))";) \
        >> $lang_pinyin_bpe/words.txt
    fi

fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ];then
    python local/prepare_char.py
fi