import re
#import jieba
import argparse
from multiprocessing import Pool
#from local.fre_dp_wordseg import WordsSegment
from local.no_fre_dp_wordseg import WordsSegment
from tqdm import tqdm
import wordninja
import string

def is_all_english(word):
    for char in word:
        if char not in string.ascii_uppercase and char != "'" :
            return False
    return True

def merge_abbreviation(word_list):
  window_size = 3 
  sliding_steps = len(word_list) - window_size + 1 
  for i in range(sliding_steps):
    cur_window = word_list[i:i+window_size]
    if is_all_english(cur_window[0]) and cur_window[1] == "'" and cur_window[2] in string.ascii_uppercase:
      word_list[i:i+window_size] = [''.join(cur_window)]
      break
    if cur_window[0] == "I" and cur_window[1] == "'" and cur_window[2] == "AM":
      word_list[i:i+window_size] = [''.join(cur_window)]
      break
  return word_list

def merge_belonging(word_list):
  window_size = 2 
  sliding_steps = len(word_list) - window_size + 1 
  for i in range(sliding_steps):
    cur_window = word_list[i:i+window_size]
    if is_all_english(cur_window[0]) and cur_window[1] == "'S":
      word_list[i:i+window_size] = [''.join(cur_window)]
      break
  return word_list
'''
def seg_sentence(sentences, dict_path):
    ret = []
    #jieba.load_userdict(dict_path)
    tokenizer = WordsSegment(dict_path)
    for sentence in sentences:
        uttid, text = sentence.split()[0], sentence.split()[1:]
        for idx, sub_text in enumerate(text):
            if is_all_english(sub_text):
                continue
            else:
                seg_words = tokenizer.cut(sub_text)
                #seg_words = jieba.cut(sub_text)
                text[idx] = ' '.join(seg_words)
        #print('{} {}'.format(uttid, ' '.join(seg_words)))
        ret.extend(['{} {}'.format(uttid, ' '.join(text))])
    return ret
'''

def seg_sentence(sentences, dict_path):
    pattern = re.compile('([\u4e00-\u9fa5]+)')
    ret = []
    tokenizer = WordsSegment(dict_path)
    for sentence in sentences:
        text = sentence
        text = pattern.split(text)
        clean_text = []
        for idx, sub_text in enumerate(text):
            if len(sub_text.strip()) == 0:
                continue
            if sub_text == "'" or sub_text == "''":
                continue
            if is_all_english(sub_text):
                seg_words = wordninja.split(sub_text)
                seg_words = merge_abbreviation(seg_words)
                seg_words = merge_belonging(seg_words)
            else:
                seg_words = tokenizer.cut(sub_text)
            clean_text.append( ' '.join(seg_words))
        # ret.extend(['{} {}'.format(uttid, ' '.join(clean_text))])
        ret.extend([' '.join(clean_text)])
    return ret
        
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', '-t', default=False, dest='text', type=str)
    parser.add_argument('--dict', '-d', required=True, dest='dict', type=str)
    parser.add_argument('--nj', default=1, dest='n_process', type=int)
    parser.add_argument('--output_text', '-o', required=True,  dest='output_text', type=str)
    args = parser.parse_args()

    pattern = "[^0-9A-Za-z\u4e00-\u9fa5\']"
    utt2clean_text = []
    with open(args.text, 'r', encoding='utf8') as f:
        for line in f.readlines():
            clean_text = re.sub(pattern, '', ''.join(line.rstrip()))
            utt2clean_text.append(clean_text)
    n_process = args.n_process
    pool = Pool(n_process)
    dict_path = args.dict
    #jieba.set_dictionary(args.dict)
    l = len(utt2clean_text)
    lines_per_process = l//n_process
    #test = seg_sentence(utt2clean_text[0:min(l, 1*lines_per_process)], dict_path)
    ret = []
    print("Start segmentation")
    for i in range(n_process+1):
        ret.append(pool.apply_async(seg_sentence, args=(utt2clean_text[i*lines_per_process:min(l, (i+1)*lines_per_process)], dict_path, )))
    pool.close()
    pool.join()
    print("Segmentation finished")
    f = open(args.output_text, 'w', encoding='utf8')
    for link in tqdm(ret):
        for sentence in link.get():
            f.write(sentence + '\n')
    f.close()
