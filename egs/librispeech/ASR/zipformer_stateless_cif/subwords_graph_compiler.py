# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../LICENSE for clarification regarding multiple authors
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


from typing import List
from pathlib import Path
import re
import wordninja

import k2
import torch

from icefall.lexicon import Lexicon
from local.no_fre_dp_wordseg import WordsSegment
import string

def read_lexicon_to_dict(filename: Path):
    ret = {}
    with open(filename, 'r') as f:
        for line in f:
            pieces = line.rstrip('\t').split(' ')
            word = pieces[0]
            tokens = " ".join(pieces[1:])
            ret[word] = tokens
    return ret

class SubwordsCtcGraphCompiler(object):
    def __init__(
        self,
        lexicon: Lexicon,
        device: torch.device,
        dict_path: Path,
        oov: str = "<unk>",
        need_repeat_flag: bool = False,
    ):
        """
        Args:
          lexicon:
            It is built from `data/lang/lexicon.txt`.
          device:
            The device to use for operations compiling transcripts to FSAs.
          oov:
            Out of vocabulary word. When a word in the transcript
            does not exist in the lexicon, it is replaced with `oov`.
          need_repeat_flag:
            If True, will add an attribute named `_is_repeat_token_` to ctc_topo
            indicating whether this token is a repeat token in ctc graph.
            This attribute is needed to implement delay-penalty for phone-based
            ctc loss. See https://github.com/k2-fsa/k2/pull/1086 for more
            details. Note: The above change MUST be included in k2 to open this
            flag.
        """
        L_inv = lexicon.L_inv.to(device)
        assert L_inv.requires_grad is False

        assert oov in lexicon.token_table

        self.L_inv = k2.arc_sort(L_inv)
        # self.oov_id = lexicon.token_table[oov]
        self.oov_id = lexicon.token_table[oov]
        self.word_table = lexicon.word_table
        self.token_table = lexicon.token_table
        self.lexicon_table = read_lexicon_to_dict(lexicon.lang_dir / "lexicon.txt")
        
        max_token_id = max(lexicon.tokens)
        ctc_topo = k2.ctc_topo(max_token_id, modified=False)

        self.ctc_topo = ctc_topo.to(device)

        if need_repeat_flag:
            self.ctc_topo._is_repeat_token_ = (
                self.ctc_topo.labels != self.ctc_topo.aux_labels
            )

        self.device = device
        self.tokenizer = WordsSegment(dict_path)

    def compile(self, texts: List[str]) -> k2.Fsa:
        """Build decoding graphs by composing ctc_topo with
        given transcripts.

        Args:
          texts:
            A list of strings. Each string contains a sentence for an utterance.
            A sentence consists of spaces separated words. An example `texts`
            looks like:

                ['hello icefall', 'CTC training with k2']

        Returns:
          An FsaVec, the composition result of `self.ctc_topo` and the
          transcript FSA.
        """
        transcript_fsa = self.convert_transcript_to_fsa(texts)

        # NOTE: k2.compose runs on CUDA only when treat_epsilons_specially
        # is False, so we add epsilon self-loops here
        fsa_with_self_loops = k2.remove_epsilon_and_add_self_loops(transcript_fsa)

        fsa_with_self_loops = k2.arc_sort(fsa_with_self_loops)

        decoding_graph = k2.compose(
            self.ctc_topo, fsa_with_self_loops, treat_epsilons_specially=False
        )

        assert decoding_graph.requires_grad is False

        return decoding_graph

    def texts_to_ids(self, texts: List[str]) -> List[List[int]]:
        """Convert a list of texts to a list-of-list of word IDs.

        Args:
          texts:
            It is a list of strings. Each string consists of space(s)
            separated words. An example containing two strings is given below:

                ['HELLO ICEFALL', 'HELLO k2']
        Returns:
          Return a list-of-list of word IDs.
        """
        token_ids_list = []
        for text in texts:
            token_ids = []
            for word in self._seg_sentence(text).split(" "):
                if word in self.word_table:
                    # word_ids.append(self.word_table[word])
                    tokens = self.lexicon_table[word].rstrip("\n")
                    for token in tokens.split(" "):
                        if token in self.token_table:
                            token_ids.append(self.token_table[token])
                else:
                    for i in range(len(word)):
                        token_ids.append(self.oov_id)
                    # word_ids.append(self.oov_id)
            token_ids_list.append(token_ids)
            
        return token_ids_list

    def convert_transcript_to_fsa(self, texts: List[str]) -> k2.Fsa:
        """Convert a list of transcript texts to an FsaVec.

        Args:
          texts:
            A list of strings. Each string contains a sentence for an utterance.
            A sentence consists of spaces separated words. An example `texts`
            looks like:

                ['hello icefall', 'CTC training with k2']

        Returns:
          Return an FsaVec, whose `shape[0]` equals to `len(texts)`.
        """
        word_ids_list = []
        for text in texts:
            word_ids = []
            for word in text.split():
                if word in self.word_table:
                    word_ids.append(self.word_table[word])
                else:
                    word_ids.append(self.oov_id)
            word_ids_list.append(word_ids)

        word_fsa = k2.linear_fsa(word_ids_list, self.device)

        word_fsa_with_self_loops = k2.add_epsilon_self_loops(word_fsa)

        fsa = k2.intersect(
            self.L_inv, word_fsa_with_self_loops, treat_epsilons_specially=False
        )
        # fsa has word ID as labels and token ID as aux_labels, so
        # we need to invert it
        ans_fsa = fsa.invert_()
        return k2.arc_sort(ans_fsa)

    def _seg_sentence(self, sentence: str) -> str:

        pattern = re.compile('([\u4e00-\u9fa5]+)')
        words = pattern.split(sentence)
        clean_text = []
        for idx, word in enumerate(words):
            
            if len(word.strip()) == 0:
                continue
            if word == "'" or word == "''":
                continue
            if self.is_all_english(word):
                seg_words = wordninja.split(word)
                seg_words = self._merge_abbreviation(seg_words)
                seg_words = self._merge_belonging(seg_words)
            else:
                seg_words = self.tokenizer.cut(word)
            clean_text.append(" ".join(seg_words))
        
        return " ".join(clean_text)
    
    def _merge_belonging(self, word_list):
        window_size = 2 
        sliding_steps = len(word_list) - window_size + 1 
        for i in range(sliding_steps):
            cur_window = word_list[i:i+window_size]
            if is_all_english(cur_window[0]) and cur_window[1] == "'S":
                word_list[i:i+window_size] = [''.join(cur_window)]
                break
        return word_list
    
    def _merge_abbreviation(self, word_list):
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
    
    def is_all_english(self, word):
        for char in word:
            if char not in string.ascii_uppercase and char != "'" :
                return False
        return True