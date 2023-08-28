# 基于词典的分词

class WordPath:
    def __init__(self, prefix, word_history):
        self.prefix = prefix
        self.word_history = word_history
        self.is_valid = True


class WordsSegment:
    def __init__(self, words_file):
        self.max_match_word_len = 7
        self.limit_word_set = {}
        with open(words_file, "r", encoding="utf-8") as f:
            for index, line in enumerate(f.readlines()):
                self.limit_word_set[line.split()[0].strip()] = index
        self.paths = [WordPath([], [])]

    def reset(self):
        self.paths = [WordPath([], [])]

    def conbine_char(self, word):
        str_word = ''
        for char in word:
            str_word += char
        max_length = 4
        new_word = []
        jump = 0
        split_pos = 0
        for i in range(len(word)):
            if jump > 0:
                jump -= 1
                continue
            old_length = len(new_word)
            for j in range(max_length, 1, -1):
                if self.limit_word_set.__contains__(str_word[i:i + j]):
                    new_word.append(str_word[i:i + j])
                    jump = j - 1
                    break
            new_length = len(new_word)
            if old_length == new_length:
                new_word.append(word[i])
        for index, word in enumerate(new_word[::-1]):
            if len(word) == 1 and len(new_word[len(new_word) - index - 2]) > 1:
                split_pos = len(new_word) - index - 1
                break
            if index == 0 and len(word) > 1:
                split_pos = len(new_word)
                break
        return new_word, split_pos

    def merge_paths(self):
        paths_merged = {}
        for path in self.paths:
            if path.is_valid:
                history_str = "".join(path.word_history)
                if history_str not in paths_merged:
                    paths_merged[history_str] = path
                elif len(path.word_history) < len(paths_merged[history_str].word_history):
                    paths_merged[history_str].word_history = path.word_history
                elif len(path.word_history) == len(paths_merged[history_str].word_history):
                    current = 0
                    past = 0
                    for word in path.word_history:
                        if len(word) > 1:
                            current += self.limit_word_set[word]
                    for word in paths_merged[history_str].word_history:
                        if len(word) > 1:
                            past += self.limit_word_set[word]
                    if current < past:
                        paths_merged[history_str].word_history = path.word_history
        self.paths = [v for v in paths_merged.values()]

    def add_new_word(self, word, index):
        new_paths = []
        for path in self.paths:
            if self.limit_word_set.__contains__("".join(path.prefix) + word):  # if match one word, add a new path.
                new_prefix = []
                new_word_history = path.word_history + ["".join(path.prefix) + word]
                new_path = WordPath(new_prefix, new_word_history)
                new_paths.append(new_path)

            path.prefix += [word]
            if len(path.prefix) > self.max_match_word_len:
                if len(path.word_history) >= 1:
                    conbine_char, split_pos = self.conbine_char(path.prefix)
                    path.word_history.extend(conbine_char[:split_pos])
                    path.prefix = conbine_char[split_pos:]
                else:
                    path.word_history.extend(path.prefix)
                    path.prefix = []
                    if len(path.word_history) == index + 1 and len(self.paths) > 1:
                        path.is_valid = False
                    elif len(self.paths) == 1:
                        conbine_char, _ = self.conbine_char(path.prefix)
                        path.word_history.extend(conbine_char)
                        path.prefix = []

        self.paths += new_paths
        self.merge_paths()

    def get_best_result(self):
        for path in self.paths:
            conbine_char, _ = self.conbine_char(path.prefix)
            path.word_history.extend(conbine_char)
            path.prefix = []
        paths = sorted(self.paths, key=lambda x: len(x.word_history))
        small_length = len(paths[0].word_history)
        same_length = []
        for path in self.paths:
            if len(path.word_history) == small_length:
                same_length.append(path)
        return same_length[-1].word_history

    def cut(self, sentence):
        self.reset()
        sentence = sentence.replace(" ", "")
        for index, word in enumerate(sentence):
            self.add_new_word(word, index)
        return self.get_best_result()


# tokenizer = WordsSegment("data/local/dict/cn_dict.txt")
# cut = tokenizer.cut("今天天气真好")
# print(cut)
