import sys
import argparse
import codecs
from tqdm import tqdm


is_python2 = sys.version_info[0] == 2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        default=False,
        type=str,
    )
    parser.add_argument(
        "--output-file",
        default=False,
    )
    args = parser.parse_args()
    return args
    
def main():
    args = get_args()
    
    if args.input_file:
        f = codecs.open(args.input_file, encoding="utf-8")
    else:
        f = codecs.getreader("utf-8")(sys.stdin if is_python2 else sys.stdin.buffer)
        
    lines = f.readlines()
    
    for i in tqdm(range(len(lines))):
        x = lines[i].rstrip()
        seg_list = jieba.cut