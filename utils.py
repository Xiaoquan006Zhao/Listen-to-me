import os
import json
import re
import string


def process_hotwords(hotword_file):
    fst_dict = {}
    hotword_msg = ""

    if hotword_file.strip() != "":
        if os.path.exists(hotword_file):
            with open(hotword_file, encoding="utf-8") as f_scp:
                hot_lines = f_scp.readlines()
                for line in hot_lines:
                    words = line.strip().split(" ")
                    if len(words) < 2:
                        print("Please check the format of hotwords")
                        continue
                    try:
                        fst_dict[" ".join(words[:-1])] = int(words[-1])
                    except ValueError:
                        print("Please check the format of hotwords")
            hotword_msg = json.dumps(fst_dict)
        else:
            hotword_msg = hotword_file

    return hotword_msg


def post_process_funasr_result(result, remove_punctuation=False):
    text = result[0]["text"]
    text = re.sub(r"<\|.*?\|>", "", text)

    if remove_punctuation:
        punctuation_chars = r"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~。，、；：？！「」『』（）《》【】…—～"
        text = re.sub(f"[{re.escape(punctuation_chars)}]", "", text)

    return text
