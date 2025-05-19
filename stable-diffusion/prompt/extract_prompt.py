import json
import random

with open("annotations/captions_val2014.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open('prompt.txt', 'w') as f:
    for i, index in enumerate(random.sample(range(len(data['annotations'])), 10000)):
        caption = data['annotations'][index]['caption']
        caption = caption.split('\n')[0].strip()
        f.write(caption +'\n')
        print(i, caption)

print('done')
