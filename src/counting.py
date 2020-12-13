import glob
from multiprocessing.pool import Pool

from tqdm import tqdm


def _mlti_read(params):
    file = params
    tweet_count = 0
    prev_line = ""
    # print(file)
    with open(file) as F:
        for li in F:
            if len(li.strip())==0:
                tweet_count += 1

    return tweet_count

files = []
for f in glob.glob("/disk1/sajad/twitter-data/*.txt"):
    files.append((f))
    # _mlti_read(files[-1])

pool = Pool(24)
# import pdb;pdb.set_trace()

total_tweets = 0
for d in tqdm(pool.imap_unordered(_mlti_read, files)):
    total_tweets += d

print('There are {} total tweets.'.format(total_tweets))