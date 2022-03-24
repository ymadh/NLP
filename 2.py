import pickle
from pathlib import Path

from tqdm import tqdm
from transformers.trainer_utils import set_seed

tqdm.pandas()

data_yelp_path = Path("data/sentiment/yelp/")

# ------------------------------------

# local?
data_yelp_path = Path("data_raw/sentiment/yelp/")

# local? - output path (base) for sentiment review yelp pairs
data_yelp_b_tdt_path = Path("data/sentiment/yelp-pair-b/")
data_yelp_b_rand_tdt_path = Path("data/sentiment/yelp-pair-rand-b/")
# local - output path for simple sentiment reviews yelp
data_yelp_tdt_sentiment_5_path = Path("data/sentiment/yelp-sentiment-5/")
data_yelp_tdt_sentiment_b_path = Path("data/sentiment/yelp-sentiment-b/")

dn_yelp_cached = data_yelp_path / "cached"

#  #### Load categories & topics
from data_prep import load_reviews, load_topics

# ##### Filter categories
from data_prep import filter_min_cat_combis, make_map_cats, make_cat_combis

# ##### Filter reviews
from data_prep import filter_min_review_freq, filter_both_good_bad

# ##### Filter businesses
from data_prep import filter_by_businesses, filter_by_businesses_not_same

# #### Load category tree
from data_prep import load_category_tree
from data_prep import get_root_category_items, get_children_category_item_list
from data_prep import get_businesses_in_category, get_businesses_in_category_branch


# #### Cache root category reviews in dataframes
from data_prep import cache_root_category_businesses_df, load_cached_root_category_businesses_df


# #### Positive + negative same-sentiment pairs
from data_prep import make_pairs_good_bad
from data_prep import make_pairs_good_bad_over_business

# #### Not same-sentiment pairs (combinations positive + negative)
from data_prep import make_pairs_negative
from data_prep import make_pairs_negative_over_business

# #### Dataframe for training etc.
from data_prep import make_or_load_pairs
from data_prep import make_or_load_pairs_over_businesses


# #### Make train/dev/test splits
from data_prep import split_df, write_pair_df_tsv, write_pair_tdt_tsv


# N positive + N negative
# --> 2N pos+neg (not same-sentiment)
num_pairs_per_class = 2

#: number of negative same-sentiment samples same as positive same-sentiment samples
num_pairs_negative = 2 * num_pairs_per_class

#: whether for a single side (good or bad) there can be multiple occurrences of the same review
#: may need to check afterwared that not by chance same pairing happens ...
repeatable_on_side = False

fn_yelp_df = data_yelp_path / "df_traindev4_typed.p"

with open(fn_yelp_df, "rb") as fp:
    traindev_df = pickle.load(fp)

fn_yelp_df = data_yelp_path / "df_traindev_test.p"


# store
traindev_df, test_df = split_df(traindev_df, ratio=0.1, do_shuffle=True, random_state=42, name_train="traindev", name_dev="test")

with open(fn_yelp_df, "wb") as fp:
    pickle.dump(traindev_df, fp, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_df, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open(fn_yelp_df, "rb") as fp:
    traindev_df = pickle.load(fp)
    test_df = pickle.load(fp)

root_path = data_yelp_b_tdt_path

write_pair_tdt_tsv(root_path, traindev_df, split_test=0.1, split_dev=0.3)

import os
os.system("ln -s test.tsv {}/pred.tsv".format(root_path))

os.system("ls -lh {}".format(root_path))

print("run train.sh")   