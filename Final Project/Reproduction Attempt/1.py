import pickle
import pandas as pd
from collections import Counter
from copy import deepcopy
from itertools import combinations, groupby
from pathlib import Path

try:
    import networkx as nx
except ImportError:
    print("No networkx installed!")

from tqdm import tqdm
from transformers.trainer_utils import set_seed

tqdm.pandas()

# download + scp to server + extract
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

def get_Ntop_cats(inv_cat_bids, n=50):
    # get most common cats
    f_cat_cnt = Counter({k: len(v) for k, v in inv_cat_bids.items()})
    f_cats = {c for c, v in f_cat_cnt.most_common(n)}
    return f_cats


def make_cat_Ntuples(f_inv_cat_combis, n=2):
    f_cat_pairs = Counter()

    for cat_group in tqdm(f_inv_cat_combis.keys()):
        if len(cat_group) < n:
            continue
        it = combinations(cat_group, n)
        # repeat (#num_businesses) + chain combis
        f_cat_pairs.update(it)
        
    return f_cat_pairs

def make_graph(f_cat_pairs):
    g_from, g_to, g_value = zip(*((k1, k2, n) for (k1, k2), n in tqdm(f_cat_pairs.most_common())))

    g_df = pd.DataFrame({"from": g_from, "to": g_to, "value": g_value})
    G = nx.from_pandas_edgelist(g_df, "from", "to", create_using=nx.Graph())
    
    return G


def make_NxN_map(f_cats, f_cat_pairs):
    f_cats = list(f_cats)
    array = list()
    for i, cat1 in enumerate(tqdm(f_cats)):
        array_row = list()
        for j, cat2 in enumerate(f_cats):
            array_row.append(f_cat_pairs.get((cat1, cat2), f_cat_pairs.get((cat2, cat1), 0)))
        array.append(array_row)
    df_cm = pd.DataFrame(array, index=list(f_cats), columns=list(f_cats))
    
    # dataframe, NxN array + labels
    return df_cm, array, f_cats


def print_category_tree(map_categories):
    root_categories = get_root_category_items(map_categories)
    
    def _print_cat_list_rec(lst_cats, level=0):
        for item in sorted(lst_cats, key=lambda x: x["title"]):
            if level:
                print("  " * level, end="")
            print(f"""{item["title"]} [{item["alias"]}]""", end="")
            if item["children"]:
                print(f""" [#{len(item["children"])} children]""")
            else:
                print()
            
            children = get_children_category_item_list(map_categories, item["alias"])
            _print_cat_list_rec(children, level=level + 1)
            
    _print_cat_list_rec(root_categories, level=0)


def print_category_tree_with_num_businesses(map_categories, inv_cat_bids):
    root_categories = get_root_category_items(map_categories)
    
    def _print_cat_list_rec(lst_cats, level=0):
        for item in sorted(lst_cats, key=lambda x: x["title"]):
            cur_line = " ." * 30
            parts = list()

            if level:
                parts.append("  " * level)
            parts.append(f"""{item["title"]} [{item["alias"]}]""")
            
            str_len = sum(len(part) for part in parts)
            print("".join(part for part in parts), end="")
            print(cur_line[str_len:], end="")
            
            if item["title"] not in inv_cat_bids:
                print(" No businesses associated!")
            else:
                print(f""" {len((inv_cat_bids[item["title"]])):>5d} businesses""")
            
            children = get_children_category_item_list(map_categories, item["alias"])
            _print_cat_list_rec(children, level=level + 1)
            
            if level == 0:
                print()
            
    _print_cat_list_rec(root_categories, level=0)
    

def print_category_tree_with_num_businesses_rec(map_categories, inv_cat_bids, map_cat_name2id):
    root_categories = get_root_category_items(map_categories)
    
    def _print_cat_list_rec(lst_cats, level=0):
        for item in sorted(lst_cats, key=lambda x: x["title"]):
            cur_line = " ." * 30
            parts = list()

            if level:
                parts.append("  " * level)
            parts.append(f"""{item["title"]} [{item["alias"]}]""")
            
            str_len = sum(len(part) for part in parts)
            print("".join(part for part in parts), end="")
            print(cur_line[str_len:], end="")
            
            businesses = get_businesses_in_category_branch(inv_cat_bids, item["title"], map_categories, map_cat_name2id)
            businesses_self = get_businesses_in_category(inv_cat_bids, item["title"])
            if not businesses:
                print(" No businesses associated!")
            else:
                businesses = set(businesses)
                print(f""" {len(businesses):>5d} businesses""", end="")
                if len(businesses) != len(businesses_self):
                    print(f""" (self: {len(businesses_self)})""", end="")
                print()
            
            children = get_children_category_item_list(map_categories, item["alias"])
            _print_cat_list_rec(children, level=level + 1)
            
            if level == 0:
                print()
            
    _print_cat_list_rec(root_categories, level=0)
    
    
def print_category_tree_with_num_businesses_root(map_categories, inv_cat_bids, map_cat_name2id):
    root_categories = get_root_category_items(map_categories)
    
    for item in sorted(root_categories, key=lambda x: x["title"]):
        cur_line = " ." * 25
        parts = [f"""{item["title"]} [{item["alias"]}] """]

        str_len = sum(len(part) for part in parts)
        print("".join(part for part in parts), end="")
        print(cur_line[str_len:], end="")

        businesses = get_businesses_in_category_branch(inv_cat_bids, item["title"], map_categories, map_cat_name2id)
        businesses_self = get_businesses_in_category(inv_cat_bids, item["title"])

        businesses = set(businesses)
        print(f""" {len(businesses):>5d} businesses""", end="")
        if len(businesses) != len(businesses_self):
            print(f""" (self: {len(businesses_self)})""", end="")
        print()
        

def print_category_tree_with_num_businesses_root2(map_categories, inv_cat_bids, map_cat_name2id):
    root_categories = get_root_category_items(map_categories)
    for item in root_categories:
        item["businesses"] = get_businesses_in_category_branch(inv_cat_bids, item["title"], map_categories, map_cat_name2id)
        item["businesses_self"] = get_businesses_in_category(inv_cat_bids, item["title"])
    
    for item in sorted(root_categories, key=lambda x: len(set(x["businesses"]))):
        cur_line = " ." * 25
        parts = [f"""{item["title"]} [{item["alias"]}] """]

        str_len = sum(len(part) for part in parts)
        print("".join(part for part in parts), end="")
        print(cur_line[str_len:], end="")

        businesses = item["businesses"]
        businesses_self = item["businesses_self"]

        businesses = set(businesses)
        print(f""" {len(businesses):>5d} businesses""", end="")
        if len(businesses) != len(businesses_self):
            print(f""" (self: {len(businesses_self)})""", end="")
        print()

def print_2category_compare(inv_cat_bids, map_categories, map_cat_name2id, cat_name_i, cat_name_j):
    businesses_i = get_businesses_in_category_branch(inv_cat_bids, cat_name_i, map_categories, map_cat_name2id)
    businesses_j = get_businesses_in_category_branch(inv_cat_bids, cat_name_j, map_categories, map_cat_name2id)
    
    cat_name_i += ":"
    cat_name_j += ":"
    width = max(12, len(cat_name_i), len(cat_name_j))

    print(f"""{cat_name_i:<{width}} {len(set(businesses_i)):>5d}""")
    print(f"""{cat_name_j:<{width}} {len(set(businesses_j)):>5d}""")
    print(f"""Both: {"same:":>{width - 6}} {len(set(businesses_i) & set(businesses_j)):>5d}""")
    print(f"""{"total:":>{width}} {len(set(businesses_i) | set(businesses_j)):>5d}""")

# N positive + N negative
# --> 2N pos+neg (not same-sentiment)
num_pairs_per_class = 2

#: number of negative same-sentiment samples same as positive same-sentiment samples
num_pairs_negative = 2 * num_pairs_per_class

#: whether for a single side (good or bad) there can be multiple occurrences of the same review
#: may need to check afterwared that not by chance same pairing happens ...
repeatable_on_side = False

fn_yelp_reviews = data_yelp_path / "review.json"
df = load_reviews(fn_yelp_reviews)

fn_yelp_topics = data_yelp_path / "business.json"
bids_not_cats = set()
inv_bid_cats = load_topics(fn_yelp_topics, bids_not_cats=bids_not_cats)

inv_cat_bids = make_map_cats(inv_bid_cats)

inv_cat_combis = make_cat_combis(inv_bid_cats)

fn_yelp_catgory_tree = data_yelp_path / "all_category_list.json"
map_categories, map_cat_name2id, lst_root_categories = load_category_tree(fn_yelp_catgory_tree)

cache_root_category_businesses_df(df, inv_cat_bids, map_categories, map_cat_name2id)

# number of businesses
print(f"Number of businesses total: {len(inv_bid_cats.keys())}")
# number of reviews (total)
print(f"Number of reviews total: {df.rid.count()}")

set_seed(42)
fn_yelp_df = data_yelp_path / "df_traindev.p"

df = filter_min_review_freq(df, min_ratings=5)
df = filter_both_good_bad(df)

df_traindev = make_or_load_pairs(df, inv_cat_bids, str(fn_yelp_df), num_pairs_per_class=2)


set_seed(42)
fn_yelp_df = data_yelp_path / "df_traindev4_typed.p"

df = filter_min_review_freq(df, min_ratings=8)
df = filter_both_good_bad(df)

df_traindev = make_or_load_pairs(df, inv_cat_bids, str(fn_yelp_df), num_pairs_per_class=4)

set_seed(42)
fn_yelp_df = data_yelp_path / "df_traindev_over_business.p"

df = filter_min_review_freq(df, min_ratings=5)
df = filter_both_good_bad(df)

df_traindev = make_or_load_pairs_over_businesses(df, inv_cat_bids, str(fn_yelp_df))

fn_yelp_df = data_yelp_path / "df_traindev4_typed.p"

with open(fn_yelp_df, "rb") as fp:
    traindev_df = pickle.load(fp)

fn_yelp_df = data_yelp_path / "df_traindev_test.p"

traindev_df, test_df = split_df(traindev_df, ratio=0.1, do_shuffle=True, random_state=42, name_train="traindev", name_dev="test")

with open(fn_yelp_df, "wb") as fp:
    pickle.dump(traindev_df, fp, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_df, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open(fn_yelp_df, "rb") as fp:
    traindev_df = pickle.load(fp)
    test_df = pickle.load(fp)

root_path = data_yelp_b_tdt_path
#root_path = data_yelp_b_rand_tdt_path

write_pair_tdt_tsv(root_path, traindev_df, split_test=0.1, split_dev=0.3)

print('run.....')
print('! ln -s test.tsv {root_path}/pred.tsv')
print('! ls -lh {root_path}')
