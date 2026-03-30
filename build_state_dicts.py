from agb_auctionnet.data.dataset import AuctionNetDataset

data_dicts = AuctionNetDataset.build('/DATA/xuehy/ad/AAB/data/trajectory/dense_raw/trajectory_data_all.csv',
                                    split=100,
                                    action_mode='pacer')

import pickle

from tqdm import tqdm

for i, dd in tqdm(enumerate(data_dicts)):
    with open(f'/DATA/xuehy/ad/AAB/data/trajectory/dense_pacer/part_{i}.pkl', 'wb') as f:
        pickle.dump(dd, f)