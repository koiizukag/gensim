import pandas as pd
import json

train_df = pd.read_csv('/Users/kojiro.iizuka/Downloads/gn_train.tsv', delimiter='\t')
    
relations_list = train_df['article_ids'] \
        .map(lambda ids: map(str, json.loads(ids))) \
        .map(list) \
        .map(lambda ids: zip(ids[:len(ids)-1], ids[1:])) \
        .map(list) 
    
relations_list = relations_list.values.flatten()

relations = []
for r in relations_list:
  relations.extend(r)
relations = relations[:10000]

from gensim.models.poincare import PoincareModel
# from poincare import PoincareModel

import logging
import time
logger = logging.getLogger()
level = logging.INFO
logger.setLevel(level)
handler = logging.StreamHandler()
handler.setLevel(level)
logger.addHandler(handler)

start_time = time.time()
model1 = PoincareModel(relations, negative=5, workers=1)
model1.train(epochs=10, print_every=2, batch_size=10)
print(model1.kv.most_similar("30725893"))
print("--- %s seconds ---" % (time.time() - start_time))
