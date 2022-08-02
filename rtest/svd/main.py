from surprise import SVD
from surprise import Dataset, Reader
import os

reader = Reader(line_format='user item rating', sep=',')

data = Dataset.load_from_file('mydata.csv', reader=reader)

trainset = data.build_full_trainset()

algo = SVD()
algo.fit(trainset)


# we can now query for specific predicions
uid = str(5)  # raw user id
iid = str(1)  # raw item id

# get a prediction for specific users and items.
pred = algo.predict(uid, iid)
print('rating of user-{0} to item-{1} is '.format(uid, iid), pred.est)# rating of user-5 to item-1
