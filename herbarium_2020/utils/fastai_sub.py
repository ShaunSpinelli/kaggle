# --- 100 characters ------------------------------------------------------------------------------
# Created by: Shaun 2019/01/01

"""Fast ai submission cause the kernel kept crashing"""

from fastai.vision import *

# In[2]:
print("running")

TRAIN_PATH = Path("/home/shaun/personal/kaggle-data/nybg2020/train/")
df = pd.read_csv(TRAIN_PATH / "train.csv")
df_train = df[["file_name", "category_id"]]

# In[3]:


# df_train = df_merged[["file_name", "family_id"]]


# In[4]:


df_train.head()

# In[5]:


df_train = df_train.rename(columns={"category_id": "label", "file_name": " name"})


# In[6]:


# def sample_class(x):
#     if len(x) < 10:
#         return x.sample(n=10, replace=True)
#     return x


# In[7]:


# counts  = df_train.label.value_counts() # Get Class counts

# labels_keep = counts.keys()[counts>9] # Get labels that have more then 9 exmaples

# df_new_train = df_train[df_train.label.isin(labels_keep)] # Select those items from df

# # Do a random sample from data frame of 20 items
# new_df = df_new_train.groupby('label').apply(lambda x: x.sample(n=20, replace=True)).reset_index(drop = True)


# In[8]:


# TRAIN_PATH = Path("/home/shaun/personal/kaggle-data/nybg2020/train/")
# df = pd.read_csv(TRAIN_PATH/"train.csv")
# df_train = df[[ "file_name", "category_id"]]


# In[9]:


new_df = df_train

# # In[10]:
#
#
# len(new_df)
#
#
# # In[11]:
#
#
# # there is a better way to do this
# def sample(df, n=4):
#     validation_set = {}
#     bool_arr = []
#     for i, row in df.iterrows():
#         if validation_set.get(row["label"], 0) == n:
#             bool_arr.append(False)
#         else:
#             validation_set[row["label"]] = validation_set.get(row["label"], 0) + 1
#             bool_arr.append(True)
#
#     return bool_arr


# In[12]:


# valid = sample(new_df, n=4)


# In[13]:


# new_df["is_valid"] = valid





# In[14]:


new_df.tail()

# In[15]:


classes = new_df["label"].unique()

# In[16]:


classes

# In[17]:


len(classes)

# In[18]:


classes_map = {classes[i]: i for i in range(len(classes))}

# In[19]:


classes_map_key = {v: k for k, v in classes_map.items()}



new_labels = [classes_map[i] for i in new_df["label"]]



new_df["new_labels"] = new_labels



data = (ImageList.from_df(new_df, TRAIN_PATH)
        .split_none()
        .label_from_df(cols="new_labels")
        .transform(size=256)
        .databunch(bs=64))



learn = cnn_learner(data, models.resnet34, metrics=accuracy).to_fp16()




learn.load("/home/shaun/personal/kaggle-data/nybg2020/models/model-all-classes-unfreeze")


# get test data
TEST_PATH = Path("/home/shaun/personal/kaggle-data/nybg2020/test/")
with open(TEST_PATH / "metadata.json", encoding="utf8", errors='ignore') as json_file:
    data = json.load(json_file)
df_test = pd.DataFrame.from_dict(data["images"])



test_data = ImageImageList.from_df(df_test, TEST_PATH, cols="file_name")
learn.data.add_test(test_data)

print("getting preds")

preds, y = learn.get_preds(ds_type=DatasetType.Test)


res_flat = np.argmax(preds, axis=1)


res_correct = [classes_map_key[i] for i in res_flat]




df_test["Predicted"] = res_correct



sub = df_test[["id", "Predicted"]]




subm_file = "/home/shaun/personal/kaggle-data/nybg2020/sub-23"




sub.to_csv(subm_file, index=False)




# get_ipython().system(
#     ' kaggle competitions submit herbarium-2020-fgvc7 -f $subm_file -m "fastai sub"')






