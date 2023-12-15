# ridge-prostate-dataprep.py
#
# Assuming that df is a pandas DataFrame of lprostate data available,
# the following code will standardize the data into matrices (X_train,
# y_train) for the training data and (X_test, y_test) for test data.

n = df.shape[0]
train_size = int(n * 0.6)
perm = np.random.choice(n, n, replace=False)
df_train = df.iloc[perm[:train_size]]
df_test = df.iloc[perm[train_size:]]

X_train = df_train.drop('lpsa', axis=1)
y_train = df_train['lpsa']
X_test = df_test.drop('lpsa', axis=1)
y_test = df_test['lpsa']

X_train_mean = X_train.mean()
X_train_sd = X_train.std()
y_train_mean = y_train.mean()

y_train -= y_train_mean
X_train = (X_train - X_train_mean) / X_train_sd
y_test -= y_train_mean
X_test = (X_test - X_train_mean) / X_train_sd
