## generate_heteroskedastic_sin
##
## Generates data (train, valid, test) with x taking values uniformly
## in [0, 1], while the value y is drawn according to
##
##  y = sin(2 * pi * x) + Uni[-.1, 1 + cos(4 * pi * x)]
##
## so that it is fairly heteroskedastic noise
import numpy as np

# Use it like this:
#
# X_train, y_train = generate()
# X_valid, y_valid = generate()
# X_test, y_test = generate()
def generate():
    n_sample = 200
    x = np.sort(np.random.uniform(size=n_sample))
    y = np.sin(2 * np.pi * x) - 0.1 + \
        (1.1 + np.cos(4 * np.pi * x)) * np.random.uniform(size=n_sample)
    return x, y
