import transplant
import numpy as np

# matlab = transplant.Matlab()


def myMFA(channel, dim, kw, kb):
    def calMatrix(x, y):
        y = [np.argmax(one_hot) for one_hot in y]
        y = np.array(y).reshape(len(y), 1)

        return matlab.MFA(x.astype(np.double), y.astype(np.double), channel, dim, kw, kb)

    return calMatrix