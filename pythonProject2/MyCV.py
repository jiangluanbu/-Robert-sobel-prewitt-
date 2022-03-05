import numpy as np


def padding(img, p):
    new_img = np.zeros((img.shape[0]+2*p, img.shape[1]+2*p))
    new_img[p:-p, p:-p] = img
    return new_img



def suanzi(img, s,  p):
    img = padding(img, p)

    k = np.zeros(img.shape)
    h = np.array(s).shape[1]
    w = np.array(s).shape[2]

    assert s is not None
    for si in s:
        for i in range(img.shape[0]-h+1):
            for j in range(img.shape[1]-h+1):
                img[i][j] = (img[i:i+h, j:j+w]*si).sum()
        k = np.add(k, img)
    # k = k/len(s)

    return k




