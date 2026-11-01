"""銘柄ごとの仲間(相関が高い同規模の上位200銘柄)を cache/peers_w.npz に保存する(weekly_rank_build.py と同じ作り方)。"""
import numpy as np
import stock_pool as sp, stock_pool2 as p2, stock_pool3 as p3
K = 200
corr, keep = p3.corr_any(sp.NT, ret="gap", regime="up")
_, zmode, _k = p2.corr_matrix(sp.NT)
corr = np.where(zmode[:, None] == zmode[None, :], corr, -9)
nn = np.argsort(-corr, axis=1)[:, :K]; valid = np.take_along_axis(corr, nn, 1) > -9
np.savez_compressed("cache/peers_w.npz", nn=nn.astype(np.int16), valid=valid, keep=keep, zmode=zmode, codes=np.array([str(c) for c in sp.CODES]))
print("saved", nn.shape, keep.sum())
