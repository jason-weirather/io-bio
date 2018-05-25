import math
import numpy as np
def correlation_matrix_to_entropy(mat):
    mat = mat.copy()
    mat.index = range(0,mat.shape[0])
    mat.columns = range(0,mat.shape[1])
    mat.columns.name = 'id_2'
    mat.index.name = 'id_1'
    tdf = mat.unstack().reset_index().rename(columns={0:'correlation'})
    tdf = tdf[tdf['id_2']<tdf['id_1']]
    tdf['corrlog'] = np.log(tdf['correlation'])
    return np.multiply(tdf['correlation'],tdf['corrlog']).multiply(-1).astype(float).sum()/tdf.shape[0]
    