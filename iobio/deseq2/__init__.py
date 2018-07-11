import numpy as np
from rpy2 import robjects
import rpy2.robjects.numpy2ri
import rpy2.robjects.pandas2ri
from rpy2.robjects import Formula, r
robjects.pandas2ri.activate()
from rpy2.robjects.packages import importr
deseq = importr('DESeq2')
as_df = r("as.data.frame")

def deseq2_basic(data_frame,
	numerator = 2,
	denominator = 1,
	category_field='Category',
	sample_field='Sample',
	batch_field=None,
	expression_name_field='Name',
	counts_field='NumReads'):
    # from a dataframe
    # https://stackoverflow.com/questions/41821100/running-deseq2-through-rpy2
    design = '~ `'+category_field+'`'
    if batch_field is not None: design = '~ `'+batch_field+'` + `'+category_field+'`'
    #print(design)
    design = Formula(design)
    mat = data_frame.pivot(columns=sample_field,index=expression_name_field,values=counts_field)
    mfields = [sample_field, category_field]
    if batch_field is not None: mfields += [batch_field]
    meta = data_frame[mfields].groupby(sample_field).first().loc[mat.columns]
    metaarr = {}
    metaarr[category_field] = robjects.IntVector(meta[category_field].apply(lambda x: _trans(x,numerator,denominator)))
    if batch_field is not None:
        metaarr[batch_field] = robjects.IntVector(meta[batch_field])    	
    dds0 = deseq.DESeqDataSetFromMatrix(countData=mat.astype(int),colData=robjects.DataFrame(metaarr),design=design)
    dds1 = deseq.DESeq(dds0)
    res = rpy2.robjects.pandas2ri.ri2py(as_df(deseq.results(dds1)))
    res.index = mat.index
    res.index.name = expression_name_field
    return (dds0,dds1,res,mat,meta)
def _trans(x,numerator,denominator):
	if x == numerator: return 2
	elif x == denominator: return 1
	return np.nan