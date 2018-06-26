import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list

class GenericElement:
    def __init__(self):
        return
    @property
    def row_count(self):
        return self._data.shape[0]
    @property
    def data(self):
        return self._data

class DiscreteMatrix(GenericElement):
    def __init__(self,data,title=''):
        super()
        self._data = data
        self.title = title
        self.type = 'DiscreteMatrix'

class ContinuousMatrix(GenericElement):
    def __init__(self,data,title='',row_standard=False,row_mean=False,center=0,cmap='RdBu_r',row_cluster=None):
        super()
        self._data = data
        self._center = center
        self._cmap = cmap
        if row_standard:
            d = self._data.T
            nd = (d-d.min())/(d.max()-d.min())
            self._data = nd.T
        if row_mean:
            d = self._data.T
            nd = (d-d.mean())/d.std()
            self._data = nd.T
        if row_cluster:
            v = leaves_list(linkage(self._data,method=row_cluster,optimal_ordering=True))
            ordered = self._data.index[v]
            #print(ordered)
            self._data = self._data.loc[self._data.index[v]]
        self.title = title
        self.type = 'ContinuousMatrix'

class StackedHeatmap(GenericElement):
    def __init__(self):
        super()
        self.columns = [] # our column order
        self._levels = [] # from top to bottom how the levels are stacked
        self._column_widths = [] # heatmap will be on left and the scales will be on the right
        self._level_heights = [] # default to the
        return
    def add_level(self,level):
        self._levels.append(level)
    def predict_height_ratios(self):
        return [x.row_count for x in self._levels]
    def draw(self,figure_size=(10,10),width_ratios=(10,1),height_ratios=None):
        height_ratios = self.predict_height_ratios() if height_ratios is None else height_ratios
        fig, ax = plt.subplots(len(self._levels),2,
                               figsize=figure_size,
                               gridspec_kw={
                                   'height_ratios': height_ratios,
                                   'width_ratios':width_ratios
                               })
        #Draw each of the subplots
        for i,level in enumerate(self._levels):
            if level.type == 'ContinuousMatrix': self._draw_continuous_heatmap(i,level,ax)
        # Finish the plot
        return fig, ax

    def _draw_continuous_heatmap(self,i,level,ax):
        if len(self._levels) > 1:
            axi = ax[i][0] # the heatmap
            axj = ax[i][1] # the legend
        else:
            axi = ax[0] # the heatmap
            axj = ax[1] # the legend
        sns.heatmap(level.data.loc[:,self.columns],cbar_ax=axj,ax=axi,cmap=level._cmap,center=level._center)
        if i < len(self._levels)-1:
            axi.get_xaxis().set_visible(False)
        labs = list(level.data.index)
        axi.set_ylabel('')
        axi.set_yticks([x+0.5 for x in range(0,len(labs))])
        axi.set_yticklabels(labs)
        for t in axi.get_yticklabels(): t.set_rotation(0)


