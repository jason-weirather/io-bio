import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
import pandas as pd
from collections import OrderedDict
from matplotlib import colors
from matplotlib.patches import Patch


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
    def __init__(self,data,title='',trans=None,aspect_ratio=None):
        super()
        self._data = data
        self._cmap = None
        self.aspect_ratio = aspect_ratio
        tkeys = trans.keys()
        if trans is not None:
            trans2 = OrderedDict([(x,{'num':i,'color':trans[x]}) for i,x in enumerate(tkeys)])
            self._data = self._data.applymap(lambda x: trans2[x]['num'])
            self.cmap = colors.ListedColormap([trans2[x]['color'] for x in trans2.keys()])
        else:
            raise ValueError("You need to provide a translation to colors")
        self.title = title
        self.type = 'DiscreteMatrix'
        self.legend_elements = [OrderedDict({'color':trans2[x]['color'],'label':x}) for x in trans2.keys()]


class ContinuousMatrix(GenericElement):
    # Add a continuous expression component
    def __init__(self,data,title='',row_range_normalize=False,row_mean_normalize=False,center=0,cmap='RdBu_r',row_cluster=None,legend=True,ticks='default'):
        super()
        self._data = data
        self._center = center
        self._cmap = cmap
        if row_range_normalize:
            d = self._data.T
            nd = (d-d.min())/(d.max()-d.min())
            self._data = nd.T
        if row_mean_normalize:
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
        self.do_legend = legend
        self.ticks = ticks

class StackedHeatmap(GenericElement):
    def __init__(self):
        super()
        self._columns = [] # our column order
        self._levels = [] # from top to bottom how the levels are stacked
        self._column_widths = [] # heatmap will be on left and the scales will be on the right
        self._level_heights = [] # default to the
        return
    def set_columns(self,columns):
        self._columns = columns
    @property
    def levels(self): return self._levels
    @property
    def columns(self):
        if len(self._columns)!=0: return list(self._columns)
        cols = set()
        for x in self._levels:
            cols.add(set(self._levels.columns))
        for x in self._levels:
            cols = cols&set(self._levels.columns)
        return sorted(list(cols))
    def cluster_columns(self,method='ward',levels=None):
        cols = self.columns
        if levels is None: levels = self._levels
        arr = []
        for l in self.levels:
            arr.append(l.data[cols])
        #print(arr)
        arr = pd.concat(arr)
        v = leaves_list(linkage(arr.T,method=method,optimal_ordering=True))
        ordered = arr.columns[v]
        self._columns = list(ordered) 
    def add_level(self,level):
        self._levels.append(level)
    def predict_height_ratios(self):
        return [x.row_count for x in self._levels]
    def draw(self,figure_size=(10,10),width_ratios=(10,1,2),height_ratios=None):
        height_ratios = self.predict_height_ratios() if height_ratios is None else height_ratios
        fig, ax = plt.subplots(len(self._levels),3,
                               figsize=figure_size,
                               gridspec_kw={
                                   'height_ratios': height_ratios,
                                   'width_ratios':width_ratios
                               })
        #Draw each of the subplots
        for i,level in enumerate(self._levels):
            if level.type == 'ContinuousMatrix': self._draw_continuous_heatmap(i,level,ax)
            if level.type == 'DiscreteMatrix': self._draw_discrete_colormap(i,level,ax)
        # Finish the plot
        return fig, ax

    def _draw_continuous_heatmap(self,i,level,ax):
        #if len(self._levels) > 1:
        axi = ax[i][0] # the heatmap
        axj = ax[i][1] # the legend
        ax[i][2].axis('off')
        #else:
        #    axi = ax[0] # the heatmap
        #    axj = ax[1] # the legend
        sns.heatmap(level.data.loc[:,self.columns],cbar_ax=axj, ax=axi, cmap=level._cmap, center=level._center)
        if not level.do_legend: 
            axj.axis('off')
            axj.set_visible(False)
        if i < len(self._levels)-1:
            axi.get_xaxis().set_visible(False)
        labs = list(level.data.index)
        axi.set_ylabel('')
        axi.set_yticks([x+0.5 for x in range(0,len(labs))])
        axi.set_yticklabels(labs)
        for t in axi.get_yticklabels(): t.set_rotation(0)
        axi.set_ylabel(level.title)
        #if level.ticks == 'minimal':
        #    continue

    def _draw_discrete_colormap(self,i,level,ax):
        #if len(self._levels)
        axi = ax[i][0]
        ax[i][1].axis('off')
        axj = ax[i][2]
        axi.matshow(level.data[self.columns],cmap=level.cmap)
        #axi.axis('off')
        legend_elements = [Patch(facecolor=x['color'],edgecolor='black',label=x['label']) for x in level.legend_elements]
        axj.legend(handles=legend_elements,loc='center')
        axj.axis('off')
        axi.set_yticklabels(list(level.data.index))
        axi.set_yticks(range(0,len(level.data.index)))
        axi.xaxis.set_ticks([])
        if level.aspect_ratio is not None: axi.set_aspect(level.aspect_ratio)
        axi.spines['top'].set_visible(False)
        axi.spines['bottom'].set_visible(False)
        axi.spines['left'].set_visible(False)
        axi.spines['right'].set_visible(False)
        axi.set_ylabel(level.title)

