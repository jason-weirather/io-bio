import matplotlib.pyplot as plt
import seaborn as sns

class DiscreteMatrix:
    def __init__(self,data,title=''):
        self._data = data
        self.title = title
        self.type = 'DiscreteMatrix'

class ContinuousMatrix:
    def __init__(self,data,title=''):
        self._data = data
        self.title = title
        self.type = 'ContinuousMatrix'

class StackedHeatmap:
    def __init__(self):
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
        fig, ax = plt.subplots(len(self._levels),
                               2,
                               figsize=figure_size,
                               gridspec_kw={
                                   'height_ratios': self.predict_height_ratios if height_ratios is None else height_ratios,
                                   'width_ratios':width_ratios
                               })
        # Draw each of the subplots
        for i,level in enumerate(self._levels):
            if level.type == 'ContinuousMatrix': self._draw_continuous_heatmap(i,level,ax)
        # Finish the plot
        return fig, ax

    def _draw_continuous_heatmap(self,i,level,ax):
        axi = ax[i][0] # the heatmap
        axj = ax[i][1] # the legend
        sns.heatmap(level.data.loc[self.columns])
        if i < len(self._levels)-1:
            axi.get_xaxis().set_visible(False)
        labs = list(level.data.index)
        axi.set_ylabel('')
        axi.set_yticks([x+0.5 for x in range(0,len(labs))])
        axi.set_yticklabels(labs)
        for t in axi.get_yticklabels(): t.set_rotation(0)


