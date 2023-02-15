import numpy as np
import matplotlib.pyplot as plt


'''
    Stores and updates the data for one matplotlib line.
    If window_x is not None, the plotted data is restricted to that many data points
'''
class DynamicLine():

    def __init__(self, window_x, line):

        self.window_x = window_x
        self.xdata = []
        self.ydata = []
        self.line = line

    def add_point(self, _x, _y):

        if self.window_x is not None and len(self.xdata) >= self.window_x:

            self.xdata.pop(0)
            self.ydata.pop(0)

        self.xdata.append(_x)
        self.ydata.append(_y)

        #Update data (with the new _and_ the old points)
        self.line.set_xdata(self.xdata)
        self.line.set_ydata(self.ydata)

'''
    A collection of DynamicLines, used to pass on data and redraw in its update function
'''
class DynamicPlot():

    def __init__(self, title = None, xlabel = None, ylabel = None, window_x = None, num_plots=1):
        plt.ion()
        self.figure, self.axes = plt.subplots(num_plots, 1, sharex=True)
        # if not isinstance(self.axes, list):
        #     self.axes = [self.axes]
        self.lines = []
        for ax in self.axes:
            ax.set_autoscaley_on(True)
            ax.grid()
        self.window_x = window_x

        if isinstance(title, list) and len(title) == len(num_plots):
            for index, ax in enumerate(self.axes):
                ax.set_title(title[index])
        else:
            self.axes[0].set_title(title)

        if xlabel:
            self.axes[-1].set_xlabel(xlabel)

        if isinstance(ylabel, list) and len(ylabel) == len(num_plots):
            for index, ax in enumerate(self.axes):
                ax.set_ylabel(ylabel[index])
        elif ylabel:
            self.axes[0].set_ylabel(ylabel)
            

    def add_line(self, label = 'lineName', ax_id=0):
        line, = self.axes[ax_id].plot([],[], label = label)
        self.lines.append(DynamicLine(self.window_x, line))
        self.axes[ax_id].legend(loc='upper center')

    ''' update
     Accepts one x data point (eg. timestep) and an data array
     which is ordered based on how the lines were added to the DynamicPlot
    '''
    def update(self, x, data):
        for i in range(len(data)):
            self.lines[i].add_point(x, data[i])

        #Need both of these in order to rescale
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


# Example Usage
# import time
# d = DynamicPlot(window_x = 30, title = 'Trigonometry', xlabel = 'X', ylabel= 'Y')
# d.add_line('sin(x)')
# d.add_line('cos(x)')
# d.add_line('cos(.5*x)')
#
# for i in np.arange(0,40, .2):
#     d.update(i, [np.sin(i), np.cos(i), np.cos(i/.5)])
#     time.sleep(.01)