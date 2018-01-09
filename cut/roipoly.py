# -*- coding: utf-8 -*-
# 以下是可以连接到的事件，在事件发生时发回给你的类实例以及事件描述：
#
# | 事件名称 | 类和描述 |
# | --- | --- | --- |
# | 'button_press_event' | MouseEvent - 鼠标按钮被按下 |
# | 'button_release_event' | MouseEvent - 鼠标按钮被释放 |
# | 'draw_event' | DrawEvent - 画布绘图 |
# | 'key_press_event' | KeyEvent - 按键被按下 |
# | 'key_release_event' | KeyEvent - 按键被释放 |
# | 'motion_notify_event' | MouseEvent - 鼠标移动 |
# | 'pick_event' | PickEvent - 画布中的对象被选中 |
# | 'resize_event' | ResizeEvent - 图形画布大小改变 |
# | 'scroll_event' | MouseEvent - 鼠标滚轮被滚动 |
# | 'figure_enter_event' | LocationEvent - 鼠标进入新的图形 |
# | 'figure_leave_event' | LocationEvent - 鼠标离开图形 |
# | 'axes_enter_event' | LocationEvent - 鼠标进入新的轴域 |
# | 'axes_leave_event' | LocationEvent - 鼠标离开轴域 |

# 事件属性
#
# 所有matplotlib事件继承自基类matplotlib.backend_bases.Event，储存以下属性：
# name事件名称
# canvas生成事件的FigureCanvas实例
# guiEvent触发
# matplotlib事件的GUI事件
# 最常见的事件是按键按下 / 释放事件、鼠标按下 / 释放和移动事件。
# 处理这些事件的KeyEvent和MouseEvent类都派生自LocationEvent，它具有以下属性：
# x位置，距离画布左端的像素
# y位置，距离画布底端的像素
# inaxes
# 如果鼠标经过轴域，则为Axes实例
# xdata鼠标的x坐标，以数据坐标为单位
# ydata鼠标的y坐标，以数据坐标为单位

import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.path as mplPath

class roipoly:

    def __init__(self, fig=[], ax=[], roicolor='b'):
        if fig == []:
            fig = plt.gcf()

        if ax == []:
            ax = plt.gca()

        self.previous_point = []
        self.allxpoints = []
        self.allypoints = []
        self.start_point = []
        self.end_point = []
        self.line = None
        self.roicolor = roicolor
        self.fig = fig
        self.ax = ax
        #self.fig.canvas.draw()

        self.__ID1 = self.fig.canvas.mpl_connect('motion_notify_event', self.__motion_notify_callback)
        self.__ID2 = self.fig.canvas.mpl_connect('button_press_event', self.__button_press_callback)

        if sys.flags.interactive:
            plt.show(block=False)
        else:
            plt.show()
    ####创建掩模
    def getMask(self, currentImage):
        ny, nx ,nz= np.shape(currentImage)
        poly_verts = [(self.allxpoints[0], self.allypoints[0])]
        for i in range(len(self.allxpoints)-1, -1, -1):
            poly_verts.append((self.allxpoints[i], self.allypoints[i]))

        # Create vertex coordinates for each grid cell...
        # (<0,0> is at the top left of the grid in this system)
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x,y)).T
        ###mplPath是路径函数，将所有的顶点记录在里面
        ROIpath = mplPath.Path(poly_verts)
        ###相当于np.extract里面的conditions
        grid = ROIpath.contains_points(points).reshape((ny,nx))
        return grid
      
    def displayROI(self,**linekwargs):
        l = plt.Line2D(self.allxpoints +
                     [self.allxpoints[0]],
                     self.allypoints +
                     [self.allypoints[0]],
                     color=self.roicolor, **linekwargs)
        ax = plt.gca()
        ax.add_line(l)
        plt.draw()

    def __motion_notify_callback(self, event):
        if event.inaxes:
            ax = event.inaxes
            x, y = event.xdata, event.ydata
            if (event.button == None or event.button == 1) and self.line != None: # Move line around
                self.line.set_data([self.previous_point[0], x],
                                   [self.previous_point[1], y])
                self.fig.canvas.draw()


    def __button_press_callback(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            ax = event.inaxes
            if event.button == 1 and event.dblclick == False:  # If you press the left button, single click
                if self.line == None: # if there is no line, create a line
                    self.line = plt.Line2D([x, x],
                                           [y, y],
                                           marker='o',
                                           color=self.roicolor)
                    self.start_point = [x,y]
                    self.previous_point =  self.start_point
                    self.allxpoints=[x]
                    self.allypoints=[y]
                                                
                    ax.add_line(self.line)
                    self.fig.canvas.draw()
                    # add a segment
                else: # if there is a line, create a segment
                    self.line = plt.Line2D([self.previous_point[0], x],
                                           [self.previous_point[1], y],
                                           marker = 'o',color=self.roicolor)
                    self.previous_point = [x,y]

                    self.allxpoints.append(x)

                    self.allypoints.append(y)
                    event.inaxes.add_line(self.line)
                    self.fig.canvas.draw()
            elif ((event.button == 1 and event.dblclick==True) or
                  (event.button == 3 and event.dblclick==False)) and self.line != None: # close the loop and disconnect
                self.fig.canvas.mpl_disconnect(self.__ID1) #joerg
                self.fig.canvas.mpl_disconnect(self.__ID2) #joerg
                        
                self.line.set_data([self.previous_point[0],
                                    self.start_point[0]],
                                   [self.previous_point[1],
                                    self.start_point[1]])
                ax.add_line(self.line)
                self.fig.canvas.draw()
                self.line = None
                if sys.flags.interactive:
                    pass
                else:
                    #figure has to be closed so that code can continue
                    plt.close(self.fig) 