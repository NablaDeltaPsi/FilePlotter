# required packages besides python standard modules (https://docs.python.org/3/library/index.html)
# parenthesis gives the version that worked with python (3.9.5) and pyinstaller (6.10.0)
# numpy (1.26.4), pandas (1.4.2), matplotlib (3.9.2), scipy (1.13.1)

import os, time, tkinter as tk, numpy as np, pandas as pd, datetime as dt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from tkinter.filedialog import askopenfilename
from configparser import ConfigParser
from ctypes import windll
import scipy.interpolate
import scipy.signal
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

version = "v1.3"

class PlotWindow(tk.Toplevel):
    def __init__(self, parent, geom):
        super().__init__(parent)

        self.parent = parent
        self.geometry(geom)
        self.title("PlotWindow")

        try:
            self.iconbitmap('FilePlotter.ico')
        except:
            print("Did not find icon... continuing...")
        try:
            windll.shcore.SetProcessDpiAwareness(1)
        except:
            print("SetProcessDpiAwareness(1) not successful... continuing...")

        print("... new PlotWindow instance")


        # colors
        linenumber = parent.linenumber
        if parent.check_savgol.get():
            linenumber = int(linenumber/2)

        if parent.check_orangeblue.get():
            mycolors = continuouscolor(linenumber, 'orangeblue')
        elif parent.check_grey.get():
            mycolors = continuouscolor(linenumber, 'grey')
        else:
            mycolors = continuouscolor(linenumber, 'distinguish')

        if parent.check_savgol.get():
            add_colors = np.zeros((linenumber, 3))
            for i in range(linenumber):
                add_colors[i, :] = 0.5 * mycolors[i, :] + 0.5 * np.array([1, 1, 1])
            mycolors = np.vstack((add_colors, mycolors))

        # plot
        fig = plt.Figure(figsize=(20, 20))
        ax = fig.add_subplot(111)
        for i in range(parent.linenumber):
            if len(parent.data[i][2]) > 0:
                ax.errorbar(parent.data[i][0], parent.data[i][1], yerr=parent.data[i][2], color=mycolors[i, :], label=parent.linelabels[i], marker='o', markersize=5, capsize=3, ls='', linewidth=0.5)
            else:
                if parent.check_marker.get():
                    ax.plot(parent.data[i][0], parent.data[i][1], color=mycolors[i, :], label=parent.linelabels[i], marker='s', ls='', markersize=3)
                else:
                    if parent.check_savgol.get() and i > linenumber/2:
                        thisline = ax.plot(parent.data[i][0], parent.data[i][1], color=mycolors[i, :], label=parent.linelabels[i])
                    else:
                        thisline = ax.plot(parent.data[i][0], parent.data[i][1], color=mycolors[i, :], label=parent.linelabels[i])
            if len(parent.data[i][3]) > 0:
                for n in range(len(parent.data[i][3])):
                    ax.text(parent.data[i][0][n], parent.data[i][1][n], parent.data[i][3][n], va='center', ha='left', fontsize=8)

        # SETTINGS ===============================================================================
        if parent.check_logy.get():
            ax.set_yscale('log')

        # x limits if both not empty:
        if (not not parent.entry_xlims_x1.get()) and (not not parent.entry_xlims_x2.get()):
            x1 = float(parent.entry_xlims_x1.get())
            x2 = float(parent.entry_xlims_x2.get())
            ax.set_xlim(xmin=x1, xmax=x2)
        # else only one side
        elif not not parent.entry_xlims_x1.get():
            x1 = float(parent.entry_xlims_x1.get())
            ax.set_xlim(xmin=x1, xmax=None)
        elif not not parent.entry_xlims_x2.get():
            x2 = float(parent.entry_xlims_x2.get())
            ax.set_xlim(xmin=None, xmax=x2)

        # after changing x-limits, adjust the y-scaling
        # if (not not parent.entry_xlims_x1.get()) or (not not parent.entry_xlims_x2.get()):
        xmin, xmax = ax.get_xlim()
        # get ymin and ymax
        if parent.check_logy.get():
            ymin, ymax = y_min_max_between(parent.data, xmin, xmax, 1)
        else:
            ymin, ymax = y_min_max_between(parent.data, xmin, xmax, 0)
        # add some space depending on log or lin and the labeling case (more space)
        small_space = 0.03
        large_space = 0.10
        if parent.check_logy.get():
            if parent.check_maxs.get():
                add_pos = large_space * (np.log10(ymax) - np.log10(ymin))
            else:
                add_pos = small_space * (np.log10(ymax) - np.log10(ymin))
            if parent.check_mins.get():
                add_neg = large_space * (np.log10(ymax) - np.log10(ymin))
            else:
                add_neg = small_space * (np.log10(ymax) - np.log10(ymin))
            ymax = ymax * 10 ** add_pos
            ymin = ymin * 10 ** -add_neg
        else:
            if parent.check_maxs.get():
                add_pos = large_space * (ymax - ymin)
            else:
                add_pos = small_space * (ymax - ymin)
            if parent.check_mins.get():
                add_neg = large_space * (ymax - ymin)
            else:
                add_neg = small_space * (ymax - ymin)
            ymax = ymax + add_pos
            ymin = ymin - add_neg

        # apply new ylimits
        if not ymin >= ymax:
            ax.set_ylim(ymin=ymin, ymax=ymax)

        # y limits if both not empty:
        if (not not parent.entry_ylims_y1.get()) and (not not parent.entry_ylims_y2.get()):
            y1 = float(parent.entry_ylims_y1.get())
            y2 = float(parent.entry_ylims_y2.get())
            ax.set_ylim(ymin=y1, ymax=y2)
        # else only one side
        elif not not parent.entry_ylims_y1.get():
            y1 = float(parent.entry_ylims_y1.get())
            ax.set_ylim(ymin=y1, ymax=None)
        elif not not parent.entry_ylims_y2.get():
            y2 = float(parent.entry_ylims_y2.get())
            ax.set_ylim(ymin=None, ymax=y2)

        if parent.check_leg.get():
            ax.legend()

        x1, x2 = ax.get_xlim()
        y1, y2 = ax.get_ylim()
        if parent.check_savgol.get():
            label_from = int(linenumber/2)
            label_to = linenumber
        else:
            label_from = 0
            label_to = linenumber

        if parent.check_mins.get():
            dominance_mins = float(parent.entry_dominance_mins.get()) / 100 * abs(x2 - x1)
            for i in range(label_from, label_to):
                mins = determine_local_minmaxs(parent.data, x1, x2, dominance_mins, 'min')
                x = parent.data[i][0]
                y = parent.data[i][1]
                if parent.check_logy.get():
                    y_off = abs(np.log10(y2) - np.log10(y1)) * 0.03
                else:
                    y_off = abs(y2 - y1) * 0.03
                for k in range(len(mins)):
                    found_index = np.where(x == mins[k])
                    found_index = int(found_index[0])
                    if parent.check_logy.get():
                        text_y = max(y[found_index] * 10 ** (-1.0 * y_off), y1)
                        line_y = max(y[found_index] * 10 ** (-0.8 * y_off), y1)
                    else:
                        text_y = max(y[found_index] - 1.0 * y_off, y1)
                        line_y = max(y[found_index] - 0.8 * y_off, y1)
                    ax.text(mins[k], text_y, '{:.5g}'.format(mins[k]),
                            fontsize=7, color=mycolors[i, :],
                            horizontalalignment='center', verticalalignment='top')
                    ax.plot([mins[k], mins[k]], [y[found_index], line_y],
                            color=mycolors[i, :], linestyle='-', linewidth=0.3)

        if parent.check_maxs.get():
            dominance_maxs = float(parent.entry_dominance_maxs.get()) / 100 * abs(x2 - x1)
            for i in range(label_from, label_to):
                maxs = determine_local_minmaxs(parent.data, x1, x2, dominance_maxs, 'max')
                x = parent.data[i][0]
                y = parent.data[i][1]
                if parent.check_logy.get():
                    y_off = abs(np.log10(y2) - np.log10(y1)) * 0.03
                else:
                    y_off = abs(y2 - y1) * 0.03
                for k in range(len(maxs)):
                    found_index = np.where(x == maxs[k])
                    found_index = int(found_index[0])
                    if parent.check_logy.get():
                        text_y = min(y[found_index] * 10 ** (1.0 * y_off), y2)
                        line_y = min(y[found_index] * 10 ** (0.8 * y_off), y2)
                    else:
                        text_y = min(y[found_index] + 1.0 * y_off, y2)
                        line_y = min(y[found_index] + 0.8 * y_off, y2)
                    ax.text(maxs[k], text_y, '{:.5g}'.format(maxs[k]),
                            fontsize=7, color=mycolors[i, :],
                            horizontalalignment='center', verticalalignment='bottom')
                    ax.plot([maxs[k], maxs[k]], [y[found_index], line_y],
                            color=mycolors[i, :], linestyle='-', linewidth=0.3)

        if parent.check_aspect.get():
            ax.set_aspect('equal')

        # PUT GRAPH INTO A WINDOW ======================================================================
        self.canvas = FigureCanvasTkAgg(fig, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().place(relx=0.01, rely=0.01, relheight=0.98, relwidth=0.98, anchor='nw')
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        self.parent.plot_window_size = self.winfo_geometry()
        self.destroy()

def determine_local_minmaxs(data, x1, x2, dominance, minmax):
    # important that x2 is larger than x1, therefore first sort:
    thisxlims = np.array([x1, x2])
    thisxlims.sort()
    x1 = thisxlims[0]
    x2 = thisxlims[1]
    linenumber = len(data)
    for i in range(linenumber):
        # sort and cut data
        ind = data[i][0].argsort()
        x = data[i][0][ind]
        y = data[i][1][ind]
        cut_x = []
        cut_y = []
        for k in range(len(x)):
            if x1 <= x[k] <= x2:
                cut_x.append(x[k])
                cut_y.append(y[k])
        # search for mins/maxs in cutted data
        minmaxlist = []
        for k in range(3, len(cut_x) - 3):
            # check if this x is min within a given width around the current x
            this_is_minmax = True
            for n in range(len(cut_x)):
                if cut_x[k] - dominance < cut_x[n] < cut_x[k] + dominance:
                    if minmax == 'min':
                        if cut_y[n] < cut_y[k]:  # cut_y[k] is no minmax
                            this_is_minmax = False
                            break
                    else:
                        if cut_y[n] > cut_y[k]:  # cut_y[k] is no minmax
                            this_is_minmax = False
                            break
                    if cut_x[n] in minmaxlist: # other x with same y already noted as minmax
                        this_is_minmax = False
                        break
            # if True after the complete for loop, it is a local minmax
            if this_is_minmax:
                minmaxlist.append(cut_x[k])
        return minmaxlist

def set_zero_between(data, from_x, to_x):
    for i in range(len(data)):
        # find closest index for from and to
        set_zero_from = np.argmin(abs(data[i][0] - from_x))
        set_zero_to = np.argmin(abs(data[i][0] - to_x))
        set_zero_from = int(set_zero_from)
        set_zero_to = int(set_zero_to)
        # same value: special case (min of a single value does not work)
        if set_zero_from == set_zero_to:
            data[i][1] = data[i][1] - data[i][1][set_zero_from]
            continue
        # put them into an array and sort
        set_zero = np.array([set_zero_from, set_zero_to])
        set_zero.sort()
        # find maximum in the range between and subtract
        data[i][1] = data[i][1] - min(data[i][1][set_zero[0]:set_zero[1]])
    return data


def set_zero_close_to(data, close_x):
    for i in range(len(data)):
        # find closest index for from and to
        set_zero_close = np.argmin(abs(data[i][0] - close_x))
        set_zero_close = int(set_zero_close)
        data[i][1] = data[i][1] - data[i][1][set_zero_close]
    return data


def normalize_between(data, from_x, to_x):
    for i in range(len(data)):
        # find closest index for from and to
        norm_from = np.argmin(abs(data[i][0] - from_x))
        norm_to = np.argmin(abs(data[i][0] - to_x))
        norm_from = int(norm_from)
        norm_to = int(norm_to)
        # same value: special case (max of a single value does not work)
        if norm_from == norm_to:
            data[i][1] = data[i][1] / abs(data[i][1][norm_from])
            continue
        # put them into an array and sort
        norm_x = np.array([norm_from, norm_to])
        norm_x.sort()
        # find maximum in the range between
        divby = max(abs(data[i][1][norm_x[0]:norm_x[1]]))
        if not divby == 0:
            data[i][1] = data[i][1] / divby
    return data


def normalize_close_to(data, close_x):
    for i in range(len(data)):
        # find closest index for from and to
        norm_close = np.argmin(abs(data[i][0] - close_x))
        norm_close = int(norm_close)
        divby = abs(data[i][1][norm_close])
        if not divby == 0:
            data[i][1] = data[i][1] / divby
    return data


def y_min_max_between(data, from_x, to_x, skip_zero_and_neg):
    all_ymin = []
    all_ymax = []
    for i in range(len(data)):
        # find closest index for from and to
        norm_from = np.argmin(abs(data[i][0] - from_x))
        norm_to = np.argmin(abs(data[i][0] - to_x))
        norm_from = int(norm_from)
        norm_to = int(norm_to)
        # same value: special case (max of a single value does not work)
        if norm_from == norm_to:
            return 0, 0
        # put them into an array and sort
        norm_x = np.array([norm_from, norm_to])
        norm_x.sort()
        # find max in the range
        thisvalues = data[i][1][norm_x[0]:norm_x[1]]
        if len(data[i][2]) > 0:
            thisvalues = np.add(thisvalues, np.abs(data[i][2][1][norm_x[0]:norm_x[1]]))
        all_ymax.append(max(thisvalues))
        # for log scaling, zero and negative should be ignored for ymin:
        if skip_zero_and_neg:
            #     | make negatives zero |  add maximum to zeros   |
            data[i][1] = data[i][1] * (data[i][1] > 0) + max(data[i][1]) * (data[i][1] <= 0)
        # now, find min in the range
        thisvalues = data[i][1][norm_x[0]:norm_x[1]]
        if len(data[i][2]) > 0:
            thisvalues = np.subtract(thisvalues, np.abs(data[i][2][0][norm_x[0]:norm_x[1]]))
        all_ymin.append(min(thisvalues))
    ymin = min(all_ymin)
    ymax = max(all_ymax)
    return ymin, ymax


def toggle_handle(handle):
    oldvalue = float(handle.get())
    if not oldvalue == 0:
        newvalue = 1239.84 / oldvalue
        handle.delete(0, 'end')
        handle.insert('end', "{0:.3g}".format(newvalue))


def toggle_handles(handle1, handle2):
    oldvalue1 = float(handle1.get())
    oldvalue2 = float(handle2.get())
    if oldvalue1 == 0:
        newvalue1 = oldvalue1
    else:
        newvalue1 = 1239.84 / oldvalue1
    if oldvalue2 == 0:
        newvalue2 = oldvalue2
    else:
        newvalue2 = 1239.84 / oldvalue2
    handle1.delete(0, 'end')
    handle2.delete(0, 'end')
    handle1.insert('end', "{0:.3g}".format(newvalue2))  # switched!
    handle2.insert('end', "{0:.3g}".format(newvalue1))  # switched!


def create_two_colors_shift(startcolor, endcolor, number, dist, skip):
    number = int(number)
    if dist == 0:
        dist = 0.0001
    if number == 1:
        mycolors = startcolor
        return mycolors
    mycolors = np.empty((number, 3))
    # dist_data = np.empty((number,2))
    for i in range(number):
        x = (i + skip) / (number - 1 + skip)
        x = 1 - x
        x_ = (10 ** (dist * x) - 1) / (10 ** dist - 1)
        # dist_data[i,:] = [i,x_]
        mycolors[i, :] = x_ * startcolor + (1 - x_) * endcolor
    # plt.figure("continuouscolor")
    # plt.plot(dist_data[:,0],dist_data[:,1])
    # plt.show()
    return mycolors


def continuouscolor(number, colorcase):
    if colorcase == 'distinguish':
        mycolors1 = np.array([0.9, 0.2, 0])
        mycolors2 = 0.2 * np.array([1, 1, 1])
        mycolors3 = np.array([0, 0.1, 0.9])
        mycolors4 = np.array([0, 0.6, 0.1])
        mycolors5 = np.array([0.9, 0.5, 0])
        mycolors6 = np.array([0, 0.5, 0.7])
        mycolors7 = np.array([0.7, 0, 0.9])
        mycolors8 = np.array([0.4, 0.7, 0])
        mycolors = np.empty((0, 3))
        for i in range(int(number / 8) + 1):
            mycolors = np.append(mycolors, np.array([mycolors1]), axis=0)
            mycolors = np.append(mycolors, np.array([mycolors2]), axis=0)
            mycolors = np.append(mycolors, np.array([mycolors3]), axis=0)
            mycolors = np.append(mycolors, np.array([mycolors4]), axis=0)
            mycolors = np.append(mycolors, np.array([mycolors5]), axis=0)
            mycolors = np.append(mycolors, np.array([mycolors6]), axis=0)
            mycolors = np.append(mycolors, np.array([mycolors7]), axis=0)
            mycolors = np.append(mycolors, np.array([mycolors8]), axis=0)
        mycolors = mycolors[0:number, :]
        return mycolors

    if colorcase == 'turquoise':
        startcolor = np.array([0, 0.4, 0.7])
        endcolor = np.array([0, 0.7, 0.4])
        mycolors = create_two_colors_shift(startcolor, endcolor, number, 0, 0)
        return mycolors

    if colorcase == 'bluegreen':
        startcolor = np.array([0, 0, 0.7])
        endcolor = np.array([0, 0.7, 0.3])
        mycolors = create_two_colors_shift(startcolor, endcolor, number, 0, 0)
        return mycolors

    if colorcase == 'fire':
        startcolor = np.array([1, 0, 0])
        endcolor = np.array([1, 0.55, 0])
        mycolors = create_two_colors_shift(startcolor, endcolor, number, 0, 0)
        return mycolors

    if colorcase == 'purple':
        startcolor = np.array([1, 0, 0])
        endcolor = np.array([0, 0, 1])
        mycolors = create_two_colors_shift(startcolor, endcolor, number, 0, 0)
        return mycolors

    if colorcase == 'grey':
        if number == 1:
            mycolors = np.array([[0, 0, 0]])
            return mycolors
        startcolor = np.array([0, 0, 0])
        endcolor = 0.8 * np.array([1, 1, 1])
        mycolors = create_two_colors_shift(startcolor, endcolor, number, 0, 0)
        return mycolors

    if colorcase == 'orangeblue':
        if number == 1:
            mycolors = np.array([[1, 0, 0]])
            return mycolors
        if number == 2:
            mycolors = np.array([[1, 0, 0], [0, 0, 1]])
            return mycolors
        if number == 3:
            mycolors = np.array([[1, 0.5, 0], [1, 0, 0], [0, 0, 1]])
            return mycolors
        startcolor = np.array([1, 0.5, 0])
        firstmidcolor = np.array([1, 0, 0])
        endcolor = np.array([0, 0, 1])
        change1 = 0.5
        mycolors1 = create_two_colors_shift(startcolor, firstmidcolor, round(change1 * number + 0.1, 0), 0, 0)
        mycolors2 = create_two_colors_shift(firstmidcolor, endcolor, round((1 - change1) * number - 0.1, 0), 0, 1)
        mycolors = mycolors1
        mycolors = np.append(mycolors, mycolors2, axis=0)
        return mycolors

    if colorcase == 'rainbow':
        startcolor = np.array([1, 0, 0])
        firstmidcolor = np.array([1, 0.7, 0])
        secondmidcolor = np.array([0, 0.6, 0.1])
        endcolor = np.array([0, 0, 1])
        change1 = 0.3
        change2 = 0.6
        mycolors1 = create_two_colors_shift(startcolor, firstmidcolor, round(change1 * number + 0.1, 0), 1, 0)
        mycolors2 = create_two_colors_shift(firstmidcolor, secondmidcolor,
                                               round((change2 - change1) * number - 0.1, 0),
                                               1, 1)
        mycolors3 = create_two_colors_shift(secondmidcolor, endcolor, round((1 - change2) * number - 0.1, 0), 1, 1)
        mycolors = mycolors1
        mycolors = np.append(mycolors, mycolors2, axis=0)
        mycolors = np.append(mycolors, mycolors3, axis=0)
        return mycolors

def user_input_to_columns(_user_string, limit=0):
    # empty -> none
    if not _user_string.strip() and limit > 0:
        return [] #list(range(1,limit+1))

    # split and cast if possible
    _columns = _user_string.split(",")
    if not type(_columns) is list:
        _columns = [_columns]
    for i in range(len(_columns)):
        _columns[i] = _columns[i].strip()
        if _columns[i].isdigit():
            _columns[i] = int(_columns[i])

    # process entries by type
    columns = []
    for n in range(len(_columns)):
        errmsg = ""
        if type(_columns[n]) is str:
            if _columns[n].upper() in ('ALL','ALLE','A'):
                columns += list(range(1, limit + 1))
            elif _columns[n].upper() in ('EVEN','GERADE','E','G'):
                columns += list(range(2, limit + 1, 2))
            elif _columns[n].upper() in ('ODD','UNGERADE','O','U'):
                columns += list(range(1, limit + 1, 2))
            elif _columns[n].find('-') > 0:
                cols_range = _columns[n].split('-')
                cols_start = cols_range[0].strip()
                cols_end = cols_range[1].strip()
                if cols_start.isdigit():
                    if not cols_end:
                        columns += list(range(int(cols_start), limit + 1))
                    elif cols_end.isdigit():
                        columns += list(range(int(cols_start),int(cols_end)+1))
                    else:
                        errmsg = cols_end
                else:
                    errmsg = cols_start
            else:
                errmsg = _columns[n]
        else:
            columns.append(_columns[n])
        if errmsg:
            errmsg = "Don't know what to do with '" + errmsg + "' in columns definition."
            tk.messagebox.showerror(title="Column error", message=errmsg)
            raise Exception(errmsg)

        print(columns)

    # limit
    columns = [x for x in columns if x <= limit]
    return columns

def match_up_columns(xcols, ycols, yecols, lbcols, limit):
    if len(xcols) > 50 or len(ycols) > 50 or len(yecols) > 50 or len(lbcols) > 50:
        errmsg = "Would plot more than 50 columns! Maybe transpose?"
        tk.messagebox.showerror(title="Column error", message=errmsg)
        raise Exception(errmsg)
    if not (len(yecols) == 2 and len(ycols) == 1):
        yecols = []
    if not (len(lbcols) == len(ycols)):
        lbcols = []
    if len(xcols) == 1:
        ycols = [y for y in ycols if y not in xcols]
        yecols = [ye for ye in yecols if ye not in xcols]
        lbcols = [lb for lb in lbcols if lb not in xcols]
    if len(xcols) != len(ycols) and len(xcols) > 0:
        xcols = [xcols[0]] * len(ycols)
    return xcols, ycols, yecols, lbcols

def contains(text, substrings, booltype='any', ignore_case=True):
    if not type(substrings) is list:
        substrings = [substrings]
    if ignore_case:
        text = text.lower()
        substrings = [substr.lower() for substr in substrings]
    contains_any = False
    contains_all = True
    for substr in substrings:
        if text.find(substr) > 0:
            contains_any = True
        else:
            contains_all = False
    if booltype == 'any':
        return contains_any
    else:
        return contains_all

def try_date(array, type="min"):
    if not isinstance(array, (list, tuple, np.ndarray)):
        array = [array]
    if not isinstance(array[0], str):
        return array
    preformatted_array = preformat_try_date(array, ':')
    this_pattern = determine_date_pattern(preformatted_array, ':')
    if this_pattern is None:
        print("Could not find any date pattern!!")
        return array

    # try to format to float year
    new_array = []
    for i in range(len(preformatted_array)):
        try:
            this_datetime = dt.datetime.strptime(preformatted_array[i], this_pattern)
        except:
            print("try date: no success with line " + preformatted_array[i])
            return array
        if type=="min":
            ref_datetime = dt.datetime(2000, 1, 1, 0, 0, 0, 0)
            minutes_since = (this_datetime - ref_datetime).total_seconds()/60
            new_array.append(minutes_since)
        else:
            ref_datetime = dt.datetime(2024, 1, 1, 0, 0, 0, 0)
            float_year = 2024 + (this_datetime - ref_datetime).total_seconds()/60/60/24/365.25
            new_array.append(float_year)
    return new_array

def replace_chars(string, chars, repl):
    str_out = string
    for char in chars:
        str_out = str_out.replace(char, repl)
    return str_out

def preformat_try_date(array, sep):
    for i in range(len(array)):
        replace_string = replace_chars(array[i], ['/', '-', '.', ' ', ','], sep)
        splitsep = replace_string.split(sep)
        array[i] = sep.join([x for x in splitsep if x]) # sort out empty fields
    return array

def get_min_sep_width(array, sep):
    min_seps = [len(x) for x in array[0].split(sep)]
    for thisstring in array:
        splitarray = thisstring.split(sep)
        for i in range(len(splitarray)):
            if len(splitarray[i]) < min_seps[i]:
                min_seps[i] = len(splitarray[i])
    return min_seps

def determine_date_pattern(array, sep):
    if not isinstance(array, (list, tuple, np.ndarray)):
        array = [array]
    min_seps = get_min_sep_width(array, sep)
    date_patterns = ["%d:%m:%Y", "%Y:%m:%d", "%d:%m:%y", "%y:%m:%d"]
    time_patterns = ["", ":%H:%M:%S", ":%H:%M:%S:%f"]
    final_pattern = None
    for d_pattern in date_patterns:
        for t_pattern in time_patterns:
            this_pattern = d_pattern+t_pattern
            this_pattern_split = this_pattern.split(sep)
            if not len(min_seps) == len(this_pattern_split):
                continue
            length_match = True
            for i in range(len(min_seps)):
                # if min_seps[i] < 2:
                #     this_pattern_split[i] = '%-' + this_pattern_split[i][-1]
                if min_seps[i] == 4 and not contains(this_pattern_split[i], 'Y', ignore_case=False):
                    length_match = False
                    break
            if not length_match:
                continue
            this_pattern = ":".join(this_pattern_split)
            try:
                print("try date using", this_pattern, "on", array[0])
                dt.datetime.strptime(array[0], this_pattern)
                print("success")
                final_pattern = this_pattern
                break
            except Exception as e:
                #print(repr(e))
                pass
        if final_pattern is not None:
            break
    return final_pattern

def myformat(x, perc=False, precision=3, thousand_sep=True):
    if x is None:
        return ""
    if isinstance(x, (tuple, str)):
        str_out = str(x)
    elif isinstance(x, int) or abs(x) > 120:
        if not isinstance(x, int):
            x = int(x)
        if abs(x) > 99999:
            if thousand_sep:
                str_out = '{:,}'.format(x).replace(',', '.')
            else:
                str_out = '{:,}'.format(x).replace(',', '')
        else:
            str_out = str(x)
    else:
        format_string = "{:#." + str(precision) + "g}"
        str_out = format_string.format(x)
    if perc:
        str_out += "%"
    return str_out

class MainGUI(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.active_filenames = []
        self.linelabels = []
        self.data = []
        self.linenumber = 0
        self.plot_window_size = ""
        self.lastpath = ""

        try:
            self.parent.iconbitmap('FilePlotter.ico')
        except:
            print("Did not find icon... continuing...")

        # SIZE DEFINITIONS
        heights = 30
        label_heights = 0.05
        checkbox_heights = 20
        dropdown_heights = 30
        width_1 = 20
        width_2 = 30
        width_3 = 80
        pos_1 = 0.72
        pos_2 = 0.84
        pos_3 = 0.6

        # LOAD FRAME =======================================================
        self.frame_load = tk.LabelFrame(parent)
        self.frame_load.place(relx=0.02, rely=0.01, relheight=0.98, relwidth=0.4, anchor='nw')
        self.frame_load["text"] = "Load files"
        pv2 = 0.10  # pos vertical
        self.load_files = tk.Button(self.frame_load, text="Load files", command=self.load_files)
        self.delete_file = tk.Button(self.frame_load, text="Delete file", command=self.delete_file)
        self.entry_delete_number = tk.Entry(self.frame_load, justify='center')
        self.clear_all = tk.Button(self.frame_load, text="Clear all", command=self.clear_filenames)
        self.load_files.place(relx=0.02, rely=pv2, height=heights, relwidth=0.25, anchor='sw')
        self.delete_file.place(relx=0.31, rely=pv2, height=heights, relwidth=0.25, anchor='sw')
        self.entry_delete_number.place(relx=0.31 + 0.25 + 0.005, rely=pv2, height=heights, relwidth=0.08, anchor='sw')
        self.clear_all.place(relx=0.94, rely=pv2, height=heights, relwidth=0.25, anchor='se')
        pv3 = 0.93  # pos vertical
        self.textfield_filenames = tk.Text(self.frame_load, height=4, width=50, wrap='none')
        self.scroll_v = tk.Scrollbar(self.frame_load)
        self.scroll_h = tk.Scrollbar(self.frame_load)
        self.scroll_v.config(command=self.textfield_filenames.yview, orient='vertical')
        self.scroll_h.config(command=self.textfield_filenames.xview, orient='horizontal')
        self.textfield_filenames.config(yscrollcommand=self.scroll_v.set)
        self.textfield_filenames.config(xscrollcommand=self.scroll_h.set)
        self.textfield_filenames.place(relx=0.02, rely=pv3, relheight=pv3 - pv2 - label_heights, relwidth=0.92, anchor='sw')
        self.scroll_v.place(relx=0.94, rely=pv3, relheight=pv3 - pv2 - label_heights, width=width_1, anchor='sw')
        self.scroll_h.place(relx=0.02, rely=pv3, height=20, relwidth=0.92, anchor='nw')

        # DATA FRAME ================================================================================
        vpos = PosSequence(-0.055, 0.065)
        self.frame_plot = tk.LabelFrame(parent)
        self.frame_plot.place(relx=0.44, rely=0.01, relheight=0.83, relwidth=0.26, anchor='nw')
        self.frame_plot['text'] = "Data definition"

        self.label_xcol = tk.Label(self.frame_plot, text='horiz. axis columns')
        self.label_xcol.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.entry_xcol = tk.Entry(self.frame_plot, justify='center')
        self.entry_xcol.place(relx=pos_3, rely=vpos.get(keep=True), height=checkbox_heights, width=width_3, anchor='nw')

        self.label_ycol = tk.Label(self.frame_plot, text='vert. axis columns')
        self.label_ycol.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.entry_ycol = tk.Entry(self.frame_plot, justify='center')
        self.entry_ycol.place(relx=pos_3, rely=vpos.get(keep=True), height=checkbox_heights, width=width_3, anchor='nw')

        self.label_yecol = tk.Label(self.frame_plot, text='error axis columns')
        self.label_yecol.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.entry_yecol = tk.Entry(self.frame_plot, justify='center')
        self.entry_yecol.place(relx=pos_3, rely=vpos.get(keep=True), height=checkbox_heights, width=width_3, anchor='nw')

        self.label_lbcol = tk.Label(self.frame_plot, text='label columns')
        self.label_lbcol.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.entry_lbcol = tk.Entry(self.frame_plot, justify='center')
        self.entry_lbcol.place(relx=pos_3, rely=vpos.get(keep=True), height=checkbox_heights, width=width_3, anchor='nw')

        self.label_skip_rows = tk.Label(self.frame_plot, text='skip rows')
        self.label_skip_rows.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.entry_skip_rows = tk.Entry(self.frame_plot, justify='center')
        self.entry_skip_rows.place(relx=pos_3, rely=vpos.get(keep=True), height=checkbox_heights, width=width_3, anchor='nw')

        self.label_delimiter = tk.Label(self.frame_plot, text='delimiter')
        self.label_delimiter.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.entry_delimiter = tk.Entry(self.frame_plot, justify='center')
        self.entry_delimiter.place(relx=pos_3, rely=vpos.get(keep=True), height=checkbox_heights, width=width_3, anchor='nw')

        self.label_decimal = tk.Label(self.frame_plot, text='decimal')
        self.label_decimal.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.entry_decimal = tk.Entry(self.frame_plot, justify='center')
        self.entry_decimal.place(relx=pos_3, rely=vpos.get(keep=True), height=checkbox_heights, width=width_3, anchor='nw')

        self.label_quotechar = tk.Label(self.frame_plot, text='quotechar')
        self.label_quotechar.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.entry_quotechar = tk.Entry(self.frame_plot, justify='center')
        self.entry_quotechar.place(relx=pos_3, rely=vpos.get(keep=True), height=checkbox_heights, width=width_3, anchor='nw')

        self.checkbox_transpose = tk.Checkbutton(self.frame_plot, text='transpose data matrix', justify='left')
        self.checkbox_transpose.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.check_transpose = tk.BooleanVar()
        self.checkbox_transpose["variable"] = self.check_transpose

        self.checkbox_integer_x = tk.Checkbutton(self.frame_plot, text='integer x axis', justify='left', command=self.check_integer_x)
        self.checkbox_integer_x.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.check_integer_x = tk.BooleanVar()
        self.checkbox_integer_x["variable"] = self.check_integer_x

        self.checkbox_date_min = tk.Checkbutton(self.frame_plot, text='date x axis (min)', justify='left', command=self.check_date_min)
        self.checkbox_date_min.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.check_date_min = tk.BooleanVar()
        self.checkbox_date_min["variable"] = self.check_date_min

        self.checkbox_date_year = tk.Checkbutton(self.frame_plot, text='date x axis (year)', justify='left', command=self.check_date_year)
        self.checkbox_date_year.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.check_date_year = tk.BooleanVar()
        self.checkbox_date_year["variable"] = self.check_date_year

        self.checkbox_energy = tk.Checkbutton(self.frame_plot, text='wvl ↔ energy', justify='left', command=self.check_energy)
        self.checkbox_energy.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.check_energy = tk.BooleanVar()
        self.checkbox_energy["variable"] = self.check_energy

        self.checkbox_ref = tk.Checkbutton(self.frame_plot, text='subtract (line # / factor):', justify='left')
        self.entry_ref = tk.Entry(self.frame_plot, justify='center')
        self.entry_reffact = tk.Entry(self.frame_plot, justify='center')
        self.checkbox_ref.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.entry_ref.place(relx=pos_1, rely=vpos.get(keep=True), height=checkbox_heights, width=width_1, anchor='nw')
        self.entry_reffact.place(relx=pos_2, rely=vpos.get(keep=True), height=checkbox_heights, width=width_2, anchor='nw')
        self.check_ref = tk.BooleanVar()
        self.checkbox_ref["variable"] = self.check_ref

        self.checkbox_savgol = tk.Checkbutton(self.frame_plot, text='Sav.-Golay (polyn./nr.)', justify='left')
        self.entry_savgol_pol = tk.Entry(self.frame_plot, justify='center')
        self.entry_savgol_num = tk.Entry(self.frame_plot, justify='center')
        self.checkbox_savgol.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.entry_savgol_pol.place(relx=pos_1, rely=vpos.get(keep=True), height=checkbox_heights, width=width_1, anchor='nw')
        self.entry_savgol_num.place(relx=pos_2, rely=vpos.get(keep=True), height=checkbox_heights, width=width_2, anchor='nw')
        self.check_savgol = tk.BooleanVar()
        self.checkbox_savgol["variable"] = self.check_savgol


        # EDIT FRAME ================================================================================
        first_box_2 = 0.68
        second_box_2 = 0.84
        vpos = PosSequence(-0.032, 0.072)
        self.frame_edit = tk.LabelFrame(parent)
        self.frame_edit.place(relx=0.72, rely=0.01, relheight=0.83, relwidth=0.26, anchor='nw')
        self.frame_edit['text'] = "Data edit and settings"

        self.label_xlims = tk.Label(self.frame_edit, text='x limits', justify='left')
        self.entry_xlims_x1 = tk.Entry(self.frame_edit, justify='center')
        self.entry_xlims_x2 = tk.Entry(self.frame_edit, justify='center')
        self.label_xlims.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.entry_xlims_x1.place(relx=first_box_2, rely=vpos.get(keep=True), height=checkbox_heights, width=width_2, anchor='nw')
        self.entry_xlims_x2.place(relx=second_box_2, rely=vpos.get(keep=True), height=checkbox_heights, width=width_2, anchor='nw')

        self.label_ylims = tk.Label(self.frame_edit, text='y limits', justify='left')
        self.entry_ylims_y1 = tk.Entry(self.frame_edit, justify='center')
        self.entry_ylims_y2 = tk.Entry(self.frame_edit, justify='center')
        self.label_ylims.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.entry_ylims_y1.place(relx=first_box_2, rely=vpos.get(keep=True), height=checkbox_heights, width=width_2, anchor='nw')
        self.entry_ylims_y2.place(relx=second_box_2, rely=vpos.get(keep=True), height=checkbox_heights, width=width_2, anchor='nw')

        self.label_set_zero = tk.Label(self.frame_edit, text='set to zero at / between', justify='left')
        self.entry_set_zero_x1 = tk.Entry(self.frame_edit, justify='center')
        self.entry_set_zero_x2 = tk.Entry(self.frame_edit, justify='center')
        self.label_set_zero.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.entry_set_zero_x1.place(relx=first_box_2, rely=vpos.get(keep=True), height=checkbox_heights, width=width_2, anchor='nw')
        self.entry_set_zero_x2.place(relx=second_box_2, rely=vpos.get(keep=True), height=checkbox_heights, width=width_2, anchor='nw')

        self.labeL_normalize = tk.Label(self.frame_edit, text='normalize at / between', justify='left')
        self.entry_normalize_x1 = tk.Entry(self.frame_edit, justify='center')
        self.entry_normalize_x2 = tk.Entry(self.frame_edit, justify='center')
        self.labeL_normalize.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.entry_normalize_x1.place(relx=first_box_2, rely=vpos.get(keep=True), height=checkbox_heights, width=width_2, anchor='nw')
        self.entry_normalize_x2.place(relx=second_box_2, rely=vpos.get(keep=True), height=checkbox_heights, width=width_2, anchor='nw')

        self.label_offset = tk.Label(self.frame_edit, text='offset every 1st / 2nd', justify='left')
        self.entry_offset_y1 = tk.Entry(self.frame_edit, justify='center')
        self.entry_offset_y2 = tk.Entry(self.frame_edit, justify='center')
        self.label_offset.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.entry_offset_y1.place(relx=first_box_2, rely=vpos.get(keep=True), height=checkbox_heights, width=width_2, anchor='nw')
        self.entry_offset_y2.place(relx=second_box_2, rely=vpos.get(keep=True), height=checkbox_heights, width=width_2, anchor='nw')

        self.checkbox_leg = tk.Checkbutton(self.frame_edit, text='show legend', justify='left')
        self.checkbox_leg.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.check_leg = tk.BooleanVar()
        self.checkbox_leg["variable"] = self.check_leg

        self.checkbox_mins = tk.Checkbutton(self.frame_edit, text='label mins (dominance %)', justify='left')
        self.checkbox_mins.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.check_mins = tk.BooleanVar()
        self.checkbox_mins["variable"] = self.check_mins
        self.entry_dominance_mins = tk.Entry(self.frame_edit, justify='center')
        self.entry_dominance_mins.place(relx=second_box_2, rely=vpos.get(keep=True), height=checkbox_heights, width=width_2, anchor='nw')

        self.checkbox_maxs = tk.Checkbutton(self.frame_edit, text='label maxs (dominance %)', justify='left')
        self.checkbox_maxs.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.check_maxs = tk.BooleanVar()
        self.checkbox_maxs["variable"] = self.check_maxs
        self.entry_dominance_maxs = tk.Entry(self.frame_edit, justify='center')
        self.entry_dominance_maxs.place(relx=second_box_2, rely=vpos.get(keep=True), height=checkbox_heights, width=width_2, anchor='nw')

        self.checkbox_logy = tk.Checkbutton(self.frame_edit, text='log y', justify='left')
        self.checkbox_logy.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.check_logy = tk.BooleanVar()
        self.checkbox_logy["variable"] = self.check_logy

        self.checkbox_orangeblue = tk.Checkbutton(self.frame_edit, text='orangeblue colors', justify='left', command=self.check_orangeblue)
        self.checkbox_orangeblue.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.check_orangeblue = tk.BooleanVar()
        self.checkbox_orangeblue["variable"] = self.check_orangeblue

        self.checkbox_grey = tk.Checkbutton(self.frame_edit, text='fading grey colors', justify='left', command=self.check_grey)
        self.checkbox_grey.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.check_grey = tk.BooleanVar()
        self.checkbox_grey["variable"] = self.check_grey

        self.checkbox_marker = tk.Checkbutton(self.frame_edit, text='markers', justify='left')
        self.checkbox_marker.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.check_marker = tk.BooleanVar()
        self.checkbox_marker["variable"] = self.check_marker

        self.checkbox_aspect = tk.Checkbutton(self.frame_edit, text='equal aspect', justify='left')
        self.checkbox_aspect.place(relx=0.02, rely=vpos.get(), height=checkbox_heights, anchor='nw')
        self.check_aspect = tk.BooleanVar()
        self.checkbox_aspect["variable"] = self.check_aspect

        # RESET BUTTONs ============================================================================
        self.default = tk.Button(parent, text="Default", command=self.set_default)
        self.default.place(relx=0.57, rely=0.88, height=heights, relwidth=0.10, anchor='nw')

        self.reload = tk.Button(parent, text="Reload", command=self.reload_config)
        self.reload.place(relx=0.69, rely=0.88, height=heights, relwidth=0.10, anchor='nw')

        # PLOT BUTTON ================================================================================
        self.plot = tk.Button(parent, text="Plot", command=self.plot)
        self.plot.place(relx=0.81, rely=0.88, height=heights, relwidth=0.16, anchor='nw')

        # INITIALIZE CONFIG FILE =======================================================
        self.set_default()
        self.reload_config()

        # IF CLOSING PARENT WINDOW
        self.parent.protocol("WM_DELETE_WINDOW", self.on_closing)

    def init_config(self, update=False):
        if update:
            print("Update config file... delete FilePlotter.conf")
            os.remove('FilePlotter.conf')
        if not os.path.exists("FilePlotter.conf"):
            print("create FilePlotter.conf")
            config_object = ConfigParser()
            config_object["FILES"] = {
                "search path": ".\\",
                "filenames": "",
            }
            config_object["DATA CHECKBOXES"] = {
                "transpose": "0",
                "integer x": "0",
                "wvl - energy": "0",
                "subtract ref": "0",
                "Savitzky-Golay": "0",
            }
            config_object["SETTINGS CHECKBOXES"] = {
                "legend": "0",
                "mins": "0",
                "maxs": "0",
                "log y": "0",
                "orangeblue": "0",
                "fading grey": "0",
                "markers": "0",
                "aspect": "0",
            }
            config_object["DATA FIELDS"] = {
                "skip_rows": "auto",
                "delimiter": "auto",
                "decimal": "auto",
                "quotechar": "auto",
                "x col": "1",
                "y col": "2-",
                "ye col": "",
                "lb col": "",
                "subtract (line)": "",
                "subtract (factor)": "",
                "Sav.-Gol. (polynomial)": "",
                "Sav.-Gol. (number)": "",
            }
            config_object["SETTINGS FIELDS"] = {
                "x limits (low)": "",
                "x limits (high)": "",
                "y limits (low)": "",
                "y limits (high)": "",
                "set zero (low)": "",
                "set zero (high)": "",
                "normalize (low)": "",
                "normalize (high)": "",
                "offset 1": "",
                "offset 2": "",
                "dominance mins": "10",
                "dominance maxs": "10",
            }
            config_object["WINDOW SIZE"] = {
                "plot window": "500x400+500+400",
                "main window": "800x400+300+300",
            }
            with open('FilePlotter.conf', 'w') as conf:
                config_object.write(conf)


    # BUTTON FUNCTIONS ================================================================================
    def save_conf_file(self):
        if os.path.exists('FilePlotter.conf'):  # in case it has been deleted during execution
            print("... save config file")
            config_object = ConfigParser()
            config_object.read("FilePlotter.conf")

            files_config = config_object["FILES"]
            files_config["search path"] = self.lastpath
            files_config["filenames"] = "\n".join(self.active_filenames)

            data_check = config_object["DATA CHECKBOXES"]
            data_check["transpose"] = str(int(self.check_transpose.get()))
            data_check["integer x"] = str(int(self.check_integer_x.get()))
            data_check["date x min"] = str(int(self.check_date_min.get()))
            data_check["date x year"] = str(int(self.check_date_year.get()))
            data_check["wvl - energy"] = str(int(self.check_energy.get()))
            data_check["subtract ref"] = str(int(self.check_ref.get()))
            data_check["Savitzky-Golay"] = str(int(self.check_savgol.get()))

            settings_check = config_object["SETTINGS CHECKBOXES"]
            settings_check["legend"] = str(int(self.check_leg.get()))
            settings_check["mins"] = str(int(self.check_mins.get()))
            settings_check["maxs"] = str(int(self.check_maxs.get()))
            settings_check["log y"] = str(int(self.check_logy.get()))
            settings_check["orangeblue"] = str(int(self.check_orangeblue.get()))
            settings_check["fading grey"] = str(int(self.check_grey.get()))
            settings_check["markers"] = str(int(self.check_marker.get()))
            settings_check["aspect"] = str(int(self.check_aspect.get()))

            data_fields = config_object["DATA FIELDS"]
            data_fields["skip_rows"] = self.entry_skip_rows.get()
            data_fields["delimiter"] = self.entry_delimiter.get()
            data_fields["decimal"] = self.entry_decimal.get()
            data_fields["quotechar"] = self.entry_quotechar.get()
            data_fields["x col"] = self.entry_xcol.get()
            data_fields["y col"] = self.entry_ycol.get()
            data_fields["ye col"] = self.entry_yecol.get()
            data_fields["lb col"] = self.entry_lbcol.get()
            data_fields["subtract (line)"] = self.entry_ref.get()
            data_fields["subtract (factor)"] = self.entry_reffact.get()
            data_fields["Sav.-Gol. (polynomial)"] = self.entry_savgol_pol.get()
            data_fields["Sav.-Gol. (number)"] = self.entry_savgol_num.get()

            settings_fields = config_object["SETTINGS FIELDS"]
            settings_fields["x limits (low)"] = self.entry_xlims_x1.get()
            settings_fields["x limits (high)"] = self.entry_xlims_x2.get()
            settings_fields["y limits (low)"] = self.entry_ylims_y1.get()
            settings_fields["y limits (high)"] = self.entry_ylims_y2.get()
            settings_fields["set zero (low)"] = self.entry_set_zero_x1.get()
            settings_fields["set zero (high)"] = self.entry_set_zero_x2.get()
            settings_fields["normalize (low)"] = self.entry_normalize_x1.get()
            settings_fields["normalize (high)"] = self.entry_normalize_x2.get()
            settings_fields["offset 1"] = self.entry_offset_y1.get()
            settings_fields["offset 2"] = self.entry_offset_y2.get()
            settings_fields["dominance mins"] = self.entry_dominance_mins.get()
            settings_fields["dominance maxs"] = self.entry_dominance_maxs.get()

            config_object["WINDOW SIZE"]["plot window"] = self.plot_window_size
            config_object["WINDOW SIZE"]["main window"] = self.parent.winfo_geometry()

            # write
            with open('FilePlotter.conf', 'w') as conf:
                config_object.write(conf)

    def on_closing(self):
        self.save_conf_file()
        self.parent.destroy()

    def load_files(self):
        self.parent.update()
        add_filenames = askopenfilename(initialdir=self.lastpath, multiple=True,
                                        filetypes=[('all', '.*'), ('.txt', '.txt'), ('.dat', '.dat'), ('.asc', '.asc')])
        self.lastpath = os.path.dirname(add_filenames[0])
        self.parent.update()
        for i in range(len(add_filenames)):
            self.active_filenames.append(add_filenames[i])
        self.entry_delete_number.delete(0, 'end')
        self.entry_delete_number.insert(0, str(len(self.active_filenames)))
        self.print_file_list()

    def delete_file(self):
        entered_number = self.entry_delete_number.get()
        try:
            entered_number = int(entered_number)
        except ValueError:
            print("Input is not an int!")
            return
        if entered_number > len(self.active_filenames):
            print("Input cannot be larger than the number of files!")
            return
        if not entered_number > 0:
            print("Int must be positive!")
            return
        self.active_filenames.pop(entered_number - 1)
        self.entry_delete_number.delete(0, 'end')
        self.entry_delete_number.insert(0, str(len(self.active_filenames)))
        self.print_file_list()

    def clear_filenames(self):
        self.active_filenames = []
        self.entry_delete_number.delete(0, 'end')
        self.entry_delete_number.insert(0, str(len(self.active_filenames)))
        self.print_file_list()

    def set_default(self):
        self.active_filenames = []
        self.linelabels = []
        self.data = []
        self.linenumber = 0
        self.plot_window_size = ""
        self.lastpath = ""

        self.entry_delete_number.delete(0, 'end')
        self.entry_delete_number.insert(0, "0")
        self.textfield_filenames.config(state='normal')
        self.textfield_filenames.delete("1.0", 'end')
        self.textfield_filenames.insert("end", "--- filenames will appear here.")
        self.textfield_filenames.configure(state='disabled')

        self.entry_skip_rows.delete(0, 'end')
        self.entry_skip_rows.insert(0, "auto")
        self.entry_delimiter.delete(0, 'end')
        self.entry_delimiter.insert(0, "auto")
        self.entry_decimal.delete(0, 'end')
        self.entry_decimal.insert(0, "auto")
        self.entry_quotechar.delete(0, 'end')
        self.entry_quotechar.insert(0, "auto")
        self.entry_xcol.delete(0, 'end')
        self.entry_xcol.insert(0, "1")
        self.entry_ycol.delete(0, 'end')
        self.entry_ycol.insert(0, "2-")
        self.entry_yecol.delete(0, 'end')
        self.entry_lbcol.delete(0, 'end')
        self.entry_ref.delete(0, 'end')
        self.entry_reffact.delete(0, 'end')
        self.entry_savgol_pol.delete(0, 'end')
        self.entry_savgol_num.delete(0, 'end')
        self.entry_dominance_mins.delete(0, 'end')
        self.entry_dominance_mins.insert(0, "10")
        self.entry_dominance_maxs.delete(0, 'end')
        self.entry_dominance_maxs.insert(0, "10")

        self.check_transpose.set(0)
        self.check_integer_x.set(0)
        self.check_date_min.set(0)
        self.check_date_year.set(0)
        self.check_energy.set(0)
        self.check_ref.set(0)
        self.check_savgol.set(0)
        self.check_leg.set(0)
        self.check_mins.set(0)
        self.check_maxs.set(0)
        self.check_logy.set(0)
        self.check_orangeblue.set(0)
        self.check_grey.set(0)
        self.check_marker.set(0)
        self.check_aspect.set(0)

    def reload_config(self):
        if not os.path.exists('FilePlotter.conf'):
            self.init_config()
        config_object = ConfigParser()
        config_object.read('FilePlotter.conf')

        try:
            files_config = config_object["FILES"]
            self.lastpath = files_config["search path"]
            if len(files_config["filenames"]) > 0:
                for file in files_config["filenames"].split("\n"):
                    self.active_filenames.append(file)
                self.entry_delete_number.delete(0, 'end')
                self.entry_delete_number.insert(0, str(len(self.active_filenames)))
                self.print_file_list()

            data_check = config_object["DATA CHECKBOXES"]
            self.check_transpose.set(int(data_check["transpose"]))
            self.check_integer_x.set(int(data_check["integer x"]))
            self.check_date_min.set(int(data_check["date x min"]))
            self.check_date_year.set(int(data_check["date x year"]))
            self.check_energy.set(int(data_check["wvl - energy"]))
            self.check_ref.set(int(data_check["subtract ref"]))
            self.check_savgol.set(int(data_check["Savitzky-Golay"]))

            settings_check = config_object["SETTINGS CHECKBOXES"]
            self.check_leg.set(int(settings_check["legend"]))
            self.check_mins.set(int(settings_check["mins"]))
            self.check_maxs.set(int(settings_check["maxs"]))
            self.check_logy.set(int(settings_check["log y"]))
            self.check_orangeblue.set(int(settings_check["orangeblue"]))
            self.check_grey.set(int(settings_check["fading grey"]))
            self.check_marker.set(int(settings_check["markers"]))
            self.check_aspect.set(int(settings_check["aspect"]))

            data_fields = config_object["DATA FIELDS"]
            self.entry_skip_rows.delete(0, 'end')
            self.entry_skip_rows.insert(0, data_fields["skip_rows"])
            self.entry_delimiter.delete(0, 'end')
            self.entry_delimiter.insert(0, data_fields["delimiter"])
            self.entry_decimal.delete(0, 'end')
            self.entry_decimal.insert(0, data_fields["decimal"])
            self.entry_quotechar.delete(0, 'end')
            self.entry_quotechar.insert(0, data_fields["quotechar"])
            self.entry_xcol.delete(0, 'end')
            self.entry_xcol.insert(0, data_fields["x col"])
            self.entry_ycol.delete(0, 'end')
            self.entry_ycol.insert(0, data_fields["y col"])
            self.entry_yecol.delete(0, 'end')
            self.entry_yecol.insert(0, data_fields["ye col"])
            self.entry_lbcol.delete(0, 'end')
            self.entry_lbcol.insert(0, data_fields["lb col"])
            self.entry_ref.delete(0, 'end')
            self.entry_ref.insert(0, data_fields["subtract (line)"])
            self.entry_reffact.delete(0, 'end')
            self.entry_reffact.insert(0, data_fields["subtract (factor)"])
            self.entry_savgol_pol.delete(0, 'end')
            self.entry_savgol_pol.insert(0, data_fields["Sav.-Gol. (polynomial)"])
            self.entry_savgol_num.delete(0, 'end')
            self.entry_savgol_num.insert(0, data_fields["Sav.-Gol. (number)"])

            settings_fields = config_object["SETTINGS FIELDS"]
            self.entry_xlims_x1.delete(0, 'end')
            self.entry_xlims_x1.insert(0, settings_fields["x limits (low)"])
            self.entry_xlims_x2.delete(0, 'end')
            self.entry_xlims_x2.insert(0, settings_fields["x limits (high)"])
            self.entry_ylims_y1.delete(0, 'end')
            self.entry_ylims_y1.insert(0, settings_fields["y limits (low)"])
            self.entry_ylims_y2.delete(0, 'end')
            self.entry_ylims_y2.insert(0, settings_fields["y limits (high)"])
            self.entry_set_zero_x1.delete(0, 'end')
            self.entry_set_zero_x1.insert(0, settings_fields["set zero (low)"])
            self.entry_set_zero_x2.delete(0, 'end')
            self.entry_set_zero_x2.insert(0, settings_fields["set zero (high)"])
            self.entry_normalize_x1.delete(0, 'end')
            self.entry_normalize_x1.insert(0, settings_fields["normalize (low)"])
            self.entry_normalize_x2.delete(0, 'end')
            self.entry_normalize_x2.insert(0, settings_fields["normalize (high)"])
            self.entry_offset_y1.delete(0, 'end')
            self.entry_offset_y1.insert(0, settings_fields["offset 1"])
            self.entry_offset_y2.delete(0, 'end')
            self.entry_offset_y2.insert(0, settings_fields["offset 2"])
            self.entry_dominance_mins.delete(0, 'end')
            self.entry_dominance_mins.insert(0, settings_fields["dominance mins"])
            self.entry_dominance_maxs.delete(0, 'end')
            self.entry_dominance_maxs.insert(0, settings_fields["dominance maxs"])

            self.plot_window_size = config_object["WINDOW SIZE"]["plot window"]
            self.parent.geometry(config_object["WINDOW SIZE"]["main window"])

        except:
            print("Warning! Create config file again.")
            self.init_config(update=True)
            #self.reload_config()


    def plot(self):
        try:
            print("... load data")
            self.load_data()
            self.edit_data()
            print("... plot data")
            PlotWindow(self, geom=self.plot_window_size)
        except Exception as e:
            errmsg = "Unknown error: " + repr(e)
            tk.messagebox.showerror(title="Load error", message=errmsg)
            raise

    def load_data(self):
        self.data = []
        self.linelabels = []
        filenumber = len(self.active_filenames)
        if filenumber == 0:
            print("no files!!")
            return 0
        quoting = 1  # 0 or csv.QUOTE_MINIMAL, 1 or csv.QUOTE_ALL, 2 or csv.QUOTE_NONNUMERIC, 3 or csv.QUOTE_NONE
        for filenr in range(filenumber):
            print("\n", self.active_filenames[filenr])

            # automatic check file format
            quotechar = self.entry_quotechar.get()
            delimiter = self.entry_delimiter.get()
            decimal = self.entry_decimal.get()
            skiprows = self.entry_skip_rows.get()
            delim_decim_candidates = ['\t', ';', ',', '.']
            with (open(self.active_filenames[filenr], 'r', encoding='utf-8') as fobj):
                found_first_line = False
                line_counter = 0
                while not found_first_line:
                    thisline = fobj.readline()
                    line_counter += 1
                    if any([x in delim_decim_candidates for x in thisline]) and not any(
                            [x.isalpha() for x in thisline]):
                        found_first_line = True
                        print("First non-skip_rows line:", line_counter)
                        if quotechar.lower() == 'auto':
                            if thisline.find('\'') > 0:
                                quotechar = '\''
                            else:
                                quotechar = '\"'
                            print("set quotechar:", repr(quotechar))
                        if skiprows.lower() == 'auto':
                            skiprows = line_counter - 1
                            print("set skiprows:", skiprows)
                        if found_first_line:
                            found_delim = False
                            for cand in delim_decim_candidates:
                                if any([x in cand for x in thisline]):
                                    if delimiter.lower() == 'auto' and not found_delim:
                                        delimiter = cand
                                        found_delim = True
                                        print("set delimiter:", repr(delimiter))
                                    if decimal.lower() == 'auto' and found_delim and not delimiter == cand:
                                        decimal = cand
                                        print("set decimal:", repr(decimal))
                                        break
            quotechar = "\"" if quotechar == 'auto' else quotechar
            delimiter = "," if delimiter == 'auto' else delimiter
            decimal   = "." if decimal   == 'auto' else decimal
            skiprows  = 0 if skiprows  == 'auto' else skiprows

            fobj = open(self.active_filenames[filenr], 'r', encoding='utf-8')
            for i in range(int(skiprows)):
                fobj.readline()
            try:
                loaddata = pd.read_csv(fobj, quoting=quoting, quotechar=quotechar, delimiter=delimiter, decimal=decimal,
                                       header=None, skipinitialspace=True)
            except:
                errmsg = "Could not parse file. Check skip rows, delimiter, decimal and quotechar."
                tk.messagebox.showerror(title="Load error", message=errmsg)
                raise
            loaddata = loaddata.to_numpy()

            # check loaded data
            print(loaddata[:30,:])
            print("File", filenr + 1, "shape:", loaddata.shape)
            loaderror = False
            if len(loaddata.shape) < 2:
                if not self.check_integer_x.get():
                    loaderror = True
            elif loaddata.shape[1] == 1:
                loaderror = True
            if loaderror:
                errmsg = "File " + str(
                    filenr) + " only has one column! Check skip rows, delimiter, decimal and quotechar."
                tk.messagebox.showerror(title="Load error", message=errmsg)
                raise Exception(errmsg)
            if self.check_transpose.get():
                loaddata = loaddata.T
            col_limit = loaddata.shape[1]

            # determine columns requested by user
            xcols = user_input_to_columns(self.entry_xcol.get(), limit=col_limit)
            ycols = user_input_to_columns(self.entry_ycol.get(), limit=col_limit)
            yecols = user_input_to_columns(self.entry_yecol.get(), limit=col_limit)
            lbcols = user_input_to_columns(self.entry_lbcol.get(), limit=col_limit)
            print("input columns: ", xcols, ycols, yecols, lbcols)

            # match up columns against each other (e.g. same xcol for different ycols)
            xcols, ycols, yecols, lbcols = match_up_columns(xcols, ycols, yecols, lbcols, col_limit)
            if self.check_integer_x.get():
                xcols = [-1] * len(ycols)
            print("use columns: ", xcols, ycols, yecols, lbcols)

            for n in range(len(ycols)):
                self.data.append([])
                if xcols[n] < 0:
                    self.data[-1].append(list(range(len(loaddata[:, ycols[n] - 1]))))
                else:
                    self.data[-1].append(loaddata[:, xcols[n] - 1])
                self.data[-1].append(loaddata[:, ycols[n] - 1])
                self.linelabels.append(
                    "file " + os.path.basename(self.active_filenames[filenr]) + ", col " + str(xcols[n]))
                self.linelabels[n] += " vs. " + str(ycols[n])
            self.data[-1].append([])
            if yecols:
                for n in range(len(yecols)):
                    self.data[-1][2].append(loaddata[:, yecols[n] - 1])
            if lbcols:
                for n in range(len(lbcols)):
                    self.data[-1].append(['  ' + myformat(x) for x in loaddata[:, lbcols[n] - 1]])
        self.linenumber = len(self.data)

        # subtract line if checked
        if self.check_ref.get():
            refnr = int(self.entry_ref.get())
            if self.entry_reffact.get():
                subtrfact = float(self.entry_reffact.get())
            else:
                subtrfact = 1
            newdata = []
            new_linelabels = []
            for i in range(self.linenumber):
                if not (i == refnr - 1):
                    newdata.append(self.data[i])
                    new_linelabels.append(self.linelabels[i])
            for i in range(len(newdata)):
                f_subtr = interp1d(self.data[refnr - 1][0], self.data[refnr - 1][1], kind='linear',
                                   fill_value='extrapolate')
                newdata[i][1] = newdata[i][1] - subtrfact * f_subtr(newdata[i][0])
                new_linelabels[i] = new_linelabels[i] + " -ref"
            self.data = newdata
            self.linenumber = len(self.data)
            self.linelabels = new_linelabels

        # further edit lines
        if self.check_integer_x.get():
            for f in range(self.linenumber):
                self.data[f][0] = np.arange(len(self.data[f][0]))

        if self.check_energy.get():
            for f in range(self.linenumber):
                self.data[f][0] = 1239.84 / self.data[f][0]

        if self.check_date_min.get():
            for f in range(self.linenumber):
                self.data[f][0] = try_date(self.data[f][0], type="min")

        if self.check_date_year.get():
            for f in range(self.linenumber):
                self.data[f][0] = try_date(self.data[f][0], type="year")

        # print final data
        for f in range(len(self.data)):
            for c in range(len(self.data[f])):
                print(self.data[f][c][:20])

    def edit_data(self):

        # Columns should not be of type string from now on (except labels)
        for f in range(len(self.data)):
            for c in range(3):
                if len(self.data[f][c]) > 0:
                    if isinstance(self.data[f][c][0], str):
                        raise Exception("Column is of type string!!")

        # SET TO ZERO
        # if both fields are not empty, search the minimum between
        if (not not self.entry_set_zero_x1.get()) and (not not self.entry_set_zero_x2.get()):
            x1 = float(self.entry_set_zero_x1.get())
            x2 = float(self.entry_set_zero_x2.get())
            self.data = set_zero_between(self.data, x1, x2)
        # if only one is not empty, find the closest value
        elif not not self.entry_set_zero_x1.get():
            x1 = float(self.entry_set_zero_x1.get())
            self.data = set_zero_close_to(self.data, x1)
        elif not not self.entry_set_zero_x2.get():
            x2 = float(self.entry_set_zero_x2.get())
            self.data = set_zero_close_to(self.data, x2)

        # NORMALIZE
        # if both fields are not empty, search the abs(max) between
        if (not not self.entry_normalize_x1.get()) and (not not self.entry_normalize_x2.get()):
            x1 = float(self.entry_normalize_x1.get())
            x2 = float(self.entry_normalize_x2.get())
            for i in range(self.linenumber):
                self.data = normalize_between(self.data, x1, x2)
        # if only one is not empty, find the closest value
        elif not not self.entry_normalize_x1.get():
            x1 = float(self.entry_normalize_x1.get())
            for i in range(self.linenumber):
                self.data = normalize_close_to(self.data, x1)
        elif not not self.entry_normalize_x2.get():
            x2 = float(self.entry_normalize_x2.get())
            for i in range(self.linenumber):
                self.data = normalize_close_to(self.data, x2)

        # OFFSET
        if not not self.entry_offset_y1.get():
            for i in range(self.linenumber):
                yoff = i * float(self.entry_offset_y1.get())
                if not not self.entry_offset_y2.get():
                    yoff += (i % 2) * float(self.entry_offset_y2.get())
                    print("add ", (i % 2) * float(self.entry_offset_y2.get()))
                if self.check_logy.get():
                    self.data[i][1] = self.data[i][1] * 10 ** yoff
                else:
                    self.data[i][1] = self.data[i][1] + yoff

        # FILTER
        if self.check_savgol.get():
            smooth_data = []
            filter_line_resolution = 10  # filtered has a higher resolution
            for i in range(self.linenumber):
                poly = int(self.entry_savgol_pol.get())
                num = int(self.entry_savgol_num.get()) * filter_line_resolution
                if (num % 2 == 0):
                    num = num + 1
                x = self.data[i][0]
                y = self.data[i][1]
                y_ip_fn = interp1d(x, y, kind='linear', fill_value='extrapolate')
                x_ip = np.linspace(min(x), max(x), len(x) * filter_line_resolution)
                y_ip = y_ip_fn(x_ip)
                y_smoothed = savgol_filter(y_ip, num, poly)
                self.data.append([x_ip, y_smoothed])
                self.linelabels.append(self.linelabels[i] + ' (smoothed)')
            self.linenumber = len(self.data)



    # CHECKBOX ACTIONS ================================================================================
    def check_energy(self):
        if self.check_energy.get() == 1:
            self.check_integer_x.set(0)
            self.check_date_year.set(0)
            self.check_date_min.set(0)
        self.toggle_entries()

    def check_integer_x(self):
        if self.check_integer_x.get() == 1:
            self.check_energy.set(0)
            self.check_date_year.set(0)
            self.check_date_min.set(0)
            # if energy is toggled, the same as clickaction should occur:
            self.toggle_entries()

    def check_date_min(self):
        if self.check_date_min.get() == 1:
            self.check_integer_x.set(0)
            self.check_date_year.set(0)
            self.check_energy.set(0)

    def check_date_year(self):
        if self.check_date_year.get() == 1:
            self.check_integer_x.set(0)
            self.check_date_min.set(0)
            self.check_energy.set(0)

    def check_orangeblue(self):
        if self.check_orangeblue.get() == 1:
            self.check_grey.set(0)

    def check_grey(self):
        if self.check_grey.get() == 1:
            self.check_orangeblue.set(0)

    def print_file_list(self):
        self.textfield_filenames.configure(state='normal')
        self.textfield_filenames.delete("1.0", 'end')
        for i in range(len(self.active_filenames)):
            self.textfield_filenames.insert('end', str(i + 1) + ": " + os.path.basename(self.active_filenames[i])
                                            + "   (" + self.active_filenames[i] + ") " + "\n")
        self.textfield_filenames.configure(state='disabled')

    def toggle_entries(self):
        # toggle normalize if both not empty:
        if (not not self.entry_normalize_x1.get()) and (not not self.entry_normalize_x2.get()):
            toggle_handles(self.entry_normalize_x1, self.entry_normalize_x2)
        # else toggle single entries:
        elif not not self.entry_normalize_x1.get():
            toggle_handle(self.entry_normalize_x1)
        elif not not self.entry_normalize_x2.get():
            toggle_handle(self.entry_normalize_x2)

        # toggle set zero if both not empty:
        if (not not self.entry_set_zero_x1.get()) and (not not self.entry_set_zero_x2.get()):
            toggle_handles(self.entry_set_zero_x1, self.entry_set_zero_x2)
        # else toggle single entries:
        elif not not self.entry_set_zero_x1.get():
            toggle_handle(self.entry_set_zero_x1)
        elif not not self.entry_set_zero_x2.get():
            toggle_handle(self.entry_set_zero_x2)

        # toggle x limits if both not empty:
        if (not not self.entry_xlims_x1.get()) and (not not self.entry_xlims_x2.get()):
            toggle_handles(self.entry_xlims_x1, self.entry_xlims_x2)
        # else toggle single entries to the other one:
        elif not not self.entry_xlims_x1.get():
            oldvalue = float(self.entry_xlims_x1.get())
            if not oldvalue == 0:
                newvalue = 1239.84 / oldvalue
                self.entry_xlims_x1.delete(0, 'end')
                self.entry_xlims_x2.delete(0, 'end')
                self.entry_xlims_x2.insert('end', "{0:.3g}".format(newvalue))
        elif not not self.entry_xlims_x2.get():
            oldvalue = float(self.entry_xlims_x2.get())
            if not oldvalue == 0:
                newvalue = 1239.84 / oldvalue
                self.entry_xlims_x1.delete(0, 'end')
                self.entry_xlims_x2.delete(0, 'end')
                self.entry_xlims_x1.insert('end', "{0:.3g}".format(newvalue))

class PosSequence():
    def __init__(self, orig, dist):
        self.orig = orig
        self.dist = dist
        self.pos = orig
    def get(self, keep=False):
        if not keep:
            self.pos += self.dist
        return self.pos
    def reset(self):
        self.pos = self.orig
    def skip(self, units=1):
        self.pos += self.dist * units

def main():
    root = tk.Tk()
    root.geometry("900x450+200+200")
    # root.configure(background='gray98')
    root.title("FilePlotter (version " + version + ")")
    MainGUI(root)  # GUI is a frame in the root window
    root.mainloop()


if __name__ == '__main__':
    main()
