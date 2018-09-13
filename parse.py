# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from bokeh.plotting import figure as _figure,show,Figure
from bokeh.models import ColumnDataSource
from bokeh.layouts import gridplot
from bokeh.palettes import brewer 
from bokeh.transform import jitter
from bokeh.models.annotations import Span,Label,BoxAnnotation

from inspect import isfunction
import sys

import re

import warnings
warnings.filterwarnings('ignore')


#查看
## 查看信息 主要在jupyter中使用
def _dataFrameInfo(data,iqr_mult):

    if not isinstance(data,pd.Series):
        data = pd.DataFrame(data)

    df = data
    index = ["数据类型","有效数据","缺失数据","异常数据","%s*iqr" %iqr_mult,"%s*iqr%%" %iqr_mult,"floor","ceil"]
    dt = pd.DataFrame(columns = df.columns.tolist(),index = index)
    dt.index.name = str(len(df)) + '条数据'
    dt.loc["数据类型"] = df.dtypes

    for col in df.columns:

        dt.loc["有效数据"][col] = len(df[df[col].notnull()])
        dt.loc["缺失数据"][col] = len(df[df[col].isnull()])
        try :
            q1 = df[col].quantile(q = 0.25)
            q3 = df[col].quantile(q = 0.75)
            iqr = q3 - q1
            ceil = q3 + iqr_mult * iqr
            floor = q1 - iqr_mult * iqr
            nor_len = len(df[(df[col] >= floor) & (df[col] <= ceil) ])
            dt.loc["异常数据"][col] = dt.loc["有效数据"][col] - nor_len
            dt.loc["%s*iqr" %iqr_mult][col] = nor_len
            dt.loc["%s*iqr%%" %iqr_mult][col] = '%0.2f%%' %(nor_len/ dt.loc["有效数据"][col] * 100)
            dt.loc["floor"][col] = floor
            dt.loc["ceil"][col] = ceil
            
        except:
            pass
    return dt



def _object_subs(o,func_name=""):
    r = re.compile(func_name,re.I)

    return sorted(filter(lambda x:r.search(x),dir(o)))

# 查看 信息
def info(o,*args,**kwargs):

    if isinstance(o,(pd.DataFrame,pd.Series)):
        return _dataFrameInfo(o,*args,**kwargs)
    elif isfunction(o):
        return help(o)
    else:
        return _object_subs(o,*args,**kwargs)
    
    

## 查看相关性信息
from scipy import stats
def lookRel(data,show_norm=False):
    
    l = []
    for col in data:
        q = stats.kstest(data[col],'norm',(data[col].mean(),data[col].std()))
        l.append(True if q.pvalue > 0.05 else False )
        l.append(q.pvalue)

    pear = data.corr()
    spear = data.corr("spearman")
    for i,col in enumerate(data.columns):
        if l[i] == False:
            pear[col] = spear[col]
            pear.loc[col] = spear.loc[col]
    rel = pear
    if show_norm: 
        rel["正太分布"] = l[::2]
        rel["pvalue"] = l[1::2]

    return rel


#处理
## 处理异常
def filterIQR(data,cols,out_cols=[],filter_count = 1,whis = 3):

    """
        
        col 需要过滤的列
        out_cols 为额外输出的列
        filter_count 循环过滤异常的次数

        ### 今后在添加
        err_per ：当异常占整体 多少时候结束，
                  还需要加最大过滤次数或当样本数量为谋值结束限制
        
    """
    
    if not isinstance(cols,(list,tuple,str)):
        raise TypeError('cols 可以为 list,tuple,str类型')
    if isinstance(cols,str):
        cols = [cols]
    if not isinstance(out_cols,(list,tuple,str)):
        raise TypeError('out_cols 可以为 list,tuple,str类型')
    if isinstance(out_cols,str):
        out_cols = [out_cols]
    
    d = {}
    for col in cols:
        new_out_cols = out_cols.copy()
        new_out_cols.append(col)
        temp_data = data[new_out_cols]
        
        for i in range(filter_count):
            q1 = temp_data[col].quantile(q = 0.25)
            q3 = temp_data[col].quantile(q = 0.75)
            iqr = q3 - q1
            floor = q1 - iqr * whis
            ceil = q3 + iqr * whis
            temp_data = temp_data[(temp_data[col] > floor) & (temp_data[col] < ceil )][new_out_cols]
        
        
        d[col] = data[(data[col] > floor) & (data[col] < ceil )][new_out_cols]
    return d

##清除左右为空的字符串
def strip(data,cols = None,inplace = False):
    pat = "\s*([^\s](?:.+?[^\s])?)\s*"
    if isinstance(cols,str):
        cols = [cols]
    
    rep_d = {}
    if cols:
        if not set(cols) <= set(data.columns):
            raise Warning("不存在: %s 列名"%(set(cols) - set(data.columns)))
        for col in cols:
            rep_d[col] = r"\1"
    else:
        #此处日后添加 为所有 object格式
        rep_d = r"\1"
    if not inplace:
        return data.replace(pat,rep_d,regex=True)    
    else:
        data.replace(pat,rep_d,regex=True,inplace = True)

##提取唯一字段 
"""
传入Series 适合需要拆分的数据
"""
def extractUnique(data,sep = '/',repl = {'\s*/\s*':"/","^\s*":""},rt_type="set"):
    l = []
    temps = data.replace(repl,regex=True)
    for temp in temps.str.split(sep):
        l.extend(temp)
    return set(l) if rt_type == "set" else list(set(l))

## 转换为gephi数据

def toGephiData(data,source_field,nor=None):
    gephi_data = data[source_field]
    gephi_data.columns = ["source","target","weight"]
    if nor != None and isinstance(nor,str):
        if nor == "nor":
            gephi_data["weight"] = (gephi_data["weight"] - gephi_data["weight"].min())\
        /(gephi_data["weight"].max() - gephi_data["weight"].min())
        elif nor == "nor_r":
            gephi_data["weight"] = (gephi_data["weight"].max() - gephi_data["weight"])\
        /(gephi_data["weight"].max() - gephi_data["weight"].min())
    
    return gephi_data


#图表分析

## 数据分析用图

def showBox(data,cols=None,by=None,figsize=(14,6),whis=1.5,vert=True,ax=None,
    patch_artist = True,meanline = False,showmeans=True,showbox = True,showcaps = True,
    showfliers=True,notch = False,
    **kwargs):
    if isinstance(data,pd.Series):
        data = pd.DataFrame(data)
    if not ax: 
        plt.rcParams['figure.figsize'] = figsize
    fs = data.boxplot(column=cols,
           by = by,
           ax=ax,
           whis = whis,
           vert = vert,
           #下面样式部分
           patch_artist = patch_artist,  # 上下四分位框内是否填充，True为填充
           meanline = meanline,showmeans=showmeans,  # 是否有均值线及其形状 其中meanline优先级更高
           showbox = showbox,  # 是否显示箱线
           showcaps = showcaps,  # 是否显示边缘线
           showfliers = showfliers,  # 是否显示异常值
           notch = False, # 中间箱体是否缺口
           return_type='dict',  # 返回类型为字典
           **kwargs
           )
    if not set(fs.keys()) < set(data.columns):
        fs = [fs]

    for f in fs:
        for box in f['boxes']:
            # print(box)
            box.set( color='aqua', linewidth=1)        # 箱体边框颜色
            box.set( facecolor = 'aqua' ,alpha=0.1)    # 箱体内部填充颜色
        for whisker in f['whiskers']:
            whisker.set(color='aqua', linewidth=0.5,linestyle='-')
        for cap in f['caps']:
            cap.set(color='aqua', linewidth=1)
        for median in f['medians']:
            median.set(color='DarkBlue', linewidth=1)
        for flier in f['fliers']:
            flier.set(marker='+',markeredgecolor="red", alpha=0.5,linewidth=0.5)


def showPie(d_pie,title=None, exp = None, labels = None, cmap=None, size=1.5,figsize=(4,4), textfmt='%.2f%%',
            textdis=0.6, labeldis = 1.1, shadow = False, rot=None, frame=False,rot_clock=False ,ax=None,**kwargs):
    
    """
        d_pie: 数据
        exp (explode):  指定每部分的偏移量
        labels:         标签
        cmap:           颜色
        textfmt:        饼图上的数据标签显示方式
        textdis:        每个饼切片的中心距离 与 text显示位置比例
        labeldis:       被画饼标记的直径,默认值：1.1
        shadow:         阴影
        rot:            开始角度
        size(radius):   半径
        frame：图框
        rot_clock:      指定指针方向，顺时针或者 (默认)逆时针

    """
    plt.rcParams["patch.force_edgecolor"] = False
    plt.rcParams['figure.figsize'] = figsize
    if isinstance(cmap,str):
        cmap = brewer[cmap][len(d_pie)]
        
    if not labels:
        labels = d_pie.index
    
    d_pie.plot.pie(
       explode = exp,
       labels = labels,
       colors=cmap,
       autopct='%.2f%%',
       pctdistance=textdis,
       labeldistance = labeldis,
       shadow = shadow,
       startangle=rot,
       radius=size,
       frame=False,
       counterclock = not rot_clock,
       ax=ax,
       **kwargs
       )
    if ax:
        ax.set(aspect="equal")
        if title:
            ax.set_title(title ,fontsize=14)
    else:
        plt.axis('equal')
        if title:
            plt.title(title ,fontsize=14)

 

# 色板

CMAP = ','.join(sorted(brewer.keys()))
pal = brewer
temp = {}

## 添加 反转 cmap
for cols in brewer:
    temp[cols+"_r"] = {}
    for i in brewer[cols]:
        temp[cols+"_r"][i] = brewer[cols][i]    #bokeh 的色板顺序和 seaborn 相反
        brewer[cols][i] = list(reversed( brewer[cols][i]))

brewer.update(temp)
del temp


def showPal(cmp = None):
    print(CMAP)
    if cmp: 
        if not isinstance(cmp,(list,tuple)):
            for s in brewer[cmp]:
                s_i = "%i" %s
                print(s_i.rjust(3),brewer[cmp][s])
                
                sns.palplot(brewer[cmp][s],size=0.6)
                plt.title(s,loc="left")
        else:
            sns.palplot(cmp,size=0.6)
        


# bokeh 的figure 无参数提示， 用有参数的函数包装下
def figure(plot_width,plot_height,title=None,
            x_axis_label = None, y_axis_label =None,  
    x_range=None,y_range=None,toolbar_location="right",tooltips = None,
    tools = None,
    **kwargs
    ):
    """
        toolbar_location="above"|"below"|"right"|"left"
        "hover,pan,box_zoom,box_select,lasso_select, wheel_zoom,crosshair,save,reset"
        p.line(df.index,df[col],line_width=2, color=color, alpha=0.8,legend = col,
           muted_color=color, muted_alpha=0.2) 

    """
    
    if not tools:
        tools = "pan,box_select, wheel_zoom,crosshair,save,reset"
    if tooltips:
        tools = "hover,"+tools

    return _figure(title=title,plot_width=plot_width,plot_height=plot_height,
            x_axis_label = x_axis_label, y_axis_label =y_axis_label,  
    x_range=x_range,y_range=y_range,toolbar_location=toolbar_location,
    tools = tools,tooltips = tooltips,**kwargs)



# 作图默认主题

## matplotlib 部分
plt_theme = {"monokai":"_plt_monokai_theme",}

def _plt_default_params(func):
    def inner(*args,**kargs):
        if len(args) > 1:
            sns.set_style(args[1])
        func(*args,**kargs) 
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei','SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 12
    return inner 

def _plt_monokai_theme():
    plt.rcParams['text.color'] = "white"
    # plt.rcParams["boxplot.flierprops.markeredgecolor"] = "red"
    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["ytick.left"] = True
    plt.rcParams["grid.color"] = "white"
    
    plt.rcParams["axes.grid"] = True
    

    plt.rcParams["patch.edgecolor"] = "#eeeeee"
    plt.rcParams["patch.force_edgecolor"] = True    #图片外边框
    # plt.rcParams["lines.solid_capstyle"] = "round"
    
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["axes.facecolor"] = "#232323"
    plt.rcParams["figure.facecolor"] = "#232323"
    plt.rcParams["grid.color"] = "#777777"
    plt.rcParams["xtick.color"] = "#eeeeee"
    plt.rcParams["ytick.color"] = "#eeeeee"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["axes.edgecolor"] = "#aaaaaa"
    plt.rcParams["axes.labelcolor"] = "white"
    plt.rcParams["legend.facecolor"] = "#232323"
    # plt.rcParams["xtick.top"] = False
    # plt.rcParams["ytick.right"] = False
    

@_plt_default_params
def _pltTheme(theme=None,*args):
    if theme == None:
        return
    if  theme not in plt_theme:
        if theme in ["white", "dark", "whitegrid", "darkgrid", "ticks"]:
            sns.set_style(theme)
            return
        print("matplot可选主题:",','.join(plt_theme.keys()))
        return

    func = getattr(sys.modules[__name__],plt_theme[theme])
    func()


## bokeh 部分
bokeh_theme = {"dash":"_fig_dash"}

def _fig_dash(fig):
    fig.xgrid.grid_line_dash = [6,4]
    fig.ygrid.grid_line_dash = [6,4]
    
def _bokehTheme(fig_objs=None,theme=None):
    if theme == None or theme not in bokeh_theme:
        print("bokeh可选主题:",','.join(bokeh_theme.keys()))
        return
    if not isinstance(fig_objs,list):
        fig_objs = [fig_objs]
    for fig in fig_objs:
        func = getattr(sys.modules[__name__],bokeh_theme[theme])
        func(fig)

## 全部由此函数统一调用
def figTheme(*args,**kargs):
    if len(args) != 0 and isinstance( args[0],(Figure,list,tuple)):
        _bokehTheme(*args)
    else:
        _pltTheme(*args,**kargs)

