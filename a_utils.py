import pandas as pd
from numpy.random import default_rng
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn import linear_model
from sklearn import feature_selection
import statsmodels.api as sm
from sklearn.feature_selection import chi2
from mpmath import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import seaborn as sns
import networkx as nx
from textwrap import wrap
from sklearn.feature_selection import SelectFromModel
import os
from sklearn.preprocessing import MinMaxScaler
def printGraph(columns, data):
    # plt.style.use('ggplot')
    columns = columns.delete(len(columns) - 1)
    data = data[columns]
    corData = data.corr()
    # links = corData.stack().reset_index()
    # links.columns = ['var1', 'var2', 'value']
    # links_filtered = links.loc[(links['value'] > 0.8) & (links['var1'] != links['var2'])]
    # G = nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')
    # nx.draw(G, with_labels=True, node_color='orange', node_size=400, edge_color='black', linewidths=1, font_size=15)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corData, annot=False, cmap=plt.cm.Reds,ax=ax)
    plt.show()


class TeamFile:
    # instance attribute
    def __init__(self, train, listFileTest, resultColName):
        self.train = train
        self.listFileTest = listFileTest
        self.resultColName = resultColName
dirname = os.path.dirname(__file__)

def getOldDataset():
    train = os.path.join(dirname, 'data/feng_x.csv')
    fileListTest = []
    fileListTest.append(os.path.join(dirname, 'data/yu_x.csv'))
    fileListTest.append(os.path.join(dirname, 'data/zeller_x.csv'))
    fileListTest.append(os.path.join(dirname, 'data/vogtmann_x.csv'))
    return TeamFile(train, fileListTest, "RS")


def getNewDataset():
    train = os.path.join(dirname, 'data/ibdfullHS_UCr_x.csv') #iCDr & UCf &iCDf &CDr&CDf
    fileListTest = []
    fileListTest.append(os.path.join(dirname, 'data/ibdfullHS_iCDr_x.csv'))
    fileListTest.append(os.path.join(dirname, 'data/ibdfullHS_UCf_x.csv'))
    fileListTest.append(os.path.join(dirname, 'data/ibdfullHS_iCDf_x.csv'))
    fileListTest.append(os.path.join(dirname, 'data/ibdfullHS_CDr_x.csv'))
    fileListTest.append(os.path.join(dirname, 'data/ibdfullHS_CDf_x.csv'))
    return TeamFile(train, fileListTest, "RS")


class color:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


