import numpy as np
import pandas as pd
from compare import generate_pairs

from metrics import Metrics



def make_data0(n=500, p=0.5, l=6):
    # n is the number of data points.
    # 0 <= p <= 1 is the sampling probability of Male (sex=1).
    keys = ["sex", "work_exp", "hair_length", "hire", "pred"]
    data = {key: [] for key in keys}
    for i in range(n):
        rand = np.random.random()
        sex = 1 if rand < p else 0
        hair_length = 35 * np.random.beta(2, 2+5*sex)
        # work_exp = int(np.random.poisson(5 + 6 * sex))
        work_exp = int(np.random.poisson(25+l*sex) - np.random.normal(20, 0.2))
        if work_exp < 0:
            work_exp = 0
        hire_prob = 1.0/(1+np.exp(25.5-2.5*work_exp))
        pred_prob = hire_prob + np.random.normal(0, 0.1)
        # hire_prob = 1.0 / (1 + np.exp(15.5  +10*sex - 2.5 * work_exp))
        # hire_prob = 1.0 / (1 + np.exp(8 - work_exp))
        data["sex"].append(sex)
        data["work_exp"].append(work_exp)
        data["hair_length"].append(hair_length)
        data["hire"].append(hire_prob)
        data["pred"].append(pred_prob)
    df = pd.DataFrame(data, columns = keys)
    return df

def make_data1(n=500, p=0.5, l=6):
    # n is the number of data points.
    # 0 <= p <= 1 is the sampling probability of Male (sex=1).
    keys = ["sex", "work_exp", "hair_length", "hire", "pred"]
    data = {key: [] for key in keys}
    for i in range(n):
        rand = np.random.random()
        sex = 1 if rand < p else 0
        hair_length = 35 * np.random.beta(2, 2+5*sex)
        # work_exp = int(np.random.poisson(5 + 6 * sex))
        work_exp = int(np.random.poisson(25+l*sex) - np.random.normal(20, 0.2))
        if work_exp < 0:
            work_exp = 0
        hire_prob = 1.0/(1+np.exp(25.5-2.5*work_exp))
        pred_prob = 1.0 / (1 + np.exp(15.5  +10*sex - 2.5 * work_exp)) + np.random.normal(0, 0.1)
        # hire_prob = 1.0 / (1 + np.exp(15.5  +10*sex - 2.5 * work_exp))
        # hire_prob = 1.0 / (1 + np.exp(8 - work_exp))
        data["sex"].append(sex)
        data["work_exp"].append(work_exp)
        data["hair_length"].append(hair_length)
        data["hire"].append(hire_prob)
        data["pred"].append(pred_prob)
    df = pd.DataFrame(data, columns = keys)
    return df



df1 = make_data0(n=1000, p=0.5, l=6)
df1["A"] = df1["sex"]
df1["Y"] = df1["hire"]
df1["Y_pred"] = df1["pred"]

df = df1


m = Metrics(df["Y"], df["Y_pred"])

gAOD = m.gAOD(df["A"])
gWithin = m.gWithin(df["A"])
gSep = m.gSep(df["A"])
MI = m.MI(df["A"])

from pdb import set_trace
# set_trace()

data_tr = generate_pairs(df)

m = Metrics(data_tr["Label"], data_tr["pred"])
# AOD_comp = m.AOD_comp(data_tr[["A", "B"]])
# Within_comp = m.Within_comp(data_tr[["A", "B"]])
# Sep_comp = m.Sep_comp(data_tr[["A", "B"]])
MI_comp = m.MI_comp(data_tr[["A", "B"]])
set_trace()