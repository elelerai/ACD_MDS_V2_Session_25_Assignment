# ACD_MDS_V2_Session_25_Assignment
## 1.​​ ​ Introduction
This assignment will help you to consolidate the concepts learnt in the session.

## 2.​​ ​ Problem Statement

In this assignment students will build the random forest model after normalizing the
variable to house pricing from boston data set.
Following the code to get data into the environment:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
boston = datasets.load_boston()
features = pd.DataFrame(boston.data, columns=boston.feature_names)
targets = boston.target
NOTE:​​​​ ​​​​The​​​​ ​​​​solution​​​​ ​​​​shared​​​​ ​​​​through​​​​ ​​​​Github​​​​ ​​​​should​​​​ ​​​​contain​​​​ ​​​​the​​​​ ​​​​source​​ ​​code​​​​ ​​​​used​​​​ ​​​​and​​​​
​​​​the​​​​ ​​​​screenshot​​​​ ​​​​of​​​​ ​​​​the​​​​ ​​​​output.
## 3.​​ ​ Output
N/A
