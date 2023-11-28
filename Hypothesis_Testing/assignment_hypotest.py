# -*- coding: utf-8 -*-
"""Assignment HYPOTEST.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1A_D3qVlxZO9M1HVUFkBXUwbPlcOb6lUE
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm

"""### Q1.     A F&B manager wants to determine whether there is any significant difference in the diameter of the cutlet between two units. A randomly selected sample of cutlets was collected from both units and measured? Analyze the data and draw inferences at 5% significance level. Please state the assumptions and tests that you carried out to check validity of the assumptions."""

cutlets=pd.read_csv('C:/Users/Masum/Downloads/Cutlets.csv')

cutlets.head()

# separating both the columns in different names
# TWO sample TWO tail test

c1=cutlets.iloc[:,0]
c1

c2=cutlets.iloc[:,1]
c2

c1

stats.ttest_ind(c1,c2)

p_value=(stats.ttest_ind(c1,c2)[1])

p_value

# compare p_value with α = 0.05 (At 5% significance level)
# p_> a , hence accepting NULL Hypothesis



"""### Q2.A hospital wants to determine whether there is any difference in the average Turn Around Time (TAT) of reports of the laboratories on their preferred list. They collected a random sample and recorded TAT for reports of 4 laboratories. TAT is defined as sample collected to report dispatch.
  
"""

lab=pd.read_csv('C:/Users/Masum/Downloads/LabTAT.mtw')

lab.head()

# separating the columns

l1=lab.iloc[:,0]
l1

l2=lab.iloc[:,1]
l2

l3=lab.iloc[:,2]
l3

l4=lab.iloc[:,3]
l4

# Using ANNOVA method because the data has more than two groups

stats.f_oneway(l1,l2,l3,l4)

p_value=stats.f_oneway(l1,l2,l3,l4)[1]

p_value

# compare p_value with α = 0.05 (At 5% significance level)
# p_< a , hence rejecting NULL Hypothesis



"""### Q3.Sales of products in four different regions is tabulated for males and females. Find if male-female buyer rations are similar across regions."""

df=pd.read_csv('C:/Users/Masum/Downloads/BuyerRatio.csv')

df.head()

dff=df.iloc[:,1:]

dff

dff.values

# here two qualitative variables are independent so we are using chi-square test.

from scipy.stats import chi2_contingency

chi_val, p_val, dof, expected =  chi2_contingency(dff)
chi_val, p_val, dof, expected

chi_val

p_val

# degree of freedom
dof

# Expected value
expected

# Critical Value
from scipy.stats import chi2
critical_value=chi2.ppf(0.95,df=3)
critical_value

# compare p_value with α = 0.05
# p_> a , hence accept NULL Hypothesis

if p_val <= 0.05:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')



"""### Q4.     TeleCall uses 4 centers around the globe to process customer order forms. They audit a certain %  of the customer order forms. Any error in order form renders it defective and has to be reworked before processing.  The manager wants to check whether the defective %  varies by centre. Please analyze the data at 5% significance level and help the manager draw appropriate inferences

"""

tele=pd.read_csv('C:/Users/Masum/Downloads/Costomer+OrderForm.csv')

tele.head()

tele.Phillippines.value_counts()

tele.Indonesia.value_counts()

tele.Malta.value_counts()

tele.India.value_counts()

TELE=np.array([[271,267,269,280],[29,33,31,20]])

TELE

chi_val, p_val, dof, expected =  chi2_contingency(TELE)
chi_val, p_val, dof, expected

p_val

# compare p_value with α = 0.05
# p_> a , hence accept NULL Hypothesis

