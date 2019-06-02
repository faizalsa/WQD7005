import pandas as pd
import numpy as np

# assemble financial data from yahoo finance
allstocks = 'portfolio.csv'

data = pd.read_csv(allstocks, sep=',')

# convert price into returns

returns = data.sort_values(['Date'], ascending = [True])

# eliminate first row
returns = returns[1:]

# focus on top 10 stocks only
stockNames = ['RGB',
 'MTRONIC',
 'CUSCAPI',
 'GPACKET',
 'NOTION',
 'DAYA',
 'TRIVE',
 'FRONTKN',
 'MYEG',
 'KEYASIC',
 'INARI',
 'PTRANS',
 'INTA',
 'AMBANK',
 'CIMB',
 'MAYBANK',
 'MBSB',
 'SUMATEC',
 'PBBANK',
 'IWCITY',
 'DRBHCOM',
 'MRCB',
 'IOICORP',
 'COMFORT',
 'ABMB',
 'HUAAN',
 'L&G',
 'GENTING',
 'IJM',
 'BJCORP',
 'JOHAN',
 'MUIIND',
 'TADMAX',
 'SIME',
 'LIONIND',
 'HENGYUAN',
 'DNEX',
 'PHB',
 'MEDIA',
 'POS',
 'YTL',
 'GENM',
 'JAKS',
 'VERSATL',
 'MIECO',
 'AIRPORT',
 'COMPUGT',
 'LUSTER',
 'HIAPTEK',
 'HEVEA',
 'MASTEEL',
 'AIRASIA',
 'CNI',
 'ALAM',
 'DAYANG',
 'UEMS',
 'XDL',
 'JCY',
 'MINDA',
 'HARTA',
 'KSTAR',
 'SIGGAS',
 'PCHEM',
 'MHB',
 'CNOUHUA',
 'APFT',
 'HIBISCS',
 'ARMADA',
 'SUNWAY',
 'SAPNRG',
 'GLOTEC',
 'FGV',
 'IHH',
 'IGBREIT',
 'AAX',
 'VELESTO',
 'BAUTO',
 'IOIPG',
 'ECONBHD',
 'BPLANT',
 'ICON',
 'REACH',
 'BIMB',
 'MALAKOF',
 'SERBADK',
 'LCTITAN',
 'SIMEPLT',
 'SIMEPROP',
 'TENAGA',
 'GAMUDA',
 'LBS',
 'MAXIS',
 'STAR',
 'MALTON',
 'ASTRO',
 'YTLPOWR',
 'AXIATA',
 'DIGI',
 'VS',
 'HUBLINE',
 'BORNOIL',
 'YONGTAI',
 'VIZIONE',
 'TIGER',
 'PERMAJU',
 'SUPERMX',
 'PERDANA',
 'TOPGLOV',
 'PWORTH',
 'CAELY',
 'SKPRES',
 'SCOMI',
 'PENTA',
 'KNM',
 'CAB',
 'DBE',
 'D&O',
 'THHEAVY',
 'DESTINI',
 'MINETEC',
 'PA',
 'DUFU',
 'BIOOSMO',
 'BARAKAH',
 'DIALOG',
 'DATAPRP',
 'SALCON',
 'SPSETIA',
 'PMETAL',
 'EKOVEST',
 'ANZO',
 'HWGB',
 'WCT']

stockNames

stockReturns = returns[stockNames]

stockReturns.head()

n = len(stockNames)

#-------------

import cvxopt as opt

from cvxopt import matrix, solvers

expectedReturn = np.mean(stockReturns)

expectedReturn = matrix(expectedReturn)

maxLoss = np.min(stockReturns)

maxLoss = -maxLoss

#convert maxloss to linear objective function
objective = matrix(maxLoss)

constraintsEqualityLHS = matrix(1.0,(1,n))
constraintsEqualityRHS = matrix(1.0)

# impose >= than constraints
constraintsInequalityLHSRow1 = matrix(-np.identity(n))
constraintsInequalityLHSRow2 = matrix(-np.transpose(np.array(expectedReturn)))

constraintsInequalityRHSRow1 = matrix(0.0, (n,1))
constraintsInequalityRHSRow2 = matrix(-np.ones((1,1))*0.02)

# combine lhs of all inequality constraints into one matrix
constraintsInequalityLHSCombined = matrix(np.concatenate((constraintsInequalityLHSRow1, constraintsInequalityLHSRow2),0))

# combine rhs of all inequality constraints into one matrix
constraintsInequalityRHSCombined = matrix(np.concatenate((constraintsInequalityRHSRow1, constraintsInequalityRHSRow2),0))

solution = solvers.lp(objective,constraintsInequalityLHSCombined,constraintsInequalityRHSCombined,constraintsEqualityLHS,constraintsEqualityRHS)

weightsLinear=solution['x']

weightsLinear=np.array(weightsLinear)

pd.DataFrame(weightsLinear).to_csv("weightsLinear.csv")
#print('\n')
print(sum(weightsLinear))


# calculate the risk of portfolio
portfolioRiskLinear = np.dot(weightsLinear.T,maxLoss)
#print('\n')
print(f'Returns sum-product of weights and max losses = {portfolioRiskLinear}')

# calculate the return of portfolio
portfolioReturnLinear = np.dot(expectedReturn.T,weightsLinear)
#print('\n')
print(f'Returns sum-product of weights and max returns = {portfolioReturnLinear}')
print('\n')


#for this portfolio return in as threshold at 2%










