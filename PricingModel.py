'''
Pricing Model

Step 1: Define the type of pricing model
I'm going to go with life insurance.
In Layman's terms life insurance cost is largely determined by 4 factors MILE

1) M - Mortality or Biometric factor
- complete health profile
2) I - Investment
- Interest date use to discount. Since term life insurance premiums are paid over the policy duration, actuaries discount the future expected cash flows (claims and expenses) to their present value using an appropraite discount rate
3) L - Lapse
- Chance of you lapsing the policy.
4) E - Expenses
-- Commission or underwriting expense. In addition to the expected claims, the premiums must cover the insurance company's expenses, such as underwriting, administration, and commissions. Actuaries incorporate these expense loadings into the pricing model.
'''

#Import Libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Mortality rate
def mortality_rate (age, smoker):
    base = 0.005 + 0.00015 * (age - 30)
    base = base * (1 + smoker)
    return np.minimum(base, 0.05)

Interest_Rate = 0.03 
Lapse_Rate = 0.05  # 5% Probability of policy terminating early each year

#Expenses
Initial_Expense = 50 # flat aquisition expense
Renewal_Expense = 10 # per year

#Profit loading
Profit_Loading = 0.10 # 10% margin

#Premium Calculation
def premium_per_policy(age, smoker, coverage_amount, term):
    pv_claims = 0
    pv_expenses = Initial_Expense #Upfront cost
    
    survival_prob = 1.0
    
    for t in range (1, term+1):
        age_t = age + t
        qx = mortality_rate(age_t, smoker)
        
        death_prob = survival_prob * qx
        
        #PV of claims
        v = 1 / (1+ Interest_Rate)**t
        pv_claims += coverage_amount * death_prob * v
        
        #Renewal expenses (discounted)
        pv_expenses += Renewal_Expense * survival_prob * v
        
        #Update survival probability for next year
        survival_prob *= (1-qx) * (1 - Lapse_Rate)
        
    total_pv = pv_claims + pv_expenses
    premium = total_pv * ( 1 + Profit_Loading)
    return premium

def annual_premium(single_premium, term, Interest_Rate):
    term = int(term)  # <-- add this line
    annuity_factor = sum(1 / (1 + Interest_Rate)**t for t in range( 1, term +1))
    return single_premium/ annuity_factor
'''
Sample Data 
Small dataset of example customers
'''
np.random.seed(50)
n =100
df = pd.DataFrame({
    'age': np.random.randint(25, 55, size = n),
    'smoker': np.random.choice([0,1], size = n, p = [0.8, 0.2]),
    'coverage_amount': np.random.randint(50000, 300000, size = n),
    'term': np.random.choice([10, 15, 20], size = n)
})

df['premium'] = df.apply(
    lambda row: premium_per_policy(
        age = row['age'],
        smoker = row['smoker'],
        coverage_amount = row['coverage_amount'],
        term = row['term']
        ), axis = 1
)

# Convert single premium to annual premium
INTEREST_RATE = 0.03
df['annual_premium'] = df.apply(
    lambda row: annual_premium(row['premium'], row['term'], Interest_Rate),
    axis=1
)

print(df)
# Example: Average premium by smoker status
print("\nAverage Premium by Smoker Status:")
print(df.groupby('smoker')['premium'].mean())


'''
The following is simply data analysis, code is repeated lots seperated by section
kept in the same file for now
'''
#Premium vs Age
plt.figure(figsize = (10,6))
plt.scatter(df['age'], df['premium'], label = 'Data points')

#Fit Line
x = df.age
y = df.premium
coefficients = np.polyfit( x, y, 1)
gradient, intercept = coefficients
poly_fn = np.poly1d(coefficients)

#Plot line of best fit
x_sorted = np.sort(x)
plt.plot(x_sorted, poly_fn(x_sorted), color = 'red', label = f'Best fit [y = {gradient:.4f}x + {intercept:.4f}]')
plt.title('Premium vs Age')
plt.xlabel('Age')
plt.ylabel('Premium')
plt.legend()
plt.show()
print(f'Data fitted with a linear line of best fit with gradient {gradient:.4f} and intercept {intercept:.4f}')

#Premium vs Coverage Amount
plt.figure(figsize = (10,6))
plt.scatter(df['coverage_amount'], df['premium'], label = 'Data points')
#Fit Line
x = df.coverage_amount
y = df.premium
coefficients = np.polyfit( x, y, 1)
gradient, intercept = coefficients
poly_fn = np.poly1d(coefficients)

#Plot line of best fit
x_sorted = np.sort(x)
plt.plot(x_sorted, poly_fn(x_sorted), color = 'red', label = f'Best fit [y = {gradient:.4f}x + {intercept:.4f}]')
plt.title('Premium vs Coverage Amount')
plt.xlabel('Coverage Amount')
plt.ylabel('Premium')
plt.legend()
plt.show()

#Save

#Annual Premium vs Term
plt.figure(figsize = (10,6))
plt.scatter(df['term'], df['annual_premium'], label = 'Data points')
plt.title('Annual Premium vs Term')
plt.xlabel('Term')
plt.ylabel('Annual Premium')
plt.legend()
plt.show()

#Smoker vs Non Smoker
prem_smoker = df.loc[df['smoker'] == 1, 'premium']
prem_nonsmoker = df.loc[df['smoker'] == 0, 'premium']

plt.violinplot([prem_nonsmoker, prem_smoker], showmeans=True)
plt.xticks([1, 2], ['Non-smoker', 'Smoker'])
plt.ylabel('Premium')
plt.title('Premium distribution by smoking status')
plt.show()

#Survival rate

# Non-smokers
AgeNonSmoker = df.loc[df['smoker'] == 0, 'age'].values

# Smokers
AgeSmoker = df.loc[df['smoker'] == 1, 'age'].values
#Although we can grab the values of 
ages = np.linspace(20,90,71)
MR_NonSmoker = mortality_rate(ages, smoker = 0)
MR_Smoker = mortality_rate(ages, smoker = 1 )
plt.figure(figsize = (10,6))
plt.plot(ages, MR_NonSmoker, label = 'Non Smoker')
plt.plot(ages, MR_Smoker, label = 'Smoker')
plt.title('Mortality rate vs age for smokers and non smokers')
plt.ylabel('Mortality Rate')
plt.xlabel('Ages')
plt.legend()
plt.show()


