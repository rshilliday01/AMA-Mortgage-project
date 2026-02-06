import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def r(t, B):
    if t <= 60:
        return 1148
    else:
        N = 300 - t  # remaining months
        a = alpha(t)
        return (a * B) / (1 - (1 + a)**(-N))

def alpha(t):
    if t<=60:
        return 0.0482 /12
    else:
        return (0.045 + 0.025*np.sin(2*np.pi*(t-10)/60)) / 12

# ODE: B' = -r + alpha B
def Bdiff(t, B):
    return -r(t, B[0]) + alpha(t) * B

# Event: stop when B(t) = 0
def balance_zero(t, B):
    return B[0]   # trigger when this hits zero

balance_zero.terminal = True   # stop integration
balance_zero.direction = -1    # only trigger when crossing from + to -

# Time interval (large upper bound, solver will stop early)
t_span = (0, 720)

# Initial balance
B0 = [200000]
print(f"1st loan Initial balance: {B0[0]:.2f}")

# Solve
sol = solve_ivp(
    Bdiff,
    t_span,
    B0,
    events=balance_zero,
    t_eval=np.linspace(0, 720, 5000)
)

# Extract and print payoff time
if sol.t_events[0].size > 0:
    payoff_time = sol.t_events[0][0]
    payoff_time_year = payoff_time/12
    t_vals = sol.t
    B_vals = sol.y[0] 
    r_vals = np.array([r(t, B) for t, B in zip(t_vals, B_vals)]) 
    mask = t_vals <= payoff_time 
    total_cost = np.trapezoid(r_vals[mask], t_vals[mask]) 
    print(f"1st loan paid off after {payoff_time:.2f} months")
    print(f"1st loan paid off after {payoff_time_year:.2f} years")
    print(f"total cost (£) paid back: {total_cost:.2f} \n")

def r2(t, B):
    if t <= 60:
        return 957
    else:
        N = 300 - t  # remaining months
        a = alpha(t)
        return (a * B) / (1 - (1 + a)**(-N))

# Initial balance 2nd mortgage
B1 = [180000]
print(f"2nd loan Initial balance: {B1[0]:.2f}")

def alpha2(t):
    if t<=60:
        return 0.0407 /12
    else:
        return (0.045 + 0.025*np.sin(2*np.pi*(t-46)/60)) / 12
    
# ODE: B' = -r + alpha(t) B
def Bdiff2(t, B):
    return -r2(t, B[0]) + alpha2(t) * B

# Solve 2nd set of mortgage variables
sol2 = solve_ivp(
    Bdiff2,
    t_span,
    B1,
    events=balance_zero,
    t_eval=np.linspace(0, 720, 5000)
)

# Extract and print payoff time
if sol2.t_events[0].size > 0:
    payoff_time = sol2.t_events[0][0]
    payoff_time_year = payoff_time/12
    t_vals = sol2.t
    B_vals = sol2.y[0] 
    r_vals = np.array([r2(t, B) for t, B in zip(t_vals, B_vals)]) 
    mask = t_vals <= payoff_time 
    total_cost = np.trapezoid(r_vals[mask], t_vals[mask]) 
    print(f"2nd loan paid off after {payoff_time:.2f} months")
    print(f"2nd loan paid off after {payoff_time_year:.2f} years")
    print(f"total cost (£) paid back: {total_cost:.2f} \n")


def r3(t, B):
    if t <= 60:
        return 1563
    else:
        N = 180 - t  # remaining months
        a = alpha(t)
        return (a * B) / (1 - (1 + a)**(-N))

# Initial balance 3rd mortgage
B2 = [200000]
print(f"3rd loan Initial balance: {B2[0]:.2f}")

def alpha3(t):
    if t<=60:
        return 0.0482 /12
    else:
        return (0.045 + 0.025*np.sin(2*np.pi*(t-10)/60)) / 12
    
# ODE: B' = -r + alpha(t) B
def Bdiff3(t, B):
    return -r3(t, B[0]) + alpha3(t) * B

# Solve 3rd set of mortgage variables
sol3 = solve_ivp(
    Bdiff3,
    t_span,
    B2,
    events=balance_zero,
    t_eval=np.linspace(0, 720, 5000)
)

# Extract and print payoff time
if sol3.t_events[0].size > 0:
    payoff_time = sol3.t_events[0][0]
    payoff_time_year = payoff_time/12
    t_vals = sol3.t
    B_vals = sol3.y[0] 
    r_vals = np.array([r3(t, B) for t, B in zip(t_vals, B_vals)]) 
    mask = t_vals <= payoff_time 
    total_cost = np.trapezoid(r_vals[mask], t_vals[mask]) 
    print(f"3rd loan paid off after {payoff_time:.2f} months")
    print(f"3rd loan paid off after {payoff_time_year:.2f} years")
    print(f"total cost (£) paid back: {total_cost:.2f} \n")

def r4(t, B):
    if t <= 60:
        return 1338
    else:
        N = 180 - t  # remaining months
        a = alpha(t)
        return (a * B) / (1 - (1 + a)**(-N))

# Initial balance 4th mortgage
B3 = [180000]
print(f"4th loan Initial balance: {B3[0]:.2f}")

def alpha4(t):
    if t<=60:
        return 0.0407 /12
    else:
        return (0.045 + 0.025*np.sin(2*np.pi*(t-46)/60)) / 12
    
# ODE: B' = -r + alpha(t) B
def Bdiff4(t, B):
    return -r4(t, B[0]) + alpha4(t) * B

# Solve 4th set of mortgage variables
sol4 = solve_ivp(
    Bdiff4,
    t_span,
    B3,
    events=balance_zero,
    t_eval=np.linspace(0, 720, 5000)
)

# Extract and print payoff time
if sol4.t_events[0].size > 0:
    payoff_time = sol4.t_events[0][0]
    payoff_time_year = payoff_time/12
    t_vals = sol4.t
    B_vals = sol4.y[0] 
    r_vals = np.array([r4(t, B) for t, B in zip(t_vals, B_vals)]) 
    mask = t_vals <= payoff_time 
    total_cost = np.trapezoid(r_vals[mask], t_vals[mask]) 
    print(f"4th loan paid off after {payoff_time:.2f} months")
    print(f"4th loan paid off after {payoff_time_year:.2f} years")
    print(f"total cost (£) paid back: {total_cost:.2f} \n")  

# Plot
plt.plot(sol.t, sol.y[0] / 1000, label='£20k deposit, 25 years')
plt.plot(sol2.t, sol2.y[0] / 1000, label='£40k deposit, 25 years')
plt.plot(sol3.t, sol3.y[0] / 1000, label='£20k deposit, 15 years')
plt.plot(sol4.t, sol4.y[0] / 1000, label='£40k deposit, 15 years')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.xlabel("Months", fontsize=14)
plt.ylabel("Balance (£1000)", fontsize=14)
plt.legend(fontsize=12)
plt.savefig('LLOYDS_years_compare.pdf', bbox_inches='tight')
plt.show()
