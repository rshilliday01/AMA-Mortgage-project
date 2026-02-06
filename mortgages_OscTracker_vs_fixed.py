import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def r(t):
        return 1400

def alpha(t):
        return (0.05 + 0.025*np.sin(2*np.pi*t/60)) / 12

# ODE: B' = -r + alpha B
def Bdiff(t, B):
    return -r(t) + alpha(t) * B

# Event: stop when B(t) = 0
def balance_zero(t, B):
    return B[0]   # trigger when this hits zero

balance_zero.terminal = True   # stop integration
balance_zero.direction = -1    # only trigger when crossing from + to -

# Time interval (large upper bound, solver will stop early)
t_span = (0, 1000)

# Initial balance
B0 = [220000]
print(f"1st loan Initial balance: {B0[0]:.2f}")

# Solve
sol = solve_ivp(
    Bdiff,
    t_span,
    B0,
    events=balance_zero,
    t_eval=np.linspace(0, 1000, 5000)
)

# Extract and print payoff time
if sol.t_events[0].size > 0:
    payoff_time = sol.t_events[0][0]
    payoff_time_year = payoff_time/12
    total_cost = payoff_time*1400
    print(f"1st loan paid off after {payoff_time:.2f} months")
    print(f"1st loan paid off after {payoff_time_year:.2f} years")
    print(f"total cost (£) paid back: {total_cost:.2f} \n")

def r2(t):
    return 1400

# Initial balance 2nd mortgage
B1 = [220000]
print(f"2nd loan Initial balance: {B1[0]:.2f}")

def alpha2(t):
        return 0.055 / 12
    
# ODE: B' = -r + alpha(t) B
def Bdiff2(t, B):
    return -r2(t) + alpha2(t) * B

# Solve 2nd set of mortgage variables
sol2 = solve_ivp(
    Bdiff2,
    t_span,
    B1,
    events=balance_zero,
    t_eval=np.linspace(0, 1000, 5000)
)

# Extract and print payoff time
if sol2.t_events[0].size > 0:
    payoff_time = sol2.t_events[0][0]
    payoff_time_year = payoff_time/12
    total_cost = payoff_time*1400
    print(f"2nd loan paid off after {payoff_time:.2f} months")
    print(f"2nd loan paid off after {payoff_time_year:.2f} years")
    print(f"total cost (£) paid back: {total_cost:.2f}")


# Plot
plt.plot(sol.t, sol.y[0] / 1000, label='Tracker interest', color = 'g')
plt.plot(sol2.t, sol2.y[0] / 1000, label='Fixed interest', color = 'r')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.xlabel("Months", fontsize=14)
plt.ylabel("Balance (£1000)", fontsize=14)
plt.legend(fontsize=12)
plt.savefig('OSC_Tracker_vs_Fixed.pdf', bbox_inches='tight')
plt.show()
