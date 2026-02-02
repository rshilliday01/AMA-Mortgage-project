import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Monthly payment
r = 1000

# Monthly interest rate
alpha = 0.0436   # 4.36%

# ODE: B' = -r + alpha B
def Bdiff(t, B):
    return -r + alpha * B

# Event: stop when B(t) = 0
def balance_zero(t, B):
    return B[0]   # trigger when this hits zero

balance_zero.terminal = True   # stop integration
balance_zero.direction = -1    # only trigger when crossing from + to -

# Time interval (large upper bound, solver will stop early)
t_span = (0, 1000)

# Initial balance
B0 = [20000]

# Solve
sol = solve_ivp(
    Bdiff,
    t_span,
    B0,
    events=balance_zero,
    t_eval=np.linspace(0, 1000, 5000)
)

# Extract payoff time
if sol.t_events[0].size > 0:
    payoff_time = sol.t_events[0][0]
    print(f"Loan paid off after {payoff_time:.2f} months")

# Plot
plt.plot(sol.t, sol.y[0])
plt.axhline(0, color='r')  # zero line
plt.xlabel("Months")
plt.ylabel("Balance")
plt.title("Mortgage Balance Over Time")
plt.grid()
plt.savefig('mortgages.pdf')
plt.show()