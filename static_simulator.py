from dataclasses import dataclass
import numpy as np

from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt




# ──────────────────────────────────────────────────────────────
# system equations
# ──────────────────────────────────────────────────────────────


def dC_dt(T_pan, t, u, constant=0.0, t_offset=0.0):
    # dC/dt crust-formation rate.
    a = 3.315151e-05
    b = 170.604
    exponent = -a * (T_pan - b) * t
    return 100 * a * ((T_pan - b) + t * u) * np.exp(exponent) + constant

def sigmoid(x):
    # sigmoid function.
    return 1 / (1 + np.exp(-x))

def dTint_dt(t, T_pan, T_int_init, u, constant=0.0, t_offset=0.0):
    # dTint/dt internal temperature rate.
    c = 1.1
    d = -3.908e-03
    f = 2.742e-05
    g = 587.7

    T_diff = T_pan - T_int_init
    H = c * (d + f * T_diff) * (t - g)
    sig = sigmoid(H)
    sig_prime = sig * (1 - sig)
    
    dH_dt = f * u * (t - g) + d + f * T_diff
    
    return u * sig + c * T_diff * sig_prime * dH_dt + constant


# ──────────────────────────────────────────────────────────────
# constants & simulation options
# ──────────────────────────────────────────────────────────────


@dataclass
class SteakTargets:
    Tsteak_init: float     # initial core temperature (°F)
    tpan_init: float # initial pan temperature (°F)
    Tgoal: float     # target core temperature  (°F)
    Cgoal: float     # target crust index       (0-1)


@dataclass
class SimOptions:
    dt_truth:  float = 0.001     # 1 ms physics step
    dt_ctrl:   float = 1.0     # 1 s  controller / measurement step
    tol_C:     float = .5     # acceptable crust error (± units)
    tol_T:     float = .5     # acceptable core-temp error (± °F)


# ──────────────────────────────────────────────────────────────
# time step
# ──────────────────────────────────────────────────────────────


def time_step(x, u, t, dt, Tinit, k_loss=0.001, Tambient=70.0):
    C, Tint, Tpan = x

    dT_heater = u
    dT_loss = k_loss * (Tpan - Tambient)
    u = dT_heater - dT_loss

    # timestep updates
    C += dC_dt(Tpan, t, u) * dt
    C = np.clip(C, 0.0, 100.0) # clamp to [0, 100]
    
    Tint += dTint_dt(t, Tpan, Tinit, u) * dt


    Tpan += u * dt
    #Tpan += u * dt

    return np.array([C, Tint, Tpan])


# ──────────────────────────────────────────────────────────────
# simulator
# ──────────────────────────────────────────────────────────────


def simulate(tgt, opt, controller, goal):

    # ──────────────────────────────────────────────────────────────
    # initialization
    # ──────────────────────────────────────────────────────────────


    # init values
    x_true  = np.array([0.0, tgt.Tsteak_init, tgt.tpan_init]) # real values
    x_meas  = x_true.copy() # measured values
    
    u = 0.0 # init control
    t = 0.0 # init time

    t_log = []
    X_log = []

    # complete flags
    cflag = False
    tflag = False

    next_ctrl_time = 0.0
    ctrl_cnt = 3000.0
    
    # ──────────────────────────────────────────────────────────────
    # simulation loop
    # ──────────────────────────────────────────────────────────────

    while 1:
        # print for user
        print(f"t={t:.3f}s, C={x_true[0]:.2f}, Tint={x_true[1]:.2f}, Tpan={x_true[2]:.2f} - [{cflag},{tflag}], ctrl={u:.2f}          ", end="\r")

        # controller on every dt_ctrl step
        if t >= next_ctrl_time:
            x_meas = x_true.copy()

            # generate control
            # UNUSED. ONLINE PLANNER
            if t >= ctrl_cnt:
                print(f"t={t:.3f}s, C={x_true[0]:.2f}, Tint={x_true[1]:.2f}, Tpan={x_true[2]:.2f} - [{cflag},{tflag}], ctrl={u:.2f}          ")
                goal, _ = optimal_pan_temp(targets, t, x_meas)
                ctrl_cnt += 60.0
            
            u = controller(goal, x_meas, opt.dt_ctrl)

            # update log
            t_log.append(t)
            X_log.append(x_meas)

            # check stopping criteria
            C_err   = abs(x_meas[0] - tgt.Cgoal)
            Tint_err= abs(x_meas[1] - tgt.Tgoal)

            if C_err <= opt.tol_C and not cflag:
                print(f"Crust index finished: {x_meas[0]:.2f} (goal: {tgt.Cgoal:.2f}) at time {t:.1f}                      ")
                
                cflag = True
            
            if Tint_err <= opt.tol_T and not tflag:
                print(f"Core temperature finished: {x_meas[1]:.2f} (goal: {tgt.Tgoal:.2f}) at time {t:.1f}                      ")
                tflag = True

            if cflag and tflag:
                print(f"Ending time: {t:.1f} s\n")
                print(f"Crust index ended at: {x_meas[0]:.2f} (goal: {tgt.Cgoal:.2f})")
                print(f"Core temperature ended at: {x_meas[1]:.2f} (goal: {tgt.Tgoal:.2f})")
                
                return (np.asarray(t_log),np.vstack(X_log), t)

            next_ctrl_time += opt.dt_ctrl


        # propagate fine dynamics
        x_true = time_step(x_true, u, t, opt.dt_truth, tgt.Tsteak_init)

        # increment time
        t += opt.dt_truth

    return (np.asarray(t_log),np.vstack(X_log), t)


# ──────────────────────────────────────────────────────────────
# controller
# ──────────────────────────────────────────────────────────────


class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.prev_error = None

    # FIX THIS
    def __call__(self, T_plan, x_meas, dt):
        
        error = T_plan - x_meas[2]
        self.integral += (error) * dt
        self.integral = np.clip(self.integral, -100, 100) # clamp integral to prevent windup



        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

def temp_PID_controller():
    k=(0.1, 0.005, 0.001)
    pid = PID(*k)

    return pid


# ------------------------------------------------------------
# steady temp solution
# ------------------------------------------------------------


def t_core(Tpan, T_goal, T_init, T_curr=0.0):
    c = 1.1
    d = -3.908e-03
    f = 2.742e-05
    g = 587.7


    dT  = Tpan - T_curr
    k   = c * (d + f * dT)
    S   = T_goal - T_curr + dT / (1.0 + np.exp(k * g))
    rhs = dT / S - 1.0
    if rhs <= 0 or k == 0: # invalid parameter combo
        return np.inf
    return g - (1.0 / k) * np.log(rhs)
   
def t_crust(Tpan, C_goal,  C_curr=0.0):
    a = 3.315151e-05
    b = 170.604

    if C_curr >= C_goal:
        return 0.0

    dC = C_goal - C_curr
    if dC >= 100 or Tpan <= b:
        return np.inf

    return -np.log(1.0 - dC/100.0) / (a * (Tpan - b))

def cost(Tpan, Tint, C, Tinit, x_meas):
    return abs(t_core(Tpan, Tint, Tinit, x_meas[1]) - t_crust(Tpan, C, x_meas[0]))

def optimal_pan_temp(targets, t, x_meas, bracket=(250.0, 450.0), xatol=1e-3):

    # unpack targets
    Tgoal   = targets.Tgoal
    Cgoal   = targets.Cgoal
    Tinit   = targets.Tsteak_init

    sol = minimize_scalar(cost, bounds=bracket, args=(Tgoal, Cgoal, Tinit, x_meas), method='bounded', options={'xatol': xatol})
    print(f"optimal pan temperature: {sol.x:.1f} °F")
    print(f"residual time difference: {sol.fun:.1f} s")
    if not sol.success:
        raise RuntimeError("optimisation failed: " + sol.message)
    return sol.x, sol.fun   # (optimal Tpan, residual |dt|)


# ------------------------------------------------------------
# plotting
# ------------------------------------------------------------


def plot_logs(t_log, X_log, targets=None):
    """
    Vibe Coded plotting function
    """
    t = np.array(t_log)
    C    = X_log[:,0]
    Tint = X_log[:,1]
    Tpan = X_log[:,2]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    
    # Crust index
    ax = axes[0]
    ax.plot(t, C,    label="Crust index")
    if targets is not None:
        ax.axhline(targets.Cgoal, color="gray", linestyle="--", label="C goal")
    ax.set_ylabel("Crust index")
    ax.legend(loc="best")
    ax.grid(True, which="both", ls=":", lw=0.5)

    # Core temperature
    ax = axes[1]
    ax.plot(t, Tint, label="Core temperature")
    if targets is not None:
        ax.axhline(targets.Tgoal, color="gray", linestyle="--", label="T goal")
    ax.set_ylabel("Core temp (°F)")
    ax.legend(loc="best")
    ax.grid(True, which="both", ls=":", lw=0.5)

    # Pan temperature
    ax = axes[2]
    ax.plot(t, Tpan, label="Pan temperature")
    ax.set_ylabel("Pan temp (°F)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="best")
    ax.grid(True, which="both", ls=":", lw=0.5)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# main
# ------------------------------------------------------------


if __name__ == "__main__":

    # set values HERE
    targets = SteakTargets(Tsteak_init=66.0, tpan_init=350, Tgoal=135.0, Cgoal=95.0)
    opts = SimOptions()


    controller = temp_PID_controller()

    goal, _ = optimal_pan_temp(targets, 0, [0.0, targets.Tsteak_init, targets.tpan_init])

    # run sim
    t_pts, X_pts, t_end = simulate(targets, opts, controller, goal)
    plot_logs(t_pts, X_pts, targets)
    print(f"Finished at t = {t_end:.1f} s with {len(t_pts)} samples.")


