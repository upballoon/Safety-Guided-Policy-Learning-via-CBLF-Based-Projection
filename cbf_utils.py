import numpy as np
import cvxpy as cp
import torch

def get_circle_centers_global(X, Y, psi, offset=1.4):
    centers_body = np.array([
        [ offset, 0.0],   # front
        [-offset, 0.0],   # rear
    ])  # (2, 2)

    cpsi, spsi = np.cos(psi), np.sin(psi)
    R = np.array([[cpsi, -spsi],
                  [spsi,  cpsi]])

    centers_world = centers_body @ R.T + np.array([X, Y])
    return centers_world


def cbf_constraints_two_discs(
    ego_state,
    other_state,
    L_wheelbase=2.875,
    r_ego=1.4,
    r_other=1.4,
    d_margin=0.2,
    k0=1.0,
    k1=2.0,
):
    
    X_e, Y_e, psi_e, v_e = ego_state["X"], ego_state["Y"], ego_state["psi"], ego_state["v"]
    X_o, Y_o, psi_o, v_o = other_state["X"], other_state["Y"], other_state["psi"], other_state["v"]

    centers_e = get_circle_centers_global(X_e, Y_e, psi_e)  # (2,2)
    centers_o = get_circle_centers_global(X_o, Y_o, psi_o)  # (2,2)

    d_safe = r_ego + r_other + d_margin

    cpsi, spsi = np.cos(psi_e), np.sin(psi_e)

    A_list, B_list = [], []

    for ce in centers_e:       # ego: front / rear
        for co in centers_o:   # other: front / rear
            dx = ce[0] - co[0]
            dy = ce[1] - co[1]

            R = dx * cpsi + dy * spsi
            T = -dx * spsi + dy * cpsi

            h = dx**2 + dy**2 - d_safe**2

            A = np.array([
                2.0 * R,                     
                2.0 * (v_e**2) / L_wheelbase * T  
            ], dtype=float)

            b = -2.0 * (v_e**2) - 2.0 * k1 * v_e * R - k0 * h

            A_list.append(A)
            B_list.append(b)

    return A_list, B_list

def cbf_lane_deviation(
    e_y,
    e_psi,
    v,
    L_wheelbase,
    e_max=3.0,
    k0=1.0,
    k1=2.0,
    kappa_ref=0.0,
):
    
    v_eff = max(v, 0.1)

    A_L = np.array([
        -e_psi,
        -(v_eff**2) / L_wheelbase
    ], dtype=float)
    b_L = (
        - (v_eff**2) * kappa_ref
        + k1 * v_eff * e_psi
        - k0 * (e_max - e_y)
    )

    A_R = np.array([
        e_psi,
        (v_eff**2) / L_wheelbase
    ], dtype=float)
    b_R = (
        (v_eff**2) * kappa_ref
        - k1 * v_eff * e_psi
        - k0 * (e_max + e_y)
    )

    A_lane = np.vstack([A_L, A_R])   # shape (2, 2)
    b_lane = np.array([b_L, b_R])    # shape (2,)

    return A_lane, b_lane

def clf_constraint(
    e_y,
    e_psi,
    e_v,
    v,
    L_wheelbase,
    k_y=1.0,
    k_psi=1.0,
    k_v=1.0,
    c_V=0.5,
    kappa_ref=0.0,
):
    V = 0.5 * (k_y * e_y**2 + k_psi * e_psi**2 + k_v * e_v**2)

    A_clf = np.array([
        k_v * e_v,                
        k_psi * e_psi * v / L_wheelbase  
    ], dtype=float)

    b_clf = (
        -c_V * V
        - k_y * v * e_y * e_psi
        + k_psi * v * kappa_ref * e_psi
    )

    return A_clf, b_clf


def solve_cbf_qp(
    ego_state,
    others_state,
    e_y,
    e_psi,
    e_v,
    kappa_ref,
    action=None,
    L_wheelbase=2.875,
    a_min=-3.0,
    a_max=3.0,
    delta_min=-0.3,
    delta_max=0.3,
    r_ego=1.5,
    r_other=1.5,
    d_margin=0.2,
    k0_cbf=1.0,
    k1_cbf=2.0,
    k3_cbf=1.0,
    k4_cbf=2.0,
    k_y=1.0,
    k_psi=1.0,
    k_v=1.0,
    c_V=0.5,
    w_a=1.0,
    w_delta=1.0,
    w_eps=10.0,
):
    
    v = ego_state["v"]

    A_cbf_all = []
    b_cbf_all = []

    for other in others_state:
        A_list, B_list = cbf_constraints_two_discs(
            ego_state=ego_state,
            other_state=other,
            L_wheelbase=L_wheelbase,
            r_ego=r_ego,
            r_other=r_other,
            d_margin=d_margin,
            k0=k0_cbf,
            k1=k1_cbf,
        )
        A_cbf_all.extend(A_list)
        b_cbf_all.extend(B_list)

    A_lane, b_lane = cbf_lane_deviation(
        e_y=e_y,
        e_psi=e_psi,
        v=v,
        L_wheelbase=L_wheelbase,
        e_max=6.0, 
        k0=k3_cbf,
        k1=k4_cbf,
        kappa_ref=kappa_ref,
    )
    A_cbf_all.append(A_lane[0])
    A_cbf_all.append(A_lane[1])
    b_cbf_all.append(b_lane[0])
    b_cbf_all.append(b_lane[1])

    if len(A_cbf_all) > 0:
        A_cbf_all = np.vstack(A_cbf_all)     # shape = (Nc, 2)
        b_cbf_all = np.array(b_cbf_all)      # shape = (Nc,)
    else:
        A_cbf_all = np.zeros((0, 2), dtype=float)
        b_cbf_all = np.zeros((0,), dtype=float)
    
    A_clf, b_clf = clf_constraint(
        e_y=e_y,
        e_psi=e_psi,
        e_v=e_v,
        v=v,
        L_wheelbase=L_wheelbase,
        k_y=k_y,
        k_psi=k_psi,
        k_v=k_v,
        c_V=c_V,
        kappa_ref=kappa_ref,
    )

    a = cp.Variable()             
    delta = cp.Variable()         
    eps = cp.Variable(nonneg=True)

    u = cp.hstack([a, delta])

    constraints = []

    # 4.1 CBF: A_cbf_all @ u >= b_cbf_all
    if A_cbf_all.shape[0] > 0:
        constraints.append(A_cbf_all @ u >= b_cbf_all)

    # 4.2 CLF: A_clf @ u <= b_clf + eps
    constraints.append(A_clf @ u <= b_clf + eps)

    constraints += [
        a >= a_min,
        a <= a_max,
        delta >= delta_min,
        delta <= delta_max,
    ]


    if action is not None:
        obj = 0.5 * (w_a * cp.square(a-action[0]) + w_delta * cp.square(delta-action[1]) + w_eps * cp.square(eps))
    else:
        obj = 0.5 * (w_a * cp.square(a) + w_delta * cp.square(delta) + w_eps * cp.square(eps))

    prob = cp.Problem(cp.Minimize(obj), constraints)

    try:
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
    except Exception:
        prob.solve(solver=cp.ECOS, warm_start=True, verbose=False)

    info = {
        "status": prob.status,
        "obj": None,
        "eps": None,
        "fallback": False,
        "A_cbf":A_cbf_all,
        "b_cbf":b_cbf_all,
    }

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        a_sol = float(np.clip(0.0, a_min, a_max))
        delta_sol = float(np.clip(0.0, delta_min, delta_max))
        info["fallback"] = True
        return a_sol, delta_sol, info

    a_sol = float(a.value)
    delta_sol = float(delta.value)
    info["obj"] = float(prob.value)
    info["eps"] = float(eps.value)

    return a_sol, delta_sol, info

from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class CBFContext:
    ego_state: Dict[str, float]
    others_state: List[Dict[str, float]]
    e_y: float
    e_psi: float
    e_v: float
    kappa_ref: float

def push_buffer(env, args):
    # 现在需要构造 CBFContext
    ego_state = {
        "X":   env.ego_car.X,
        "Y":   env.ego_car.Y,
        "psi": env.ego_car.psi,
        "v":   env.ego_car.vx,  # 或 np.hypot(vx, vy)
    }

    others_state = []
    for veh in (env.actor.lane_1_vehicles
                + env.actor.lane_2_vehicles
                + env.actor.lane_3_vehicles):
        others_state.append({
            "X":   veh.X,
            "Y":   veh.Y,
            "psi": veh.psi,
            "v":   veh.vx,
        })

    cbf_ctx = CBFContext(
        ego_state=ego_state.copy(),               # 防止后续 env 修改
        others_state=[d.copy() for d in others_state],
        e_y=float(env.e_y),
        e_psi=float(env.e_psi),
        e_v=float(env.ego_car.vx - args.v_ref),
        kappa_ref=float(env.curv_ref),
    )
    return cbf_ctx

def safe_action(env, args, a_raw=None,
                k0_cbf=1.0,
                k1_cbf=1.6,
                k3_cbf=1.0,
                k4_cbf=1.6,
                k_y=1,
                k_psi=0.1,
                k_v=0.1,
                ):
    ego_state = {
        "X":  env.ego_car.X,
        "Y":  env.ego_car.Y,
        "psi": env.ego_car.psi,
        "v":  env.ego_car.vx,  # 或 np.hypot(vx, vy)
    }

    others_state = []
    for veh in (env.actor.lane_1_vehicles
                + env.actor.lane_2_vehicles
                + env.actor.lane_3_vehicles):
        others_state.append({
            "X":  veh.X,
            "Y":  veh.Y,
            "psi": veh.psi,
            "v":  veh.vx,
        })
    
    a_safe, delta_safe, info = solve_cbf_qp(
            ego_state=ego_state,
            others_state=others_state,
            e_y=env.e_y,
            e_psi=env.e_psi,
            e_v=env.ego_car.vx - args.v_ref, 
            kappa_ref=env.curv_ref,  
            action=a_raw,        
            L_wheelbase=env.ego_car.L,
            k0_cbf=k0_cbf,
            k1_cbf=k1_cbf,
            k3_cbf=k3_cbf,
            k4_cbf=k4_cbf,
            k_y=k_y,
            k_psi=k_psi,
            k_v=k_v,
        )
    action = [a_safe, delta_safe]
    
    if a_raw is None:
        return action
    else:
        A_i = torch.as_tensor(info["A_cbf"], dtype=torch.float)
        b_i = torch.as_tensor(info["b_cbf"], dtype=torch.float).view(-1)
        g_i = A_i @ a_raw - b_i  # [Ni]

        return action, g_i