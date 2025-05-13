import numpy as np
import time 
def main():
    np.random.seed(0)
    obs = np.array([[-12.5, -5,2.5],
                    [-15,0,1],
                    [-5, 2, 2],
                    [-20,5, 4]
                    ])
    safe_zones = np.array([[-30.0, 0.0,  0.0],
                        [-10.0, 0.0, 0.0],
                        [-6.0, -8.0, 0.0],
                        [-10.0, -8.0, 0.0],
                        [-15.0, -8.0, 0.0],
                        [-20.0, -8.0, 0.0],
                        [-24.0, -8.0, 0.0],
                        [-26.0, -6.0, 0.0],
                        [-6.0, -3.0, 0.0],
                        [-4.0, -4.0, 0.0], 
                        [3.5, 1.0, 0.0],
                        [-2.0, -6.0, 0.0],
                        [1.0,-3.0,0.0]])  # Safe States

    resolution = 0.4
    origin = np.array([-40,-10])
    wh = np.array([50/resolution, 20/resolution]) # width, height
    boundary = [[origin[0], origin[0]+wh[0]*resolution], [origin[1], origin[1]+wh[1]*resolution]]
    grid = OccupGrid(boundary, resolution)
    grid.find_occupancy_grid(obs, buffer=0.05)
    occupied = grid.find_all_occupied(obs)
    planner = TopoPRM(None, resolution=resolution, max_raw_path=20, max_raw_path2=20,reserve_num=6, ratio_to_short=10)
    planner.occup_grid = grid.occup_grid
    planner.origin = origin
    planner.resolution = resolution
    planner.wh = wh
    planner.safe_zones = safe_zones
    start=np.array([-24.0, -8.0, -np.pi]) 
    # start=np.array([-11.0, -7.0, -np.pi*0.75]) 
    # start=np.array([3.5, 3.5, -np.pi*0.75]) 
    # start=np.array([3.5, 3.5, 0]) 
    end = np.array([-30.0, 0.0])  # Reference state
    theta = np.arctan2((end[:2]-start[:2])[1], (end[:2]-start[:2])[0])
    end = np.array([-31.0, 0.0, theta])  # Reference state


    st = time.time()
    paths, samples = planner.findTopoPaths(start, end)
    print(f"Plan time: {time.time()-st}")

    dt = 0.2  # Time step
    Nt = 30  # horizon for MPPIn
    # dt = 0.4  # Time step
    # Nt = 15  # horizon for MPPIn
    nominal_params = {"L":1.0}
    halfcar = HalfCar(nominal_params, dt=dt)

    halfcar = Unicycle({"lb":np.array([-3.1415, -2.0]), "ub":np.array([3.1415, 2.0])}, dt=dt)

    dis = np.array(halfcar.control_bounds[1][1]) * halfcar.dt * Nt*0.8

    # Parameters for Collocation
    # state = np.array([3.5, 3.5, 0.0])  # [x, y, theta]
    Q = np.diag([10.0,10.0, 0.0])  # Weight for stz`te
    QT = Q.copy() * Nt / 3.0  # Weight for terminal state
    R = np.diag([0.0000001, 0.00000001])  # Weight for control
    # q_ref = np.array([-30.0, 0.0, 0.0])  # Reference state

    dxdt, state, control = halfcar.cas_ode()
    ode = ca.Function('ode', [state, control], [dxdt]) 
    f_nonlin = cas_shooting_solver(halfcar, Nt, ode=ode)
    f_lin = cas_shooting_linearized_solver(halfcar, Nt)
    dxdt, state, control = halfcar.cas_ode()
    jit_options = {"flags": ["-Ofast", "-march=native"]}
    options = {'jit':True, 'compiler':'shell', 'jit_options':jit_options}
    ode = ca.Function('ode', [state, control], [dxdt], options)
    jac_x = ca.Function('jac_x', [state, control], [ca.jacobian(dxdt, state)], options)
    jac_u = ca.Function('jax_u', [state, control], [ca.jacobian(dxdt, control)], options)

    qp_solver = QPShoot(halfcar, Nt, ns=6)
    box = np.array([[1,2]])

    X0_list = []
    u0_list = []
    start_list = []
    goal_list = []
    A_np_list = []
    b_np_list = []

    st = time.time()
    print(len(paths))
    for path in paths:
        iter_start = time.time()
        add_plan_time = time.time()
        path = planner.cutToMax(path, dis)
        # path = planner.cutToSafe(path)
        print(f"Additional time: {time.time()-add_plan_time}")
        sim_state = start
        # box = np.array([[6,6]])
        # U =  np.kron(np.ones((1, Nt+1)), [0.0, halfcar.control_bounds[1][1]]).ravel()
        U =  np.kron(np.ones((1, Nt+1)), [0.0, 0.0]).ravel()
        # U =  np.kron(np.ones((1, Nt+1)), [0.0, 2.0]).ravel()
        u0 = U.reshape((-1,2))

        p = np.array(path)[:,0:2]
        decompose_time = time.time()
        A, b = pdc.convex_decomposition_2D(occupied, p, box)
        X0, idx =  planner.discretizePath(p,Nt+1)

        thetas1 = -(np.array(X0)[1:] - np.array(X0)[:-1])
        thetas1 = np.arctan2(thetas1[:,1], thetas1[:,0])
        thetas1 = np.append(np.array(start[2]), thetas1)

        # thetas = np.array(X0[-1]) - np.array(X0[-2])
        # # thetas = np.array(X0[-2]) - np.array(X0[-2])
        # thetas = np.arctan2(thetas[1], thetas[0])
        # # thetas = np.linspace(wrap_to_pi(start[2]), wrap_to_pi(thetas), Nt+1)
        # thetas = linspace_theta_wrap_to_pi(start[2], thetas, Nt+1)

        thetas = (np.array(X0)[1:] - np.array(X0)[:-1])
        thetas = np.arctan2(thetas[:,1], thetas[:,0])
        thetas = np.append(np.array(start[2]), thetas)

        thetas = thetas1+np.pi

        X0 = np.hstack((np.array(X0), thetas.reshape(-1,1)))
        X0_b = np.copy(X0)
        X0_b[:,2] = thetas1
        obs_halfplane = [[A[i],b[i]] for i in idx]
        print(f"Decomposition time: {time.time()-decompose_time}")

        A_np = []
        b_np = []
        for Ab in obs_halfplane:
            A_np.append(np.vstack((Ab[0], np.zeros((6-len(Ab[0]),2)))))
            b_np.append(np.vstack((Ab[1], np.ones((6-len(Ab[1]),1)))))
        A_np = np.array(A_np)        
        b_np = np.array(b_np)        
        X0_list.append(X0.reshape(-1,))
        u0_list.append(u0.reshape(-1)[:-2])
        start_list.append(ca.DM(start.reshape(1,-1)))
        goal_list.append(ca.DM(X0[-1,:].reshape(1,-1)))
        A_np_list.append(ca.DM(A_np.reshape(-1,2)))
        b_np_list.append(ca.DM(b_np.reshape(-1)))
        rX0 = rotate_2dvectors(X0, start[2], start[:2])
        rX0[:,2] = wrap_to_pi(rX0[:,2] - start[2])

        rX0_b = rotate_2dvectors(X0_b, start[2], start[:2])
        rX0_b[:,2] = wrap_to_pi(rX0_b[:,2] - start[2])
        robs_hp = [rotate_polyhedral(A_np[i],b_np[i], start[2], start[:2]) for i in range(len(A_np))]
        # sol = cas_coll(halfcar, start, X0[-1], obs_halfplane, R, Q, QT, Nt, X0, u0)
        # sol = cas_shooting(halfcar, start, X0[-1,:], obs_halfplane, R, Q, QT, Nt, X0, u0)
        # sol = cas_coll_sparse(halfcar, start, X0[-1,:], obs_halfplane, R, Q, QT, Nt, X0, u0)
        # sol = cas_shooting_sparse(halfcar, start, X0[-1,:], obs_halfplane, R, Q, QT, Nt, X0, u0)
        
        # CAS OPTI SOLVER
        # sol = cas_shooting_linearized(halfcar, np.array([0.0,0.0,0.0]), rX0[-1,:], robs_hp, Nt, X0=rX0, u0=u0) 
        # sol = cas_shooting_linearized(halfcar, np.array([0.0,0.0,0.0]), rX0[-1,:], robs_hp, Nt, X0=sol[0], u0=sol[1])
        # sol = cas_shooting_linearized(halfcar, np.array([0.0,0.0,0.0]), rX0[-1,:], robs_hp, Nt, X0=sol[0], u0=sol[1])
        # sol = cas_shooting_linearized(halfcar, np.array([0.0,0.0,0.0]), rX0[-1,:], robs_hp, Nt, X0=sol[0], u0=sol[1])
        # sol = cas_shooting_linearized(halfcar, np.array([0.0,0.0,0.0]), rX0[-1,:], robs_hp, Nt, X0=sol[0], u0=sol[1])

        # CAS QPSOL
        # sol = qp_shooting_linearized(halfcar, ode, jac_x, jac_u, np.array([0.0,0.0,0.0]), rX0[-1,:], robs_hp, Nt, X0=rX0, u0=u0)
        # sol = qp_shooting_linearized(halfcar, ode, jac_x, jac_u, np.array([0.0,0.0,0.0]), rX0[-1,:], robs_hp, Nt, X0=sol[0], u0=sol[1])
        # sol = qp_shooting_linearized(halfcar, ode, jac_x, jac_u, np.array([0.0,0.0,0.0]), rX0[-1,:], robs_hp, Nt, X0=sol[0], u0=sol[1])
        # sol = qp_shooting_linearized(halfcar, ode, jac_x, jac_u, np.array([0.0,0.0,0.0]), rX0[-1,:], robs_hp, Nt, X0=sol[0], u0=sol[1])
        # sol = qp_shooting_linearized(halfcar, ode, jac_x, jac_u, np.array([0.0,0.0,0.0]), rX0[-1,:], robs_hp, Nt, X0=sol[0], u0=sol[1])

        # CAS QPSOL CLASS
        # sol = qp_solver.solve(np.array([0.0,0.0,0.0]), rX0[-1,:], robs_hp, Nt, X0=rX0, u0=u0)
        # sol = qp_solver.solve(np.array([0.0,0.0,0.0]), rX0[-1,:], robs_hp, Nt, X0=rX0, u0=u0)
        # sol = qp_solver.solve(np.array([0.0,0.0,0.0]), rX0[-1,:], robs_hp, Nt, X0=sol[0], u0=sol[1])
        # sol = qp_solver.solve(np.array([0.0,0.0,0.0]), rX0[-1,:], robs_hp, Nt, X0=sol[0], u0=sol[1])
        # sol = qp_solver.solve(np.array([0.0,0.0,0.0]), rX0[-1,:], robs_hp, Nt, X0=sol[0], u0=sol[1])
        # sol = qp_solver.solve(np.array([0.0,0.0,0.0]), rX0[-1,:], robs_hp, Nt, X0=sol[0], u0=sol[1])
        # sol = qp_solver.solve(np.array([0.0,0.0,0.0]), rX0[-1,:], robs_hp, Nt, X0=sol[0], u0=sol[1])
        # sol = qp_solver.solve(np.array([0.0,0.0,0.0]), rX0[-1,:], robs_hp, Nt, X0=sol[0], u0=sol[1])
        # sol = qp_solver.solve(np.array([0.0,0.0,0.0]), rX0[-1,:], robs_hp, Nt, X0=sol[0], u0=sol[1])
        # x_sol, u_sol = sol

        # breakpoint()
        # f_time = time.time()
        # x_sol, u_sol =f_lin(
        #   rX0.reshape(-1,), 
        #   u0.reshape(-1), 
        #   np.array([0.0,0.0,0.0]),
        #   rX0[-1,:].reshape(1,-1), 
        #   np.array([robs_hp[i][0] for i in range(Nt+1)]).reshape(-1,2), 
        #   np.array([robs_hp[i][1] for i in range(Nt+1)]).reshape(-1),
        #   rX0,
        #   u0,)
        # x_sol = np.array(x_sol).reshape((-1,3))
        # u_sol = np.array(u_sol).reshape((-1,2))
        # # print(f"Solve time: {time.time()-f_time}")

        # x_sol = rotate_2dvectors(x_sol, -start[2])
        # x_sol[:,:2] = x_sol[:,:2]+start[:2]

        # f_time = time.time()
        # x_sol, u_sol =f_nonlin(
        #     X0.reshape(-1,), 
        #     u0.reshape(-1)[:], 
        #     start.reshape(1,-1),
        #     X0[-1,:].reshape(1,-1), 
        #     A_np.reshape(-1,2), 
        #     b_np.reshape(-1),
        #     X0.reshape(-1,)) 
        # x_sol = np.array(x_sol).reshape((-1,3))
        # u_sol = np.array(u_sol).reshape((-1,2))

        x_sol, u_sol =f_nonlin(
          rX0.reshape(-1,), 
          u0.reshape(-1), 
          np.array([0.0,0.0,0.0]),
          rX0[-1,:].reshape(1,-1), 
          np.array([robs_hp[i][0] for i in range(Nt+1)]).reshape(-1,2), 
          np.array([robs_hp[i][1] for i in range(Nt+1)]).reshape(-1),
          rX0.reshape(-1,),
          )
        x_sol = np.array(x_sol).reshape((-1,3))
        u_sol = np.array(u_sol).reshape((-1,2))
        
        # breakpoint()

        # '''
        # x_sol, u_sol = sol
        states = []
        for j, u in enumerate(u_sol):
            if j == len(u_sol)-1: 
                break
            for i in range(100):
                sim_state = halfcar.dynamics(sim_state, u, 0, dt=dt/100, params=halfcar.nominal_params)
                # sim_state = halfcar.dynamics(sim_state, u+i/100*(u_sol[j+1]-u), 0, dt=dt/100, params=halfcar.nominal_params)
            states.append(sim_state)
        # ax = pdc.visualize_environment(Al=A, bl=b, p=p, planar=True)
        # plot_simulation_result(x_sol, obs, safe_zones, text="Collocation", max_arrows=5, ax=ax)  
        # plot_simulation_result(states, obs, safe_zones, text="simulated", max_arrows=5, ax = ax)  
        # plot_simulation_result(X0, obs, safe_zones, text="guess", max_arrows=15, ax=ax)  
        # breakpoint()
        # '''
        # print(f"Iter time:{time.time()-iter_start}")
        # breakpoint()
    print(f"NMPC time: {time.time()-st}")
    # breakpoint()
    # mp.set_start_method('spawn')
    pool = mp.Pool(3)
    thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)
    que = mp.Queue()
    find_halfcar_controls = functools.partial(find_Nonlin_Controls, solver=f_nonlin, dis=dis, 
                                        system=halfcar, Nt=Nt, start=start, 
                                        occupied=occupied, box=box,planner=planner)
    # find_halfcar_controls = functools.partial(find_lin_Controls, qp_solver=qp_solver, dis=dis, 
    #                                     system=halfcar, Nt=Nt, start=start, 
    #                                     occupied=occupied, box=box,planner=planner)
    # find_halfcar_controls(paths[0])

    # result = pool.map(find_halfcar_controls, paths[0:2])
    # solve_time = time.time()
    # result = pool.map(find_halfcar_controls, paths)
    # print(f"Pool time: {time.time()-solve_time}")

    # solve_time = time.time()
    # result = thread_pool.map(find_halfcar_controls, paths)
    # solve_time = time.time()
    # result = thread_pool.map(find_halfcar_controls, paths)
    # thread_pool.shutdown(wait=True)
    # print(f"Thread Pool time: {time.time()-solve_time}")

    states = []
    # for res in result:
    #     sim_state=start
    #     x_sol, u_sol = res
    #     for j, u in enumerate(u_sol):
    #         if j == len(u_sol)-1:
    #             break
    #         for i in range(100):
    #             sim_state = halfcar.dynamics(sim_state, u, 0, dt=dt/100, params=halfcar.nominal_params)
    #             # sim_state = halfcar.dynamics(sim_state, u+i/100*(u_sol[j+1]-u), 0, dt=dt/100, params=halfcar.nominal_params)
    #         states.append(sim_state)
    #     ax = pdc.visualize_environment(Al=A, bl=b, p=p, planar=True)
    #     plot_simulation_result(x_sol, obs, safe_zones, text="Collocation", max_arrows=5, ax=ax)  
    #     plot_simulation_result(states, obs, safe_zones, text="simulated", max_arrows=5, ax = ax)  
    #     breakpoint()

    jobs = []
    solve_time = time.time()
    # for path in paths:
    #     p = mp.Process(target=find_halfcar_controls, kwargs={"path":path, "que":que})
    #     jobs.append(p)
    #     p.start()
    # for proc in jobs:
    #     proc.join()
    #     if proc.is_alive():
    #         proc.terminate()
    controls= []
    for path in paths:
        x_sol, u_sol, _ = find_halfcar_controls(path)
        # breakpoint()
        controls.append(u_sol)
        break


    print(f"Process time: {time.time()-solve_time}")
    # for j in range(que.qsize()):
    #     sim_state=start
    #     u_sol = que.get()
    for u_sol in controls:
        for j, u in enumerate(u_sol):
            if j == len(u_sol)-1:
                break
            for i in range(100):
                sim_state = halfcar.dynamics(sim_state, u, 0, dt=dt/100, params=halfcar.nominal_params)
                # sim_state = halfcar.dynamics(sim_state, u+i/100*(u_sol[j+1]-u), 0, dt=dt/100, params=halfcar.nominal_params)
            states.append(sim_state)
        ax = pdc.visualize_environment(Al=A, bl=b, planar=True)
        # plot_simulation_result(x_sol, obs, safe_zones, text="Collocation", max_arrows=5, ax=ax)  
        plot_simulation_result(states, obs, safe_zones, text="simulated", max_arrows=5, ax = ax)  
        breakpoint()

if __name__ == "__main__":
    main()