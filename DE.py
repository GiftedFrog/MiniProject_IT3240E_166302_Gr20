import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import time

# --- 1. THAM SỐ MÔ HÌNH PHA THỰC TẾ ---
BETA_MIN = 0.2
K_PARAM = 1.6
PHI_PARAM = 0.43 * np.pi

def get_beta(theta):
    return (1 - BETA_MIN) * ((np.sin(theta - PHI_PARAM) + 1) / 2) ** K_PARAM + BETA_MIN

# ========================================================
# --- 2. THUẬT TOÁN AO (PROPOSITION 1 & 1D SEARCH) -------
# ========================================================
def f_objective_element(theta, Psi_nn, phi_n):
    beta = get_beta(theta)
    return (beta**2) * Psi_nn + beta * np.abs(phi_n) * np.cos(np.angle(phi_n) - theta)

def solve_p2_proposition1(Psi_nn, phi_n):
    arg_phi = np.angle(phi_n)
    theta_C = np.pi if arg_phi >= 0 else -np.pi
    theta_A = arg_phi
    theta_B = (arg_phi + theta_C) / 2.0
    
    f1 = f_objective_element(theta_A, Psi_nn, phi_n)
    f2 = f_objective_element(theta_B, Psi_nn, phi_n)
    f3 = f_objective_element(theta_C, Psi_nn, phi_n)
    
    denominator = 4 * (f1 - 2*f2 + f3)
    candidates = [theta_A, theta_B, theta_C]
    
    if np.abs(denominator) > 1e-12:
        numerator = theta_C * (3*f1 - 4*f2 + f3) + theta_A * (f1 - 4*f2 + 3*f3)
        theta_opt = numerator / denominator
        low, high = min(theta_A, theta_C), max(theta_A, theta_C)
        candidates.append(np.clip(theta_opt, low, high))
        
    best_theta = theta_A
    best_val = f1
    for t in candidates:
        val = f_objective_element(t, Psi_nn, phi_n)
        if val > best_val:
            best_val = val
            best_theta = t
    return best_theta

def optimize_irs_ao(h_d, h_r, G, mode='prop1', max_iters=15, tol=1e-4):
    N = len(h_r)
    theta = np.random.choice([np.pi, -np.pi], size=N).astype(float)
    
    if mode == 'ideal':
        v = np.exp(1j * theta)
    else:
        v = get_beta(theta) * np.exp(1j * theta)
        
    Phi_mat = np.diag(h_r.flatten().conj()) @ G
    Psi = Phi_mat @ Phi_mat.conj().T
    Ghd = Phi_mat @ h_d.flatten()
    
    prev_obj = 0
    for iteration in range(max_iters):
        for n in range(N):
            Psi_nn = float(Psi[n, n].real)
            sum_m = np.dot(Psi[n, :], v) - Psi[n, n] * v[n] 
            phi_n = complex(2 * sum_m + 2 * Ghd[n]) 
            
            if mode == 'ideal':
                theta[n] = np.angle(phi_n)
                v[n] = np.exp(1j * theta[n])
            elif mode == 'prop1':
                theta[n] = solve_p2_proposition1(Psi_nn, phi_n)
                v[n] = get_beta(theta[n]) * np.exp(1j * theta[n])
            elif mode == '1d_search':
                arg_phi = np.angle(phi_n)
                theta_C = np.pi if arg_phi >= 0 else -np.pi
                t_grid = np.linspace(min(arg_phi, theta_C), max(arg_phi, theta_C), 50)
                objs = [f_objective_element(t, Psi_nn, phi_n) for t in t_grid]
                theta[n] = t_grid[np.argmax(objs)]
                v[n] = get_beta(theta[n]) * np.exp(1j * theta[n])
                
        eff_channel = v.conj() @ Phi_mat + h_d.flatten().conj()
        current_obj = float(np.linalg.norm(eff_channel) ** 2)
        if np.abs(current_obj - prev_obj) < tol:
            break
        prev_obj = current_obj
    return theta 

# ========================================================
# --- 3. THUẬT TOÁN TIẾN HÓA VI PHÂN CHUẨN CAO (DE) -----
# ========================================================
def de_objective_function(theta, Phi_mat, h_d_flat):
    v = get_beta(theta) * np.exp(1j * theta)
    eff_channel = v.conj() @ Phi_mat + h_d_flat.conj()
    gain = np.linalg.norm(eff_channel) ** 2
    return -float(gain)

def optimize_irs_de(h_d, h_r, G):
    N = len(h_r)
    Phi_mat = np.diag(h_r.flatten().conj()) @ G
    h_d_flat = h_d.flatten()
    bounds = [(-np.pi, np.pi) for _ in range(N)]
    
    # KHÔNG ÉP GIẢM THÔNG SỐ: Cài đặt cấu hình tìm kiếm toàn cục mạnh nhất
    result = differential_evolution(
        de_objective_function, 
        bounds, 
        args=(Phi_mat, h_d_flat),
        strategy='best1bin', 
        maxiter=200,          # Tăng số thế hệ lặp tiến hóa
        popsize=15,           # Kích thước quần thể chuẩn kích cỡ không gian N chiều
        tol=1e-4,             # Siết chặt điều kiện hội tụ pha
        mutation=(0.5, 1.0), 
        recombination=0.9, 
        polish=True,          # Bật thuật toán mài nhẵn L-BFGS-B ở bước cuối để chạm chóp đỉnh
        workers=-1,           # Kích hoạt tính năng chạy đa nhân CPU để tối ưu thời gian xử lý
        disp=False
    )
    return result.x

# ========================================================
# --- 4. KHỞI TẠO KÊNH TRUYỀN & ĐÁNH GIÁ HỆ THỐNG -------
# ========================================================
def generate_channels(M, N, d_horizontal):
    d_AI, d_AU, d_IU = 500.0, np.sqrt(d_horizontal**2 + 2.0**2), np.sqrt((500.0 - d_horizontal)**2 + 2.0**2)
    PL_0 = 10**(-40/10)
    G = np.sqrt(PL_0 * (d_AI ** -2.2)) * (np.random.randn(N, M) + 1j * np.random.randn(N, M)) / np.sqrt(2)
    h_r = np.sqrt(PL_0 * (d_IU ** -2.8)) * (np.random.randn(N, 1) + 1j * np.random.randn(N, 1)) / np.sqrt(2)
    h_d = np.sqrt(PL_0 * (d_AU ** -3.8)) * (np.random.randn(M, 1) + 1j * np.random.randn(M, 1)) / np.sqrt(2)
    return h_d, h_r, G

def evaluate_6_curves(h_d, h_r, G, P_T, sigma_squared):
    Phi_mat = np.diag(h_r.flatten().conj()) @ G
    hd_flat = h_d.flatten()
    
    # [Curve 5] No IRS
    w_no_irs = np.sqrt(P_T) * hd_flat.conj() / np.linalg.norm(hd_flat)
    rate_5 = np.log2(1 + np.abs(hd_flat @ w_no_irs)**2 / sigma_squared)
    
    # [Curve 1] Upper Bound: Ideal IRS
    theta_ideal = optimize_irs_ao(h_d, h_r, G, mode='ideal')
    v_ideal = np.exp(1j * theta_ideal)
    eff_chan_1 = v_ideal.conj() @ Phi_mat + hd_flat.conj()
    w_ideal = np.sqrt(P_T) * eff_chan_1.conj() / np.linalg.norm(eff_chan_1)
    rate_1 = np.log2(1 + np.abs(eff_chan_1 @ w_ideal)**2 / sigma_squared)
    
    # [Curve 4] Practical IRS with Ideal Assumption
    v_prac_eval = get_beta(theta_ideal) * np.exp(1j * theta_ideal)
    eff_chan_4 = v_prac_eval.conj() @ Phi_mat + hd_flat.conj()
    rate_4 = np.log2(1 + np.abs(eff_chan_4 @ w_ideal)**2 / sigma_squared)
    
    # [Curve 2] Practical IRS (AO with Proposition 1)
    theta_p1 = optimize_irs_ao(h_d, h_r, G, mode='prop1')
    v_p1 = get_beta(theta_p1) * np.exp(1j * theta_p1)
    eff_chan_2 = v_p1.conj() @ Phi_mat + hd_flat.conj()
    rate_2 = np.log2(1 + P_T * np.linalg.norm(eff_chan_2)**2 / sigma_squared)
    
    # [Curve 3] Practical IRS (AO with 1D Search)
    theta_1d = optimize_irs_ao(h_d, h_r, G, mode='1d_search')
    v_1d = get_beta(theta_1d) * np.exp(1j * theta_1d)
    eff_chan_3 = v_1d.conj() @ Phi_mat + hd_flat.conj()
    rate_3 = np.log2(1 + P_T * np.linalg.norm(eff_chan_3)**2 / sigma_squared)

    # [Curve 6] Practical IRS (Differential Evolution - Global Optimum)
    theta_de = optimize_irs_de(h_d, h_r, G)
    v_de = get_beta(theta_de) * np.exp(1j * theta_de)
    eff_chan_de = v_de.conj() @ Phi_mat + hd_flat.conj()
    rate_6 = np.log2(1 + P_T * np.linalg.norm(eff_chan_de)**2 / sigma_squared)
    
    return rate_1, rate_2, rate_3, rate_4, rate_5, rate_6

# ========================================================
# --- 5. ĐIỀU PHỐI MÔ PHỎNG VÀ XUẤT ĐỒ THỊ ---------------
# ========================================================
if __name__ == "__main__":
    M = 2
    P_T = 10 ** ((36 - 30) / 10)
    sigma_squared = 10 ** ((-94 - 30) / 10)
    
    # Do DE chạy full thông số cực kỳ nặng, đặt số mẫu là 15-20 để đảm bảo 
    # thời gian chạy máy cá nhân từ 3-5 phút nhưng vẫn mượt đồ thị.
    num_realizations = 15 
    
    print(f"Bắt đầu chạy mô phỏng liên hợp 6 đường (AO baselines + Full DE)")
    print(f"Số lượng mẫu Monte Carlo: {num_realizations}\n")
    
    # --- MÔ PHỎNG HÌNH 5 (Thay đổi d, N=40) ---
    print("--- Đang xử lý kịch bản Hình 5 (Khảo sát theo khoảng cách d) ---")
    distances = [480, 485, 490, 495, 500]
    res_fig5 = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
    for d in distances:
        start = time.time()
        r1, r2, r3, r4, r5, r6 = [], [], [], [], [], []
        for _ in range(num_realizations):
            h_d, h_r, G = generate_channels(M, 40, d)
            rates = evaluate_6_curves(h_d, h_r, G, P_T, sigma_squared)
            r1.append(rates[0]); r2.append(rates[1]); r3.append(rates[2])
            r4.append(rates[3]); r5.append(rates[4]); r6.append(rates[5])
        for i, r_list in enumerate([r1, r2, r3, r4, r5, r6], 1):
            res_fig5[i].append(np.mean(r_list))
        print(f"  Hoàn thành mốc d = {d}m | Thời gian chặng: {time.time() - start:.2f}s")

    # --- MÔ PHỎNG HÌNH 6 (Thay đổi N, d=498m) ---
    print("\n--- Đang xử lý kịch bản Hình 6 (Khảo sát số phần tử N từ 10 - 50) ---")
    elements = [10, 20, 30, 40, 50]
    res_fig6 = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
    for N in elements:
        start = time.time()
        r1, r2, r3, r4, r5, r6 = [], [], [], [], [], []
        for _ in range(num_realizations):
            h_d, h_r, G = generate_channels(M, N, 498.0)
            rates = evaluate_6_curves(h_d, h_r, G, P_T, sigma_squared)
            r1.append(rates[0]); r2.append(rates[1]); r3.append(rates[2])
            r4.append(rates[3]); r5.append(rates[4]); r6.append(rates[5])
        for i, r_list in enumerate([r1, r2, r3, r4, r5, r6], 1):
            res_fig6[i].append(np.mean(r_list))
        print(f"  Hoàn thành mốc N = {N} | Thời gian chặng: {time.time() - start:.2f}s")

    # --- TIẾN HÀNH VẼ ĐỒ THỊ ĐỐI CHIẾU ---
    plt.figure(figsize=(15, 6))
    
    def draw_plots(ax, x_data, data_dict, title, xlabel):
        ax.plot(x_data, data_dict[1], 'r--', label='1) Upper Bound: Ideal IRS')
        ax.plot(x_data, data_dict[6], marker='D', color='magenta', linestyle='-', linewidth=2.5, markersize=6, label='6) Practical IRS: Global Optimum (DE)')
        ax.plot(x_data, data_dict[2], 'g-', linewidth=2, label='2) Practical IRS: AO with Proposition 1')
        ax.plot(x_data, data_dict[3], 'b--', dashes=(5, 5), label='3) Practical IRS: AO with 1D Search')
        ax.plot(x_data, data_dict[4], color='orange', linestyle=':', linewidth=2, label='4) Practical IRS with Ideal Assumption')
        ax.plot(x_data, data_dict[5], 'k--*', markersize=8, label='5) Lower Bound: No IRS')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Achievable rate (bits/s/Hz)')
        ax.grid(True, linestyle='--')
        ax.legend(fontsize=8.5, loc='upper left')

    ax1 = plt.subplot(1, 2, 1)
    draw_plots(ax1, distances, res_fig5, 'Fig 5: Rate vs AP-user horizontal distance (N=40)', 'Distance d (m)')
    
    ax2 = plt.subplot(1, 2, 2)
    draw_plots(ax2, elements, res_fig6, 'Fig 6: Rate vs Number of elements (d=498m)', 'Number of reflecting elements: N')
    
    plt.tight_layout()
    plt.show()