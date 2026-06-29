# IRS Beamforming Optimization: AO vs DE

Dự án này là mã nguồn thực thi lại (reproduce) các kết quả mô phỏng trong bài báo khoa học: **"Intelligent Reflecting Surface: Practical Phase Shift Model and Beamforming Optimization"** của nhóm tác giả S. Abeywickrama, R. Zhang, và C. Yuen.

Dự án tập trung vào việc tối ưu hóa luồng búp sóng (beamforming) cho hệ thống MISO được hỗ trợ bởi Bề mặt phản xạ thông minh (IRS), có xét đến **mô hình sai pha thực tế (Practical Phase Shift Model)** - nơi biên độ phản xạ bị suy giảm phụ thuộc vào góc dịch pha.

## 🚀 Các tính năng chính (Features)
Repository này cung cấp 2 giải thuật tối ưu hóa để đối chiếu và so sánh:
1. **Thuật toán Alternating Optimization (AO) (`ao_optimization.py`)**:
   - Triển khai thuật toán Tối ưu hóa luân phiên đề xuất trong bài báo.
   - Sử dụng phương pháp **Xấp xỉ bậc hai (Quadratic Estimation - Proposition 1)** để tìm nghiệm dạng đóng cực kỳ nhanh.
   - Hỗ trợ mô phỏng đối chứng 5 đường (Curves) để tái tạo chính xác **Hình 5** và **Hình 6** trong bài báo.
2. **Thuật toán Differential Evolution (DE) (`de_optimization.py`)**:
   - Sử dụng giải thuật metaheuristic **Tiến hóa vi phân (DE)** để quét không gian $N$ chiều.
   - Đóng vai trò là mốc đối chứng (Benchmark) để kiểm tra nghiệm toàn cục (Global Optimum), chứng minh độ chính xác của thuật toán AO.

## 🛠️ Yêu cầu hệ thống (Prerequisites)
Đảm bảo bạn đã cài đặt Python 3.x trên máy. Các thư viện cần thiết bao gồm `numpy`, `scipy` và `matplotlib`.

Cài đặt các thư viện bằng lệnh:
```bash
pip install numpy scipy matplotlib
