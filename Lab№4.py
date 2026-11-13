import math
from scipy.optimize import minimize

m1, m2, m3 = 20, 40, 60
s1, s2, s3 = 10, 18, 30
r12, r13, r23 = 1, -1, -1

def doxid(x):
    return x[0]*m1 + x[1]*m2 + x[2]*m3

def ryzyk(x):
    d = (x[0]**2)*(s1**2) + (x[1]**2)*(s2**2) + (x[2]**2)*(s3**2) + \
        2*x[0]*x[1]*r12*s1*s2 + 2*x[0]*x[2]*r13*s1*s3 + 2*x[1]*x[2]*r23*s2*s3
    if d < 0:
        d = 0
    return math.sqrt(d)

umovy1 = ({'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] - 1})
meji = ((0, 1), (0, 1), (0, 1))
poch = [0.33, 0.33, 0.34]

res_min = minimize(ryzyk, poch, method='SLSQP', bounds=meji, constraints=umovy1)
min_s = res_min.fun

umovy2 = (
    {'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] - 1},
    {'type': 'ineq', 'fun': lambda x: min_s + 0.00001 - ryzyk(x)}
)

res = minimize(lambda x: -doxid(x), poch, method='SLSQP', bounds=meji, constraints=umovy2)

x1 = round(res.x[0], 3)
x2 = round(res.x[1], 3)
x3 = round(res.x[2], 3)

m_p = round(doxid([x1, x2, x3]), 2)
s_p = round(ryzyk([x1, x2, x3]), 2)

print("==========================================")
print("         ЗАВДАННЯ 9. РОЗВ'ЯЗАННЯ          ")
print("==========================================")
print("Вихідні дані:")
print(f"• Сподівані прибутки: m1 = {m1}%, m2 = {m2}%, m3 = {m3}%")
print(f"• Оцінки ризику:      s1 = {s1}%, s2 = {s2}%, s3 = {s3}%")
print(f"• Кореляція:          r12 = {r12}, r13 = {r13}, r23 = {r23}")
print("-" * 42)

print("а) Структура оптимального ПЦП:")
print(f"   Частка акцій A1 (x1*) = {x1:.3f} ({x1*100:.1f}%)")
print(f"   Частка акцій A2 (x2*) = {x2:.3f} ({x2*100:.1f}%)")
print(f"   Частка акцій A3 (x3*) = {x3:.3f} ({x3*100:.1f}%)")
print(f"   Вектор структури X*  = ({x1:.3f}; {x2:.3f}; {x3:.3f})")
print("-" * 42)

print("б) Характеристики отриманого ПЦП:")
print(f"   m_П* = {x1}*{m1} + {x2}*{m2} + {x3}*{m3} = {m_p}%")
print(f"   sigma_П* = {s_p}%")
print("==========================================")