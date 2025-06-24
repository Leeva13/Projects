# Функція для обчислення НСД та коефіцієнтів Безу
def extended_gcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        gcd, x, y = extended_gcd(b % a, a)
        return (gcd, y - (b // a) * x, x)

# Функція для знаходження модульного оберненого
def modinv(a, m):
    gcd, x, y = extended_gcd(a, m)
    if gcd != 1:
        raise Exception('Модульне обернене не існує')
    else:
        return x % m

# Функція для швидкого піднесення до степеня за модулем
def power_mod(base, exponent, modulus):
    result = 1
    base = base % modulus
    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % modulus
        exponent = exponent >> 1
        base = (base * base) % modulus
    return result

# Параметри задачі (варіант 2)
p = 43
q = 97
n = p * q
phi_n = (p-1) * (q-1)
e = 5  # Відкрита експонента
d = modinv(e, phi_n)  # Секретна експонента

# Перевірка коректності ключів
assert (e * d) % phi_n == 1

# Повідомлення: "BR" (перші літери прізвища Бражник)
# B=2, R=18 (позиції в алфавіті)
M = 218

# Обчислення хеш-значення
m = (M**3 + 9) % (p-1)

# Створення підпису
S = power_mod(m, d, n)

# Верифікація підпису
m_prime = power_mod(S, e, n)

# Вивід результатів
print(f"Параметри RSA:")
print(f"p = {p}, q = {q}")
print(f"n = p*q = {n}")
print(f"φ(n) = {phi_n}")
print(f"Відкритий ключ (e, n) = ({e}, {n})")
print(f"Секретний ключ (d, n) = ({d}, {n})")
print(f"\nПовідомлення M = 'BR' = {M}")
print(f"Хеш повідомлення m = h(M) = {m}")
print(f"Цифровий підпис S = {S}")
print(f"\nВерифікація:")
print(f"Розшифрований хеш m' = {m_prime}")
print(f"Результат перевірки: {'Підпис вірний' if m == m_prime else 'Підпис невірний'}")