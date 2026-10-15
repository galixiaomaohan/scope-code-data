# -*- coding: utf-8 -*-
"""V31 exact number-theory primitives (stdlib only; all-integer/rational exact channel).

Discipline (copied from svokernel):
  - every returned value travels as an exact integer/Fraction; float never impersonates the
    exact channel (float appears only in residual reports);
  - primality testing and factorization are **two independent implementations**, used on
    separate paths by generator and verifier:
      * is_prime        -- deterministic Miller-Rabin (base table fixed, see below);
      * is_prime_trial  -- trial-division proof (the verifier-side independent path);
    so that a certificate check "factor product + per-prime primality" does not depend on the
    generator's primality code;
  - randomness always comes from a deterministic LCG (seed fixed), independent of
    PYTHONHASHSEED / wall clock (§17.6 reproducibility: two runs with the same seed must agree
    bit for bit);
  - periodic ctl.check_time on long loops is the operator's business; this module only
    provides pure functions.

Honesty boundary: Pollard-rho is a **probabilistic** factorization, but every factorization
from this module is independently re-checked on the verifier side (product reconciliation +
per-prime primality proof along a different path), so "the factorization is correct" never
rests on rho's probabilistic claim.
"""
from __future__ import annotations

import math
from fractions import Fraction

__all__ = [
    "gcd", "lcm", "modinv", "powmod", "is_prime", "is_prime_trial", "sieve_primes",
    "factorize", "valuation", "valuation_squaring", "divisors_from_factors",
    "divisors_brute", "phi_from_factors", "mu_from_factors", "tau_from_factors",
    "sigma_from_factors", "jacobi", "legendre_euler", "sqrt_mod_prime", "is_squarefree",
    "is_perfect_square", "rational_sqrt", "log_bracket", "LCG",
]

#: Deterministic Miller-Rabin base table: the first 12 primes suffice to decide n < 3.317e24
#: (Sorenson-Webster). Beyond the bound the code **raises instead of taking a position** --
#: no soft "probably prime" answer.
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
MR_BOUND = 3317044064679887385961981  # ~3.317e24: valid upper bound of the base table (exclusive)


def gcd(a: int, b: int) -> int:
    return math.gcd(a, b)


def lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a // math.gcd(a, b) * b)


def _egcd(a: int, b: int):
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return old_r, old_s, old_t


def modinv(a: int, m: int):
    """Modular inverse via extended Euclid (does not rely on pow(a,-1,m), so it is
    version-independent). Returns None when not invertible."""
    if m <= 0:
        return None
    a %= m
    g, x, _ = _egcd(a, m)
    if g != 1:
        return None
    return x % m


def powmod(a: int, e: int, m: int) -> int:
    if m <= 0:
        raise ValueError("modulus must be positive")
    if e < 0:
        inv = modinv(a, m)
        if inv is None:
            raise ValueError("negative exponent requires invertible base")
        return powmod(inv, -e, m)
    return pow(a, e, m)


def is_perfect_square(n: int) -> bool:
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n


def rational_sqrt(f: Fraction):
    """Exact square root of a rational; returns None if it is not a perfect square."""
    if f < 0:
        return None
    n, d = f.numerator, f.denominator
    rn, rd = math.isqrt(n), math.isqrt(d)
    if rn * rn == n and rd * rd == d:
        return Fraction(rn, rd)
    return None


def is_prime(n: int) -> bool:
    """Deterministic primality test (base table _MR_BASES); above n >= 3.3e24 it raises
    ValueError for the caller to fold.

    Honesty note: inside the base-table range the answer is **deterministic** (no probability
    claim); beyond the bound it refuses to answer (raises) rather than returning a "very
    likely". The caller decides which §15 error code the out-of-range path folds into.
    """
    if n < 0:
        raise ValueError("is_prime: 负数")
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    if n >= MR_BOUND:
        raise ValueError("is_prime: n 超出确定性基表界 3.317e24 (n=%d)" % n)
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def is_prime_trial(n: int, limit=None):
    """Primality proof by trial division (the verifier-side independent path). Returns
    True/False; returns None when the limit is exhausted without a proof.

    limit=None uses isqrt(n) (complete proof). None means "not proven" -- it never impersonates
    True (honesty discipline). limit must be a complete bound of the isqrt(n) order, otherwise
    None is returned.
    """
    if n < 0:
        return False
    if n < 2:
        return False
    if limit is None:
        limit = math.isqrt(n) + 1
    if limit * limit < n:
        return None
    if n % 2 == 0:
        return n == 2
    if limit < 3:
        return True      # n ∈ {2,3} handled above (2 via the even branch); here n=3 => prime
    d = 3
    while d <= limit:
        if n % d == 0:
            return False
        d += 2
    return True


def sieve_primes(n: int) -> list:
    """Sieve of Eratosthenes: all primes <= n."""
    if n < 2:
        return []
    bs = bytearray([1]) * (n + 1)
    bs[0] = 0
    bs[1] = 0
    for i in range(2, math.isqrt(n) + 1):
        if bs[i]:
            bs[i * i:: i] = bytearray(len(bs[i * i:: i]))
    return [i for i in range(n + 1) if bs[i]]


class LCG:
    """Deterministic linear congruential generator (Numerical Recipes constants). The seed is
    fixed => reproducibility is a hard guarantee."""

    __slots__ = ("s",)

    def __init__(self, seed: int) -> None:
        self.s = (seed ^ 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF

    def next(self) -> int:
        self.s = (6364136223846793005 * self.s + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return self.s >> 11

    def randbelow(self, m: int) -> int:
        return self.next() % m if m > 0 else 0


def _pollard_brent(n: int, seed: int) -> int:
    """Brent-style Pollard-rho: returns a nontrivial factor of n (returns n on failure).
    Deterministic seed."""
    if n % 2 == 0:
        return 2
    rng = LCG(seed)
    for _attempt in range(8):
        y = rng.randbelow(n - 2) + 1
        c = rng.randbelow(n - 1) + 1
        m = 128
        g = 1
        r = 1
        q = 1
        x = ys = y
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
            if r > (1 << 18):
                break
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
        if 1 < g < n:
            return g
    return n


def _trial_factor(n: int) -> int:
    d = 41
    while d * d <= n:
        if n % d == 0:
            return d
        d += 2
    return n


def _factor_rec(n: int, out: list, seed: int) -> None:
    if n == 1:
        return
    if is_prime(n):
        for t in out:
            if t[0] == n:
                t[1] += 1
                return
        out.append([n, 1])
        return
    d = _pollard_brent(n, seed)
    if d == n:
        # rho found no factor: degrade to trial division (should not happen in theory, but
        # never do it silently)
        d = _trial_factor(n)
        if d == n:
            out.append([n, 1])
            return
    _factor_rec(d, out, seed + 1)
    _factor_rec(n // d, out, seed + 1)


def factorize(n: int, trial_limit: int = 100000, seed: int = 20260920) -> list:
    """Complete factorization of |n| -> [[p, e], ...] (p increasing). |n| <= 1 -> [] (empty
    factorization).

    Path: trial division up to trial_limit, then Pollard-rho recursion on the remaining
    cofactor > 1. The result **must** be independently re-checked by the caller/verifier
    (product + per-prime); this function does not claim it is proven.
    """
    n = abs(n)
    if n <= 1:
        return []
    out: list = []
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            out.append([p, e])
    d = 41
    while d <= trial_limit and d * d <= n:
        if n % d == 0:
            e = 0
            while n % d == 0:
                n //= d
                e += 1
            out.append([d, e])
        d += 2
    if n > 1:
        _factor_rec(n, out, seed)
    out.sort(key=lambda t: t[0])
    return out


def valuation(p: int, n: int) -> int:
    """v_p(n): the exponent of p in n (n != 0; p >= 2)."""
    if n == 0:
        raise ValueError("v_p(0) = +inf 不可表示")
    if p < 2:
        raise ValueError("p 必须 >= 2")
    n = abs(n)
    e = 0
    while n % p == 0:
        n //= p
        e += 1
    return e


def valuation_squaring(p: int, n: int) -> int:
    """v_p(n) along an independent path: bisect on p^2, p^4, ... first, then refine step by
    step (a different algorithm from the naive loop)."""
    if n == 0:
        raise ValueError("v_p(0) = +inf")
    n = abs(n)
    if n % p != 0:
        return 0
    k = 0
    pk = p
    while k < 200 and n % (pk * pk) == 0:
        pk = pk * pk
        k += 1
    lo, hi = 1 << k, 1 << (k + 1)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if n % pow(p, mid) == 0:
            lo = mid
        else:
            hi = mid
    return lo


def divisors_from_factors(factors) -> list:
    """All positive divisors from the factorization table (increasing)."""
    ds = [1]
    for p, e in factors:
        cur = []
        pe = 1
        for _ in range(e + 1):
            for d in ds:
                cur.append(d * pe)
            pe *= p
        ds = cur
    return sorted(ds)


def divisors_brute(n: int) -> list:
    """All divisors via trial division (independent of the factorization table; for the
    verifier / small exhaustive runs)."""
    n = abs(n)
    if n == 0:
        return []
    small, large = [], []
    d = 1
    while d * d <= n:
        if n % d == 0:
            small.append(d)
            if d != n // d:
                large.append(n // d)
        d += 1
    return small + large[::-1]


def phi_from_factors(factors) -> int:
    out = 1
    for p, e in factors:
        out *= (p - 1) * p ** (e - 1)
    return out


def mu_from_factors(factors) -> int:
    for _, e in factors:
        if e >= 2:
            return 0
    return -1 if len(factors) % 2 else 1


def tau_from_factors(factors) -> int:
    out = 1
    for _, e in factors:
        out *= e + 1
    return out


def sigma_from_factors(factors, k: int = 1) -> int:
    """σ_k(n) = Σ_{d|n} d^k (exact integer). k=1 is σ(n)."""
    out = 1
    for p, e in factors:
        s = 0
        pe = 1
        for _ in range(e + 1):
            s += pe ** k
            pe *= p
        out *= s
    return out


def jacobi(a: int, n: int) -> int:
    """Jacobi symbol (a/n), n a positive odd number. Binary reciprocity algorithm (a
    different path from the Euler criterion)."""
    if n <= 0 or n % 2 == 0:
        raise ValueError("Jacobi 符号要求 n 为正奇数")
    a %= n
    result = 1
    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n
    return result if n == 1 else 0


def legendre_euler(a: int, p: int) -> int:
    """(a/p) via the Euler criterion, p an odd prime: a^((p-1)/2) mod p ∈ {0,1,p-1} ->
    {0,1,-1}."""
    a %= p
    if a == 0:
        return 0
    r = pow(a, (p - 1) // 2, p)
    return 1 if r == 1 else (-1 if r == p - 1 else 0)


def sqrt_mod_prime(a: int, p: int):
    """Tonelli-Shanks: one solution of x² ≡ a (mod p); None if there is none (p an odd prime,
    a arbitrary)."""
    a %= p
    if a == 0:
        return 0
    if p == 2:
        return a
    if legendre_euler(a, p) != 1:
        return None
    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while legendre_euler(z, p) != -1:
        z += 1
    m, c, t, r = s, pow(z, q, p), pow(a, q, p), pow(a, (q + 1) // 2, p)
    while t != 1:
        i, t2 = 0, t
        while t2 != 1:
            t2 = t2 * t2 % p
            i += 1
        b = pow(c, 1 << (m - i - 1), p)
        m, c = i, b * b % p
        t = t * c % p
        r = r * b % p
    return r


def is_squarefree(n: int) -> bool:
    """Square-free test (n >= 1): trial division to the cube root + a perfect-square check on
    the remainder (does not rely on factorize)."""
    if n <= 0:
        return False
    if n == 1:
        return True
    d = 2
    m = n
    while d * d * d <= m:
        if m % d == 0:
            m //= d
            if m % d == 0:
                return False
        d += 1 if d == 2 else 2
    return not is_perfect_square(m)


def log_bracket(p: int, terms: int = 40) -> tuple:
    """**Strict rational enclosure** of log(p): (lo, hi) ⊂ Q.

    Why not raw atanh: t = (p−1)/(p+1) -> 1 for large p, so the series converges extremely
    slowly; with 30 terms the enclosure can be 0.8 wide -- a useless "strict enclosure"
    (a trap we actually hit). Binary reduction instead: take j = p.bit_length()−1 and
    m = p/2^j ∈ [1,2) =>
        log p = j·log 2 + log m,   log m = 2·atanh((m−1)/(m+1)),  t' = (p−2^j)/(p+2^j) ≤ 1/3
    t' ≤ 1/3 => the tail shrinks exponentially with the number of terms (~1e-15 at 30 terms).
    Tail bound (standard geometric comparison): |tail| <= 2·t^(K+1)/((K+1)(1−t²)), K = terms.
    Returns [lo, hi] strictly containing log p -- strict in the interval-arithmetic sense,
    not a numerical estimate.
    """
    if p <= 0:
        raise ValueError("log_bracket 仅对 p >= 1")
    if p == 1:
        return Fraction(0), Fraction(0)
    if terms < 1:
        raise ValueError("terms >= 1")
    j = p.bit_length() - 1
    q = 1 << j                      # 2^j <= p < 2^(j+1)
    if q == p:
        # p = 2^j: log p = j·log 2, only the enclosure of log 2 is needed
        lo2, hi2 = _atanh_bracket(Fraction(1, 3), terms)
        return j * lo2, j * hi2
    t = Fraction(p - q, p + q)      # ∈ [0, 1/3)
    lom, him = _atanh_bracket(t, terms)
    lo2, hi2 = _atanh_bracket(Fraction(1, 3), terms)
    return j * lo2 + lom, j * hi2 + him


def _atanh_bracket(t: Fraction, terms: int) -> tuple:
    """Strict enclosure of 2·atanh(t) = 2·Σ_{k odd} t^k/k (t ∈ [0,1))."""
    acc = Fraction(0)
    for k in range(1, terms + 1, 2):
        acc += t ** k / k
    lo = 2 * acc
    tail = 2 * t ** (terms + 1) / ((terms + 1) * (1 - t * t))
    return lo, lo + tail


def log_bracket_fixed(p: int, terms: int, den: int) -> tuple:
    """**Fixed-point** strict interval for log(p) (integer pair): returns (lo_i, hi_i) with
    lo_i/den <= log p <= hi_i/den (outward rounding, so inclusion survives the rounding).

    Why fixed point (performance and strictness both win): when accumulating prime powers
    with exact rationals the denominator snowballs (at x=1e6, 78734 log terms to add -> the
    denominator explodes; measured: 18 minutes without convergence). A fixed denominator with
    outward rounding reduces accumulation to integer addition (O(1) per term), and inclusion
    is guaranteed by floor/ceil.

    The returned integers can be large (den·log p ≈ 6.9e17·log p), but they are plain
    big-integer operations; Python integer addition costs the same.
    """
    if den < 1:
        raise ValueError("den >= 1")
    lo, hi = log_bracket(p, terms)
    li = (lo * den).__floor__()
    hi_i = -((-hi * den).__floor__())          # ceil(x) = -floor(-x)
    return int(li), int(hi_i)


def _lcm_odd_upto(K: int) -> int:
    """lcm(1,3,5,...,K) (K odd) -- used to put Σ 1/k over a common denominator."""
    out = 1
    for k in range(1, K + 1, 2):
        out = out // math.gcd(out, k) * k
    return out


def _atanh_int(num: int, den_: int, terms: int, den: int) -> tuple:
    """Fixed-point strict interval of 2·atanh(num/den_) (integer arithmetic).

    2·atanh(t) = 2·Σ_{k odd} t^k/k, t = num/den_ ∈ (0, 1).
    Common denominator: let L = lcm(1,3,...,K) (K = terms, odd), then
        Σ_{k odd} t^k/k = [Σ_k 2·(L/k)·num^k·den_^{K−k}] / (L·den_^K)
    Numerator and denominator are all integers => one integer division with floor/ceil gives
    the fixed-point endpoints (no Fraction, no gcd cascade).
    Tail bound (standard geometric comparison): |tail| <= 2·t^(K+1)/((K+1)(1−t²)), integerized
    and rounded **up** (the tail bound must not be smaller than the true tail), added to the
    upper endpoint -- this guarantees strict inclusion.
    """
    if terms % 2 == 0:
        terms -= 1
    if terms < 1:
        terms = 1
    K = terms
    L = _lcm_odd_upto(K)
    S = 0
    for k in range(1, K + 1, 2):
        S += 2 * (L // k) * (num ** k) * (den_ ** (K - k))
    # value = S / (L·den_^K) => fixed-point lower bound = floor(S·den / (L·den_^K))
    denom = L * (den_ ** K)
    lo_i = (S * den) // denom
    # tail bound (rounded up): 2·num^(K+1)·den_²·den / ((K+1)·den_^(K+1)·(den_²−num²))
    tn = 2 * (num ** (K + 1)) * (den_ * den_) * den
    td = (K + 1) * (den_ ** (K + 1)) * (den_ * den_ - num * num)
    if td <= 0:
        raise ValueError("atanh 尾界参数域外")
    tail_i = -((-tn) // td)                    # ceil(tn/td)
    # The +2 is insurance for the floor/ceil interaction: lo_i is floor(S·den/denom) while the
    # true value is at most (S·den/denom)+1 (the fractional part), so the upper endpoint must be
    # lo_i + ceil(tail) + 2 to strictly cover the true value. The lower endpoint floor is already
    # <= the true value and needs no correction.
    return int(lo_i), int(lo_i + tail_i + 2)


def log_bracket_int(p: int, terms: int, den: int) -> tuple:
    """**All-integer** fixed-point strict interval for log(p) (fast path, mathematically
    equivalent to log_bracket_fixed).

    The binary reduction matches log_bracket: log p = j·log 2 + log m, m = p/2^j ∈ [1,2),
    t' = (p−2^j)/(p+2^j) ≤ 1/3 => the series converges quickly.
    """
    if den < 1:
        raise ValueError("den >= 1")
    if p < 1:
        raise ValueError("log 仅对 p >= 1")
    if p == 1:
        return 0, 0
    j = p.bit_length() - 1
    q = 1 << j
    l2lo, l2hi = _atanh_int(1, 3, terms, den)       # log 2 = 2·atanh(1/3)
    lo_i = j * l2lo
    hi_i = j * l2hi
    if q != p:
        a, b = _atanh_int(p - q, p + q, terms, den)  # log(p/2^j)
        lo_i += a
        hi_i += b
    return int(lo_i), int(hi_i)
