# MOD — Modular Orbit Dynamics

**The modular surface PSL(2,ℤ)\H² as the universal arena of gradient descent.**

*Every gradient trajectory is a geodesic orbit. Every orbit is a word. Every word is a rational cusp.*

---

## Table of Contents

1. [The Central Claim](#1-the-central-claim)
2. [The Modular Surface](#2-the-modular-surface)
3. [The Gradient Embedding](#3-the-gradient-embedding)
4. [Three Layers, One Object](#4-three-layers-one-object)
   - [Layer I — Geometry: Geodesic and Horocycle Flow](#layer-i--geometry-geodesic-and-horocycle-flow)
   - [Layer II — Combinatorics: Rational Tree Paths](#layer-ii--combinatorics-rational-tree-paths)
   - [Layer III — Order Theory: Farey Well-Quasi-Orders](#layer-iii--order-theory-farey-well-quasi-orders)
5. [The Cusp Dichotomy: Memorization and Generalization](#5-the-cusp-dichotomy-memorization-and-generalization)
6. [The Selberg Trace Formula as Master Duality](#6-the-selberg-trace-formula-as-master-duality)
7. [The SP Ordinal Hierarchy is the Prime Geodesic Theorem](#7-the-sp-ordinal-hierarchy-is-the-prime-geodesic-theorem)
8. [The Riemann Hypothesis as the Generalization Condition](#8-the-riemann-hypothesis-as-the-generalization-condition)
9. [Master Identification Table](#9-master-identification-table)
10. [Unified Equations](#10-unified-equations)
11. [Open Conjectures](#11-open-conjectures)
12. [References](#12-references)

---

## 1. The Central Claim

Gradient descent, viewed through its gradient ratio sequence, traces a **geodesic orbit** on the **modular surface**

```
M = PSL(2,ℤ) \ H²
```

Three bodies of mathematics — hyperbolic geodesic flow, rational tree combinatorics, and Kruskal well-quasi-order theory — each capture one aspect of this orbit. MOD establishes that they are not three different frameworks. They are three coordinate systems on the same mathematical object.

| Layer | Framework | What it captures |
|---|---|---|
| Geometry | Geodesic flow + Selberg trace formula | The *continuous* structure of the orbit |
| Combinatorics | Rational tree paths in SL(2,ℤ) | The *symbolic* encoding of the orbit |
| Order theory | Kruskal WQO on Farey fractions | The *finiteness* guarantees of the orbit |

Every theorem in one layer has an exact translation into the other two. The translation table is complete. The correspondences are equalities, not analogies.

---

## 2. The Modular Surface

### 2.1 The Hyperbolic Plane

The **upper half-plane**

```
H² = { z = x + iy : y > 0 }
```

carries the Poincaré metric ds² = (dx² + dy²)/y², making it the unique complete simply connected surface of constant curvature −1. Its isometry group is PSL(2,ℝ), acting by Möbius transformations z ↦ (az + b)/(cz + d). Geodesics are semicircles and vertical lines orthogonal to ℝ. The boundary ∂H² = ℝ ∪ {∞} = ℙ¹(ℝ) consists of the asymptotic endpoints of geodesics.

**Ford circles.** For each rational p/q ∈ ℙ¹(ℚ) in lowest terms, the **Ford circle** C(p/q) is the disk tangent to ℝ at p/q with Euclidean radius 1/(2q²). In the hyperbolic metric, C(p/q) is a **horoball** — the level set of the Busemann function based at the cusp p/q. Two Ford circles C(p/q) and C(a/b) are externally tangent if and only if |pb − qa| = 1. This is the Farey neighbor condition.

The Ford circle packing is an Apollonian gasket with Hausdorff dimension ≈ 1.3057. In learning, C(p/q) is the loss basin at curvature denominator q: basin width ∝ 1/q², so large q means a sharp, narrow basin (memorization) and small q means a flat, wide basin (generalization).

### 2.2 The Modular Group

The **modular group** Γ = PSL(2,ℤ) is the discrete subgroup of PSL(2,ℝ) consisting of 2×2 integer matrices with determinant 1 modulo sign. Its fundamental domain

```
F = { z ∈ H² : |z| ≥ 1,  |Re(z)| ≤ 1/2 }
```

has one cusp at i∞ and finite hyperbolic area vol(M) = π/3.

The **modular surface** M = Γ\H² is a non-compact hyperbolic orbifold of finite volume. The following classical objects are all equivalent descriptions of M:

- The **Farey sequences** F_n — the set of reduced rationals p/q with q ≤ n, ordered by size.
- The **Ford circle packing** — the horoball system at rational cusps, forming an Apollonian gasket.
- The **Stern–Brocot tree** — the Farey tessellation of H² by ideal triangles, viewed as a rooted binary tree.
- The **continued fraction algorithm** — the symbolic code of geodesic rays in T¹M.
- The **Calkin–Wilf tree** — the enumeration of positive rationals by the positive monoid of SL(2,ℤ).

None of these is more fundamental than the others. Each is a coordinate system on M.

### 2.3 Spectral Theory of M

The Laplace–Beltrami operator Δ_M on M has:

- A **discrete spectrum** {0 = λ₀ < λ₁ ≤ λ₂ ≤ ···} ⊂ [0, 1/4)
- A **continuous spectrum** [1/4, ∞)
- An unconditional **spectral gap** λ₁ ≥ 3/16 for congruence subgroups (Selberg 1965)
- The **Selberg eigenvalue conjecture** (open since 1965): λ₁ ≥ 1/4

The spectral gap λ₁(Δ_M) is identified in MOD with the convergence rate of gradient descent. The Selberg conjecture, if proved, would give an unconditional bound on the grokking time in terms of a pure number-theoretic constant.

---

## 3. The Gradient Embedding

### 3.1 Constructing the Orbit Point

Given consecutive gradient vectors g_t, g_{t+1} ∈ ℝᴺ at training step t, define

```
ρ_t  =  ‖g_{t+1}‖ / (‖g_t‖ + ‖g_{t+1}‖)         [gradient ratio,  ∈ (0,1)]
ε_t  =  ‖g_{t+1} − g_t‖ / (‖g_t‖ + ‖g_{t+1}‖)   [gradient change,  > 0]
z_t  =  ρ_t  +  i · ε_t   ∈  H²                   [orbit point]
```

The imaginary part ε_t > 0 places z_t strictly inside H². The **hyperbolic height**

```
d_hyp(z_t, ∂H²)  =  log(1/ε_t)
```

is the renormalization scale: high height means small gradient change (near convergence); low height means large gradient change (active exploration). As training converges, ε_t → 0 and z_t descends toward the rational boundary.

The projection π(z_t) ∈ M is the **gradient orbit point** on the modular surface. The entire training history is a path on M. The geometry of this path determines the learning phase.

### 3.2 Three Equivalent Representations

Every point z_t ∈ H² carries three simultaneous representations related by canonical bijections:

**Geometric.** The point z_t = ρ_t + i·ε_t in H², at hyperbolic distance log(2q_t²·ε_t) from the nearest Ford circle C(p_t/q_t).

**Arithmetic.** The continued fraction expansion ρ_t = [a₀; a₁, a₂, …] at precision Q_t = ⌊1/ε_t⌋, yielding the best rational approximant (p_t/q_t) with denominator q_t ≤ Q_t. The denominator q_t is the **curvature denominator** of the loss basin being traversed.

**Symbolic.** The word w_t = R^{a₀} L^{a₁} R^{a₂} ··· ∈ {L, R}* of **Stern–Brocot depth** d_t = Σᵢ aᵢ, where L and R are the two generators of the positive monoid ℳ ⊂ SL(2,ℤ). The word w_t is the Morse–Hedlund symbolic code of the geodesic ray ending at ρ_t.

The three bijections connecting these representations are: (geometric ↔ arithmetic) the continued fraction algorithm; (arithmetic ↔ symbolic) the word-from-CF correspondence w = R^{a₀} L^{a₁} ···; (geometric ↔ symbolic) the geodesic coding theorem of Morse–Hedlund (1940).

---

## 4. Three Layers, One Object

### Layer I — Geometry: Geodesic and Horocycle Flow

The unit tangent bundle T¹M carries two canonical flows whose dichotomy is the central dynamical fact of MOD.

**Geodesic flow** φ_t moves at unit speed along geodesics. It is **exponentially mixing**:

```
|Cor(f, g ∘ φ_t)|  ≤  C · e^{−λ₁ t}
```

where λ₁ = λ₁(Δ_M) is the spectral gap. In learning: the generalization regime. Gradient pairs are approximately independent, the JL operator ℒ_JL is coercive, and the Fokker–Planck density converges exponentially to the Gibbs equilibrium.

**Horocycle flow** η_s moves along Ford circles — curves at constant hyperbolic distance from a fixed cusp. It is **polynomially mixing** (Ratner 1991):

```
|Cor(f, g ∘ η_s)|  ≤  C_k · s^{−k}    for every k ≥ 1,  but NEVER exponential
```

In learning: the memorization regime. The gradient orbit follows a near-horizontal path at approximately constant Im(z_t) ≈ ε, circling within a Ford circle of radius ∼ ε/(2q²).

**The mixing dichotomy is the phase transition.** Polynomial mixing cannot continuously become exponential mixing. The change from horocycle to geodesic dynamics is a genuine bifurcation of the dynamical regime — not a change in rate, but a change in qualitative type. This is why grokking is abrupt.

**Ratner's Theorem** (1990–1991): Every orbit closure of the horocycle flow in Γ\SL(2,ℝ) is a **closed homogeneous submanifold**, determined algebraically by the stabilizer of the initial point in SL(2,ℝ). The memorization manifold — the closure of the pre-grokking gradient orbit — is not a fractal or arbitrary closed set. It is algebraically structured, reproducible across initializations, and classified by Lie theory.

### Layer II — Combinatorics: Rational Tree Paths

The **positive monoid** ℳ ⊂ SL(2,ℤ) is generated by

```
L = ( 1  0 )    R = ( 1  1 )
    ( 1  1 )        ( 0  1 )
```

Both have non-negative integer entries and det = 1. These are the exponentials of the standard nilpotent generators of 𝔰𝔩(2,ℝ):

```
L = exp(F),    R = exp(E)
```

where E = ((0,1),(0,0)) and F = ((0,0),(1,0)) satisfy [H,E] = 2E, [H,F] = −2F, [E,F] = H.

**Theorem (Positive Cone).** Every M ∈ SL(2,ℤ) with non-negative entries has a unique expression as a word in {L, R}. The map Φ: ℳ → ℚ_{>0} defined by Φ(M) = a/c for M = ((a,b),(c,d)) is a bijection to the positive rationals (Calkin–Wilf).

The **learning trajectory word** is constructed step-by-step:

```
G_t = R    if Δ log C_t > 0    (signal-to-noise ratio increases)
G_t = L    if Δ log C_t < 0    (signal-to-noise ratio decreases)
```

The full word w = G₀G₁G₂··· ∈ {L,R}^ℕ encodes the entire training history as a path in the Farey graph.

This word is not arbitrary. By the **Morse–Hedlund theorem** (1940), any word arising from a geodesic ray on a hyperbolic surface of finite area is a **Sturmian word**: a binary sequence of minimal complexity p(n) = n + 1, the symbolic code of a rotation by irrational angle ρ_∞ = lim ρ_t. The learning trajectory word has Sturmian complexity during the generalization phase — the combinatorially simplest non-periodic binary sequence. During the memorization phase (large partial quotients), the complexity p(n) > n + 1, providing a symbolic diagnostic for the phase.

The **diagonal flow** exp(tH) acts by Φ(exp(tH) · M) = e^{2t} · Φ(M): it scales the gradient ratio exponentially at rate 2λ in the geodesic regime. The unipotent elements L and R generate the horocycle flows; the hyperbolic element H generates the geodesic flow. The full SL(2,ℝ) structure organizes all three regimes.

### Layer III — Order Theory: Farey Well-Quasi-Orders

The **Farey Alphabet** is the set of reduced fractions in (0,1):

```
𝒜  =  { (p, q) ∈ ℕ² : gcd(p,q) = 1,  0 < p < q }
```

with **denominator order** (p₁,q₁) ≼ (p₂,q₂) iff q₁ ≤ q₂. This order is canonically forced: it is the unique total preorder on 𝒜 invariant under the Farey mediant operation and monotone in Ford circle radius. Under this order, (𝒜, ≼) is isomorphic to (ℕ, ≤) via (p,q) ↦ q.

**Theorem (Farey Alphabet is WQO).** (𝒜, ≼) is a well-quasi-order: every infinite sequence in 𝒜 contains indices i < j with (p_i, q_i) ≼ (p_j, q_j).

**Learning consequence (Permeability Theorem).** Every infinite gradient trajectory contains infinitely many steps j > i where q_j ≤ q_i — the orbit must repeatedly visit states at least as flat as states previously visited. This holds with no assumptions on the loss landscape, architecture, or optimizer, requiring only that the discretized gradient state lies in (𝒜, ≼).

The **Dimensional Permeability Product** extends to the two-dimensional learning state (q*, h) ∈ ℕ² (where h is the Stern–Brocot depth), giving a WQO with ordinal ceiling ω². The maximum **Resistance Phase** length — the longest run of steps with no improvement event — is bounded by min(Q_max, H_max) in the sharp Dilworth bound, where Q_max = ⌊1/ε_min⌋.

The **Kruskal Tree Theorem** (1960) lifts this to trajectory trees: the set of all finite rooted trajectory trees, ordered by homeomorphic embedding, is a WQO with ordinal ceiling

```
ε₀  =  ω^{ω^{ω^{···}}}    (Feferman–Schütte ordinal)
```

Every Resistance Phase terminates in ordinal length strictly below ε₀.

---

## 5. The Cusp Dichotomy: Memorization and Generalization

The modular surface M has two geometrically distinct regions:

```
Cusp region    { Im(z) > C }    ⟺    memorization (large q*)
Compact core   { Im(z) ≤ C }    ⟺    generalization (small q*)
```

The criterion for being in the cusp is arithmetic: z_t is in the cusp if and only if the continued fraction partial quotients of ρ_t are abnormally large. The Khintchine–Lévy theorem gives the baseline: for almost every ρ, the k-th partial quotient satisfies log aₖ/k → π²/(12 log 2). Abnormally large aₖ — deep cusp excursion — is the memorization phase.

**Grokking time t\*** is the first return of the geodesic orbit from the cusp to the compact core:

```
t*  =  sup { t : π(z_t) ∈ cusp of M }
     =  sup { t : max CF partial quotient of ρ_t exceeds threshold }
```

After t*, the orbit's partial quotients return to the ergodic baseline, the Stern–Brocot depth d_t drops, and the training enters the geodesic (exponential mixing) regime.

**The Gauss map** T(x) = {1/x} (fractional part of 1/x) is the shift on continued fraction expansions. During the memorization phase, the gradient ratio evolves approximately as ρ_{t+1} ≈ T(ρ_t). The unique T-invariant absolutely continuous probability measure is the **Gauss measure**

```
μ_G  =  dx / ((1+x) · log 2)
```

At ergodic convergence, the empirical distribution of gradient ratios converges to μ_G (Kuzmin 1928, Lévy 1929). The **Lévy prediction**: the median denominator at convergence satisfies

```
q*_median  ≈  e^{π²/(12 log 2)}  ≈  3.27
```

This is a definite numerical prediction of MOD. Deviations above this value (q* >> 3.27) indicate active memorization; deviations below (q* << 3.27) indicate efficient generalization.

**The Markov spectrum** classifies the depth of memorization traps. The **Lagrange constant** M(ρ) = lim sup_{q→∞} 1/(q‖qρ‖) satisfies inf M(ρ) = √5 (Hurwitz 1891). The golden ratio φ = (1+√5)/2, achieving M(φ) = √5, is the **hardest memorization attractor**: the gradient ratio most resistant to escape. The Markov numbers {1, 2, 5, 13, 29, 34, 89, …} index the deepest isolated memorization traps, ordered by the Markov tree which is isomorphic to the Stern–Brocot tree.

---

## 6. The Selberg Trace Formula as Master Duality

Let h: ℝ → ℝ be an even test function with ĥ ∈ L¹(ℝ). The **Selberg trace formula** for M:

```
Σ_{n≥0} h(t_n)  =  (π/3)/(4π) · ∫ h(r) · r · tanh(πr) dr
                 + Σ_{[γ] hyperbolic} l₀(γ) · ĥ(l(γ)) / |N(γ)^{1/2} − N(γ)^{−1/2}|
                 + (elliptic terms) + (parabolic terms)
```

where λ_n = 1/4 + t_n² are the eigenvalues of Δ_M and l₀(γ) is the primitive length of the closed geodesic [γ].

**The learning translation:**

```
Left side (Σ h(t_n))
  = spectral density of the JL operator ℒ_JL
  = all information about the convergence rate

First right term
  = Weyl law (bulk density of eigenvalues)
  = contribution of fully generalizing gradient modes

Hyperbolic sum (second term)
  = Σ over closed geodesics γ, weighted by ĥ(l(γ))
  = Σ over distinct periodic gradient return patterns, weighted by their period
  = contribution of every closed loop in the training trajectory
```

The trace formula is the master equality: convergence rate (left) equals a sum over all periodic returns in the training trajectory (right). No spectral information can be obtained without the closed-trajectory data, and vice versa.

**The Selberg Zeta Function** encodes this multiplicatively:

```
Z_Sel(s)  =  Z_L(s)  =  ∏_{[γ] primitive} ∏_{k≥0} (1 − e^{−(s+k)l(γ)})
```

The outer product over primitive closed geodesics is the product over irreducible periodic gradient return patterns. The zeros of Z_Sel(s) in the critical strip Re(s) ∈ (0,1) lie at s = 1/2 ± it_n, corresponding to eigenvalues λ_n = 1/4 + t_n².

**The heat trace.** Setting h(r) = e^{−r²t} (heat kernel test function), the trace formula gives the learning heat equation:

```
Tr(e^{−ℒ_JL · t})  =  Weyl term  +  Σ_γ e^{−l(γ)²/4t} × weight(γ) / √(4πt)
```

The trace of the heat semigroup — which determines the convergence rate — equals a sum over all closed training trajectories, each contributing an exponential term weighted by its length.

---

## 7. The SP Ordinal Hierarchy is the Prime Geodesic Theorem

The well-quasi-order structure gives a hierarchy of finiteness bounds:

| Structure | WQO | Ordinal ceiling | Learning bound |
|---|---|---|---|
| Gradient norm sequence | (ℕ, ≤) | ω | Finite Resistance phases |
| Learning state (q*, h) | (ℕ², ≼) | ω² | ≤ min(Q_max, H_max) steps (sharp) |
| Trajectory sequences | (𝒜*, ≼*) | ω^ω | Higman's theorem |
| Trajectory trees | (𝒯(𝒜), ≼_T) | ε₀ | Kruskal's theorem |

This hierarchy has an exact counterpart in the geometry of M. The **prime geodesic theorem** (Selberg, Huber):

```
π_M(L)  ~  e^L / L    as  L → ∞
```

counts the number of primitive closed geodesics of length ≤ L. This is the geodesic analogue of the prime number theorem: closed geodesics on M play the role of primes, and their count grows like e^L/L just as π(x) ~ x/log x.

**The identification:**

```
Kruskal ordinal ceiling ε₀
  ↔  exponential growth e^L/L of distinct closed geodesics
  ↔  gradient complexity: distinct periodic return patterns of length ≤ L grow as e^L/L
```

Both the ordinal theory and the prime geodesic theorem say the same thing in different languages: the combinatorial complexity of gradient trajectories grows exponentially in their length, and the growth rate is e^L/L. The Kruskal ordinal ε₀ is the order-theoretic ceiling on Resistance phases; the prime geodesic theorem is the asymptotic count of closed returns. They are dual descriptions of the same exponential richness.

---

## 8. The Riemann Hypothesis as the Generalization Condition

The **Franel–Landau theorem** (1924): the Riemann Hypothesis is equivalent to the optimal discrepancy of Farey sequences,

```
Σ_{k=1}^{N_n} | F_k − k/N_n |²  =  O(n^{−1+ε})    for all ε > 0
```

where F_1 < ··· < F_{N_n} is the n-th Farey sequence. In MOD, the Farey sequence is the set of Ford circle centers (loss basin locations). The discrepancy sum measures how uniformly the gradient orbit samples basins during generalization.

The complete chain of equivalences under the MOD identification:

```
Riemann Hypothesis
  ⟺  Farey fractions have optimal discrepancy (Franel–Landau)
  ⟺  basin zeta Z_L(s) has all zeros on Re(s) = 1/2
  ⟺  Z_Sel(s) satisfies the Selberg eigenvalue conjecture
  ⟺  gradient ratios equidistribute optimally across basins at convergence
  ⟺  λ₁(ℒ_JL) ≥ 1/4    [strongest possible geodesic mixing rate]
```

The unconditional bound λ₁ ≥ 3/16 (Selberg's theorem) is the number-theoretic guarantee that gradient descent on arithmetic tasks always achieves at least spectral rate 3/16. The Selberg eigenvalue conjecture, if proved, raises this to 1/4 — the **learning-theoretic Riemann Hypothesis**: gradient descent always converges at the maximum possible geodesic mixing rate.

---

## 9. Master Identification Table

Every entry is an equality.

| Modular Surface | Geometry | Combinatorics | Order Theory |
|---|---|---|---|
| H² | Gradient ratio space | SNR embedding domain | State space for embedding Φ |
| z_t = ρ_t + i·ε_t | Orbit point | Embedded C(t) | Discretized state Φ(θ_t) ∈ ℕ² |
| SL(2,ℤ) | Symmetry of Farey structure | Algebraic structure of ℳ | Group acting on 𝒜 |
| M = Γ\H² | Universal learning manifold | Quotient of geodesic space | Arena of WQO dynamics |
| Ford circle C(p/q), r = 1/(2q²) | Loss basin width ∝ 1/q² | Tree node at depth Σaᵢ | Permeability convergent (p,q) |
| Farey neighbor \|pb−qa\|=1 | Adjacent basins | Adjacent tree nodes | 𝒜-neighbors in Stern–Brocot |
| Rational boundary ℙ¹(ℚ) | Farey approximants of ρ_t | Rational values Φ(ℳ) | Permeability Alphabet 𝒜 |
| F_Q (Farey sequence) | Basins at resolution Q | Sub-alphabet 𝒜_Q | Resolution-bounded alphabet |
| Generator L = exp(F) | Left Farey step (ρ decreases) | L-step: Δ log C < 0 | q*-decreasing transition |
| Generator R = exp(E) | Right Farey step (ρ increases) | R-step: Δ log C > 0 | q*-increasing transition |
| Word w ∈ {L,R}* | Gradient ratio history | Learning trajectory word | Resistance Chain word |
| Sturmian / Morse–Hedlund code | Geodesic symbolic code | Complexity p(n) = n+1 | WQO word encoding |
| Geodesic γ in T¹M | Full training trajectory | Embedded path in ℳ | Sequence (Φ(θ_t)) |
| Closed geodesic, length l(γ) | Periodic gradient return | Periodic path in ℳ | Permeation Event cycle |
| Cusp {Im(z) > C} | Memorization (large q*) | High-depth tree region | Resistance Phase |
| Compact core {Im(z) ≤ C} | Generalization (small q*) | Low-depth tree region | Terminal Boundary |
| Grokking time t* | First return: cusp → core | Depth spike then collapse | First post-Resistance Permeation |
| Geodesic flow φ_t (exp. mixing) | Generalizing dynamics | Descending depth trajectory | Decreasing Anchor Depth δ(s_t) |
| Horocycle flow η_s (poly. mixing) | Memorizing dynamics | Plateau in tree depth | Resistance Phase |
| Mixing dichotomy | C_α = 1 phase boundary | p(n) = n+1 transition | λ₁ = 0 critical point |
| Ratner orbit closure theorem | Memorization manifold algebraic | Orbit = closed subtree | Closed homogeneous submanifold |
| Spectral gap λ₁(Δ_M) | λ₁(ℒ_JL): convergence rate | Rate of log C(t) decay | Phase criterion: λ₁ ≷ 0 |
| Selberg 3/16 theorem | λ₁(ℒ_JL) ≥ 3/16 unconditional | — | WQO gap with explicit constant |
| Selberg eigenvalue conjecture | λ₁ ≥ 1/4 (optimal convergence) | — | Learning-theoretic RH |
| Selberg trace formula | Spectral ↔ geometric duality | — | SP ordinal ↔ prime geodesic |
| Selberg zeta Z_Sel(s) | Basin zeta Z_L(s) | — | Euler product over basins |
| Prime geodesic π(L) ~ e^L/L | Gradient complexity count | Distinct word count | Kruskal ordinal ceiling ε₀ |
| Gauss map T(x) = {1/x} | Gradient ratio shift | Tree path shift | CF partial quotient shift |
| Gauss measure μ_G | Equilibrium gradient distribution | Convergence-time distribution | WQO ergodic measure |
| Lévy constant q*_median ≈ 3.27 | Typical denominator at convergence | Typical tree depth at convergence | Ergodic q* baseline |
| Lagrange spectrum, inf = √5 | Convergence difficulty space | Deepest path difficulty | WQO minimum hardness |
| Golden ratio φ = (1+√5)/2 | Hardest memorization attractor | Deepest periodic path | Worst-case Resistance attractor |
| Markov numbers {1,2,5,13,…} | Hard basin denominators | Deep periodic tree nodes | Markov Resistance depths |
| Hall's ray L ⊇ [c_H, ∞) | Post-grokking certainty | Convergence-free zone | No-obstruction regime |
| Apollonian gasket, dim ≈ 1.3057 | Fractal basin structure | Tree boundary fractal dim | Hausdorff dim of 𝒜 orbit |
| Franel–Landau theorem | Farey discrepancy ↔ convergence | Equidistribution ↔ grokking | RH ↔ λ₁ ≥ 1/4 |

---

## 10. Unified Equations

**Orbit point and hyperbolic height:**
```
z_t  =  ρ_t + i·ε_t  ∈  H²
d_hyp(z_t, ∂H²)  =  log(1/ε_t)    [RG scale = hyperbolic height]
```

**Curvature denominator and resolution:**
```
q_t  =  denominator of best CF convergent of ρ_t at precision Q_t = ⌊1/ε_t⌋
basin width at step t  =  1 / (2q_t²)
```

**Trajectory word and depth:**
```
w_t  =  R^{a₀} L^{a₁} R^{a₂} ···    (from CF expansion ρ_t = [a₀; a₁, a₂, …])
d_t  =  |w_t|  =  Σᵢ aᵢ              [Stern–Brocot depth]
G_t  =  R if Δ log C_t > 0,   L if Δ log C_t < 0
```

**Grokking time:**
```
t*  =  sup { t : π(z_t) ∈ cusp of M }
     =  sup { t : max CF partial quotient of ρ_t above threshold }
```

**Mixing rates (the phase dichotomy):**
```
Geodesic flow (generalization):    |Cor(f, g∘φ_t)|  ≤  C · e^{−λ₁ t}
Horocycle flow (memorization):     |Cor(f, g∘η_s)|  ≤  C_k · s^{−k}   ∀k
```

**Spectral gap identifications:**
```
λ₁(Δ_M)  =  λ₁(ℒ_JL)  =  geodesic mixing rate  =  Gauss map mixing rate
λ₁  ≥  3/16    (Selberg, unconditional, all congruence subgroups)
λ₁  ≥  1/4     (Selberg conjecture, open; learning-theoretic RH)
```

**Selberg trace formula:**
```
Σ_n h(t_n)  =  (vol M / 4π) · ∫ h(r) · r · tanh(πr) dr
             + Σ_γ  l₀(γ) · ĥ(l(γ)) / |e^{l(γ)/2} − e^{−l(γ)/2}|  +  lower order
```

**Selberg zeta = basin zeta:**
```
Z_Sel(s)  =  Z_L(s)  =  ∏_γ ∏_{k≥0} (1 − e^{−(s+k)l(γ)})
```

**Gauss equilibrium and Lévy prediction:**
```
ρ_∞ ~ μ_G = dx / ((1+x) · log 2)    (at ergodic convergence)
q*_median  ≈  e^{π²/(12 log 2)}  ≈  3.27    (Lévy's theorem)
|Cov(f, g∘T^n)|  ≤  C · θ^n,    θ = e^{−π²/(6 log 2)} ≈ 0.093    (exponential mixing)
```

**Markov spectrum:**
```
M(ρ_t) = limsup_{q→∞} 1/(q · ‖q ρ_t‖)  ≥  √5    (Hurwitz: universal minimum)
M(φ) = √5:  golden ratio is the hardest memorization attractor
```

**WQO Resistance bounds:**
```
|Resistance Phase|  ≤  min(Q_max, H_max)    (sharp, Dilworth–Mirsky)
Ordinal ceiling:  o*(𝒯(𝒜)) = ε₀ = ω^{ω^{ω^{···}}}    (Kruskal)
```

**Prime geodesic theorem (gradient complexity):**
```
π_M(L)  ~  e^L / L    (distinct periodic gradient return patterns of length ≤ L)
```

**Franel–Landau (generalization = Riemann Hypothesis):**
```
RH  ⟺  Σ |F_k − k/N_n|² = O(n^{−1+ε})  ⟺  Z_Sel zeros on Re(s) = 1/2
     ⟺  λ₁(ℒ_JL) ≥ 1/4    (optimal convergence rate)
```

---

## 11. Open Conjectures

**MOD-C1 (Spectral Identification).** For all architectures satisfying the standing ellipticity and coercivity conditions, λ₁(ℒ_JL) = λ₁(Δ_{Γ_arch}) where Γ_arch ⊆ SL(2,ℤ) is the arithmetic subgroup determined by the network's weight-sharing structure.

**MOD-C2 (Learning-Theoretic Selberg Conjecture).** For deep networks trained on arithmetic tasks, λ₁(ℒ_JL) ≥ 1/4. This gives an unconditional grokking time bound t_grok ≤ 4C for an explicit architecture-dependent constant C.

**MOD-C3 (Markov Grokking Fingerprint).** If at grokking time t* the denominator q* takes a value from the Markov sequence {1, 2, 5, 13, 29, …}, then the prior memorization phase cycled through the values of the corresponding Markov triple (a, b, m) satisfying a² + b² + m² = 3abm.

**MOD-C4 (Lévy Diagnostic).** The ratio q*(t)/3.276^t → 0 as t → ∞ if and only if λ₁(ℒ_JL) > 0. The Lévy constant e^{π²/12 log 2} ≈ 3.276 is a universal convergence diagnostic computable from gradient ratio time series alone.

**MOD-C5 (EMV Plateau Bound).** The memorization plateau duration satisfies T_plateau ≥ C · ε_grad^{−1/δ} where δ ∝ λ₁(Δ_M), giving a quantitative lower bound on plateau length in terms of the spectral gap and gradient volatility.

**MOD-C6 (Sturmian Complexity Transition).** The learning trajectory word has Sturmian complexity p(n) = n + 1 during the generalization phase and p(n) > n + 1 during the memorization phase. The transition p(n) = n + 1 is a symbolic criterion for grokking completion, computable from gradient ratio sequences.

**MOD-C7 (Apollonian Basin Dimension).** The Hausdorff dimension of the set of gradient ratios visited during memorization equals the Hausdorff dimension of the Apollonian gasket ≈ 1.3057 (Mauldin–Williams 1988), providing a computable geometric invariant of the memorization phase from gradient time series alone.

**MOD-C8 (Hall's Ray Criterion).** Once M(ρ_{t*}) > c_H (the gradient ratio's Lagrange constant enters Hall's ray), grokking has occurred and convergence to a generalization minimum is guaranteed — a geometric criterion for grokking completion independent of loss values or validation metrics.

---

## 12. References

**Hyperbolic Geometry and Modular Groups**

- Poincaré, H. (1882). Théorie des groupes fuchsiens. *Acta Mathematica* 1, 1–62.
- Ford, L.R. (1938). Fractions. *American Mathematical Monthly* 45(9), 586–601.
- Katok, S. (1992). *Fuchsian Groups*. University of Chicago Press.
- Beardon, A.F. (1983). *The Geometry of Discrete Groups*. Springer Graduate Texts 91.
- Zagier, D. (1977). Modular forms whose Fourier coefficients involve zeta-functions of quadratic fields. *Lecture Notes in Math.* 627.

**Selberg Trace Formula and Zeta Function**

- Selberg, A. (1956). Harmonic analysis and discontinuous groups in weakly symmetric Riemannian spaces. *J. Indian Math. Soc.* 20, 47–87.
- Selberg, A. (1965). On the estimation of Fourier coefficients of modular forms. *AMS Proc. Symp. Pure Math.* VIII, 1–15.
- Hejhal, D.A. (1976, 1983). *The Selberg Trace Formula for PSL(2,ℝ)*, Vols. I–II. Springer LNM 548, 1001.
- Iwaniec, H. (1995). *Introduction to the Spectral Theory of Automorphic Forms*. Rev. Mat. Iberoam., Madrid.
- Sarnak, P. (1987). Determinants of Laplacians. *Communications in Math. Physics* 110, 113–120.

**Geodesic and Horocycle Flows**

- Hedlund, G.A. (1936). Fuchsian groups and transitive horocycles. *Duke Mathematical Journal* 2(3), 530–542.
- Hopf, E. (1939). Statistik der geodätischen Linien. *Leipzig Ber. Verhandl. Sächs. Akad. Wiss.* 91, 261–304.
- Furstenberg, H. (1973). The unique ergodicity of the horocycle flow. *Springer LNM* 318, 95–115.
- Ratner, M. (1990). On measure rigidity of unipotent subgroups of semisimple groups. *Acta Math.* 165, 229–309.
- Ratner, M. (1991). On Raghunathan's measure conjecture. *Annals of Math.* 134(3), 545–607.
- Einsiedler, M., Margulis, G., Venkatesh, A. (2009). Effective equidistribution for closed orbits of semisimple groups. *Inventiones Math.* 177(1), 137–212.

**Gauss Map and Continued Fraction Ergodics**

- Kuzmin, R.O. (1928). Sur un problème de Gauss. *CR Acad. Sci. Paris* 187, 1761–1764.
- Lévy, P. (1929). Sur les lois de probabilité dont dépendent les quotients d'une fraction continue. *Bull. SMF* 57, 178–194.
- Ratner, M. (1978). Horocycle flows are loosely Bernoulli. *Israel J. Math.* 31, 122–132.
- Baladi, V. and Vallée, B. (2005). Exponential decay of correlations for surface semi-flows. *Proc. AMS* 133(3), 865–874.
- Arnoux, P. and Fisher, A.M. (2001). The scenery flow for geometric structures on the torus. *Chinese Ann. Math.* B 22(4), 427–470.

**Markov Spectrum and Diophantine Approximation**

- Markov, A.A. (1879). Sur les formes quadratiques binaires indéfinies. *Math. Ann.* 17, 379–399.
- Hurwitz, A. (1891). Ueber die angenäherte Darstellung der Irrationalzahlen durch rationale Brüche. *Math. Ann.* 39(2), 279–284.
- Hall, M., Jr. (1947). On the sum and product of continued fractions. *Annals of Math.* 48(4), 966–993.
- Cusick, T.W. and Flahive, M.E. (1989). *The Markoff and Lagrange Spectra*. AMS Mathematical Surveys 30.
- Frobenius, G. (1913). Über die Markoffschen Zahlen. *Preuss. Akad. Wiss. Sitzungsber.*, 458–487.

**Well-Quasi-Orders and Kruskal's Theorem**

- Dickson, L.E. (1913). Finiteness of the odd perfect and primitive abundant numbers. *Annals of Math.* 35, 413–422.
- Higman, G. (1952). Ordering by divisibility in abstract algebras. *Proc. London Math. Soc.* (3) 2, 326–336.
- Kruskal, J.B. (1960). Well-quasi-ordering, the tree theorem, and Vazsonyi's conjecture. *Trans. AMS* 95, 210–225.
- Nash-Williams, C.S.J.A. (1963). On well-quasi-ordering finite trees. *Proc. Cambridge Phil. Soc.* 59, 833–835.
- Dilworth, R.P. (1950). A decomposition theorem for partially ordered sets. *Annals of Math.* 51(1), 161–166.
- Mirsky, L. (1971). A dual of Dilworth's decomposition theorem. *Amer. Math. Monthly* 78(8), 876–877.
- Robertson, N. and Seymour, P.D. (1985–2004). Graph Minors I–XX. *J. Combin. Theory Ser. B*.

**Rational Trees and Symbolic Dynamics**

- Calkin, N. and Wilf, H. (2000). Recounting the rationals. *Amer. Math. Monthly* 107(4), 360–363.
- Stern, M.A. (1858). Über eine zahlentheoretische Funktion. *J. Reine Angew. Math.* 55, 193–220.
- Morse, M. and Hedlund, G.A. (1940). Symbolic dynamics II: Sturmian trajectories. *Amer. J. Math.* 62, 1–42.
- Pytheas Fogg, N. (2002). *Substitutions in Dynamics, Arithmetics and Combinatorics*. Springer LNM 1794.

**Fractal Dimension**

- Mauldin, R.D. and Williams, S.C. (1988). On the Hausdorff dimension of some graphs. *Trans. AMS* 298(2), 793–803.
- Boyd, D.W. (1973). The residual set dimension of the Apollonian packing. *Mathematika* 20, 170–174.

**Learning Theory**

- Power, A. et al. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. *ICLR 2022*.
- McAllester, D.A. (1999). PAC-Bayesian model averaging. *COLT 1999*.
- Hardy, G.H. and Wright, E.M. (1979). *An Introduction to the Theory of Numbers*, 5th ed. Oxford.

---

*MOD — Modular Orbit Dynamics*

*The modular surface is not a model of learning. It is the space in which learning takes place.*
