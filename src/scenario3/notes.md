# For the future?

Sparse feedback from the user, with time to time
negative feedback, sometimes positive feedback


## Notes on implementation

For belief update, both implementations are used:
```math
KL(Q(z) || P(x, z))
```
and
```math
- E_{Q(z)} [ln P(x | z)] + KL(Q(z) || P(z))
```

with `x` the observable variable(s) 
and `z` the latent variable(s)

KL estimated via Monte Carlo sampling

```math
z \sim Q
```

```math
log(Q(z)) - log(P(x, z))
```

For action, 

```math
E_Q(z) [ln T(x | z)] + KL(Q(x) || P(z))  
```
with `T` the target distribution, and 