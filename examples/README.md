# Examples — Reference set (B3LYP / 6-311+G(d,p))

Each molecule below ships with **four files**:

| Suffix | Role | Type |
|---|---|---|
| `*_rho_dist.dat` | Distance matrix (Bohr) — **AroX input** | Input |
| `*_rho_ldm.dat`  | LDM matrix — **AroX input (optional)** | Input |
| `*_rho_dist.arx` | AroX report — **reference output** | Expected output |
| `*_rho_ldm.LDM`  | Canonized LDM — **reference output** | Expected output |

## Molecule list

1. Benzene
2. Naphthalene
3. Anthracene
4. Phenanthrene
5. Naphtacene (tetracene)
6. Chrysene
7. Triphenylene
8. Pyrene
9. Biphenyl
10. Biphenylene
11. Benzocyclobutadiene
12. Acenaphtylene
13. Pyracylene
14. Dibenz[a,j]anthracene
15. Coronene

## Reproduce a calculation

From the repository root:

```bash
python AroX.v0.2.0.py
# When prompted, enter:
#   geometry/distances file : examples/Benzen_rho_dist.dat
#   LDM file (optional)     : examples/Benzen_rho_ldm.dat
#   reference parameters    : EG (default)
```

The new `Benzen_rho_dist.arx` produced in your working directory should
be **identical** (up to insignificant float formatting) to
`examples/Benzen_rho_dist.arx`.

This is the simplest sanity check that your installation works.
