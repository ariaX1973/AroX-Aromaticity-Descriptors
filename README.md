# AroX — Aromaticity Descriptors (HOMA + LDM)

> Programme Python pour le calcul automatisé des descripteurs d'aromaticité
> (HOMA et LDM) à partir d'une géométrie moléculaire ou d'une matrice de
> distances. Conçu au **Laboratoire de Chimie Théorique (LCT)** par
> **Aria Noroozi**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.3.5-green.svg)](AroX.v0.3.5.py)

---

## Sommaire

- [Présentation](#présentation)
- [Démarche scientifique](#démarche-scientifique)
- [Installation](#installation)
- [Démarrage rapide](#démarrage-rapide)
- [Formats d'entrée acceptés](#formats-dentrée-acceptés)
- [Fichiers de sortie](#fichiers-de-sortie)
- [Exemples fournis](#exemples-fournis)
- [Comment citer AroX](#comment-citer-arox)
- [Licence](#licence)
- [Auteur & contact](#auteur--contact)

---

## Présentation

**AroX** est un programme autonome qui analyse l'aromaticité de cycles
organiques en combinant **trois familles de descripteurs complémentaires** :

| Descripteur | Nature | Référence |
|---|---|---|
| **HOMA** *(Harmonic Oscillator Model of Aromaticity)* | Basé sur les **distances de liaison** géométriques par rapport à une référence aromatique idéale. | Kruszewski & Krygowski, *Tetrahedron Lett.* 1972. |
| **LDM** *(Local Delocalization Matrix)* | Basé sur la **matrice de densité locale** (indices de délocalisation δ_ij) projetée sur les liaisons du cycle. Comparaison à la référence EG / EG-hetero / CUSTOM. | Implémentation LCT. |
| **H** *(descripteurs entropiques, v0.3)* | **Entropies de Shannon** en bits sur les termes de la LDM : `H_LDM` (complet : λ_ii + δ_ij, décomposé en H_loc, H_deloc, H_part), `H_Q` (δ_ij/R_ij, favorise les contributions courtes), `H_S` (δ_ij·R_ij, favorise les contributions étendues). Normalisation μ_LDM entre la limite localisée chimique (H_min_chem) et la borne maximale (H_max_ref = log₂(N_loc + N_deloc)). | Cahier des charges LCT v2.0. |

Les trois familles sont calculées **localement** sur chaque cycle et
**globalement** sur le sous-graphe des atomes lourds (Z > 1) de la
molécule. Le résumé final du programme affiche successivement les trois
tableaux : **LDM**, **H**, puis **HOMA**.

Le programme gère :

- les cycles **élémentaires**, **périphériques** et les cycles **fusionnés**
  (générés par différence symétrique des arêtes des cycles auto/manuels) ;
- l'**adaptation automatique des rayons covalents** selon l'hybridation
  détectée (sp / sp² / sp³) ;
- les **molécules hétéroatomiques** (H, C, N, O, F, P, S, Cl) ;
- une **analyse globale "atomes lourds"** sur la molécule entière ;
- un **mode trajectoire MD** (HOMA suivi le long d'une dynamique
  moléculaire, format `.traj`, avec export CSV).

## Démarche scientifique

AroX a été développé pour répondre à un besoin concret au LCT : pouvoir
quantifier l'aromaticité de **systèmes polycycliques** (PAH) de manière
*reproductible*, *automatisée*, et *vérifiable* à partir des seuls
fichiers produits en sortie d'un calcul de chimie quantique (matrice de
distances et matrice LDM exportées en `.dat`).

Le flux de travail est volontairement simple :

1. Préparer la matrice de distances (Bohr) du calcul DFT (`*_rho_dist.dat`).
2. *(optionnel)* préparer la matrice LDM associée (`*_rho_ldm.dat`).
3. Lancer AroX : le programme détecte la connectivité, énumère les cycles,
   propose une analyse cycle par cycle puis un récapitulatif global.
4. Le rapport complet est écrit dans un fichier `.arx` (texte brut,
   lisible et diff-able).

Les fichiers d'exemple commités dans `examples/` correspondent à la
**série de référence** utilisée pendant le développement : benzène,
naphtalène, anthracène, phénanthrène, naphtacène, chrysène, triphénylène,
pyrène, biphényle, biphénylène, benzocyclobutadiène, acénaphtylène,
pyracylène, dibenz[a,j]anthracène et coronène, tous obtenus au niveau
**B3LYP / 6-311+G(d,p)**.

## Installation

AroX n'a qu'une seule dépendance externe : **NumPy**.

```bash
git clone https://github.com/ariaX1973/AroX-Aromaticity-Descriptors.git
cd AroX-Aromaticity-Descriptors
pip install numpy
```

Python ≥ 3.8 suffit (annotations de type modernes utilisées).

## Démarrage rapide

```bash
python AroX.v0.3.5.py
```

Le programme est entièrement **interactif** : il vous demande dans
l'ordre les fichiers, paramètres et options nécessaires.

### Exemple — benzène (HOMA + LDM)

```text
$ python AroX.v0.3.5.py
Entrez le fichier géométrie / distances / trajectoire
(.xyz, .traj, .dat ou .dt) : examples/Benzen_rho_dist.dat
Entrez le fichier LDM (.dat) [optionnel]   : examples/Benzen_rho_ldm.dat
```

Choisissez ensuite les paramètres de référence :

- **EG** (Estrada-Goodman pur, fractions 7/12, 1/6, 1/12) ;
- **EG-hetero** (généralisation aux cycles hétéroatomiques) ;
- **CUSTOM** (vos propres valeurs ε_o, ε_p, ε_m).

Le rapport est écrit dans `Benzen_rho_dist.arx`.

### Exemple — HOMA seul (sans LDM)

Si vous ne renseignez pas de fichier LDM (touche `Entrée` à l'invite),
AroX bascule en **mode HOMA seul** : seuls les rayons d'équilibre et le
HOMA généralisé sont calculés.

### Exemple — Trajectoire MD

Si le fichier d'entrée est un `.traj` (ou contient `traj` dans le nom),
AroX active le **mode trajectoire** : il vous demande le pas de temps
(en fs), suit le HOMA frame par frame et exporte un CSV au format
`molecule_homa_trajectoire.csv`.

## Formats d'entrée acceptés

| Extension | Contenu attendu | Unités |
|---|---|---|
| `.dat` (rho_dist) | Première ligne : `0` + numéros atomiques. Lignes suivantes : `Z_i d_i1 d_i2 ...` (matrice carrée). | distances en **Bohr** (converties en Å en interne) |
| `.dat` (rho_ldm) | Même structure matricielle, valeurs LDM (densité électronique délocalisée). | sans dimension |
| `.xyz` | Format XYZ standard `Symbole x y z` (avec ou sans ligne d'en-tête). | Å |
| `.traj` / `.dt` | Concaténation de blocs XYZ (1 par frame) pour une trajectoire MD. | Å |

Voir `examples/Benzen_rho_dist.dat` pour un fichier minimal.

## Fichiers de sortie

| Fichier | Contenu |
|---|---|
| `*.arx` | Rapport texte complet : connectivité détectée, types d'atomes, cycles élémentaires/périphériques/fusés, analyse cycle par cycle (HOMA + LDM si fourni), analyse globale atomes lourds, classement final des cycles. |
| `*.LDM` *(optionnel)* | Re-écriture de la matrice LDM canonisée avec annotations sur les cycles traités. |
| `*_homa_trajectoire.csv` | (Mode MD) HOMA(t) pour chaque cycle suivi. |

Les fichiers `.arx` sont des **rapports lisibles à l'œil** (texte UTF-8)
et donc faciles à versionner / diff-er pour vérifier qu'une modification
de l'algorithme n'a pas changé un résultat scientifique.

## Exemples fournis

Le dossier [`examples/`](examples/) contient 15 molécules de référence
au niveau B3LYP / 6-311+G(d,p) ; pour chacune :

- `*_rho_dist.dat` — matrice de distances (entrée AroX) ;
- `*_rho_ldm.dat`  — matrice LDM (entrée AroX, optionnelle) ;
- `*_rho_dist.arx` — rapport AroX de référence (sortie attendue) ;
- `*_rho_ldm.LDM`  — LDM canonisée de référence.

Vous pouvez relancer AroX sur n'importe lequel de ces couples de
fichiers et comparer votre `.arx` au `.arx` de référence pour vérifier
votre installation.

## Comment citer AroX

Si vous utilisez AroX dans un travail académique, merci de citer :

> **Noroozi, A.** (2026). *AroX — Aromaticity Descriptors (HOMA + LDM)*,
> v0.3.5 [computer software]. Laboratoire de Chimie Théorique (LCT).
> https://github.com/ariaX1973/AroX-Aromaticity-Descriptors

### BibTeX

```bibtex
@software{Noroozi_AroX_2026,
  author      = {Noroozi, Aria},
  title       = {{AroX} --- Aromaticity Descriptors (HOMA + LDM)},
  year        = {2026},
  version     = {0.3.5},
  institution = {Laboratoire de Chimie Th{\'e}orique (LCT)},
  url         = {https://github.com/ariaX1973/AroX-Aromaticity-Descriptors}
}
```

GitHub détecte automatiquement le fichier [`CITATION.cff`](CITATION.cff)
fourni à la racine : un bouton **"Cite this repository"** apparaît sur
la page du dépôt.

## Licence

AroX est distribué sous **licence MIT** (voir [`LICENSE`](LICENSE)).
Vous êtes libre de l'utiliser, le modifier et le redistribuer, y compris
dans un cadre commercial, à la seule condition de conserver la mention
de copyright et la licence.

## Auteur & contact

**Aria Noroozi** — Laboratoire de Chimie Théorique (LCT), 2026.

Pour signaler un bug, proposer une amélioration ou poser une question
scientifique, ouvrez une **[issue GitHub](https://github.com/ariaX1973/AroX-Aromaticity-Descriptors/issues)**.
Les pull requests sont les bienvenues.
