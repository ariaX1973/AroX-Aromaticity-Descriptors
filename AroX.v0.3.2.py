# ==============================================================
#  AroX — Aromaticity Descriptors (HOMA + LDM)
# ==============================================================
#  Author      : Aria Noroozi
#  Affiliation : Laboratoire de Chimie Théorique (LCT), 2026
#  Version     : 0.3.2
#  License     : MIT (see LICENSE file)
#  Repository  : https://github.com/ariaX1973/AroX-Aromaticity-Descriptors
#
#  How to cite
#  -----------
#  Noroozi, A. (2026). AroX — Aromaticity Descriptors (HOMA + LDM),
#  v0.3.2 [computer software]. Laboratoire de Chimie Théorique (LCT).
#  https://github.com/ariaX1973/AroX-Aromaticity-Descriptors
#
#  BibTeX
#  ------
#  @software{Noroozi_AroX_2026,
#    author      = {Noroozi, Aria},
#    title       = {{AroX} --- Aromaticity Descriptors (HOMA + LDM)},
#    year        = {2026},
#    version     = {0.3.2},
#    institution = {Laboratoire de Chimie Th{\'e}orique (LCT)},
#    url         = {https://github.com/ariaX1973/AroX-Aromaticity-Descriptors}
#  }
# ==============================================================


# ============================================================
#  TABLE DES MATIÈRES
# ============================================================
#  BLOC 1   — Constantes globales, tables de rayons, étiquettes
#  BLOC 2   — Lecture et validation des fichiers d'entrée (.dat)
#  BLOC 3   — Paramètres de référence (EG, EG-hetero, CUSTOM)
#  BLOC 4   — Connectivité moléculaire et adaptation des rayons
#  BLOC 5   — Détection, canonisation, filtrage, classification des cycles
#  BLOC 6   — Construction de la matrice LDM de référence
#  BLOC 7   — Extraction des sous-matrices LDM et contrôle de symétrie
#  BLOC 8   — Calcul des distances, matrice différence, contributions globales
#  BLOC 8B  — Analyse spatiale pondérée et homogénéité par orbites
#  BLOC 9   — Contributions par orbites et moyennes réelles
#  BLOC 9B  — Analyse Q / CT / homogénéité d'orbite
#  BLOC 10  — Analyse complète d'un cycle traitable
#  BLOC 11  — Classement global des cycles
#  BLOC 12  — Rapport préliminaire (structure, types d'atomes)
#  BLOC 13  — Écriture du fichier .arx (mode LDM historique)
#  BLOC 13B — Analyse globale atomes lourds (Q(G), S(G), S-hom(G))
#  BLOC 13C — Descripteurs entropiques H_LDM / H_Q / H_S  (v0.3)
#  BLOC 14  — Affichage terminal du résumé LDM
#  BLOC 15  — main() historique (LDM seul)
#  BLOC 16  — HOMA intégré sur les mêmes cycles que LDM
#  BLOC 17A — Mode trajectoire MD HOMA
#  BLOC 17  — main_integre_homa_ldm()  ← point d'entrée du programme
# ============================================================


# ============================================================
# BLOC 1 — LIBRAIRIES ET CONSTANTES GLOBALES
# ============================================================
# Numpy pour l'algèbre linéaire, math pour log2, os pour la lecture
# de fichiers. Toutes les constantes numériques utilisées ailleurs
# dans le programme sont regroupées ici pour être facilement ajustées.
# ============================================================

import numpy as np
import math
import os
from typing import List, Dict, Tuple, Set


# ==============================
# CONSTANTES NUMÉRIQUES GÉNÉRALES
# ==============================

BOHR_TO_ANGSTROM = 0.529177
TOLERANCE_CONNECTIVITE = 0.45
TOLERANCE_SYMETRIE = 1.0e-8
PRECISION_RAPPORT = 6


# ==============================
# PARAMÈTRES PAR DÉFAUT DE LA RÉFÉRENCE EG
# ==============================

# Fractions exactes en interne ; affichage décimal à l'utilisateur.
EPSILON_O_DEFAUT = 7 / 12
EPSILON_P_DEFAUT = 1 / 6
EPSILON_M_DEFAUT = 1 / 12
N_A_DEFAUT = 6  # Conservé pour compat (= Z du carbone, cas benzène homo-atomique).
                # En V06, N_A est généralisé par site : N_A[k] = Z_k lu depuis le .dat.


# ==============================
# TABLE DES RAYONS COVALENTS DE BASE
# ==============================

RAYONS_COVALENTS_BASE = {
    1: 0.31,   # H
    6: 0.76,   # C
    7: 0.71,   # N
    8: 0.66,   # O
    9: 0.57,   # F
    15: 1.07,  # P
    16: 1.05,  # S
    17: 1.02   # Cl
}


# ==============================
# VALEURS ET ÉTIQUETTES DE CONTRÔLE
# ==============================

REFERENCE_NAME_EG = "EG"
REFERENCE_NAME_EG_HETERO = "EG-hetero"
REFERENCE_NAME_CUSTOM = "CUSTOM"

STATUT_CYCLE_TRAITE = "TRAITE"
STATUT_CYCLE_NON_TRAITE = "NON TRAITE EN V2"

TYPE_CYCLE_ELEMENTAIRE = "ELEMENTAIRE"
TYPE_CYCLE_PERIPHERAL = "PERIPHERAL"
TYPE_CYCLE_NON_TRAITABLE = "NON TRAITABLE"

ORIGINE_AUTO = "AUTO"
ORIGINE_MANUEL = "MANUEL"
ORIGINE_AUTO_MANUEL = "AUTO+MANUEL"
ORIGINE_FUSED = "FUSED"


# ============================================================
# BLOC 2 — LECTURE ET VALIDATION DES FICHIERS D'ENTRÉE (.dat)
# ============================================================
# Deux fichiers .dat sont attendus : la matrice LDM et la matrice
# de distances (en Bohr). Les fonctions ci-dessous valident la
# cohérence des deux fichiers (même nombre d'atomes, même ordre)
# et convertissent les distances en Ångström.
# ============================================================

# ==============================
# FONCTION DE DEMANDE DU FICHIER LDM
# ==============================

def demander_fichier_ldm() -> str:
    while True:
        nom_fichier = input("Entrez le nom du fichier LDM (.dat) : ").strip()
        if os.path.exists(nom_fichier):
            return nom_fichier
        print("Erreur : fichier LDM introuvable. Réessayez.")


# ==============================
# FONCTION DE DEMANDE DU FICHIER DIST
# ==============================

def demander_fichier_dist() -> str:
    while True:
        nom_fichier = input("Entrez le nom du fichier de distances (.dat) : ").strip()
        if os.path.exists(nom_fichier):
            return nom_fichier
        print("Erreur : fichier de distances introuvable. Réessayez.")


# ==============================
# FONCTION DE LECTURE DU FICHIER .DAT
# ==============================

def lire_fichier_dat(nom_fichier: str) -> Tuple[List[int], np.ndarray]:
    with open(nom_fichier, "r", encoding="utf-8") as fichier:
        lignes = [ligne.strip() for ligne in fichier if ligne.strip()]

    if len(lignes) < 2:
        raise ValueError(f"Erreur : le fichier {nom_fichier} est trop court.")

    premiere_ligne = lignes[0].split()
    if len(premiere_ligne) < 2:
        raise ValueError(f"Erreur : en-tête invalide dans {nom_fichier}.")

    numeros_atomiques = [int(x) for x in premiere_ligne[1:]]
    n = len(numeros_atomiques)

    if len(lignes) < n + 1:
        raise ValueError(f"Erreur : nombre de lignes insuffisant dans {nom_fichier}.")

    matrice_lignes = []
    for i in range(1, n + 1):
        morceaux = lignes[i].split()
        if len(morceaux) < n + 1:
            raise ValueError(f"Erreur : ligne {i + 1} incomplète dans le fichier {nom_fichier}.")
        valeurs = [float(x) for x in morceaux[1:n + 1]]
        matrice_lignes.append(valeurs)

    matrice = np.array(matrice_lignes, dtype=float)
    valider_format_matrice(nom_fichier, numeros_atomiques, matrice)
    return numeros_atomiques, matrice


# ==============================
# FONCTION DE VALIDATION DU FORMAT DE MATRICE
# ==============================

def valider_format_matrice(
    nom_fichier: str,
    numeros_atomiques: List[int],
    matrice: np.ndarray
) -> None:
    n = len(numeros_atomiques)
    if matrice.shape != (n, n):
        raise ValueError(f"Erreur : la matrice du fichier {nom_fichier} n'est pas de taille {n} x {n}.")


# ==============================
# FONCTION DE VÉRIFICATION DE LA COMPATIBILITÉ ENTRE LDM ET DIST
# ==============================

def verifier_compatibilite_fichiers(
    numeros_atomiques_ldm: List[int],
    matrice_ldm: np.ndarray,
    numeros_atomiques_dist: List[int],
    matrice_dist: np.ndarray
) -> None:
    verifier_nombre_atomes(numeros_atomiques_ldm, numeros_atomiques_dist)
    verifier_ordre_et_indexation(numeros_atomiques_ldm, numeros_atomiques_dist)
    if matrice_ldm.shape != matrice_dist.shape:
        raise ValueError("Erreur : les matrices LDM et DIST n'ont pas la même dimension.")


# ==============================
# FONCTION DE CONTRÔLE DU NOMBRE D’ATOMES, DE L’ORDRE ET DE L’INDEXATION
# ==============================

def verifier_nombre_atomes(
    numeros_atomiques_ldm: List[int],
    numeros_atomiques_dist: List[int]
) -> None:
    if len(numeros_atomiques_ldm) != len(numeros_atomiques_dist):
        raise ValueError("Erreur : les fichiers LDM et DIST n'ont pas le même nombre d'atomes.")


def verifier_ordre_et_indexation(
    numeros_atomiques_ldm: List[int],
    numeros_atomiques_dist: List[int]
) -> None:
    if numeros_atomiques_ldm != numeros_atomiques_dist:
        raise ValueError("Erreur : les fichiers LDM et DIST n'ont pas le même ordre atomique.")


# ==============================
# FONCTION DE CONVERSION ÉVENTUELLE DES DISTANCES EN ANGSTRÖM
# ==============================

def convertir_matrice_bohr_en_angstrom(matrice: np.ndarray) -> np.ndarray:
    return matrice * BOHR_TO_ANGSTROM


# ============================================================
# BLOC 3 — PARAMÈTRES DE RÉFÉRENCE (EG, EG-hetero, CUSTOM)
# ============================================================
# Trois valeurs epsilon (ortho, para, meta) définissent la matrice
# LDM idéale du benzène. Trois modes possibles : EG (défauts 7/12,
# 1/6, 1/12), EG-hetero (généralisation aux hétéroatomes) ou
# CUSTOM (valeurs saisies par l'utilisateur).
# ============================================================

# ==============================
# FONCTION DE DEMANDE DES PARAMÈTRES UTILISATEUR
# ==============================

def demander_parametres_reference() -> Dict[str, float]:
    print()
    print(" Paramètres de référence (Entrée = valeur EG par défaut)")
    print(" Note : N_A est lu automatiquement depuis le fichier .dat,")
    print("        chaque atome du cycle utilise son propre Z comme N_A.")
    print()

    entree_epsilon_o = input(
        f"Entrez epsilon_O (Entrée pour valeur par défaut = {EPSILON_O_DEFAUT:.6f}) : "
    ).strip()
    epsilon_o = EPSILON_O_DEFAUT if entree_epsilon_o == "" else float(entree_epsilon_o)

    entree_epsilon_p = input(
        f"Entrez epsilon_P (Entrée pour valeur par défaut = {EPSILON_P_DEFAUT:.6f}) : "
    ).strip()
    epsilon_p = EPSILON_P_DEFAUT if entree_epsilon_p == "" else float(entree_epsilon_p)

    entree_epsilon_m = input(
        f"Entrez epsilon_M (Entrée pour valeur par défaut = {EPSILON_M_DEFAUT:.6f}) : "
    ).strip()
    epsilon_m = EPSILON_M_DEFAUT if entree_epsilon_m == "" else float(entree_epsilon_m)

    # N_A indicatif (carbone) pour l'affichage et la détection EG/CUSTOM.
    # La vraie valeur N_A par site sera Z_k de chaque atome du cycle.
    n_a = N_A_DEFAUT

    valeur_lambda = calculer_lambda(epsilon_o, epsilon_p, epsilon_m, n_a)
    nom_reference = identifier_type_reference(epsilon_o, epsilon_p, epsilon_m, n_a)
    verifier_parametres_inhabituels(epsilon_o, epsilon_p, epsilon_m, n_a, valeur_lambda)

    return construire_dictionnaire_parametres(
        epsilon_o,
        epsilon_p,
        epsilon_m,
        n_a,
        valeur_lambda,
        nom_reference
    )


# ==============================
# FONCTION D’IDENTIFICATION DU TYPE DE RÉFÉRENCE
# ==============================

def identifier_type_reference(
    epsilon_o: float,
    epsilon_p: float,
    epsilon_m: float,
    n_a: float
) -> str:
    if (
        abs(epsilon_o - EPSILON_O_DEFAUT) <= 1e-15 and
        abs(epsilon_p - EPSILON_P_DEFAUT) <= 1e-15 and
        abs(epsilon_m - EPSILON_M_DEFAUT) <= 1e-15 and
        abs(n_a - N_A_DEFAUT) <= 1e-15
    ):
        return REFERENCE_NAME_EG
    return REFERENCE_NAME_CUSTOM


# ==============================
# FONCTION DE CALCUL DE LAMBDA
# ==============================

def calculer_lambda(
    epsilon_o: float,
    epsilon_p: float,
    epsilon_m: float,
    n_a: float
) -> float:
    return n_a - (2.0 * epsilon_o + 2.0 * epsilon_m + epsilon_p)


# ==============================
# FONCTION DE VÉRIFICATION DES PARAMÈTRES INHABITUELS
# ==============================

def verifier_parametres_inhabituels(
    epsilon_o: float,
    epsilon_p: float,
    epsilon_m: float,
    n_a: float,
    valeur_lambda: float
) -> None:
    avertissements = []
    if epsilon_o < 0 or epsilon_p < 0 or epsilon_m < 0:
        avertissements.append("Avertissement : au moins un epsilon est négatif.")
    if n_a <= 0:
        avertissements.append("Avertissement : N_A est inférieur ou égal à zéro.")
    if valeur_lambda < 0:
        avertissements.append("Avertissement : lambda est négatif.")
    if epsilon_o < epsilon_p:
        avertissements.append("Avertissement : epsilon_O est plus petit que epsilon_P.")
    for message in avertissements:
        print(message)


# ==============================
# FONCTION DE CONSTRUCTION DU DICTIONNAIRE DE PARAMÈTRES
# ==============================

def construire_dictionnaire_parametres(
    epsilon_o: float,
    epsilon_p: float,
    epsilon_m: float,
    n_a: float,
    valeur_lambda: float,
    nom_reference: str
) -> Dict[str, float]:
    return {
        "epsilon_o": epsilon_o,
        "epsilon_p": epsilon_p,
        "epsilon_m": epsilon_m,
        "n_a": n_a,
        "lambda": valeur_lambda,
        "reference_name": nom_reference
    }


# ============================================================
# BLOC 4 — CONNECTIVITÉ MOLÉCULAIRE ET ADAPTATION DES RAYONS
# ============================================================
# Reconstruction du graphe de liaisons à partir des distances et
# de rayons covalents adaptés selon l'hybridation (sp / sp² / sp³)
# détectée par le nombre de voisins.
# ============================================================

# ==============================
# FONCTION DE CONSTRUCTION DE LA CONNECTIVITÉ DE BASE
# ==============================

def construire_connectivite_base(
    numeros_atomiques: List[int],
    matrice_distances: np.ndarray,
    rayons_covalents: Dict[int, float],
    tolerance: float = TOLERANCE_CONNECTIVITE
) -> Dict[int, List[int]]:
    nombre_atomes = len(numeros_atomiques)
    connectivite = {indice_atome: [] for indice_atome in range(1, nombre_atomes + 1)}

    for i in range(nombre_atomes):
        for j in range(i + 1, nombre_atomes):
            z_i = numeros_atomiques[i]
            z_j = numeros_atomiques[j]
            if z_i not in rayons_covalents or z_j not in rayons_covalents:
                continue

            rayon_i = rayons_covalents[z_i]
            rayon_j = rayons_covalents[z_j]
            distance_ij = matrice_distances[i, j]
            seuil_liaison = rayon_i + rayon_j + tolerance

            if distance_ij <= seuil_liaison:
                indice_i = i + 1
                indice_j = j + 1
                connectivite[indice_i].append(indice_j)
                connectivite[indice_j].append(indice_i)

    return connectivite


# ==============================
# FONCTION DE DÉTECTION DU TYPE DE CARBONE
# ==============================

def detecter_type_carbone(nombre_voisins: int) -> str:
    if nombre_voisins == 4:
        return "sp3"
    elif nombre_voisins == 3:
        return "sp2"
    elif nombre_voisins == 2:
        return "sp"
    return "inconnu"


# ==============================
# FONCTION DE CALCUL DES RAYONS ADAPTÉS
# ==============================

def calculer_rayons_adaptes(
    numeros_atomiques: List[int],
    connectivite_provisoire: Dict[int, List[int]]
) -> Dict[int, float]:
    rayons_adaptes = {}

    for indice_atome in range(1, len(numeros_atomiques) + 1):
        z = numeros_atomiques[indice_atome - 1]
        if z != 6:
            rayons_adaptes[indice_atome] = RAYONS_COVALENTS_BASE.get(z, None)
        else:
            nombre_voisins = len(connectivite_provisoire[indice_atome])
            type_carbone = detecter_type_carbone(nombre_voisins)
            if type_carbone == "sp3":
                rayons_adaptes[indice_atome] = 0.76
            elif type_carbone == "sp2":
                rayons_adaptes[indice_atome] = 0.73
            elif type_carbone == "sp":
                rayons_adaptes[indice_atome] = 0.69
            else:
                rayons_adaptes[indice_atome] = 0.76

    return rayons_adaptes


# ==============================
# FONCTION DE CONSTRUCTION DE LA CONNECTIVITÉ AVEC RAYONS ADAPTÉS
# ==============================

def construire_connectivite_avec_rayons_adaptes(
    numeros_atomiques: List[int],
    matrice_distances: np.ndarray,
    rayons_adaptes: Dict[int, float],
    tolerance: float = TOLERANCE_CONNECTIVITE
) -> Dict[int, List[int]]:
    nombre_atomes = len(numeros_atomiques)
    connectivite = {indice_atome: [] for indice_atome in range(1, nombre_atomes + 1)}

    for i in range(nombre_atomes):
        for j in range(i + 1, nombre_atomes):
            indice_i = i + 1
            indice_j = j + 1
            rayon_i = rayons_adaptes[indice_i]
            rayon_j = rayons_adaptes[indice_j]
            if rayon_i is None or rayon_j is None:
                continue

            distance_ij = matrice_distances[i, j]
            seuil_liaison = rayon_i + rayon_j + tolerance
            if distance_ij <= seuil_liaison:
                connectivite[indice_i].append(indice_j)
                connectivite[indice_j].append(indice_i)

    return connectivite


# ==============================
# FONCTION DE CONSTRUCTION AUTOMATIQUE DE LA CONNECTIVITÉ
# ==============================

def construire_connectivite_automatique(
    numeros_atomiques: List[int],
    matrice_distances: np.ndarray,
    tolerance: float = TOLERANCE_CONNECTIVITE
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, float]]:
    connectivite_provisoire = construire_connectivite_base(
        numeros_atomiques=numeros_atomiques,
        matrice_distances=matrice_distances,
        rayons_covalents=RAYONS_COVALENTS_BASE,
        tolerance=tolerance
    )

    rayons_adaptes = calculer_rayons_adaptes(
        numeros_atomiques=numeros_atomiques,
        connectivite_provisoire=connectivite_provisoire
    )

    connectivite_finale = construire_connectivite_avec_rayons_adaptes(
        numeros_atomiques=numeros_atomiques,
        matrice_distances=matrice_distances,
        rayons_adaptes=rayons_adaptes,
        tolerance=tolerance
    )

    return connectivite_provisoire, connectivite_finale, rayons_adaptes


# ==============================
# FONCTION DE CALCUL DES TYPES DE CARBONE
# ==============================

def calculer_types_carbone(
    numeros_atomiques: List[int],
    connectivite_provisoire: Dict[int, List[int]]
) -> Dict[int, str]:
    types_carbone = {}
    for indice_atome in range(1, len(numeros_atomiques) + 1):
        if numeros_atomiques[indice_atome - 1] == 6:
            nombre_voisins = len(connectivite_provisoire[indice_atome])
            types_carbone[indice_atome] = detecter_type_carbone(nombre_voisins)
    return types_carbone


# ==============================
# FONCTION DE FILTRAGE DES CYCLES
# ==============================
def filtrer_cycles_sans_contrainte_hybridation(
    cycles: List[List[int]]
) -> List[List[int]]:
    return cycles


# ==============================
# FONCTION D’AFFICHAGE DES TYPES DE CARBONE
# ==============================

def afficher_types_carbone(
    numeros_atomiques: List[int],
    connectivite_provisoire: Dict[int, List[int]],
    rayons_adaptes: Dict[int, float]
) -> str:
    sortie = " Détection automatique des carbones\n"
    sortie += " ------------------------------\n"

    for indice_atome in range(1, len(numeros_atomiques) + 1):
        z = numeros_atomiques[indice_atome - 1]
        if z == 6:
            nombre_voisins = len(connectivite_provisoire[indice_atome])
            type_carbone = detecter_type_carbone(nombre_voisins)
            rayon = rayons_adaptes[indice_atome]
            sortie += (
                f" Atome {indice_atome:>3d} | C | voisins = {nombre_voisins} "
                f"| type = {type_carbone:<7s} | rayon = {rayon:.2f} Å\n"
            )

    sortie += "\n"
    return sortie


# ==============================
# FONCTION D’AFFICHAGE DE LA CONNECTIVITÉ
# ==============================

def afficher_connectivite(connectivite: Dict[int, List[int]]) -> str:
    sortie = " Connectivité détectée\n"
    sortie += " ------------------------------\n"
    for indice_atome in sorted(connectivite.keys()):
        voisins = sorted(connectivite[indice_atome])
        sortie += f" Atome {indice_atome:>3d} -> voisins : {voisins}\n"
    sortie += "\n"
    return sortie


# ============================================================
# BLOC 5 — DÉTECTION, CANONISATION, FILTRAGE ET CLASSIFICATION DES CYCLES
# ============================================================
# Énumération de tous les cycles atomiques du graphe de connectivité,
# canonisation (élimination des doublons issus des rotations/parcours),
# filtrage, saisie manuelle optionnelle, fusion des cycles auto/manuels
# et génération des cycles fusionnés (différence symétrique des arêtes).
# ============================================================

# ==============================
# FONCTION DE CANONISATION DES CYCLES
# ==============================

# Cette fonction transforme un cycle en représentation canonique
# pour éviter les doublons dus aux rotations et au sens de parcours.
def canoniser_cycle(cycle: List[int]) -> Tuple[int, ...]:
    # Taille du cycle.
    n = len(cycle)

    # Liste de toutes les représentations possibles.
    rotations = []

    # Génère toutes les rotations du cycle direct.
    for i in range(n):
        rotation = cycle[i:] + cycle[:i]
        rotations.append(tuple(rotation))

    # Génère toutes les rotations du cycle inversé.
    cycle_inverse = list(reversed(cycle))
    for i in range(n):
        rotation = cycle_inverse[i:] + cycle_inverse[:i]
        rotations.append(tuple(rotation))

    # Renvoie la plus petite représentation lexicographique.
    return min(rotations)


# ==============================
# FONCTION DE RECHERCHE DES CYCLES
# ==============================

# Cette fonction cherche tous les cycles simples du graphe
# entre une longueur minimale et maximale.
def trouver_cycles(
    connectivite: Dict[int, List[int]],
    longueur_min: int = 3,
    longueur_max: int = 24
) -> List[List[int]]:
    # Ensemble des cycles uniques sous forme canonique.
    cycles_uniques: Set[Tuple[int, ...]] = set()

    # Liste finale des cycles.
    cycles = []

    # Fonction interne de parcours en profondeur.
    def dfs(depart: int, courant: int, chemin: List[int]) -> None:
        # Stoppe si le chemin devient trop long.
        if len(chemin) > longueur_max:
            return

        # Parcourt tous les voisins du sommet courant.
        for voisin in connectivite[courant]:
            # Si on revient au point de départ avec une taille suffisante,
            # un cycle a été trouvé.
            if voisin == depart and len(chemin) >= longueur_min:
                cycle_canonique = canoniser_cycle(chemin)

                if cycle_canonique not in cycles_uniques:
                    cycles_uniques.add(cycle_canonique)
                    cycles.append(list(cycle_canonique))

            # Continue seulement si le voisin n'est pas déjà dans le chemin.
            elif voisin not in chemin:
                dfs(depart, voisin, chemin + [voisin])

    # Lance la recherche depuis chaque atome.
    for atome_depart in sorted(connectivite.keys()):
        dfs(atome_depart, atome_depart, [atome_depart])

    # Renvoie la liste brute des cycles trouvés.
    return cycles


# ==============================
# FONCTION DE FILTRAGE DES HYDROGÈNES
# ==============================

# Cette fonction retire les hydrogènes de la connectivité avant
# la logique de détection des cycles.
def filtrer_connectivite_sans_hydrogenes(
    numeros_atomiques: List[int],
    connectivite: Dict[int, List[int]]
) -> Dict[int, List[int]]:
    # Ensemble des atomes non hydrogène.
    atomes_conserves = set()

    # Identifie les atomes à conserver.
    for indice_atome, numero_atomique in enumerate(numeros_atomiques, start=1):
        if numero_atomique != 1:
            atomes_conserves.add(indice_atome)

    # Construit la connectivité sans hydrogènes.
    connectivite_filtree = {}

    for indice_atome in atomes_conserves:
        voisins_valides = [
            voisin for voisin in connectivite[indice_atome]
            if voisin in atomes_conserves
        ]
        connectivite_filtree[indice_atome] = voisins_valides

    # Renvoie la connectivité filtrée.
    return connectivite_filtree


# ==============================
# FONCTION D’EXTRACTION DES CORDES D’UN CYCLE
# ==============================

# Cette fonction extrait toutes les cordes d'un cycle.
# Une corde est une liaison entre deux sommets du cycle
# qui ne sont pas consécutifs dans le contour du cycle.
# Pour chaque corde, on stocke :
# - les deux atomes
# - la séparation directe en nombre d'arêtes sur le cycle
# - la séparation par l'autre côté du cycle
def extraire_cordes_cycle(
    cycle: List[int],
    connectivite: Dict[int, List[int]]
) -> List[Dict[str, object]]:
    # Taille du cycle.
    n = len(cycle)

    # Liste des cordes détectées.
    cordes = []

    # Parcourt toutes les paires de sommets du cycle.
    for i in range(n):
        for j in range(i + 1, n):
            atome_i = cycle[i]
            atome_j = cycle[j]

            # Les sommets consécutifs appartiennent au contour du cycle
            # et ne doivent pas être comptés comme cordes.
            sont_consecutifs = (
                abs(i - j) == 1 or
                (i == 0 and j == n - 1)
            )

            if sont_consecutifs:
                continue

            # Si une liaison existe entre deux sommets non consécutifs,
            # on a une corde.
            if atome_j in connectivite[atome_i]:
                separation_directe = abs(i - j)
                separation_cycle = n - separation_directe

                cordes.append({
                    "paire_chimique": tuple(sorted((atome_i, atome_j))),
                    "positions_locales": (i, j),
                    "separation_directe": separation_directe,
                    "separation_cycle": separation_cycle
                })

    # Renvoie la liste des cordes.
    return cordes
def construire_ensemble_cordes_cycle(
    indices_cycle: List[int],
    connectivite: Dict[int, List[int]]
) -> Set[Tuple[int, int]]:
    cordes = extraire_cordes_cycle(indices_cycle, connectivite)
    return {
        tuple(sorted(corde["paire_chimique"]))
        for corde in cordes
    }


# ==============================
# FONCTION DE TEST D’UNE CORDE TRAITABLE EN V2
# ==============================

# Une corde est traitable si elle découpe au moins un sous-cycle
# élémentaire de taille 4, 5 ou 6.
def est_corde_traitable_v2(
    separation_directe: int,
    separation_cycle: int
) -> bool:
    taille_sous_cycle_1 = separation_directe + 1
    taille_sous_cycle_2 = separation_cycle + 1

    return (
        taille_sous_cycle_1 in [4, 5, 6] or
        taille_sous_cycle_2 in [4, 5, 6]
    )

def obtenir_valeur_reference_paire(
    paire_chimique: Tuple[int, int],
    indices_cycle: List[int],
    valeurs_orbite: Dict[str, float],
    cordes_cycle: Set[Tuple[int, int]]
) -> float:
    paire_chimique = tuple(sorted(paire_chimique))

    if paire_chimique in cordes_cycle:
        return valeurs_orbite["ortho"]

    position_locale = {
        indice_atome: position
        for position, indice_atome in enumerate(indices_cycle)
    }

    i = position_locale[paire_chimique[0]]
    j = position_locale[paire_chimique[1]]

    distance_topologique = calculer_distance_topologique_cyclique(
        i, j, len(indices_cycle)
    )
    nom_orbite = nommer_orbite(distance_topologique)

    return valeurs_orbite[nom_orbite]


# ==============================
# FONCTION DE DÉTERMINATION DU TYPE DE CYCLE V2
# ==============================

# Cette fonction attribue un type V2 à chaque cycle :
# - ELEMENTAIRE : aucune corde
# - PERIPHERAL  : au moins une corde, et toutes les cordes sont à une seule liaison
# - NON TRAITABLE : tout autre cas
def determiner_type_cycle_v2(
    cycle: List[int],
    connectivite: Dict[int, List[int]]
) -> str:
    cordes = extraire_cordes_cycle(cycle, connectivite)

    if len(cordes) == 0:
        return TYPE_CYCLE_ELEMENTAIRE

    toutes_cordes_valides = all(
        est_corde_a_une_seule_liaison(
            separation_directe=corde["separation_directe"],
            separation_cycle=corde["separation_cycle"]
        )
        for corde in cordes
    )

    if toutes_cordes_valides:
        return TYPE_CYCLE_PERIPHERAL

    return TYPE_CYCLE_NON_TRAITABLE


# ==============================
# FONCTION DE DÉTERMINATION DU STATUT DE CYCLE V2
# ==============================

# Cette fonction détermine si le cycle sera traité ou non en V2.
def determiner_statut_cycle_v2(
    cycle: List[int],
    connectivite: Dict[int, List[int]]
) -> str:
    # Détermine d'abord le type du cycle.
    type_cycle = determiner_type_cycle_v2(cycle, connectivite)

    # Les cycles élémentaires et périphériques traitables sont analysés.
    if type_cycle in [TYPE_CYCLE_ELEMENTAIRE, TYPE_CYCLE_PERIPHERAL]:
        return STATUT_CYCLE_TRAITE

    # Les autres ne sont pas traités.
    return STATUT_CYCLE_NON_TRAITE


# ==============================
# FONCTION DE TRI DES CYCLES
# ==============================

# Cette fonction trie les cycles selon les règles :
# d'abord par taille, puis élémentaire avant peripheral avant non traitable,
# puis indices croissants.
def trier_cycles(cycles_classes: List[Dict[str, object]]) -> List[Dict[str, object]]:
    # Ordre des types de cycle.
    ordre_types = {
        TYPE_CYCLE_ELEMENTAIRE: 0,
        TYPE_CYCLE_PERIPHERAL: 1,
        TYPE_CYCLE_NON_TRAITABLE: 2
    }

    # Trie la liste finale.
    cycles_classes.sort(
        key=lambda x: (
            x["taille_cycle"],
            ordre_types.get(x["type_cycle"], 99),
            x["indices_cycle"]
        )
    )

    # Renvoie la liste triée.
    return cycles_classes


# ==============================
# FONCTION DE PRÉPARATION DE LA LISTE FINALE DES CYCLES
# ==============================

# Cette fonction construit la liste finale des cycles classés,
# triés et enrichis avec leur type, taille, indices, origine et statut.
# Elle attend en entrée une liste d'objets contenant :
# - indices_cycle
# - origine_cycle
def analyser_cycle_v2_legacy(
    matrice_ldm_complete: np.ndarray,
    cycle_info: Dict[str, object],
    parametres_reference: Dict[str, float]
) -> Dict[str, object]:
    numero_cycle = cycle_info["numero_cycle"]
    type_cycle = cycle_info["type_cycle"]
    indices_cycle = cycle_info["indices_cycle"]
    taille_cycle = cycle_info["taille_cycle"]
    statut_cycle = cycle_info["statut_cycle"]
    origine_cycle = cycle_info["origine_cycle"]

    donnees_sous_matrice = preparer_donnees_sous_matrice(
        matrice_ldm_complete=matrice_ldm_complete,
        indices_cycle=indices_cycle
    )
    sous_matrice_ldm = donnees_sous_matrice["sous_matrice_ldm"]

    matrice_reference, orbites, valeurs_orbite, valeur_lambda = construire_matrice_reference_cycle(
        indices_cycle=indices_cycle,
        parametres_reference=parametres_reference,
        connectivite=cycle_info["connectivite_cycle"],
        numeros_atomiques_cycle=cycle_info.get("numeros_atomiques_cycle")
    )

    verifier_coherence_matrice_reference(matrice_reference, valeur_lambda)

    contributions_globales = preparer_contributions_globales(
        sous_matrice_ldm,
        matrice_reference
    )

    contributions_orbites = preparer_contributions_orbites(
        sous_matrice_ldm=sous_matrice_ldm,
        matrice_reference=matrice_reference,
        indices_cycle=indices_cycle,
        orbites=orbites
    )

    return {
        "numero_cycle": numero_cycle,
        "type_cycle": type_cycle,
        "taille_cycle": taille_cycle,
        "indices_cycle": indices_cycle,
        "statut_cycle": statut_cycle,
        "origine_cycle": origine_cycle,
        "reference_name": parametres_reference["reference_name"],
        "n_a": parametres_reference["n_a"],
        "lambda": valeur_lambda,
        "nombre_paires_pi_reference": calculer_nombre_paires_pi_reference(taille_cycle),
        "nombre_orbites": len(orbites),
        "sous_matrice_ldm": sous_matrice_ldm,
        "est_symetrique": donnees_sous_matrice["est_symetrique"],
        "asymetrie_maximale": donnees_sous_matrice["asymetrie_maximale"],
        "matrice_reference": matrice_reference,
        "matrice_difference": contributions_globales["matrice_difference"],
        "distance_frobenius_complete": contributions_globales["distance_frobenius_complete"],
        "contribution_diagonale": contributions_globales["contribution_diagonale"],
        "contribution_hors_diagonale": contributions_globales["contribution_hors_diagonale"],
        "rms_matricielle": contributions_globales["rms_matricielle"],
        "rms_hors_diagonale": contributions_globales["rms_hors_diagonale"],
        "orbites": orbites,
        "valeurs_orbite_reference": valeurs_orbite,
        "contributions_orbites": contributions_orbites["contributions_orbites"],
        "moyennes_reelles_orbites": contributions_orbites["moyennes_reelles_orbites"],
        "tableaux_orbites": contributions_orbites["tableaux_orbites"],
        "tableau_detaille_paires": contributions_orbites["tableau_detaille_paires"]
    }


# ==============================
# FONCTION D’AFFICHAGE DES CYCLES DÉTECTÉS
# ==============================

# Cette fonction génère une chaîne de résumé de tous les cycles détectés.
def afficher_cycles_detectes(cycles_classes: List[Dict[str, object]]) -> str:
    # Initialise la chaîne de sortie.
    sortie = " Cycles détectés automatiquement\n"
    sortie += " ------------------------------\n"

    # Cas où aucun cycle n'a été détecté.
    if len(cycles_classes) == 0:
        sortie += " Aucun cycle détecté.\n\n"
        return sortie

    # Parcourt les cycles classés.
    for element in cycles_classes:
        numero_cycle = element["numero_cycle"]
        type_cycle = element["type_cycle"]
        taille_cycle = element["taille_cycle"]
        indices_cycle = element["indices_cycle"]
        origine_cycle = element["origine_cycle"]
        statut_cycle = element["statut_cycle"]
        nombre_cordes = element["nombre_cordes"]

        sortie += (
            f" Cycle {numero_cycle:>2d} | type = {type_cycle:<13s} "
            f"| taille = {taille_cycle:<2d} | indices = {indices_cycle} "
            f"| origine = {origine_cycle:<11s} "
            f"| cordes = {nombre_cordes:<2d} "
            f"| statut = {statut_cycle}\n"
        )

    # Ajoute une ligne vide finale.
    sortie += "\n"

    # Renvoie la chaîne.
    return sortie


# ==============================
# FONCTIONS DE VALIDATION ET DE SAISIE DES CYCLES MANUELS
# ==============================

def est_corde_a_une_seule_liaison(
    separation_directe: int,
    separation_cycle: int
) -> bool:
    """Compatibilité avec l'ancienne logique V2.

    Une corde est dite ici "à une seule liaison" si elle découpe
    au moins un sous-cycle de taille 4, 5 ou 6, ce qui correspond
    à la règle utilisée dans est_corde_traitable_v2().
    """
    return est_corde_traitable_v2(separation_directe, separation_cycle)


def valider_indices_cycle_manuels(
    cycle: List[int],
    connectivite: Dict[int, List[int]],
    nombre_total_atomes: int
) -> Tuple[bool, str]:
    """Validation V06 : les cycles manuels sont facultatifs et peuvent être
    saisis même s'ils ne respectent pas la connectivité détectée.

    On vérifie seulement :
    - au moins 3 indices
    - pas de doublons
    - indices dans les bornes
    """
    if len(cycle) < 3:
        return False, "Un cycle doit contenir au moins 3 indices."

    if len(set(cycle)) != len(cycle):
        return False, "Un cycle ne doit pas contenir de doublons."

    for indice in cycle:
        if indice < 1 or indice > nombre_total_atomes:
            return False, f"Indice hors bornes : {indice}."

    return True, "OK"


def parser_ligne_cycle_manuel(entree: str) -> List[int]:
    texte = entree.replace(';', ' ').replace(',', ' ')
    morceaux = [m for m in texte.split() if m]
    return [int(morceau) for morceau in morceaux]


def demander_cycles_manuels_optionnels(
    connectivite: Dict[int, List[int]],
    nombre_total_atomes: int
) -> List[List[int]]:
    print()
    print("Saisie optionnelle de cycles manuels.")
    print("- Appuyez simplement sur Entrée pour ne rien ajouter.")
    print("- Sinon, entrez les indices d'un cycle séparés par des espaces ou des virgules.")
    print("- Entrez une ligne vide quand vous avez terminé.")
    print("- V09 : un cycle manuel est accepté même s'il ne respecte pas la connectivité détectée.")
    print("       Il sera traité selon sa taille, ses orbites et ses cordes internes éventuelles.")
    print("       Même manuel libre, il peut donc contribuer aux cas fusionnés/périphériques.")

    cycles_manuels: List[List[int]] = []
    numero_saisie = 1

    while True:
        invite = f"Cycle manuel #{numero_saisie} : "
        entree = input(invite).strip()

        if entree == "":
            break

        try:
            cycle = parser_ligne_cycle_manuel(entree)
        except ValueError:
            print("Erreur : seuls des entiers sont autorisés.")
            continue

        cycle_canonique = list(canoniser_cycle(cycle))
        est_valide, message = valider_indices_cycle_manuels(
            cycle=cycle_canonique,
            connectivite=connectivite,
            nombre_total_atomes=nombre_total_atomes
        )

        if not est_valide:
            print(f"Cycle refusé : {message}")
            continue

        if cycle_canonique in cycles_manuels:
            print("Cycle déjà ajouté.")
            continue

        cycles_manuels.append(cycle_canonique)
        print(f"Cycle manuel ajouté : {cycle_canonique}")
        numero_saisie += 1

    print()
    return cycles_manuels


def fusionner_cycles_automatiques_et_manuels(
    cycles_automatiques: List[List[int]],
    cycles_manuels: List[List[int]]
) -> List[Dict[str, object]]:
    cycles_auto_canoniques = {tuple(canoniser_cycle(cycle)) for cycle in cycles_automatiques}
    cycles_manuels_canoniques = {tuple(canoniser_cycle(cycle)) for cycle in cycles_manuels}

    tous_les_cycles = sorted(
        cycles_auto_canoniques | cycles_manuels_canoniques,
        key=lambda cycle: (len(cycle), cycle)
    )

    cycles_fusionnes: List[Dict[str, object]] = []
    for cycle in tous_les_cycles:
        dans_auto = cycle in cycles_auto_canoniques
        dans_manuel = cycle in cycles_manuels_canoniques

        if dans_auto and dans_manuel:
            origine = ORIGINE_AUTO_MANUEL
        elif dans_auto:
            origine = ORIGINE_AUTO
        else:
            origine = ORIGINE_MANUEL

        cycles_fusionnes.append({
            "indices_cycle": list(cycle),
            "origine_cycle": origine
        })

    return cycles_fusionnes


# ==============================
# FONCTIONS DE GÉNÉRATION DE CYCLES FUSIONNÉS À PARTIR DES CYCLES AUTO + MANUELS
# ==============================

def arete_cycle_key(a: int, b: int) -> Tuple[int, int]:
    return tuple(sorted((a, b)))


def aretes_cycle_non_orientees(cycle: List[int]) -> Set[Tuple[int, int]]:
    return {arete_cycle_key(a, b) for a, b in zip(cycle, cycle[1:] + cycle[:1])}


def reconstruire_cycle_depuis_aretes(aretes: Set[Tuple[int, int]]) -> object:
    """Reconstruit un cycle simple ordonné depuis un ensemble d'arêtes.

    Retourne None si les arêtes ne forment pas un seul contour cyclique simple.
    Cette règle est volontairement topologique : elle ne vérifie pas que chaque
    arête existe dans la connectivité moléculaire détectée. Elle permet donc
    aux cycles manuels d'agir comme blocs de fusion.
    """
    if len(aretes) < 3:
        return None

    adj: Dict[int, List[int]] = {}
    for a, b in aretes:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    if any(len(voisins) != 2 for voisins in adj.values()):
        return None

    depart = min(adj)
    cycle = [depart]
    precedent = None
    courant = depart

    while True:
        voisins = adj[courant]
        suivant = voisins[0] if voisins[0] != precedent else voisins[1]
        if suivant == depart:
            break
        if suivant in cycle:
            return None
        cycle.append(suivant)
        precedent, courant = courant, suivant
        if len(cycle) > len(adj):
            return None

    if len(cycle) != len(adj):
        return None

    return list(canoniser_cycle(cycle))


def generer_cycles_fusionnes_depuis_cycles(
    cycles_base: List[Dict[str, object]],
    longueur_min: int = 3,
    longueur_max: int = 48
) -> List[Dict[str, object]]:
    """Génère A+B, A+B+C, etc. depuis cycles auto et cycles manuels.

    Algorithme repris du programme HOMAg-MD.auto.V12 : deux cycles qui partagent
    au moins une arête sont fusionnés par différence symétrique des arêtes.
    Les arêtes communes deviennent des cordes internes supprimées, et le contour
    restant devient un nouveau cycle périphérique/fusionné.
    """
    cycles_connus: Dict[Tuple[int, ...], Dict[str, object]] = {}
    aretes_par_cycle: Dict[Tuple[int, ...], Set[Tuple[int, int]]] = {}

    for item in cycles_base:
        can = tuple(canoniser_cycle(list(item["indices_cycle"])))
        item_can = {
            "indices_cycle": list(can),
            "origine_cycle": item.get("origine_cycle", ORIGINE_AUTO),
        }
        cycles_connus[can] = item_can
        aretes_par_cycle[can] = aretes_cycle_non_orientees(list(can))

    cycles_ajoutes: List[Dict[str, object]] = []
    changement = True

    while changement:
        changement = False
        cles = list(aretes_par_cycle.keys())
        for i in range(len(cles)):
            for j in range(i + 1, len(cles)):
                c1, c2 = cles[i], cles[j]
                e1, e2 = aretes_par_cycle[c1], aretes_par_cycle[c2]

                if not (e1 & e2):
                    continue

                aretes_fusion = e1 ^ e2
                cycle_fusion = reconstruire_cycle_depuis_aretes(aretes_fusion)
                if cycle_fusion is None:
                    continue
                if not (longueur_min <= len(cycle_fusion) <= longueur_max):
                    continue

                can_fusion = tuple(canoniser_cycle(cycle_fusion))
                if can_fusion in cycles_connus:
                    continue

                item_fusion = {
                    "indices_cycle": list(can_fusion),
                    "origine_cycle": ORIGINE_FUSED,
                    "source_cycle": "fused_from_auto_or_manual",
                }
                cycles_connus[can_fusion] = item_fusion
                aretes_par_cycle[can_fusion] = aretes_cycle_non_orientees(list(can_fusion))
                cycles_ajoutes.append(item_fusion)
                changement = True

    cycles_ajoutes.sort(key=lambda x: (len(x["indices_cycle"]), x["indices_cycle"]))
    return cycles_ajoutes


def fusionner_cycles_auto_manuels_et_generer_fusionnes(
    cycles_automatiques: List[List[int]],
    cycles_manuels: List[List[int]]
) -> List[Dict[str, object]]:
    cycles_base = fusionner_cycles_automatiques_et_manuels(
        cycles_automatiques=cycles_automatiques,
        cycles_manuels=cycles_manuels
    )

    cycles_fusionnes = generer_cycles_fusionnes_depuis_cycles(cycles_base)

    cycles_par_canonique: Dict[Tuple[int, ...], Dict[str, object]] = {}
    for item in cycles_base + cycles_fusionnes:
        can = tuple(canoniser_cycle(list(item["indices_cycle"])))
        if can not in cycles_par_canonique:
            item2 = dict(item)
            item2["indices_cycle"] = list(can)
            cycles_par_canonique[can] = item2

    return sorted(
        cycles_par_canonique.values(),
        key=lambda item: (len(item["indices_cycle"]), item["indices_cycle"], item.get("origine_cycle", ""))
    )


# ============================================================
# BLOC 6 — CONSTRUCTION DE LA MATRICE LDM DE RÉFÉRENCE
# ============================================================
# La matrice de référence LDM_ref représente la molécule "parfaitement
# aromatique" de même topologie que le cycle étudié : les orbites
# (ortho / meta / para / plus lointaines) reçoivent les valeurs
# theoriques epsilon_o, epsilon_m, epsilon_p...
# ============================================================

# ==============================
# FONCTION DE CALCUL DU NOMBRE DE PAIRES PI DE RÉFÉRENCE
# ==============================

def calculer_nombre_paires_pi_reference(taille_cycle: int) -> int:
    if taille_cycle % 2 == 0:
        return taille_cycle // 2
    return (taille_cycle // 2) + 1


# ==============================
# FONCTION DE CALCUL DE LA DISTANCE TOPOLOGIQUE CYCLIQUE
# ==============================

def calculer_distance_topologique_cyclique(i: int, j: int, n: int) -> int:
    return min(abs(i - j), n - abs(i - j))


# ==============================
# FONCTION DE NOMMAGE D’UNE ORBITE
# ==============================

def nommer_orbite(distance_topologique: int) -> str:
    if distance_topologique == 1:
        return "ortho"
    if distance_topologique == 2:
        return "meta"
    if distance_topologique == 3:
        return "para"
    return f"d{distance_topologique}"


# ==============================
# FONCTION DE GÉNÉRATION DES ORBITES DE PAIRES DU CYCLE
# ==============================
def generer_orbites_paires_cycle(
    indices_cycle: List[int]
) -> Dict[str, List[Tuple[int, int]]]:
    n = len(indices_cycle)
    orbites: Dict[str, List[Tuple[int, int]]] = {}

    for i in range(n):
        for j in range(i + 1, n):
            distance_topologique = calculer_distance_topologique_cyclique(i, j, n)
            nom_orbite = nommer_orbite(distance_topologique)
            paire = tuple(sorted((indices_cycle[i], indices_cycle[j])))
            orbites.setdefault(nom_orbite, []).append(paire)

    for nom_orbite in orbites:
        orbites[nom_orbite] = sorted(list(set(orbites[nom_orbite])))

    return dict(
        sorted(
            orbites.items(),
            key=lambda item: (orbite_vers_distance_topologique(item[0]), item[0])
        )
    )


# ==============================
# FONCTION DE CONVERSION NOM ORBITE → DISTANCE TOPOLOGIQUE
# ==============================

def orbite_vers_distance_topologique(nom_orbite: str) -> int:
    if nom_orbite == "ortho":
        return 1
    if nom_orbite == "meta":
        return 2
    if nom_orbite == "para":
        return 3
    if nom_orbite.startswith("d"):
        return int(nom_orbite[1:])
    raise ValueError(f"Nom d'orbite inconnu : {nom_orbite}")


# ==============================
# FONCTION DE CALCUL DE LA VALEUR DE RÉFÉRENCE PAR ORBITE
# ==============================
def calculer_valeurs_reference_par_orbite(
    indices_cycle: List[int],
    parametres_reference: Dict[str, float]
) -> Dict[str, float]:
    orbites = generer_orbites_paires_cycle(indices_cycle)
    nombre_orbites = len(orbites)
    nombre_paires_pi = calculer_nombre_paires_pi_reference(len(indices_cycle))
    paires_pi_par_orbite = nombre_paires_pi / nombre_orbites

    valeurs_orbite: Dict[str, float] = {}
    for nom_orbite, paires_orbite in orbites.items():
        nombre_paires_orbite = len(paires_orbite)
        delta_pi_par_paire = paires_pi_par_orbite / nombre_paires_orbite

        if nom_orbite == "ortho":
            epsilon_orbite = 0.5 * (1.0 + delta_pi_par_paire)
        else:
            epsilon_orbite = 0.5 * delta_pi_par_paire

        valeurs_orbite[nom_orbite] = float(epsilon_orbite)

    return valeurs_orbite


# ==============================
# FONCTION DE CONSTRUCTION DE LA MATRICE DE RÉFÉRENCE GÉNÉRALE
# ==============================
#
# Généralisation V05 : N_A est désormais site-dépendant (N_A[k] = Z_k = numéro
# atomique de l'atome à la position k du cycle). Les ε_O/ε_P/ε_M restent
# globaux au cycle et sont calculés à partir de la topologie comme avant.
# La diagonale devient : λ_k = Z_k − Σ_{j≠k} ε(k,j).
# Pour un cycle homo-atomique de C (benzène), on retombe exactement sur la
# version précédente (λ uniforme = 6 − somme_hors_diagonale_ligne).
#
# Le paramètre numeros_atomiques_cycle est OPTIONNEL pour rétro-compatibilité :
# s'il est None, la diagonale uniforme historique est utilisée avec
# parametres_reference["n_a"].
def construire_matrice_reference_cycle(
    indices_cycle: List[int],
    parametres_reference: Dict[str, float],
    connectivite: Dict[int, List[int]] = None,
    numeros_atomiques_cycle: List[int] = None
) -> Tuple[np.ndarray, Dict[str, List[Tuple[int, int]]], Dict[str, float], object]:
    n = len(indices_cycle)

    orbites = generer_orbites_paires_cycle(indices_cycle)
    valeurs_orbite = calculer_valeurs_reference_par_orbite(
        indices_cycle=indices_cycle,
        parametres_reference=parametres_reference
    )

    somme_hors_diagonale_ligne = 0.0
    for nom_orbite, valeur in valeurs_orbite.items():
        distance_topologique = orbite_vers_distance_topologique(nom_orbite)
        multiplicite_ligne = 1 if (n % 2 == 0 and distance_topologique == n // 2) else 2
        somme_hors_diagonale_ligne += multiplicite_ligne * valeur

    # Diagonale : uniforme si pas de Z fournis (cas legacy), sinon site-dépendante.
    if numeros_atomiques_cycle is None:
        n_a_par_site = [float(parametres_reference["n_a"])] * n
    else:
        if len(numeros_atomiques_cycle) != n:
            raise ValueError(
                f"Erreur : numeros_atomiques_cycle ({len(numeros_atomiques_cycle)}) "
                f"ne correspond pas à indices_cycle ({n})."
            )
        n_a_par_site = [float(z) for z in numeros_atomiques_cycle]

    valeur_lambda_par_site = [
        n_a_par_site[k] - somme_hors_diagonale_ligne
        for k in range(n)
    ]

    matrice_reference = np.zeros((n, n), dtype=float)
    for k in range(n):
        matrice_reference[k, k] = valeur_lambda_par_site[k]

    position_locale = {
        indice_atome: position
        for position, indice_atome in enumerate(indices_cycle)
    }

    cordes_cycle = set()
    if connectivite is not None:
        cordes_cycle = construire_ensemble_cordes_cycle(indices_cycle, connectivite)

    for i in range(n):
        for j in range(i + 1, n):
            paire = tuple(sorted((indices_cycle[i], indices_cycle[j])))
            valeur = obtenir_valeur_reference_paire(
                paire_chimique=paire,
                indices_cycle=indices_cycle,
                valeurs_orbite=valeurs_orbite,
                cordes_cycle=cordes_cycle
            )
            matrice_reference[i, j] = valeur
            matrice_reference[j, i] = valeur

    # Cas homo-atomique : lambda unique (float) ; cas hétéro : liste de floats.
    if len(set(n_a_par_site)) == 1:
        valeur_lambda_retour = float(valeur_lambda_par_site[0])
    else:
        valeur_lambda_retour = [float(x) for x in valeur_lambda_par_site]

    return matrice_reference, orbites, valeurs_orbite, valeur_lambda_retour


# ==============================
# FONCTION DE VÉRIFICATION DE LA COHÉRENCE DE LA MATRICE DE RÉFÉRENCE
# ==============================

def verifier_coherence_matrice_reference(
    matrice_reference: np.ndarray,
    valeur_lambda
) -> None:
    if matrice_reference.shape[0] != matrice_reference.shape[1]:
        raise ValueError("Erreur : la matrice de référence n'est pas carrée.")
    if not np.allclose(matrice_reference, matrice_reference.T, atol=TOLERANCE_SYMETRIE):
        raise ValueError("Erreur : la matrice de référence n'est pas symétrique.")
    diagonale = np.diag(matrice_reference)
    # valeur_lambda peut être un float (homo-atomique) ou une liste/array (hétéro).
    if isinstance(valeur_lambda, (list, tuple, np.ndarray)):
        valeur_lambda_array = np.asarray(valeur_lambda, dtype=float)
        if valeur_lambda_array.shape != diagonale.shape:
            raise ValueError(
                "Erreur : taille de valeur_lambda incohérente avec la diagonale."
            )
        if not np.allclose(diagonale, valeur_lambda_array, atol=TOLERANCE_SYMETRIE):
            raise ValueError("Erreur : la diagonale de la matrice de référence est incohérente.")
    else:
        if not np.allclose(diagonale, valeur_lambda, atol=TOLERANCE_SYMETRIE):
            raise ValueError("Erreur : la diagonale de la matrice de référence est incohérente.")


# ==============================
# FONCTION D’AFFICHAGE DE LA MATRICE DE RÉFÉRENCE
# ==============================

def afficher_matrice_reference(matrice_reference: np.ndarray) -> str:
    sortie = f" Matrice de référence {matrice_reference.shape[0]}x{matrice_reference.shape[1]}\n"
    sortie += " ------------------------------\n"
    for ligne in matrice_reference:
        sortie += " " + " ".join(f"{valeur:.6f}" for valeur in ligne) + "\n"
    sortie += "\n"
    return sortie


# ============================================================
# BLOC 7 — EXTRACTION DES SOUS-MATRICES LDM ET CONTRÔLE DE SYMÉTRIE
# ============================================================
# Découpe la matrice LDM complète pour ne garder que les lignes/colonnes
# des atomes du cycle, puis vérifie la symétrie (LDM = LDM^T à
# TOLERANCE_SYMETRIE près).
# ============================================================

# ==============================
# FONCTION D’EXTRACTION DE LA SOUS-MATRICE LDM D’UN CYCLE
# ==============================

def extraire_sous_matrice_ldm(
    matrice_ldm_complete: np.ndarray,
    indices_cycle: List[int]
) -> np.ndarray:
    indices_python = [indice - 1 for indice in indices_cycle]
    return matrice_ldm_complete[np.ix_(indices_python, indices_python)]


# ==============================
# FONCTION DE TEST DE SYMÉTRIE DE LA SOUS-MATRICE
# ==============================

def tester_symetrie_sous_matrice(
    sous_matrice_ldm: np.ndarray,
    tolerance: float = TOLERANCE_SYMETRIE
) -> bool:
    return np.allclose(sous_matrice_ldm, sous_matrice_ldm.T, atol=tolerance)


# ==============================
# FONCTION DE CALCUL DE L’ASYMÉTRIE MAXIMALE
# ==============================

def calculer_asymetrie_maximale(sous_matrice_ldm: np.ndarray) -> float:
    matrice_ecarts = np.abs(sous_matrice_ldm - sous_matrice_ldm.T)
    return float(np.max(matrice_ecarts))


# ==============================
# FONCTION DE PRÉPARATION DES DONNÉES DE SOUS-MATRICE
# ==============================

def preparer_donnees_sous_matrice(
    matrice_ldm_complete: np.ndarray,
    indices_cycle: List[int]
) -> Dict[str, object]:
    sous_matrice_ldm = extraire_sous_matrice_ldm(matrice_ldm_complete, indices_cycle)
    est_symetrique = tester_symetrie_sous_matrice(sous_matrice_ldm, TOLERANCE_SYMETRIE)
    asymetrie_maximale = calculer_asymetrie_maximale(sous_matrice_ldm)
    return {
        "sous_matrice_ldm": sous_matrice_ldm,
        "est_symetrique": est_symetrique,
        "asymetrie_maximale": asymetrie_maximale
    }


# ==============================
# FONCTION D’AFFICHAGE DE LA SOUS-MATRICE LDM
# ==============================

def afficher_sous_matrice_ldm(sous_matrice_ldm: np.ndarray) -> str:
    sortie = " Sous-matrice LDM réelle\n"
    sortie += " ------------------------------\n"
    for ligne in sous_matrice_ldm:
        sortie += " " + " ".join(f"{valeur:.6f}" for valeur in ligne) + "\n"
    sortie += "\n"
    return sortie


def extraire_sous_matrice_distances(
    matrice_distances_complete: np.ndarray,
    indices_cycle: List[int]
) -> np.ndarray:
    indices_python = [indice - 1 for indice in indices_cycle]
    return matrice_distances_complete[np.ix_(indices_python, indices_python)]


# ============================================================
# BLOC 8 — DISTANCES, MATRICE DIFFÉRENCE ET CONTRIBUTIONS GLOBALES
# ============================================================
# Calcule LDM_ref - LDM_obs sur le cycle, en extrait des scalaires
# globaux (norme de Frobenius, RMS, ratios diagonaux/hors-diagonaux)
# qui serviront aux descripteurs LDM synthétiques.
# ============================================================

# ==============================
# FONCTION DE CALCUL DE LA MATRICE DIFFÉRENCE
# ==============================

def calculer_matrice_difference(
    sous_matrice_ldm: np.ndarray,
    matrice_reference: np.ndarray
) -> np.ndarray:
    return sous_matrice_ldm - matrice_reference


# ==============================
# FONCTION DE CALCUL DE LA DISTANCE DE FROBENIUS COMPLÈTE
# ==============================

def calculer_distance_frobenius_complete(
    sous_matrice_ldm: np.ndarray,
    matrice_reference: np.ndarray
) -> float:
    matrice_difference = calculer_matrice_difference(sous_matrice_ldm, matrice_reference)
    return float(np.linalg.norm(matrice_difference, ord="fro"))


# ==============================
# FONCTION DE CALCUL DE LA CONTRIBUTION DIAGONALE
# ==============================

def calculer_contribution_diagonale(
    sous_matrice_ldm: np.ndarray,
    matrice_reference: np.ndarray
) -> float:
    difference_diagonale = np.diag(sous_matrice_ldm) - np.diag(matrice_reference)
    return float(np.linalg.norm(difference_diagonale))


# ==============================
# FONCTION DE CALCUL DE LA CONTRIBUTION HORS DIAGONALE
# ==============================

def calculer_contribution_hors_diagonale(
    sous_matrice_ldm: np.ndarray,
    matrice_reference: np.ndarray
) -> float:
    matrice_difference = calculer_matrice_difference(sous_matrice_ldm, matrice_reference)
    matrice_hors_diagonale = matrice_difference.copy()
    np.fill_diagonal(matrice_hors_diagonale, 0.0)
    return float(np.linalg.norm(matrice_hors_diagonale, ord="fro"))

def calculer_rms_matricielle(
    sous_matrice_ldm: np.ndarray,
    matrice_reference: np.ndarray
) -> float:
    matrice_difference = calculer_matrice_difference(sous_matrice_ldm, matrice_reference)
    n = matrice_difference.shape[0]
    if n == 0:
        return 0.0
    return float(np.linalg.norm(matrice_difference, ord="fro") / n)


def calculer_rms_hors_diagonale(
    sous_matrice_ldm: np.ndarray,
    matrice_reference: np.ndarray
) -> float:
    matrice_difference = calculer_matrice_difference(sous_matrice_ldm, matrice_reference)
    n = matrice_difference.shape[0]

    if n <= 1:
        return 0.0

    matrice_hors_diagonale = matrice_difference.copy()
    np.fill_diagonal(matrice_hors_diagonale, 0.0)

    nombre_termes_hors_diagonale = n * (n - 1)
    norme_frobenius_hors_diagonale = np.linalg.norm(matrice_hors_diagonale, ord="fro")

    return float(norme_frobenius_hors_diagonale / math.sqrt(nombre_termes_hors_diagonale))


# ==============================
# FONCTION DE PRÉPARATION DES CONTRIBUTIONS GLOBALES
# ==============================

def preparer_contributions_globales(
    sous_matrice_ldm: np.ndarray,
    matrice_reference: np.ndarray
) -> Dict[str, object]:
    matrice_difference = calculer_matrice_difference(sous_matrice_ldm, matrice_reference)
    distance_frobenius_complete = calculer_distance_frobenius_complete(sous_matrice_ldm, matrice_reference)
    contribution_diagonale = calculer_contribution_diagonale(sous_matrice_ldm, matrice_reference)
    contribution_hors_diagonale = calculer_contribution_hors_diagonale(sous_matrice_ldm, matrice_reference)
    rms_matricielle = calculer_rms_matricielle(sous_matrice_ldm, matrice_reference)
    rms_hors_diagonale = calculer_rms_hors_diagonale(sous_matrice_ldm, matrice_reference)

    return {
        "matrice_difference": matrice_difference,
        "distance_frobenius_complete": distance_frobenius_complete,
        "contribution_diagonale": contribution_diagonale,
        "contribution_hors_diagonale": contribution_hors_diagonale,
        "rms_matricielle": rms_matricielle,
        "rms_hors_diagonale": rms_hors_diagonale
    }


# ==============================
# FONCTION D’AFFICHAGE DE LA MATRICE DIFFÉRENCE
# ==============================

def afficher_matrice_difference(matrice_difference: np.ndarray) -> str:
    sortie = " Matrice différence : Delta = LDM réelle - LDM référence\n"
    sortie += " ------------------------------\n"
    for ligne in matrice_difference:
        sortie += " " + " ".join(f"{valeur:.6f}" for valeur in ligne) + "\n"
    sortie += "\n"
    return sortie


# ============================================================
# BLOC 8B — ANALYSE SPATIALE PONDÉRÉE ET HOMOGÉNÉITÉ PAR ORBITES
# ============================================================
# Pondération de chaque contribution par la distance R_ij et
# regroupement par orbites topologiques. Fournit S(cycle), S_hom
# et des indicateurs de dispersion.
# ============================================================

def construire_matrice_q_ponderee(
    sous_matrice_ldm: np.ndarray,
    sous_matrice_distances: np.ndarray
) -> np.ndarray:
    n = sous_matrice_ldm.shape[0]
    matrice_q = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                matrice_q[i, j] = 0.0
            else:
                distance_ij = sous_matrice_distances[i, j]
                if abs(distance_ij) <= 1.0e-15:
                    matrice_q[i, j] = 0.0
                else:
                    matrice_q[i, j] = sous_matrice_ldm[i, j] / distance_ij

    return matrice_q


def calculer_frobenius_hors_diagonale_matrice(matrice: np.ndarray) -> float:
    matrice_hors_diagonale = matrice.copy()
    np.fill_diagonal(matrice_hors_diagonale, 0.0)
    return float(np.linalg.norm(matrice_hors_diagonale, ord="fro"))


def calculer_rms_hors_diagonale_matrice(matrice: np.ndarray) -> float:
    n = matrice.shape[0]
    if n <= 1:
        return 0.0

    matrice_hors_diagonale = matrice.copy()
    np.fill_diagonal(matrice_hors_diagonale, 0.0)

    nombre_termes = n * (n - 1)
    norme = np.linalg.norm(matrice_hors_diagonale, ord="fro")
    return float(norme / math.sqrt(nombre_termes))


def calculer_nombre_paires_uniques_cycle(taille_cycle: int) -> int:
    return (taille_cycle * (taille_cycle - 1)) // 2


def calculer_valeurs_x_orbite(
    sous_matrice_ldm: np.ndarray,
    sous_matrice_distances: np.ndarray,
    indices_cycle: List[int],
    liste_paires_chimiques: List[Tuple[int, int]]
) -> List[float]:
    valeurs_x = []

    for paire_chimique in liste_paires_chimiques:
        i_local, j_local = convertir_paire_chimique_en_paire_locale(paire_chimique, indices_cycle)
        valeur_ldm = float(sous_matrice_ldm[i_local, j_local])
        distance_ij = float(sous_matrice_distances[i_local, j_local])
        valeurs_x.append(valeur_ldm * distance_ij)

    return valeurs_x


def calculer_moyenne_liste(valeurs: List[float]) -> float:
    if len(valeurs) == 0:
        return 0.0
    return float(sum(valeurs) / len(valeurs))


def calculer_ecart_type_population(valeurs: List[float]) -> float:
    if len(valeurs) == 0:
        return 0.0

    moyenne = calculer_moyenne_liste(valeurs)
    variance = sum((valeur - moyenne) ** 2 for valeur in valeurs) / len(valeurs)
    return float(math.sqrt(variance))


def calculer_facteur_homogeneite_orbite(
    s_orb_k: float,
    sigma_k: float
) -> float:
    if s_orb_k <= 1.0e-15:
        return 0.0
    return float(1.0 / (1.0 + sigma_k / s_orb_k))


def preparer_indice_homogeneite_cycle(
    sous_matrice_ldm: np.ndarray,
    sous_matrice_distances: np.ndarray,
    indices_cycle: List[int],
    orbites: Dict[str, List[Tuple[int, int]]]
) -> Dict[str, object]:
    s_orbites: Dict[str, float] = {}
    sigmas_orbites: Dict[str, float] = {}
    h_orbites: Dict[str, float] = {}
    tableaux_x_orbites: Dict[str, List[float]] = {}

    somme_ponderee = 0.0
    nombre_total_paires = calculer_nombre_paires_uniques_cycle(len(indices_cycle))

    for nom_orbite, paires in orbites.items():
        valeurs_x = calculer_valeurs_x_orbite(
            sous_matrice_ldm=sous_matrice_ldm,
            sous_matrice_distances=sous_matrice_distances,
            indices_cycle=indices_cycle,
            liste_paires_chimiques=paires
        )

        s_orb_k = calculer_moyenne_liste(valeurs_x)
        sigma_k = calculer_ecart_type_population(valeurs_x)
        h_k = calculer_facteur_homogeneite_orbite(s_orb_k, sigma_k)

        tableaux_x_orbites[nom_orbite] = valeurs_x
        s_orbites[nom_orbite] = s_orb_k
        sigmas_orbites[nom_orbite] = sigma_k
        h_orbites[nom_orbite] = h_k

        somme_ponderee += len(paires) * s_orb_k * h_k

    if nombre_total_paires <= 0:
        s_cycle_hom = 0.0
    else:
        s_cycle_hom = float(somme_ponderee / nombre_total_paires)

    return {
        "s_orbites": s_orbites,
        "sigmas_orbites": sigmas_orbites,
        "h_orbites": h_orbites,
        "tableaux_x_orbites": tableaux_x_orbites,
        "nombre_total_paires": nombre_total_paires,
        "s_cycle_hom": s_cycle_hom
    }


def preparer_analyse_spatiale_cycle(
    sous_matrice_ldm: np.ndarray,
    matrice_reference: np.ndarray,
    sous_matrice_distances: np.ndarray,
    indices_cycle: List[int],
    orbites: Dict[str, List[Tuple[int, int]]],
    taille_cycle: int,
    numeros_atomiques_cycle: List[int] = None
) -> Dict[str, object]:
    matrice_q_reelle = construire_matrice_q_ponderee(
        sous_matrice_ldm=sous_matrice_ldm,
        sous_matrice_distances=sous_matrice_distances
    )

    matrice_q_reference = construire_matrice_q_ponderee(
        sous_matrice_ldm=matrice_reference,
        sous_matrice_distances=sous_matrice_distances
    )

    matrice_q_difference = matrice_q_reelle - matrice_q_reference
    q_frob_offdiag = calculer_frobenius_hors_diagonale_matrice(matrice_q_difference)
    q_rms_offdiag = calculer_rms_hors_diagonale_matrice(matrice_q_difference)

    homogeneite = preparer_indice_homogeneite_cycle(
        sous_matrice_ldm=sous_matrice_ldm,
        sous_matrice_distances=sous_matrice_distances,
        indices_cycle=indices_cycle,
        orbites=orbites
    )

    n_ref_cycle = calculer_n_ref_cycle(taille_cycle, numeros_atomiques_cycle)
    donnees_ct = calculer_transfert_charge_cycle(
        sous_matrice_ldm=sous_matrice_ldm,
        n_ref_cycle=n_ref_cycle
    )

    return {
        "sous_matrice_distances": sous_matrice_distances,
        "matrice_q_reelle": matrice_q_reelle,
        "matrice_q_reference": matrice_q_reference,
        "matrice_q_difference": matrice_q_difference,
        "q_frob_offdiag": q_frob_offdiag,
        "q_rms_offdiag": q_rms_offdiag,
        "nombre_total_paires": homogeneite["nombre_total_paires"],
        "s_orbites": homogeneite["s_orbites"],
        "sigmas_orbites": homogeneite["sigmas_orbites"],
        "h_orbites": homogeneite["h_orbites"],
        "tableaux_x_orbites": homogeneite["tableaux_x_orbites"],
        "s_cycle_hom": homogeneite["s_cycle_hom"],
        "n_ref_cycle": n_ref_cycle,
        "population_totale_reelle": donnees_ct["population_totale_reelle"],
        "population_totale_reference": donnees_ct["population_totale_reference"],
        "ct_absolu": donnees_ct["ct_absolu"],
        "ct_relatif_pourcent": donnees_ct["ct_relatif_pourcent"]
    }


# ============================================================
# BLOC 9 — CONTRIBUTIONS PAR ORBITES ET MOYENNES RÉELLES
# ============================================================
# Décompose LDM_obs par orbite topologique (ortho, meta, para, ...)
# et calcule les moyennes / écarts par orbite pour comparaison au
# modèle EG.
# ============================================================

# ==============================
# FONCTION DE CONVERSION PAIRE CHIMIQUE → PAIRE LOCALE
# ==============================

def convertir_paire_chimique_en_paire_locale(
    paire_chimique: Tuple[int, int],
    indices_cycle: List[int]
) -> Tuple[int, int]:
    positions_locales = {indice_chimique: position_locale for position_locale, indice_chimique in enumerate(indices_cycle)}
    atome_i, atome_j = paire_chimique
    return positions_locales[atome_i], positions_locales[atome_j]


# ==============================
# FONCTION D’EXTRACTION DES VALEURS D’UNE LISTE DE PAIRES
# ==============================

def extraire_valeurs_paires(
    sous_matrice_ldm: np.ndarray,
    matrice_reference: np.ndarray,
    indices_cycle: List[int],
    liste_paires_chimiques: List[Tuple[int, int]],
    type_paire: str,
    connectivite: Dict[int, List[int]] = None
) -> List[Dict[str, object]]:
    resultats_paires = []

    cordes_cycle = set()
    if connectivite is not None:
        cordes_cycle = construire_ensemble_cordes_cycle(indices_cycle, connectivite)

    for paire_chimique in liste_paires_chimiques:
        i_local, j_local = convertir_paire_chimique_en_paire_locale(paire_chimique, indices_cycle)
        valeur_reelle = float(sous_matrice_ldm[i_local, j_local])
        valeur_reference = float(matrice_reference[i_local, j_local])
        difference = valeur_reelle - valeur_reference

        est_corde = tuple(sorted(paire_chimique)) in cordes_cycle
        type_paire_affiche = "ortho" if est_corde else type_paire

        resultats_paires.append({
            "type_paire": type_paire_affiche,
            "paire_chimique": paire_chimique,
            "paire_locale": (i_local, j_local),
            "valeur_reelle": valeur_reelle,
            "valeur_reference": valeur_reference,
            "difference": difference
        })

    return resultats_paires


# ==============================
# FONCTION DE CALCUL D’UNE CONTRIBUTION D’ORBITE
# ==============================

def calculer_contribution_orbite(
    tableau_paires_orbite: List[Dict[str, object]]
) -> float:
    somme_carres = 0.0
    for element in tableau_paires_orbite:
        somme_carres += element["difference"] ** 2
    return float(math.sqrt(somme_carres))


# ==============================
# FONCTION DE CALCUL DE LA MOYENNE RÉELLE D’UNE ORBITE
# ==============================

def calculer_moyenne_reelle_orbite(
    tableau_paires_orbite: List[Dict[str, object]]
) -> float:
    return float(sum(element["valeur_reelle"] for element in tableau_paires_orbite) / len(tableau_paires_orbite))


# ==============================
# FONCTION DE CONSTRUCTION DU TABLEAU DÉTAILLÉ DES PAIRES PAR ORBITES
# ==============================

def construire_tableau_detaille_paires_orbites(
    sous_matrice_ldm: np.ndarray,
    matrice_reference: np.ndarray,
    indices_cycle: List[int],
    orbites: Dict[str, List[Tuple[int, int]]]
) -> List[Dict[str, object]]:
    tableau_complet = []
    for nom_orbite, paires in orbites.items():
        tableau_complet.extend(
            extraire_valeurs_paires(
                sous_matrice_ldm=sous_matrice_ldm,
                matrice_reference=matrice_reference,
                indices_cycle=indices_cycle,
                liste_paires_chimiques=paires,
                type_paire=nom_orbite
            )
        )
    return tableau_complet


# ==============================
# FONCTION DE PRÉPARATION DES CONTRIBUTIONS PAR ORBITES
# ==============================

def preparer_contributions_orbites(
    sous_matrice_ldm: np.ndarray,
    matrice_reference: np.ndarray,
    indices_cycle: List[int],
    orbites: Dict[str, List[Tuple[int, int]]],
    connectivite: Dict[int, List[int]] = None
) -> Dict[str, object]:
    contributions_orbites: Dict[str, float] = {}
    moyennes_reelles_orbites: Dict[str, float] = {}
    tableaux_orbites: Dict[str, List[Dict[str, object]]] = {}
    tableau_detaille_paires = []

    for nom_orbite, paires in orbites.items():
        tableau_paires_orbite = extraire_valeurs_paires(
            sous_matrice_ldm=sous_matrice_ldm,
            matrice_reference=matrice_reference,
            indices_cycle=indices_cycle,
            liste_paires_chimiques=paires,
            type_paire=nom_orbite,
            connectivite=connectivite
        )
        tableaux_orbites[nom_orbite] = tableau_paires_orbite
        contributions_orbites[nom_orbite] = calculer_contribution_orbite(tableau_paires_orbite)
        moyennes_reelles_orbites[nom_orbite] = calculer_moyenne_reelle_orbite(tableau_paires_orbite)
        tableau_detaille_paires.extend(tableau_paires_orbite)

    return {
        "contributions_orbites": contributions_orbites,
        "moyennes_reelles_orbites": moyennes_reelles_orbites,
        "tableaux_orbites": tableaux_orbites,
        "tableau_detaille_paires": tableau_detaille_paires
    }

# ============================================================
# BLOC 9B — ANALYSE Q / CT / HOMOGÉNÉITÉ D'ORBITE
# ============================================================
# Descripteurs synthétiques Q_Frob, Q_RMS, indice de transfert de
# charge (CT) et homogénéité S_hom au niveau du cycle isolé.
# ============================================================

def extraire_sous_matrice_distances(
    matrice_distances_complete: np.ndarray,
    indices_cycle: List[int]
) -> np.ndarray:
    indices_python = [indice - 1 for indice in indices_cycle]
    return matrice_distances_complete[np.ix_(indices_python, indices_python)]


def construire_matrice_q_ponderee(
    sous_matrice_ldm: np.ndarray,
    sous_matrice_distances: np.ndarray
) -> np.ndarray:
    n = sous_matrice_ldm.shape[0]
    matrice_q = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                matrice_q[i, j] = 0.0
            else:
                distance_ij = sous_matrice_distances[i, j]
                if abs(distance_ij) <= 1.0e-15:
                    matrice_q[i, j] = 0.0
                else:
                    matrice_q[i, j] = sous_matrice_ldm[i, j] / distance_ij

    return matrice_q


def calculer_frobenius_hors_diagonale_matrice(matrice: np.ndarray) -> float:
    matrice_hors_diagonale = matrice.copy()
    np.fill_diagonal(matrice_hors_diagonale, 0.0)
    return float(np.linalg.norm(matrice_hors_diagonale, ord="fro"))


def calculer_rms_hors_diagonale_matrice(matrice: np.ndarray) -> float:
    n = matrice.shape[0]
    if n <= 1:
        return 0.0

    matrice_hors_diagonale = matrice.copy()
    np.fill_diagonal(matrice_hors_diagonale, 0.0)

    nombre_termes = n * (n - 1)
    norme = np.linalg.norm(matrice_hors_diagonale, ord="fro")
    return float(norme / math.sqrt(nombre_termes))


def calculer_n_ref_cycle(
    taille_cycle: int,
    numeros_atomiques_cycle: List[int] = None
) -> float:
    """
    Population électronique de référence totale du cycle.

    V05 : si numeros_atomiques_cycle est fourni, N_ref = Σ Z_k
    (somme des numéros atomiques des atomes du cycle).
    Sinon, fallback historique : N_ref = 6 × taille_cycle (carbone uniquement).
    """
    if numeros_atomiques_cycle is not None:
        if len(numeros_atomiques_cycle) != taille_cycle:
            raise ValueError(
                f"Erreur : numeros_atomiques_cycle ({len(numeros_atomiques_cycle)}) "
                f"ne correspond pas à taille_cycle ({taille_cycle})."
            )
        return float(sum(numeros_atomiques_cycle))
    # Fallback legacy : tout-carbone.
    return float(6.0 * taille_cycle)


def calculer_transfert_charge_cycle(
    sous_matrice_ldm: np.ndarray,
    n_ref_cycle: float
) -> Dict[str, float]:
    """
    Charge Transfer du cycle.

    Convention V05 (Option A, cohérente avec la théorie LDM) :
      - population_totale_reelle = Σ(tous éléments de la LDM)
        (somme totale = nombre total d'électrons distribués sur le cycle)
      - n_ref_cycle = Σ Z_k = Σ(tous éléments de la matrice de référence)
      - CT = population_totale_reelle − n_ref_cycle

    Avec cette convention, si LDM = M_ref exactement, on a CT = 0 (cohérence
    parfaite). En V04, on comparait Tr(LDM) à 6×n, ce qui donnait des CT
    biaisés (≠ 0 même pour la référence parfaite).
    """
    population_totale_reelle = float(np.sum(sous_matrice_ldm))
    ct_absolu = population_totale_reelle - n_ref_cycle

    if abs(n_ref_cycle) <= 1.0e-15:
        ct_relatif = 0.0
    else:
        ct_relatif = 100.0 * ct_absolu / n_ref_cycle

    return {
        "population_totale_reelle": population_totale_reelle,
        "population_totale_reference": float(n_ref_cycle),
        "ct_absolu": float(ct_absolu),
        "ct_relatif_pourcent": float(ct_relatif)
    }


def construire_valeurs_x_par_orbite(
    sous_matrice_ldm: np.ndarray,
    sous_matrice_distances: np.ndarray,
    indices_cycle: List[int],
    orbites: Dict[str, List[Tuple[int, int]]]
) -> Dict[str, List[float]]:
    valeurs_x_par_orbite: Dict[str, List[float]] = {}

    for nom_orbite, paires in orbites.items():
        valeurs_x = []
        for paire_chimique in paires:
            i_local, j_local = convertir_paire_chimique_en_paire_locale(
                paire_chimique=paire_chimique,
                indices_cycle=indices_cycle
            )
            ldm_ij = float(sous_matrice_ldm[i_local, j_local])
            r_ij = float(sous_matrice_distances[i_local, j_local])
            x_ij = ldm_ij * r_ij
            valeurs_x.append(x_ij)

        valeurs_x_par_orbite[nom_orbite] = valeurs_x

    return valeurs_x_par_orbite


def calculer_moyenne(liste_valeurs: List[float]) -> float:
    if len(liste_valeurs) == 0:
        return 0.0
    return float(sum(liste_valeurs) / len(liste_valeurs))


def calculer_ecart_type_population(liste_valeurs: List[float]) -> float:
    if len(liste_valeurs) == 0:
        return 0.0
    moyenne = calculer_moyenne(liste_valeurs)
    variance = sum((x - moyenne) ** 2 for x in liste_valeurs) / len(liste_valeurs)
    return float(math.sqrt(variance))


def calculer_homogeneite_orbite(
    s_orb_k: float,
    sigma_k: float
) -> float:
    if s_orb_k <= 1.0e-15:
        return 0.0
    return float(1.0 / (1.0 + sigma_k / s_orb_k))


def calculer_nombre_total_paires_cycle(taille_cycle: int) -> int:
    return (taille_cycle * (taille_cycle - 1)) // 2


def preparer_analyse_homogeneite_cycle(
    sous_matrice_ldm: np.ndarray,
    sous_matrice_distances: np.ndarray,
    indices_cycle: List[int],
    orbites: Dict[str, List[Tuple[int, int]]]
) -> Dict[str, object]:
    valeurs_x_par_orbite = construire_valeurs_x_par_orbite(
        sous_matrice_ldm=sous_matrice_ldm,
        sous_matrice_distances=sous_matrice_distances,
        indices_cycle=indices_cycle,
        orbites=orbites
    )

    s_orbites: Dict[str, float] = {}
    sigmas_orbites: Dict[str, float] = {}
    h_orbites: Dict[str, float] = {}
    details_orbites_homogeneite: Dict[str, List[Dict[str, object]]] = {}

    somme_ponderee_cycle = 0.0

    for nom_orbite, paires in orbites.items():
        valeurs_x = valeurs_x_par_orbite[nom_orbite]
        s_orb_k = calculer_moyenne(valeurs_x)
        sigma_k = calculer_ecart_type_population(valeurs_x)
        h_k = calculer_homogeneite_orbite(s_orb_k, sigma_k)

        s_orbites[nom_orbite] = s_orb_k
        sigmas_orbites[nom_orbite] = sigma_k
        h_orbites[nom_orbite] = h_k

        somme_ponderee_cycle += len(paires) * s_orb_k * h_k

        details = []
        for paire_chimique, x_ij in zip(paires, valeurs_x):
            details.append({
                "paire_chimique": paire_chimique,
                "x_ij": float(x_ij)
            })
        details_orbites_homogeneite[nom_orbite] = details

    n = len(indices_cycle)
    nombre_total_paires = calculer_nombre_total_paires_cycle(n)

    if nombre_total_paires == 0:
        s_cycle_hom = 0.0
    else:
        s_cycle_hom = float(somme_ponderee_cycle / nombre_total_paires)

    return {
        "s_orbites": s_orbites,
        "sigmas_orbites": sigmas_orbites,
        "h_orbites": h_orbites,
        "details_orbites_homogeneite": details_orbites_homogeneite,
        "s_cycle_hom": s_cycle_hom,
        "nombre_total_paires": nombre_total_paires,
        "nombre_orbites": len(orbites)
    }


def preparer_analyse_q_et_ct(
    sous_matrice_ldm: np.ndarray,
    matrice_reference: np.ndarray,
    sous_matrice_distances: np.ndarray,
    taille_cycle: int,
    numeros_atomiques_cycle: List[int] = None
) -> Dict[str, object]:
    matrice_q_reelle = construire_matrice_q_ponderee(
        sous_matrice_ldm=sous_matrice_ldm,
        sous_matrice_distances=sous_matrice_distances
    )

    matrice_q_reference = construire_matrice_q_ponderee(
        sous_matrice_ldm=matrice_reference,
        sous_matrice_distances=sous_matrice_distances
    )

    matrice_q_difference = matrice_q_reelle - matrice_q_reference

    q_frob_offdiag = calculer_frobenius_hors_diagonale_matrice(matrice_q_difference)
    q_rms_offdiag = calculer_rms_hors_diagonale_matrice(matrice_q_difference)

    n_ref_cycle = calculer_n_ref_cycle(taille_cycle, numeros_atomiques_cycle)
    donnees_ct = calculer_transfert_charge_cycle(
        sous_matrice_ldm=sous_matrice_ldm,
        n_ref_cycle=n_ref_cycle
    )

    return {
        "matrice_q_reelle": matrice_q_reelle,
        "matrice_q_reference": matrice_q_reference,
        "matrice_q_difference": matrice_q_difference,
        "q_frob_offdiag": q_frob_offdiag,
        "q_rms_offdiag": q_rms_offdiag,
        "n_ref_cycle": n_ref_cycle,
        "population_totale_reelle": donnees_ct["population_totale_reelle"],
        "population_totale_reference": donnees_ct["population_totale_reference"],
        "ct_absolu": donnees_ct["ct_absolu"],
        "ct_relatif_pourcent": donnees_ct["ct_relatif_pourcent"]
    }
# ============================================================
# BLOC 10 — ANALYSE COMPLÈTE D'UN CYCLE TRAITABLE
# ============================================================
# Orchestrateur : pour un cycle donné, enchaîne extraction LDM,
# construction de la référence, calcul des différences, ratios,
# Q, S, S_hom, et remplit un dictionnaire résultat.
# ============================================================

# ==============================
def analyser_cycle_v2(
    matrice_ldm_complete: np.ndarray,
    matrice_distances_complete: np.ndarray,
    cycle_info: Dict[str, object],
    parametres_reference: Dict[str, float]
) -> Dict[str, object]:
    numero_cycle = cycle_info["numero_cycle"]
    type_cycle = cycle_info["type_cycle"]
    indices_cycle = cycle_info["indices_cycle"]
    taille_cycle = cycle_info["taille_cycle"]
    statut_cycle = cycle_info["statut_cycle"]
    origine_cycle = cycle_info["origine_cycle"]

    donnees_sous_matrice = preparer_donnees_sous_matrice(
        matrice_ldm_complete=matrice_ldm_complete,
        indices_cycle=indices_cycle
    )
    sous_matrice_ldm = donnees_sous_matrice["sous_matrice_ldm"]

    sous_matrice_distances = extraire_sous_matrice_distances(
        matrice_distances_complete=matrice_distances_complete,
        indices_cycle=indices_cycle
    )

    matrice_reference, orbites, valeurs_orbite, valeur_lambda = construire_matrice_reference_cycle(
        indices_cycle=indices_cycle,
        parametres_reference=parametres_reference,
        connectivite=cycle_info["connectivite_cycle"],
        numeros_atomiques_cycle=cycle_info.get("numeros_atomiques_cycle")
    )

    verifier_coherence_matrice_reference(matrice_reference, valeur_lambda)

    contributions_globales = preparer_contributions_globales(
        sous_matrice_ldm=sous_matrice_ldm,
        matrice_reference=matrice_reference
    )

    contributions_orbites = preparer_contributions_orbites(
        sous_matrice_ldm=sous_matrice_ldm,
        matrice_reference=matrice_reference,
        indices_cycle=indices_cycle,
        orbites=orbites,
        connectivite=cycle_info["connectivite_cycle"]
    )

    analyse_q_ct = preparer_analyse_q_et_ct(
        sous_matrice_ldm=sous_matrice_ldm,
        matrice_reference=matrice_reference,
        sous_matrice_distances=sous_matrice_distances,
        taille_cycle=taille_cycle,
        numeros_atomiques_cycle=cycle_info.get("numeros_atomiques_cycle")
    )

    analyse_homogeneite = preparer_analyse_homogeneite_cycle(
        sous_matrice_ldm=sous_matrice_ldm,
        sous_matrice_distances=sous_matrice_distances,
        indices_cycle=indices_cycle,
        orbites=orbites
    )

    return {
        "numero_cycle": numero_cycle,
        "type_cycle": type_cycle,
        "taille_cycle": taille_cycle,
        "indices_cycle": indices_cycle,
        "statut_cycle": statut_cycle,
        "origine_cycle": origine_cycle,
        "reference_name": parametres_reference["reference_name"],
        "n_a": parametres_reference["n_a"],
        "n_a_par_site": cycle_info.get("numeros_atomiques_cycle"),
        "lambda": valeur_lambda,
        "nombre_paires_pi_reference": calculer_nombre_paires_pi_reference(taille_cycle),
        "nombre_orbites": analyse_homogeneite["nombre_orbites"],
        "nombre_total_paires": analyse_homogeneite["nombre_total_paires"],
        "sous_matrice_ldm": sous_matrice_ldm,
        "sous_matrice_distances": sous_matrice_distances,
        "est_symetrique": donnees_sous_matrice["est_symetrique"],
        "asymetrie_maximale": donnees_sous_matrice["asymetrie_maximale"],
        "matrice_reference": matrice_reference,
        "matrice_difference": contributions_globales["matrice_difference"],
        "distance_frobenius_complete": contributions_globales["distance_frobenius_complete"],
        "contribution_diagonale": contributions_globales["contribution_diagonale"],
        "contribution_hors_diagonale": contributions_globales["contribution_hors_diagonale"],
        "rms_matricielle": contributions_globales["rms_matricielle"],
        "rms_hors_diagonale": contributions_globales["rms_hors_diagonale"],
        "orbites": orbites,
        "valeurs_orbite_reference": valeurs_orbite,
        "contributions_orbites": contributions_orbites["contributions_orbites"],
        "moyennes_reelles_orbites": contributions_orbites["moyennes_reelles_orbites"],
        "tableaux_orbites": contributions_orbites["tableaux_orbites"],
        "tableau_detaille_paires": contributions_orbites["tableau_detaille_paires"],

        # Q
        "matrice_q_reelle": analyse_q_ct["matrice_q_reelle"],
        "matrice_q_reference": analyse_q_ct["matrice_q_reference"],
        "matrice_q_difference": analyse_q_ct["matrice_q_difference"],
        "q_frob_offdiag": analyse_q_ct["q_frob_offdiag"],
        "q_rms_offdiag": analyse_q_ct["q_rms_offdiag"],

        # CT
        "n_ref_cycle": analyse_q_ct["n_ref_cycle"],
        "population_totale_reelle": analyse_q_ct["population_totale_reelle"],
        "population_totale_reference": analyse_q_ct["population_totale_reference"],
        "ct_absolu": analyse_q_ct["ct_absolu"],
        "ct_relatif_pourcent": analyse_q_ct["ct_relatif_pourcent"],

        # Homogénéité
        "s_orbites": analyse_homogeneite["s_orbites"],
        "sigmas_orbites": analyse_homogeneite["sigmas_orbites"],
        "h_orbites": analyse_homogeneite["h_orbites"],
        "details_orbites_homogeneite": analyse_homogeneite["details_orbites_homogeneite"],
        "s_cycle_hom": analyse_homogeneite["s_cycle_hom"]
    }


# ==============================
# FONCTION DE FILTRAGE DES CYCLES TRAITÉS EN V2
# ==============================

def filtrer_cycles_traites_v2(
    cycles_classes: List[Dict[str, object]]
) -> List[Dict[str, object]]:
    return [cycle_info for cycle_info in cycles_classes if cycle_info["statut_cycle"] == STATUT_CYCLE_TRAITE]


# ==============================
# FONCTION DE PRÉPARATION DE LA LISTE DES RÉSULTATS D’ANALYSE
# ==============================

def preparer_resultats_cycles_analyses(
    matrice_ldm_complete: np.ndarray,
    matrice_distances_complete: np.ndarray,
    cycles_classes: List[Dict[str, object]],
    parametres_reference: Dict[str, float]
) -> List[Dict[str, object]]:
    cycles_traites = filtrer_cycles_traites_v2(cycles_classes)
    if len(cycles_traites) == 0:
        raise ValueError("Erreur : aucun cycle traitable détecté. Aucun traitement possible en V2.")

    resultats_cycles = []
    for cycle_info in cycles_traites:
        resultats_cycles.append(
            analyser_cycle_v2(
                matrice_ldm_complete=matrice_ldm_complete,
                matrice_distances_complete=matrice_distances_complete,
                cycle_info=cycle_info,
                parametres_reference=parametres_reference
            )
        )
    return resultats_cycles


# ============================================================
# BLOC 11 — CLASSEMENT GLOBAL DES CYCLES
# ============================================================
# Ordonne l'ensemble des résultats de cycles pour l'affichage final
# (par type puis par indices).
# ============================================================

# ==============================
# FONCTION DE TRI DES RÉSULTATS PAR DISTANCE DE FROBENIUS COMPLÈTE
# ==============================

def trier_resultats_par_distance_frobenius(
    resultats_cycles: List[Dict[str, object]]
) -> List[Dict[str, object]]:
    return sorted(resultats_cycles, key=lambda x: x["distance_frobenius_complete"])


# ==============================
# FONCTION D’ATTRIBUTION DES RANGS AVEC GESTION DES EX ÆQUO SANS SAUT
# ==============================

def attribuer_rangs_avec_exaequo(
    resultats_tries: List[Dict[str, object]],
    tolerance: float = 1.0e-12
) -> List[Dict[str, object]]:
    current_rank = 0
    previous_distance = None

    for index, resultat in enumerate(resultats_tries):
        distance_actuelle = resultat["distance_frobenius_complete"]
        if index == 0 or abs(distance_actuelle - previous_distance) > tolerance:
            current_rank += 1
        resultat["rang_final"] = current_rank
        previous_distance = distance_actuelle

    return resultats_tries


# ==============================
# FONCTION DE PRÉPARATION DU CLASSEMENT FINAL
# ==============================

def preparer_classement_final(
    resultats_cycles: List[Dict[str, object]]
) -> List[Dict[str, object]]:
    resultats_tries = trier_resultats_par_distance_frobenius(resultats_cycles)
    return attribuer_rangs_avec_exaequo(resultats_tries)


# ============================================================
# BLOC 12 — RAPPORT PRÉLIMINAIRE (structure moléculaire)
# ============================================================
# Construit la partie descriptive du rapport : types de carbone
# détectés, rayons adaptés, connectivité, liste des cycles avant
# toute analyse LDM/HOMA.
# ============================================================

# ==============================
# FONCTION DE GÉNÉRATION DU RAPPORT PRÉLIMINAIRE
# ==============================

def generer_rapport_preliminaire(
    numeros_atomiques: List[int],
    connectivite_provisoire: Dict[int, List[int]],
    rayons_adaptes: Dict[int, float],
    connectivite_finale: Dict[int, List[int]],
    cycles_classes: List[Dict[str, object]]
) -> str:
    rapport = ""
    rapport += afficher_types_carbone(numeros_atomiques, connectivite_provisoire, rayons_adaptes)
    rapport += afficher_connectivite(connectivite_finale)
    rapport += afficher_cycles_detectes(cycles_classes)
    return rapport


# ============================================================
# BLOC 13 — ÉCRITURE DU RAPPORT LDM (.arx historique)
# ============================================================
# Format texte lisible du rapport LDM autonome (avant intégration
# HOMA + entropie). Encore utilisé par le main() historique.
# ============================================================

# ==============================
# FONCTION DE GÉNÉRATION DU NOM DU FICHIER .LDM
# ==============================

def generer_nom_fichier_ldm(nom_fichier_entree: str) -> str:
    nom_simple = os.path.basename(nom_fichier_entree)
    nom_sans_extension = os.path.splitext(nom_simple)[0]
    return nom_sans_extension + ".LDM"


# ==============================
# FONCTION DE CALCUL DE LA LARGEUR DE LA COLONNE INDICES
# ==============================

def calculer_largeur_colonne_indices(
    cycles_classes: List[Dict[str, object]],
    largeur_minimale: int = 22,
    marge: int = 2
) -> int:
    if len(cycles_classes) == 0:
        return largeur_minimale
    textes_indices = [str(cycle_info["indices_cycle"]) for cycle_info in cycles_classes]
    largeur_maximale = max(len(texte) for texte in textes_indices) + marge
    return max(largeur_minimale, largeur_maximale)


# ==============================
# FONCTION D’ÉCRITURE DE L’EN-TÊTE GÉNÉRAL
# ==============================

def ecrire_entete_general_ldm(
    fichier,
    nom_fichier_ldm: str,
    nom_fichier_dist: str,
    nom_fichier_sortie: str,
    parametres_reference: Dict[str, float]
) -> None:
    fichier.write("===============================================================\n")
    fichier.write("                 LDM AROMATICITY ANALYSIS REPORT\n")
    fichier.write("===============================================================\n")
    fichier.write(f" Input LDM file      : {nom_fichier_ldm}\n")
    fichier.write(f" Input DIST file     : {nom_fichier_dist}\n")
    fichier.write(f" Output file         : {nom_fichier_sortie}\n")
    fichier.write(f" Reference type      : {parametres_reference['reference_name']}\n")
    fichier.write(f" epsilon_O           : {parametres_reference['epsilon_o']:.6f}\n")
    fichier.write(f" epsilon_P           : {parametres_reference['epsilon_p']:.6f}\n")
    fichier.write(f" epsilon_M           : {parametres_reference['epsilon_m']:.6f}\n")
    fichier.write(f" N_A (default C)     : {parametres_reference['n_a']:.6f}\n")
    fichier.write( " N_A per site        : Z of each atom in the cycle (auto from .dat)\n")
    fichier.write(f" lambda (benzene EG) : {parametres_reference['lambda']:.6f}\n")
    fichier.write(" lambda formula      : lambda_k = Z_k - sum(off-diagonal row k)\n")
    fichier.write("===============================================================\n\n")


# ==============================
# FONCTION D’ÉCRITURE DU TABLEAU DÉTAILLÉ DES ORBITES
# ==============================

def ecrire_tableau_detaille_orbites(
    fichier,
    tableau_detaille_paires: List[Dict[str, object]]
) -> None:
    fichier.write("DETAILED ORBIT PAIR TABLE\n")
    fichier.write("------------------------------\n")
    fichier.write(
        f" {'Orbit':<10} {'Pair':<14} {'Real':>12} {'Reference':>12} {'Difference':>12}\n"
    )
    fichier.write(" ----------------------------------------------------------------\n")

    for element in tableau_detaille_paires:
        type_paire = element["type_paire"]
        paire_chimique = str(element["paire_chimique"])
        valeur_reelle = element["valeur_reelle"]
        valeur_reference = element["valeur_reference"]
        difference = element["difference"]
        fichier.write(
            f" {type_paire:<10} {paire_chimique:<14} "
            f"{valeur_reelle:12.6f} {valeur_reference:12.6f} {difference:12.6f}\n"
        )

    fichier.write("\n")


# ==============================
# FONCTION D’ÉCRITURE DU RÉSUMÉ DES ORBITES POUR UN CYCLE
# ==============================

def ecrire_resume_orbites_cycle(
    fichier,
    resultat_cycle: Dict[str, object]
) -> None:
    fichier.write("ORBIT SUMMARY\n")
    fichier.write("------------------------------\n")
    fichier.write(
        f" {'Orbit':<10} {'n pairs':>8} {'Ref. val':>12} {'Mean real':>12} {'Contrib.':>12}\n"
    )
    fichier.write(" --------------------------------------------------------------\n")

    orbites = resultat_cycle["orbites"]
    valeurs_ref = resultat_cycle["valeurs_orbite_reference"]
    moyennes = resultat_cycle["moyennes_reelles_orbites"]
    contributions = resultat_cycle["contributions_orbites"]

    for nom_orbite in sorted(orbites.keys(), key=orbite_vers_distance_topologique):
        fichier.write(
            f" {nom_orbite:<10} {len(orbites[nom_orbite]):8d} "
            f"{valeurs_ref[nom_orbite]:12.6f} {moyennes[nom_orbite]:12.6f} {contributions[nom_orbite]:12.6f}\n"
        )

    fichier.write("\n")


# ==============================
# FONCTION D’ÉCRITURE DU TABLEAU D’HOMOGÉNÉITÉ PAR ORBITES
# ==============================

def ecrire_tableau_homogeneite_orbites(
    fichier,
    resultat_cycle: Dict[str, object]
) -> None:
    fichier.write("ORBIT HOMOGENEITY SUMMARY\n")
    fichier.write("------------------------------\n")
    fichier.write(
        f" {'Orbit':<10} {'n pairs':>8} {'S_orb':>12} {'sigma':>12} {'H':>12}\n"
    )
    fichier.write(" ------------------------------------------------------------\n")

    orbites = resultat_cycle["orbites"]
    s_orbites = resultat_cycle["s_orbites"]
    sigmas_orbites = resultat_cycle["sigmas_orbites"]
    h_orbites = resultat_cycle["h_orbites"]

    for nom_orbite in sorted(orbites.keys(), key=orbite_vers_distance_topologique):
        fichier.write(
            f" {nom_orbite:<10} {len(orbites[nom_orbite]):8d} "
            f"{s_orbites[nom_orbite]:12.6f} {sigmas_orbites[nom_orbite]:12.6f} {h_orbites[nom_orbite]:12.6f}\n"
        )

    fichier.write("\n")


# ==============================
# FONCTION D’ÉCRITURE DU DÉTAIL D’UN CYCLE TRAITÉ
# ==============================

def ecrire_detail_cycle_traite(
    fichier,
    resultat_cycle: Dict[str, object]
) -> None:
    fichier.write("===============================================================\n")
    fichier.write(f" Cycle number        : {resultat_cycle['numero_cycle']}\n")
    fichier.write(f" Cycle type          : {resultat_cycle['type_cycle']}\n")
    fichier.write(f" Cycle size          : {resultat_cycle['taille_cycle']}\n")
    fichier.write(f" Cycle indices       : {resultat_cycle['indices_cycle']}\n")
    n_a_par_site = resultat_cycle.get("n_a_par_site")
    if n_a_par_site is not None:
        fichier.write(f" N_A per site (Z)    : {list(n_a_par_site)}\n")
    fichier.write(f" Cycle status        : {resultat_cycle['statut_cycle']}\n")
    fichier.write(f" Cycle origin        : {resultat_cycle['origine_cycle']}\n")
    fichier.write(f" N_pi reference      : {resultat_cycle['nombre_paires_pi_reference']}\n")
    fichier.write(f" Number of orbits    : {resultat_cycle['nombre_orbites']}\n")
    valeur_lambda = resultat_cycle['lambda']
    if isinstance(valeur_lambda, (list, tuple, np.ndarray)):
        valeurs_str = ", ".join(f"{float(v):.6f}" for v in valeur_lambda)
        fichier.write(f" Lambda per site     : [{valeurs_str}]\n")
    else:
        fichier.write(f" Lambda              : {float(valeur_lambda):.6f}\n")
    fichier.write(f" Symmetric submatrix : {resultat_cycle['est_symetrique']}\n")
    fichier.write(f" Max asymmetry       : {resultat_cycle['asymetrie_maximale']:.6e}\n")
    fichier.write("\n")

    fichier.write("REAL LDM SUBMATRIX\n")
    fichier.write("------------------------------\n")
    for ligne in resultat_cycle["sous_matrice_ldm"]:
        fichier.write(" " + " ".join(f"{valeur:.6f}" for valeur in ligne) + "\n")
    fichier.write("\n")

    fichier.write(f"REFERENCE MATRIX ({resultat_cycle['taille_cycle']}x{resultat_cycle['taille_cycle']})\n")
    fichier.write("------------------------------\n")
    for ligne in resultat_cycle["matrice_reference"]:
        fichier.write(" " + " ".join(f"{valeur:.6f}" for valeur in ligne) + "\n")
    fichier.write("\n")

    fichier.write("DIFFERENCE MATRIX : DELTA = REAL - REFERENCE\n")
    fichier.write("------------------------------\n")
    for ligne in resultat_cycle["matrice_difference"]:
        fichier.write(" " + " ".join(f"{valeur:.6f}" for valeur in ligne) + "\n")
    fichier.write("\n")

    fichier.write("DISTANCES AND CONTRIBUTIONS\n")
    fichier.write("------------------------------\n")
    fichier.write(f" Frobenius distance (full) : {resultat_cycle['distance_frobenius_complete']:.6f}\n")
    fichier.write(f" Diagonal contribution     : {resultat_cycle['contribution_diagonale']:.6f}\n")
    fichier.write(f" Off-diagonal contribution : {resultat_cycle['contribution_hors_diagonale']:.6f}\n")
    fichier.write(f" RMS matricielle           : {resultat_cycle['rms_matricielle']:.6f}\n")
    fichier.write(f" RMS hors diagonale        : {resultat_cycle['rms_hors_diagonale']:.6f}\n")
    fichier.write(f" Final rank                : {resultat_cycle.get('rang_final', 'NA')}\n")
    fichier.write("\n")

    fichier.write("Q WEIGHTED ANALYSIS\n")
    fichier.write("------------------------------\n")
    fichier.write(f" Q Frobenius offdiag       : {resultat_cycle['q_frob_offdiag']:.6f}\n")
    fichier.write(f" Q RMS offdiag             : {resultat_cycle['q_rms_offdiag']:.6f}\n")
    fichier.write("\n")

    fichier.write("CHARGE TRANSFER ANALYSIS\n")
    fichier.write("------------------------------\n")
    fichier.write(f" N_ref cycle               : {resultat_cycle['n_ref_cycle']:.6f}\n")
    fichier.write(f" Population real (sum LDM) : {resultat_cycle['population_totale_reelle']:.6f}\n")
    fichier.write(f" CT absolute               : {resultat_cycle['ct_absolu']:.6f}\n")
    fichier.write(f" CT relative (%)           : {resultat_cycle['ct_relatif_pourcent']:.6f}\n")
    fichier.write("\n")

    fichier.write("AROMATIC HOMOGENEITY INDEX\n")
    fichier.write("------------------------------\n")
    fichier.write(f" Number of unique pairs    : {resultat_cycle['nombre_total_paires']}\n")
    fichier.write(f" S_cycle_hom               : {resultat_cycle['s_cycle_hom']:.6f}\n")
    fichier.write("\n")

    ecrire_analyse_homogeneite_cycle(fichier, resultat_cycle)

    ecrire_resume_orbites_cycle(fichier, resultat_cycle)
    ecrire_tableau_homogeneite_orbites(fichier, resultat_cycle)
    ecrire_tableau_detaille_orbites(fichier, resultat_cycle["tableau_detaille_paires"])


# ==============================
# FONCTION D’ÉCRITURE DU SUMMARY FINAL
# ==============================

def ecrire_summary_final_ldm(
    fichier,
    cycles_classes: List[Dict[str, object]],
    resultats_classes: List[Dict[str, object]],
    parametres_reference: Dict[str, float],
    analyse_globale: Dict[str, object] = None
) -> None:
    largeur_indices = calculer_largeur_colonne_indices(cycles_classes)

    fichier.write("===============================================================\n")
    fichier.write("FINAL SUMMARY\n")
    fichier.write("===============================================================\n")
    fichier.write(f" Reference type : {parametres_reference['reference_name']}\n")
    fichier.write(f" epsilon_O      : {parametres_reference['epsilon_o']:.6f}\n")
    fichier.write(f" epsilon_P      : {parametres_reference['epsilon_p']:.6f}\n")
    fichier.write(f" epsilon_M      : {parametres_reference['epsilon_m']:.6f}\n")
    fichier.write(f" lambda (6 EG)  : {parametres_reference['lambda']:.6f}\n")
    fichier.write(f" N_A (def C)    : {parametres_reference['n_a']:.6f}\n")
    fichier.write(" N_A per site   : Z of each atom in the cycle\n")
    fichier.write("\n")

    ligne_titre = (
        f" {'No.':<4} {'Type':<13} {'Taille':<5} "
        f"{'Indices':<{largeur_indices}} "
        f"{'Origine':<11} {'Statut':<18} "
        f"{'Frob full':>12} {'Offdiag.':>10} {'Diag.':>10} "
        f"{'RMS mat':>10} {'RMS off':>10} "
        f"{'CT':>10} {'CT %':>10} "
        f"{'Q Frob':>10} {'Q RMS':>10} "
        f"{'S_hom':>10} "
        f"{'Q(G)':>10} {'S(G)':>10} {'S-hom(G)':>10}"
    )
    fichier.write(ligne_titre + "\n")
    fichier.write(" " + "-" * (len(ligne_titre) - 1) + "\n")

    resultats_par_numero = {resultat["numero_cycle"]: resultat for resultat in resultats_classes}

    for cycle_info in cycles_classes:
        numero_cycle = cycle_info["numero_cycle"]
        type_cycle   = cycle_info["type_cycle"]
        taille_cycle = cycle_info["taille_cycle"]
        indices_cycle = str(cycle_info["indices_cycle"])
        origine_cycle = cycle_info["origine_cycle"]
        statut_cycle  = cycle_info["statut_cycle"]

        if numero_cycle in resultats_par_numero:
            resultat = resultats_par_numero[numero_cycle]
            fichier.write(
                f" {numero_cycle:<4d} {type_cycle:<13s} {taille_cycle:<5d} "
                f"{indices_cycle:<{largeur_indices}s} "
                f"{origine_cycle:<11s} {statut_cycle:<18s} "
                f"{resultat['distance_frobenius_complete']:12.6f} "
                f"{resultat['contribution_hors_diagonale']:10.6f} "
                f"{resultat['contribution_diagonale']:10.6f} "
                f"{resultat['rms_matricielle']:10.6f} "
                f"{resultat['rms_hors_diagonale']:10.6f} "
                f"{resultat['ct_absolu']:10.6f} "
                f"{resultat['ct_relatif_pourcent']:10.6f} "
                f"{resultat['q_frob_offdiag']:10.6f} "
                f"{resultat['q_rms_offdiag']:10.6f} "
                f"{resultat['s_cycle_hom']:10.6f} "
                f"{(analyse_globale['q_global_normalise'] if analyse_globale is not None else float('nan')):10.6f} "
                f"{(analyse_globale['s_global'] if analyse_globale is not None else float('nan')):10.6f} "
                f"{(analyse_globale['s_hom_global'] if analyse_globale is not None else float('nan')):10.6f}\n"
            )
        else:
            fichier.write(
                f" {numero_cycle:<4d} {type_cycle:<13s} {taille_cycle:<5d} "
                f"{indices_cycle:<{largeur_indices}s} "
                f"{origine_cycle:<11s} {statut_cycle:<18s} "
                f"{'NA':>12} {'NA':>10} {'NA':>10} {'NA':>10} {'NA':>10} "
                f"{'NA':>10} {'NA':>10} {'NA':>10} {'NA':>10} {'NA':>10} "
                f"{(analyse_globale['q_global_normalise'] if analyse_globale is not None else 'NA'):>10} "
                f"{(analyse_globale['s_global'] if analyse_globale is not None else 'NA'):>10} "
                f"{(analyse_globale['s_hom_global'] if analyse_globale is not None else 'NA'):>10}\n"
            )

    fichier.write("\n")

def ecrire_analyse_homogeneite_cycle(
    fichier,
    resultat_cycle: Dict[str, object]
) -> None:
    fichier.write("HOMOGENEITY ANALYSIS BY ORBIT\n")
    fichier.write("------------------------------\n")
    fichier.write(
        f" {'Orbit':<10} {'n pairs':>8} {'S_orb':>14} {'sigma':>14} {'H_k':>14}\n"
    )
    fichier.write(" ----------------------------------------------------------------------\n")

    orbites = resultat_cycle["orbites"]
    s_orbites = resultat_cycle["s_orbites"]
    sigmas_orbites = resultat_cycle["sigmas_orbites"]
    h_orbites = resultat_cycle["h_orbites"]

    for nom_orbite in sorted(orbites.keys(), key=orbite_vers_distance_topologique):
        fichier.write(
            f" {nom_orbite:<10} "
            f"{len(orbites[nom_orbite]):8d} "
            f"{s_orbites[nom_orbite]:14.6f} "
            f"{sigmas_orbites[nom_orbite]:14.6f} "
            f"{h_orbites[nom_orbite]:14.6f}\n"
        )

    fichier.write("\n")
    fichier.write("DETAIL OF x_ij = LDM_ij * r_ij BY ORBIT\n")
    fichier.write("------------------------------\n")
    fichier.write(f" {'Orbit':<10} {'Pair':<14} {'x_ij':>14}\n")
    fichier.write(" -------------------------------------------------\n")

    details_orbites = resultat_cycle["details_orbites_homogeneite"]
    for nom_orbite in sorted(details_orbites.keys(), key=orbite_vers_distance_topologique):
        for element in details_orbites[nom_orbite]:
            fichier.write(
                f" {nom_orbite:<10} "
                f"{str(element['paire_chimique']):<14} "
                f"{element['x_ij']:14.6f}\n"
            )

    fichier.write("\n")
    fichier.write(f" Corrected orbit-homogeneity index (S_hom) : {resultat_cycle['s_cycle_hom']:.6f}\n")
    fichier.write("\n")
# ==============================
# FONCTION D’ENREGISTREMENT COMPLET DU FICHIER .LDM
# ==============================


# ============================================================
# BLOC 13B — ANALYSE GLOBALE ATOMES LOURDS (Q(G), S(G), S-hom(G))
# ============================================================
# Applique les descripteurs Q, S et S_hom non plus à un cycle
# individuel mais au sous-graphe complet des atomes lourds.
# Sert de référence globale à la molécule.
# ============================================================

def obtenir_indices_atomes_lourds(numeros_atomiques: List[int]) -> List[int]:
    return [i + 1 for i, z in enumerate(numeros_atomiques) if z != 1]


def extraire_sous_matrice_par_indices_chimiques(
    matrice_complete: np.ndarray,
    indices_chimiques: List[int]
) -> np.ndarray:
    indices_python = [indice - 1 for indice in indices_chimiques]
    return matrice_complete[np.ix_(indices_python, indices_python)]


def construire_connectivite_lourde(
    connectivite: Dict[int, List[int]],
    indices_atomes_lourds: List[int]
) -> Dict[int, List[int]]:
    ensemble_lourd = set(indices_atomes_lourds)
    return {
        indice: [voisin for voisin in connectivite.get(indice, []) if voisin in ensemble_lourd]
        for indice in indices_atomes_lourds
    }


def calculer_distances_topologiques_globales(
    connectivite_lourde: Dict[int, List[int]],
    indices_atomes_lourds: List[int]
) -> Dict[Tuple[int, int], int]:
    distances = {}
    for source in indices_atomes_lourds:
        file = [(source, 0)]
        visites = {source}
        while file:
            noeud, dist = file.pop(0)
            for voisin in connectivite_lourde.get(noeud, []):
                if voisin not in visites:
                    visites.add(voisin)
                    file.append((voisin, dist + 1))
                    distances[(source, voisin)] = dist + 1
    distances_sym = {}
    for i in indices_atomes_lourds:
        for j in indices_atomes_lourds:
            if i == j:
                continue
            if (i, j) in distances:
                distances_sym[(min(i, j), max(i, j))] = distances[(i, j)]
    return distances_sym


def generer_orbites_globales_par_distance_topologique(
    indices_atomes_lourds: List[int],
    distances_topologiques: Dict[Tuple[int, int], int]
) -> Dict[str, List[Tuple[int, int]]]:
    orbites = {}
    for pos_i, i in enumerate(indices_atomes_lourds):
        for j in indices_atomes_lourds[pos_i + 1:]:
            cle = (min(i, j), max(i, j))
            distance = distances_topologiques.get(cle)
            if distance is None:
                nom_orbite = 'disc'
            else:
                nom_orbite = f'd{distance}'
            orbites.setdefault(nom_orbite, []).append(cle)

    def cle_tri(item):
        nom = item[0]
        if nom == 'disc':
            return (10**9, nom)
        return (int(nom[1:]), nom)

    return dict(sorted(orbites.items(), key=cle_tri))


def calculer_s_global(
    sous_matrice_ldm_lourde: np.ndarray,
    sous_matrice_distances_lourde: np.ndarray
) -> float:
    n = sous_matrice_ldm_lourde.shape[0]
    if n <= 1:
        return 0.0
    somme = 0.0
    nb = 0
    for i in range(n):
        for j in range(i + 1, n):
            somme += float(sous_matrice_ldm_lourde[i, j] * sous_matrice_distances_lourde[i, j])
            nb += 1
    return float(somme / nb) if nb > 0 else 0.0


def calculer_q_global_normalise(
    sous_matrice_ldm_lourde: np.ndarray,
    sous_matrice_distances_lourde: np.ndarray
) -> float:
    n = sous_matrice_ldm_lourde.shape[0]
    if n <= 1:
        return 0.0
    somme = 0.0
    nb = 0
    for i in range(n):
        for j in range(i + 1, n):
            distance = float(sous_matrice_distances_lourde[i, j])
            if abs(distance) <= 1.0e-15:
                continue
            somme += float(sous_matrice_ldm_lourde[i, j] / distance)
            nb += 1
    return float(somme / nb) if nb > 0 else 0.0


def preparer_analyse_globale_atomes_lourds(
    numeros_atomiques: List[int],
    matrice_ldm_complete: np.ndarray,
    matrice_distances_complete: np.ndarray,
    connectivite_finale: Dict[int, List[int]]
) -> Dict[str, object]:
    indices_atomes_lourds = obtenir_indices_atomes_lourds(numeros_atomiques)
    sous_matrice_ldm_lourde = extraire_sous_matrice_par_indices_chimiques(
        matrice_complete=matrice_ldm_complete,
        indices_chimiques=indices_atomes_lourds
    )
    sous_matrice_distances_lourde = extraire_sous_matrice_par_indices_chimiques(
        matrice_complete=matrice_distances_complete,
        indices_chimiques=indices_atomes_lourds
    )
    connectivite_lourde = construire_connectivite_lourde(connectivite_finale, indices_atomes_lourds)
    distances_topologiques = calculer_distances_topologiques_globales(connectivite_lourde, indices_atomes_lourds)
    orbites_globales = generer_orbites_globales_par_distance_topologique(indices_atomes_lourds, distances_topologiques)

    homogeneite = preparer_analyse_homogeneite_cycle(
        sous_matrice_ldm=sous_matrice_ldm_lourde,
        sous_matrice_distances=sous_matrice_distances_lourde,
        indices_cycle=indices_atomes_lourds,
        orbites=orbites_globales
    )

    return {
        'indices_atomes_lourds': indices_atomes_lourds,
        'nombre_atomes_lourds': len(indices_atomes_lourds),
        'nombre_paires_lourdes': homogeneite['nombre_total_paires'],
        'connectivite_lourde': connectivite_lourde,
        'orbites_globales': orbites_globales,
        'sous_matrice_ldm_lourde': sous_matrice_ldm_lourde,
        'sous_matrice_distances_lourde': sous_matrice_distances_lourde,
        'q_global_normalise': calculer_q_global_normalise(sous_matrice_ldm_lourde, sous_matrice_distances_lourde),
        's_global': calculer_s_global(sous_matrice_ldm_lourde, sous_matrice_distances_lourde),
        's_hom_global': homogeneite['s_cycle_hom'],
        's_orbites_globales': homogeneite['s_orbites'],
        'sigmas_orbites_globales': homogeneite['sigmas_orbites'],
        'h_orbites_globales': homogeneite['h_orbites'],
        'details_orbites_globales': homogeneite['details_orbites_homogeneite']
    }


def ecrire_section_analyse_globale_atomes_lourds(
    fichier,
    analyse_globale: Dict[str, object]
) -> None:
    fichier.write("===============================================================\n")
    fichier.write("GLOBAL HEAVY-ATOM ANALYSIS\n")
    fichier.write("===============================================================\n")
    fichier.write(f" Heavy atom indices        : {analyse_globale['indices_atomes_lourds']}\n")
    fichier.write(f" Number of heavy atoms     : {analyse_globale['nombre_atomes_lourds']}\n")
    fichier.write(f" Number of heavy pairs     : {analyse_globale['nombre_paires_lourdes']}\n")
    fichier.write(f" Q(G) normalized           : {analyse_globale['q_global_normalise']:.6f}\n")
    fichier.write(f" S(G) normalized           : {analyse_globale['s_global']:.6f}\n")
    fichier.write(f" S-hom(G)                  : {analyse_globale['s_hom_global']:.6f}\n")
    fichier.write("\n")
    fichier.write("GLOBAL ORBIT SUMMARY (topological distance on heavy-atom graph)\n")
    fichier.write("----------------------------------------------------------------\n")
    fichier.write(f" {'Orbit':<10} {'n pairs':>8} {'S_orb(G)':>14} {'sigma':>14} {'H_k':>14}\n")
    fichier.write(" ----------------------------------------------------------------------\n")

    orbites = analyse_globale['orbites_globales']
    s_orbites = analyse_globale['s_orbites_globales']
    sigmas = analyse_globale['sigmas_orbites_globales']
    h_orbites = analyse_globale['h_orbites_globales']

    def cle_tri(nom):
        if nom == 'disc':
            return (10**9, nom)
        return (int(nom[1:]), nom)

    for nom_orbite in sorted(orbites.keys(), key=cle_tri):
        fichier.write(
            f" {nom_orbite:<10} {len(orbites[nom_orbite]):8d} "
            f"{s_orbites[nom_orbite]:14.6f} {sigmas[nom_orbite]:14.6f} {h_orbites[nom_orbite]:14.6f}\n"
        )

    fichier.write("\n")
    fichier.write("DETAIL OF x_ij = LDM_ij * r_ij BY GLOBAL ORBIT\n")
    fichier.write("-----------------------------------------------\n")
    fichier.write(f" {'Orbit':<10} {'Pair':<14} {'x_ij':>14}\n")
    fichier.write(" -------------------------------------------------\n")
    details = analyse_globale['details_orbites_globales']
    for nom_orbite in sorted(details.keys(), key=cle_tri):
        for element in details[nom_orbite]:
            fichier.write(
                f" {nom_orbite:<10} {str(element['paire_chimique']):<14} {element['x_ij']:14.6f}\n"
            )
    fichier.write("\n")

def enregistrer_fichier_ldm(
    nom_fichier_sortie: str,
    nom_fichier_ldm: str,
    nom_fichier_dist: str,
    rapport_preliminaire: str,
    parametres_reference: Dict[str, float],
    cycles_classes: List[Dict[str, object]],
    resultats_classes: List[Dict[str, object]],
    analyse_globale: Dict[str, object] = None
) -> None:
    with open(nom_fichier_sortie, "w", encoding="utf-8") as fichier:
        ecrire_entete_general_ldm(
            fichier=fichier,
            nom_fichier_ldm=nom_fichier_ldm,
            nom_fichier_dist=nom_fichier_dist,
            nom_fichier_sortie=nom_fichier_sortie,
            parametres_reference=parametres_reference
        )

        fichier.write(rapport_preliminaire)
        fichier.write("\n")

        for resultat_cycle in resultats_classes:
            ecrire_detail_cycle_traite(fichier=fichier, resultat_cycle=resultat_cycle)

        if analyse_globale is not None:
            ecrire_section_analyse_globale_atomes_lourds(fichier, analyse_globale)

        ecrire_summary_final_ldm(
            fichier=fichier,
            cycles_classes=cycles_classes,
            resultats_classes=resultats_classes,
            parametres_reference=parametres_reference,
            analyse_globale=analyse_globale
        )


# ============================================================
# BLOC 13C — DESCRIPTEURS ENTROPIQUES H_LDM / H_Q / H_S
# ============================================================
# Cahier des charges v2.0. Trois familles de descripteurs :
#   - H_LDM : entropie complète (lambda_ii diagonaux + delta_ij hors diag.)
#             décomposée en H_loc, H_deloc, H_part, w_loc, w_deloc,
#             normalisée mu_LDM entre H_min_chem (limite localisée) et
#             H_max_ref = log2(N_loc + N_deloc).
#   - H_Q   : entropie de délocalisation pondérée par 1/R_ij.
#   - H_S   : entropie de délocalisation pondérée par R_ij.
#
# Chaque descripteur est calculé sur deux domaines. Même convention
# de paires qu'en LDM : toutes les paires atomiques possibles dans
# chaque domaine.
#   - local  : atomes d'un cycle détecté   → toutes paires i<j du cycle
#   - global : atomes lourds du graphe (G) → toutes paires i<j lourds
#
# Convention d'indexation : indices_cycle est en base 1 (identifiants
# d'atomes affichés à l'utilisateur) ; les matrices numpy sont indexées
# en base 0. La conversion est faite explicitement à l'entrée du bloc.
# ============================================================

ENTROPY_EPSILON = 1.0e-12
ENTROPY_LOCAL_MODE_DEFAUT = "all"
ENTROPY_GLOBAL_MODE_DEFAUT = "all"


# ------------------------------------------------------------
# UTILITAIRES : entropie de Shannon en bits
# ------------------------------------------------------------

def entropy_compute_shannon_bits(valeurs) -> float:
    # H = -sum(p log2 p) avec p_k = valeurs_k / sum(valeurs).
    # Ignore les p_k <= epsilon ; retourne 0.0 si la somme est nulle.
    tableau = np.asarray(valeurs, dtype=float)
    if tableau.size == 0:
        return 0.0
    total = float(np.sum(tableau))
    if total <= ENTROPY_EPSILON:
        return 0.0
    probabilites = tableau / total
    entropie = 0.0
    for p in probabilites:
        if p > ENTROPY_EPSILON:
            entropie -= p * math.log2(p)
    return entropie


# ------------------------------------------------------------
# CONSTRUCTION DES ENSEMBLES V_X ET D_X
# ------------------------------------------------------------

def entropy_liste_atomes_lourds(numeros_atomiques) -> List[int]:
    # Renvoie les indices base 1 des atomes lourds (Z > 1).
    return [i + 1 for i, z in enumerate(numeros_atomiques) if z > 1]


def entropy_paires_toutes_dans_ensemble(atomes) -> List[Tuple[int, int]]:
    # Toutes les paires i < j de la liste d'atomes.
    atomes_tries = sorted(atomes)
    paires = []
    for i_pos in range(len(atomes_tries)):
        for j_pos in range(i_pos + 1, len(atomes_tries)):
            paires.append((atomes_tries[i_pos], atomes_tries[j_pos]))
    return paires


def entropy_paires_bond_graphe(atomes, connectivite) -> List[Tuple[int, int]]:
    # Uniquement les paires liées entre atomes lourds.
    ensemble = set(atomes)
    paires_vues = set()
    for i in atomes:
        voisins = connectivite.get(i, [])
        for j in voisins:
            if j in ensemble and j != i:
                paires_vues.add(tuple(sorted((i, j))))
    return sorted(paires_vues)


# ------------------------------------------------------------
# EXTRACTION DES TERMES DE LA MATRICE LDM
# ------------------------------------------------------------

def entropy_extraire_lambda(matrice_ldm, atomes) -> List[float]:
    # Termes diagonaux (base 1 -> base 0).
    return [float(matrice_ldm[i - 1, i - 1]) for i in atomes]


def entropy_extraire_delta(matrice_ldm, paires) -> List[float]:
    # Termes hors diagonale (base 1 -> base 0). Valeur absolue clippée en zéro pour rester >= 0.
    valeurs = []
    for (i, j) in paires:
        v = float(matrice_ldm[i - 1, j - 1])
        if abs(v) < ENTROPY_EPSILON:
            v = 0.0
        valeurs.append(v)
    return valeurs


def entropy_extraire_distances(matrice_distances, paires) -> List[float]:
    # Distances R_ij en Å (base 1 -> base 0).
    return [float(matrice_distances[i - 1, j - 1]) for (i, j) in paires]


# ------------------------------------------------------------
# ESTIMATION DES POPULATIONS N_i
# ------------------------------------------------------------

def entropy_estimer_populations_bassin(matrice_ldm, atomes) -> Dict[int, float]:
    # N_i = lambda_ii + 1/2 sum_{j != i, j in atomes} delta_ij.
    populations = {}
    ensemble = set(atomes)
    for i in atomes:
        lam = float(matrice_ldm[i - 1, i - 1])
        somme_delta = 0.0
        for j in ensemble:
            if j == i:
                continue
            somme_delta += float(matrice_ldm[i - 1, j - 1])
        populations[i] = lam + 0.5 * somme_delta
    return populations


# ------------------------------------------------------------
# CALCUL DE H_LDM (COMPLET) — décomposition + normalisation mu_LDM
# ------------------------------------------------------------

def entropy_calculer_H_LDM(matrice_ldm, atomes, paires) -> Dict[str, object]:
    lambdas = entropy_extraire_lambda(matrice_ldm, atomes)
    deltas = entropy_extraire_delta(matrice_ldm, paires)

    n_loc = len(lambdas)
    n_deloc = len(deltas)

    L = float(sum(lambdas))
    D = float(sum(deltas))
    T = L + D

    if T <= ENTROPY_EPSILON:
        # Cas dégénéré (matrice LDM nulle sur le domaine)
        return {
            "n_loc": n_loc, "n_deloc": n_deloc,
            "L": L, "D": D, "T": T,
            "w_loc": 0.0, "w_deloc": 0.0,
            "H_part": 0.0, "H_loc": 0.0, "H_deloc": 0.0,
            "H_LDM": 0.0,
            "H_min_chem": 0.0, "H_max_ref": 0.0,
            "mu_LDM_raw_percent": 0.0, "mu_LDM_clipped_percent": 0.0,
            "populations": {}
        }

    w_loc = L / T
    w_deloc = D / T

    H_loc = entropy_compute_shannon_bits(lambdas)
    H_deloc = entropy_compute_shannon_bits(deltas) if D > ENTROPY_EPSILON else 0.0

    # H_part = -(w_loc log2 w_loc + w_deloc log2 w_deloc)
    H_part = 0.0
    for w in (w_loc, w_deloc):
        if w > ENTROPY_EPSILON:
            H_part -= w * math.log2(w)

    H_LDM = H_part + w_loc * H_loc + w_deloc * H_deloc

    # Référence localisée chimique : delta = 0, lambda_ii = N_i.
    populations = entropy_estimer_populations_bassin(matrice_ldm, atomes)
    liste_Ni = [populations[i] for i in atomes]
    H_min_chem = entropy_compute_shannon_bits(liste_Ni)

    # Référence maximale : distribution uniforme sur N_loc + N_deloc bins.
    total_bins = n_loc + n_deloc
    H_max_ref = math.log2(total_bins) if total_bins > 1 else 0.0

    denom = H_max_ref - H_min_chem
    if abs(denom) > ENTROPY_EPSILON:
        mu_raw = 100.0 * (H_LDM - H_min_chem) / denom
    else:
        mu_raw = 0.0
    mu_clipped = max(0.0, min(100.0, mu_raw))

    return {
        "n_loc": n_loc, "n_deloc": n_deloc,
        "L": L, "D": D, "T": T,
        "w_loc": w_loc, "w_deloc": w_deloc,
        "H_part": H_part, "H_loc": H_loc, "H_deloc": H_deloc,
        "H_LDM": H_LDM,
        "H_min_chem": H_min_chem, "H_max_ref": H_max_ref,
        "mu_LDM_raw_percent": mu_raw,
        "mu_LDM_clipped_percent": mu_clipped,
        "populations": populations
    }


# ------------------------------------------------------------
# CALCULS DE H_Q (delta/R) ET H_S (delta*R)
# ------------------------------------------------------------

def entropy_calculer_H_pondere(matrice_ldm, matrice_distances, paires, ponderation: str) -> Dict[str, object]:
    # ponderation ∈ {"Q", "S"}. Q = delta_ij / R_ij ; S = delta_ij * R_ij.
    deltas = entropy_extraire_delta(matrice_ldm, paires)
    distances = entropy_extraire_distances(matrice_distances, paires)

    contributions = []
    for delta_ij, R_ij in zip(deltas, distances):
        if R_ij <= ENTROPY_EPSILON:
            continue
        if ponderation == "Q":
            contributions.append(delta_ij / R_ij)
        else:
            contributions.append(delta_ij * R_ij)

    n_deloc = len(contributions)
    D_w = float(sum(contributions))
    H = entropy_compute_shannon_bits(contributions)
    H_max = math.log2(n_deloc) if n_deloc > 1 else 0.0
    mu = (100.0 * H / H_max) if H_max > ENTROPY_EPSILON else 0.0
    mu = max(0.0, min(100.0, mu))

    return {
        "descriptor": "H_Q" if ponderation == "Q" else "H_S",
        "weighting": "delta_ij/R_ij" if ponderation == "Q" else "delta_ij*R_ij",
        "n_deloc": n_deloc,
        "D_weighted": D_w,
        "H": H,
        "H_max": H_max,
        "mu_percent": mu
    }


# ------------------------------------------------------------
# ORCHESTRATION LOCALE (par cycle) ET GLOBALE (atomes lourds)
# ------------------------------------------------------------

def entropy_analyser_cycle(matrice_ldm, matrice_distances, cycle_info) -> Dict[str, object]:
    indices_cycle = cycle_info["indices_cycle"]
    numero_cycle = cycle_info.get("numero_cycle", 0)

    atomes = list(indices_cycle)
    # Convention identique au LDM : toutes les paires i<j entre atomes du cycle
    # (donc les cordes internes en plus des liaisons du contour).
    paires = entropy_paires_toutes_dans_ensemble(atomes)

    resultat_H_LDM = entropy_calculer_H_LDM(matrice_ldm, atomes, paires)
    resultat_H_Q = entropy_calculer_H_pondere(matrice_ldm, matrice_distances, paires, "Q")
    resultat_H_S = entropy_calculer_H_pondere(matrice_ldm, matrice_distances, paires, "S")

    return {
        "scope": "cycle",
        "cycle_id": numero_cycle,
        "atoms": atomes,
        "pairs_mode": ENTROPY_LOCAL_MODE_DEFAUT,
        "H_LDM": resultat_H_LDM,
        "H_Q": resultat_H_Q,
        "H_S": resultat_H_S
    }


def entropy_analyser_global(matrice_ldm, matrice_distances, numeros_atomiques) -> Dict[str, object]:
    atomes = entropy_liste_atomes_lourds(numeros_atomiques)
    paires = entropy_paires_toutes_dans_ensemble(atomes)

    resultat_H_LDM = entropy_calculer_H_LDM(matrice_ldm, atomes, paires)
    resultat_H_Q = entropy_calculer_H_pondere(matrice_ldm, matrice_distances, paires, "Q")
    resultat_H_S = entropy_calculer_H_pondere(matrice_ldm, matrice_distances, paires, "S")

    return {
        "scope": "global",
        "cycle_id": "-",
        "atoms": atomes,
        "pairs_mode": ENTROPY_GLOBAL_MODE_DEFAUT,
        "H_LDM": resultat_H_LDM,
        "H_Q": resultat_H_Q,
        "H_S": resultat_H_S
    }


def entropy_preparer_resultats(
    matrice_ldm,
    matrice_distances,
    numeros_atomiques,
    cycles_classes
) -> List[Dict[str, object]]:
    # Renvoie une liste : [ligne par cycle traitable, ..., ligne global].
    resultats = []
    for cycle_info in cycles_classes:
        if cycle_info.get("statut_cycle") != STATUT_CYCLE_TRAITE:
            continue
        resultats.append(entropy_analyser_cycle(matrice_ldm, matrice_distances, cycle_info))
    resultats.append(entropy_analyser_global(matrice_ldm, matrice_distances, numeros_atomiques))
    return resultats


# ------------------------------------------------------------
# ÉCRITURE DES DÉTAILS ENTROPIE PAR CYCLE (style LDM/HOMA)
# ------------------------------------------------------------

def entropy_type_cycle_str(r, cycles_classes) -> Tuple[str, str, str]:
    """Renvoie (type_cycle, origine, statut) pour la ligne de résumé."""
    if r["scope"] == "global":
        return ("GLOBAL", "-", "-")
    if not cycles_classes:
        return ("-", "-", "-")
    for c in cycles_classes:
        if c.get("numero_cycle") == r["cycle_id"]:
            return (c.get("type_cycle", "-"),
                    c.get("origine_cycle", "-"),
                    c.get("statut_cycle", "-"))
    return ("-", "-", "-")


def entropy_ecrire_intro(fichier) -> None:
    fichier.write("===============================================================\n")
    fichier.write("              ENTROPY DESCRIPTORS REPORT (H_LDM / H_Q / H_S)\n")
    fichier.write("===============================================================\n")
    fichier.write(" Formulas    : H = - sum p_k log2 p_k   [bits]\n")
    fichier.write(" H_LDM       : full entropy on lambda_ii (diagonal) + delta_ij (off-diag.)\n")
    fichier.write("              H_LDM = H_part + w_loc * H_loc + w_deloc * H_deloc\n")
    fichier.write("              normalized u_LDM in [0,100] % between H_min_chem and H_max_ref\n")
    fichier.write(" H_Q         : delocalization entropy, weight q_ij = delta_ij / R_ij\n")
    fichier.write(" H_S         : delocalization entropy, weight s_ij = delta_ij * R_ij\n")
    fichier.write(" Convention  : same as LDM  ->  all pairs i<j inside the domain\n")
    fichier.write("               Local domain  : atoms of the detected cycle\n")
    fichier.write("               Global domain : heavy atoms (Z > 1) of the molecule\n")
    fichier.write("===============================================================\n\n")


def entropy_ecrire_detail_cycle(fichier, r, type_cycle_txt, origine_txt) -> None:
    h = r["H_LDM"]
    q = r["H_Q"]
    s = r["H_S"]

    fichier.write(" ------------------------------------------------------------\n")
    if r["scope"] == "cycle":
        fichier.write(f" Cycle number        : {r['cycle_id']}\n")
        fichier.write(f" Cycle type          : {type_cycle_txt}\n")
        fichier.write(f" Cycle origin        : {origine_txt}\n")
        fichier.write(f" Cycle size          : {h['n_loc']}\n")
        fichier.write(f" Atom indices        : {r['atoms']}\n")
    else:
        fichier.write(f" GLOBAL scope        : heavy-atom subgraph\n")
        fichier.write(f" Atoms (heavy)       : {r['atoms']}\n")
    fichier.write(f" Domain              : all pairs i<j\n")
    fichier.write(f" N_loc  (diagonals)  : {h['n_loc']}\n")
    fichier.write(f" N_deloc (off-diag.) : {h['n_deloc']}\n")
    fichier.write("\n")

    fichier.write(" H_LDM decomposition (complete entropy)\n")
    fichier.write(" ------------------------------\n")
    fichier.write(f" L (sum lambda_ii)         : {h['L']:>14.6f}\n")
    fichier.write(f" D (sum delta_ij)          : {h['D']:>14.6f}\n")
    fichier.write(f" T = L + D                 : {h['L'] + h['D']:>14.6f}\n")
    fichier.write(f" w_loc = L / T             : {h['w_loc']:>14.6f}\n")
    fichier.write(f" w_deloc = D / T           : {h['w_deloc']:>14.6f}\n")
    fichier.write(f" H_part                    : {h['H_part']:>14.6f} bits\n")
    fichier.write(f" H_loc                     : {h['H_loc']:>14.6f} bits\n")
    fichier.write(f" H_deloc                   : {h['H_deloc']:>14.6f} bits\n")
    fichier.write(f" H_LDM                     : {h['H_LDM']:>14.6f} bits\n")
    fichier.write(f" H_min_chem (localized)    : {h['H_min_chem']:>14.6f} bits\n")
    fichier.write(f" H_max_ref                 : {h['H_max_ref']:>14.6f} bits   [log2(N_loc+N_deloc)]\n")
    fichier.write(f" u_LDM raw     %           : {h['mu_LDM_raw_percent']:>14.3f}\n")
    fichier.write(f" u_LDM clipped %           : {h['mu_LDM_clipped_percent']:>14.3f}\n")
    fichier.write("\n")

    fichier.write(" H_Q analysis (weight q_ij = delta_ij / R_ij)\n")
    fichier.write(" ------------------------------\n")
    fichier.write(f" D_Q (sum q_ij)            : {q['D_weighted']:>14.6f}\n")
    fichier.write(f" H_Q                       : {q['H']:>14.6f} bits\n")
    fichier.write(f" H_max_Q                   : {q['H_max']:>14.6f} bits   [log2(N_deloc)]\n")
    fichier.write(f" u_Q %                     : {q['mu_percent']:>14.3f}\n")
    fichier.write("\n")

    fichier.write(" H_S analysis (weight s_ij = delta_ij * R_ij)\n")
    fichier.write(" ------------------------------\n")
    fichier.write(f" D_S (sum s_ij)            : {s['D_weighted']:>14.6f}\n")
    fichier.write(f" H_S                       : {s['H']:>14.6f} bits\n")
    fichier.write(f" H_max_S                   : {s['H_max']:>14.6f} bits   [log2(N_deloc)]\n")
    fichier.write(f" u_S %                     : {s['mu_percent']:>14.3f}\n")
    fichier.write(" ------------------------------------------------------------\n\n")


# ------------------------------------------------------------
# SUMMARY FINAL — TROIS TABLEAUX SÉPARÉS (H_LDM, H_Q, H_S)
#   Même style que le summary LDM et HOMA.
# ------------------------------------------------------------

def _entropy_largeur_indices(resultats_entropie, mini: int = 22, marge: int = 2) -> int:
    if not resultats_entropie:
        return mini
    largeurs = [len(str(r["atoms"])) for r in resultats_entropie]
    return max(mini, max(largeurs) + marge)


def entropy_ecrire_summary_H_LDM(fichier, resultats_entropie, cycles_classes) -> None:
    largeur = _entropy_largeur_indices(resultats_entropie)
    titre = (
        f" {'No.':<5} {'Type':<13} {'Size':<5} {'Indices':<{largeur}} "
        f"{'Origine':<10} "
        f"{'Nloc':>5} {'Ndel':>5} "
        f"{'L':>10} {'D':>10} "
        f"{'w_loc':>7} {'w_del':>7} "
        f"{'H_part':>8} {'H_loc':>8} {'H_del':>8} "
        f"{'H_LDM':>8} {'Hmin_ch':>9} {'Hmax_rf':>9} "
        f"{'u_raw %':>9} {'u_clip %':>9}"
    )
    fichier.write(" --- H_LDM (complete entropy : lambda_ii + delta_ij) ---\n")
    fichier.write(titre + "\n")
    fichier.write(" " + "-" * (len(titre) - 1) + "\n")
    for r in resultats_entropie:
        h = r["H_LDM"]
        type_cyc, origine, _ = entropy_type_cycle_str(r, cycles_classes)
        no_str = str(r["cycle_id"]) if r["scope"] == "cycle" else "TOT"
        fichier.write(
            f" {no_str:<5s} {type_cyc:<13s} {h['n_loc']:<5d} "
            f"{str(r['atoms']):<{largeur}s} "
            f"{origine:<10s} "
            f"{h['n_loc']:>5d} {h['n_deloc']:>5d} "
            f"{h['L']:>10.4f} {h['D']:>10.4f} "
            f"{h['w_loc']:>7.4f} {h['w_deloc']:>7.4f} "
            f"{h['H_part']:>8.4f} {h['H_loc']:>8.4f} {h['H_deloc']:>8.4f} "
            f"{h['H_LDM']:>8.4f} {h['H_min_chem']:>9.4f} {h['H_max_ref']:>9.4f} "
            f"{h['mu_LDM_raw_percent']:>9.3f} {h['mu_LDM_clipped_percent']:>9.3f}\n"
        )
    fichier.write("\n")


def _entropy_ecrire_summary_HQS(fichier, resultats_entropie, cycles_classes, cle: str, titre_court: str, ponderation_txt: str) -> None:
    largeur = _entropy_largeur_indices(resultats_entropie)
    titre = (
        f" {'No.':<5} {'Type':<13} {'Size':<5} {'Indices':<{largeur}} "
        f"{'Origine':<10} "
        f"{'Ndel':>5} "
        f"{'D_w':>12} "
        f"{'H':>10} "
        f"{'H_max':>10} "
        f"{'u %':>10}"
    )
    fichier.write(f" --- {titre_court} (delocalization entropy, weight {ponderation_txt}) ---\n")
    fichier.write(titre + "\n")
    fichier.write(" " + "-" * (len(titre) - 1) + "\n")
    for r in resultats_entropie:
        hh = r[cle]
        type_cyc, origine, _ = entropy_type_cycle_str(r, cycles_classes)
        no_str = str(r["cycle_id"]) if r["scope"] == "cycle" else "TOT"
        n_loc = r["H_LDM"]["n_loc"]
        fichier.write(
            f" {no_str:<5s} {type_cyc:<13s} {n_loc:<5d} "
            f"{str(r['atoms']):<{largeur}s} "
            f"{origine:<10s} "
            f"{hh['n_deloc']:>5d} "
            f"{hh['D_weighted']:>12.6f} "
            f"{hh['H']:>10.4f} "
            f"{hh['H_max']:>10.4f} "
            f"{hh['mu_percent']:>10.3f}\n"
        )
    fichier.write("\n")


def entropy_ecrire_summary_H_Q(fichier, resultats_entropie, cycles_classes) -> None:
    _entropy_ecrire_summary_HQS(fichier, resultats_entropie, cycles_classes,
                                cle="H_Q", titre_court="H_Q",
                                ponderation_txt="q_ij = delta_ij / R_ij")


def entropy_ecrire_summary_H_S(fichier, resultats_entropie, cycles_classes) -> None:
    _entropy_ecrire_summary_HQS(fichier, resultats_entropie, cycles_classes,
                                cle="H_S", titre_court="H_S",
                                ponderation_txt="s_ij = delta_ij * R_ij")


# ------------------------------------------------------------
# POINT D'ENTRÉE UNIQUE : intro + details + 3 summaries
# ------------------------------------------------------------

def entropy_ecrire_section(fichier, resultats_entropie, cycles_classes=None) -> None:
    if not resultats_entropie:
        return

    entropy_ecrire_intro(fichier)

    # --- Détail par cycle (style HOMA/LDM par cycle) ---
    fichier.write("===============================================================\n")
    fichier.write("              DETAILED ENTROPY BY CYCLE\n")
    fichier.write("===============================================================\n\n")
    for r in resultats_entropie:
        type_cyc, origine, _ = entropy_type_cycle_str(r, cycles_classes)
        entropy_ecrire_detail_cycle(fichier, r, type_cyc, origine)

    # --- Summary final : 3 tableaux séparés (résultats seulement) ---
    fichier.write("===============================================================\n")
    fichier.write("              ENTROPY FINAL SUMMARY\n")
    fichier.write("===============================================================\n")
    fichier.write(" All entropies in bits (log2). u in percent.\n\n")

    entropy_ecrire_summary_H_LDM(fichier, resultats_entropie, cycles_classes)
    entropy_ecrire_summary_H_Q(fichier, resultats_entropie, cycles_classes)
    entropy_ecrire_summary_H_S(fichier, resultats_entropie, cycles_classes)


# ------------------------------------------------------------
# AFFICHAGE TERMINAL COMPACT (3ème tableau du résumé terminal)
# ------------------------------------------------------------

def entropy_afficher_terminal(resultats_entropie) -> None:
    if not resultats_entropie:
        return
    print("=" * 130)
    print("RÉSUMÉ TERMINAL — DESCRIPTEURS ENTROPIQUES (bits)")
    print("=" * 130)
    print(
        f"{'Scope':<7} {'Cycle':<6} {'Atoms':<20} "
        f"{'H_LDM':>8} {'u_LDM%':>9} {'H_loc':>8} {'H_del':>8} "
        f"{'H_Q':>8} {'u_Q %':>8} {'H_S':>8} {'u_S %':>8}"
    )
    for r in resultats_entropie:
        atomes_txt = str(r["atoms"])
        if len(atomes_txt) > 18:
            atomes_txt = atomes_txt[:15] + "...]"
        h = r["H_LDM"]
        q = r["H_Q"]
        s = r["H_S"]
        no_str = str(r["cycle_id"]) if r["scope"] == "cycle" else "-"
        print(
            f"{r['scope']:<7} {no_str:<6} {atomes_txt:<20} "
            f"{h['H_LDM']:>8.4f} {h['mu_LDM_clipped_percent']:>9.3f} "
            f"{h['H_loc']:>8.4f} {h['H_deloc']:>8.4f} "
            f"{q['H']:>8.4f} {q['mu_percent']:>8.3f} "
            f"{s['H']:>8.4f} {s['mu_percent']:>8.3f}"
        )


# ============================================================
# BLOC 14 — AFFICHAGE TERMINAL DU RÉSUMÉ (tableaux LDM)
# ============================================================
# Formatage sur mesure du grand tableau récapitulatif LDM affiché
# en fin d'exécution. Largeurs de colonnes adaptatives selon les
# indices de cycle.
# ============================================================

# ==============================
# FONCTION DE CALCUL DE LA LARGEUR DE LA COLONNE INDICES POUR LE TERMINAL
# ==============================

def calculer_largeur_indices_terminal(
    resultats_classes: List[Dict[str, object]],
    largeur_minimale: int = 22,
    marge: int = 2
) -> int:
    if len(resultats_classes) == 0:
        return largeur_minimale
    textes_indices = [str(resultat["indices_cycle"]) for resultat in resultats_classes]
    largeur_maximale = max(len(texte) for texte in textes_indices) + marge
    return max(largeur_minimale, largeur_maximale)


# ==============================
# FONCTION D’AFFICHAGE DU RÉSUMÉ TERMINAL DES CYCLES TRAITÉS
# ==============================
def afficher_resume_terminal_cycles(
    resultats_classes: List[Dict[str, object]],
    analyse_globale: Dict[str, object] = None
) -> None:
    if len(resultats_classes) == 0:
        print("Aucun cycle traité à afficher.")
        return

    largeur_indices = calculer_largeur_indices_terminal(resultats_classes, largeur_minimale=22)

    qg = analyse_globale["q_global_normalise"] if analyse_globale is not None else float("nan")
    sg = analyse_globale["s_global"] if analyse_globale is not None else float("nan")
    shg = analyse_globale["s_hom_global"] if analyse_globale is not None else float("nan")

    print("=" * 220)
    print("RÉSUMÉ TERMINAL DES CYCLES TRAITÉS")
    print("=" * 220)
    print(
        f"{'No.':<4} {'Type':<13} {'Size':<5} "
        f"{'Indices':<{largeur_indices}} "
        f"{'Frob full':>12} {'Offdiag.':>10} {'Diag.':>10} "
        f"{'RMS mat':>10} {'RMS off':>10} "
        f"{'CT':>10} {'CT %':>10} "
        f"{'Q Frob':>10} {'Q RMS':>10} "
        f"{'S_hom':>10} "
        f"{'Q(G)':>10} {'S(G)':>10} {'S-hom(G)':>10}"
    )
    print("-" * 220)

    for resultat in resultats_classes:
        print(
            f"{resultat['numero_cycle']:<4d} "
            f"{resultat['type_cycle']:<13s} "
            f"{resultat['taille_cycle']:<5d} "
            f"{str(resultat['indices_cycle']):<{largeur_indices}s} "
            f"{resultat['distance_frobenius_complete']:12.6f} "
            f"{resultat['contribution_hors_diagonale']:10.6f} "
            f"{resultat['contribution_diagonale']:10.6f} "
            f"{resultat['rms_matricielle']:10.6f} "
            f"{resultat['rms_hors_diagonale']:10.6f} "
            f"{resultat['ct_absolu']:10.6f} "
            f"{resultat['ct_relatif_pourcent']:10.6f} "
            f"{resultat['q_frob_offdiag']:10.6f} "
            f"{resultat['q_rms_offdiag']:10.6f} "
            f"{resultat['s_cycle_hom']:10.6f} "
            f"{qg:10.6f} "
            f"{sg:10.6f} "
            f"{shg:10.6f}"
        )

    print("=" * 220)


# ==============================
# FONCTION D’AFFICHAGE DU MESSAGE FINAL
# ==============================

def afficher_message_final(nom_fichier_sortie: str) -> None:
    print(f"Résultats détaillés enregistrés dans : {nom_fichier_sortie}")


# ============================================================
# BLOC 15 — FONCTION main() HISTORIQUE (LDM SEUL)
# ============================================================
# Point d'entrée historique du programme LDM autonome (avant le
# passage à HOMA+LDM+entropie). Conservé pour rétro-compatibilité
# mais non utilisé par __main__.
# ============================================================

def preparer_cycles_classes(
    cycles_dicts: List[Dict[str, object]],
    connectivite: Dict[int, List[int]],
    numeros_atomiques: List[int] = None
) -> List[Dict[str, object]]:
    cycles_classes = []

    for element in cycles_dicts:
        cycle = element["indices_cycle"]
        origine_cycle = element["origine_cycle"]

        type_cycle = determiner_type_cycle_v2(cycle, connectivite)
        statut_cycle = determiner_statut_cycle_v2(cycle, connectivite)
        cordes = extraire_cordes_cycle(cycle, connectivite)

        # V09 : un cycle manuel libre est toujours traité, même si son contour
        # ne respecte pas la connectivité détectée. Les cordes internes restent
        # évaluées avec la connectivité réelle pour permettre aux cycles manuels
        # de contribuer aussi aux cas fusionnés/périphériques quand c'est possible.
        est_cycle_manuel_libre = (origine_cycle == ORIGINE_MANUEL)
        est_cycle_fusionne_force = (origine_cycle == ORIGINE_FUSED)
        if est_cycle_manuel_libre:
            statut_cycle = STATUT_CYCLE_TRAITE
            if len(cordes) == 0:
                type_cycle = TYPE_CYCLE_ELEMENTAIRE
            else:
                type_cycle = TYPE_CYCLE_PERIPHERAL
        elif est_cycle_fusionne_force:
            # Cycle périphérique créé par différence symétrique des arêtes
            # depuis cycles auto/manuels. Il doit rester calculable même si
            # son contour n'est pas un cycle détecté par la connectivité.
            statut_cycle = STATUT_CYCLE_TRAITE
            type_cycle = TYPE_CYCLE_PERIPHERAL

        # Z des atomes du cycle dans l'ordre des positions, lu depuis le .dat.
        # indices_cycle est 1-based ; numeros_atomiques est 0-based.
        if numeros_atomiques is not None:
            numeros_atomiques_cycle = [
                int(numeros_atomiques[indice_atome - 1]) for indice_atome in cycle
            ]
        else:
            numeros_atomiques_cycle = None

        cycles_classes.append({
            "type_cycle": type_cycle,
            "indices_cycle": cycle,
            "taille_cycle": len(cycle),
            "statut_cycle": statut_cycle,
            "origine_cycle": origine_cycle,
            "nombre_cordes": len(cordes),
            "cordes_cycle": cordes,
            "connectivite_cycle": connectivite,
            "numeros_atomiques_cycle": numeros_atomiques_cycle,
            "est_cycle_manuel_libre": est_cycle_manuel_libre,
            "est_cycle_fusionne_force": est_cycle_fusionne_force
        })

    cycles_classes = trier_cycles(cycles_classes)

    for numero_cycle, element in enumerate(cycles_classes, start=1):
        element["numero_cycle"] = numero_cycle

    return cycles_classes


def main() -> None:
    print("===============================================================")
    print("         PROGRAMME D’ANALYSE DE L’AROMATICITÉ PAR LDM")
    print("===============================================================")
    print()

    nom_fichier_ldm = demander_fichier_ldm()
    nom_fichier_dist = demander_fichier_dist()

    numeros_atomiques_ldm, matrice_ldm_complete = lire_fichier_dat(nom_fichier_ldm)
    numeros_atomiques_dist, matrice_distances = lire_fichier_dat(nom_fichier_dist)

    matrice_distances = convertir_matrice_bohr_en_angstrom(matrice_distances)

    verifier_compatibilite_fichiers(
        numeros_atomiques_ldm=numeros_atomiques_ldm,
        matrice_ldm=matrice_ldm_complete,
        numeros_atomiques_dist=numeros_atomiques_dist,
        matrice_dist=matrice_distances
    )

    parametres_reference = demander_parametres_reference()

    connectivite_provisoire, connectivite_finale, rayons_adaptes = construire_connectivite_automatique(
        numeros_atomiques=numeros_atomiques_dist,
        matrice_distances=matrice_distances,
        tolerance=TOLERANCE_CONNECTIVITE
    )

    types_carbone = calculer_types_carbone(
        numeros_atomiques=numeros_atomiques_dist,
        connectivite_provisoire=connectivite_provisoire
    )

    connectivite_cycles = filtrer_connectivite_sans_hydrogenes(
        numeros_atomiques=numeros_atomiques_dist,
        connectivite=connectivite_finale
    )

    cycles_detectes_auto = trouver_cycles(
        connectivite=connectivite_cycles,
        longueur_min=3,
        longueur_max=24
    )

    cycles_detectes_auto = filtrer_cycles_sans_contrainte_hybridation(
        cycles=cycles_detectes_auto
    )

    cycles_manuels = demander_cycles_manuels_optionnels(
        connectivite=connectivite_cycles,
        nombre_total_atomes=len(numeros_atomiques_dist)
    )

    cycles_manuels = filtrer_cycles_sans_contrainte_hybridation(
        cycles=cycles_manuels
    )

    cycles_detectes = fusionner_cycles_auto_manuels_et_generer_fusionnes(
        cycles_automatiques=cycles_detectes_auto,
        cycles_manuels=cycles_manuels
    )

    cycles_classes = preparer_cycles_classes(
        cycles_dicts=cycles_detectes,
        connectivite=connectivite_cycles,
        numeros_atomiques=numeros_atomiques_dist
    )

    rapport_preliminaire = generer_rapport_preliminaire(
        numeros_atomiques=numeros_atomiques_dist,
        connectivite_provisoire=connectivite_provisoire,
        rayons_adaptes=rayons_adaptes,
        connectivite_finale=connectivite_finale,
        cycles_classes=cycles_classes
    )

    resultats_cycles = preparer_resultats_cycles_analyses(
        matrice_ldm_complete=matrice_ldm_complete,
        matrice_distances_complete=matrice_distances,
        cycles_classes=cycles_classes,
        parametres_reference=parametres_reference
    )

    resultats_classes = preparer_classement_final(resultats_cycles)

    analyse_globale = preparer_analyse_globale_atomes_lourds(
        numeros_atomiques=numeros_atomiques_dist,
        matrice_ldm_complete=matrice_ldm_complete,
        matrice_distances_complete=matrice_distances,
        connectivite_finale=connectivite_finale
    )

    nom_fichier_sortie = generer_nom_fichier_ldm(nom_fichier_ldm)

    enregistrer_fichier_ldm(
        nom_fichier_sortie=nom_fichier_sortie,
        nom_fichier_ldm=nom_fichier_ldm,
        nom_fichier_dist=nom_fichier_dist,
        rapport_preliminaire=rapport_preliminaire,
        parametres_reference=parametres_reference,
        cycles_classes=cycles_classes,
        resultats_classes=resultats_classes,
        analyse_globale=analyse_globale
    )

    afficher_resume_terminal_cycles(
        resultats_classes=resultats_classes,
        analyse_globale=analyse_globale
    )
    afficher_message_final(nom_fichier_sortie=nom_fichier_sortie)




# ============================================================
# BLOC 16 — HOMA INTÉGRÉ SUR LES MÊMES CYCLES QUE LDM
# ============================================================
# Implémentation généralisée du HOMA (Kruszewski-Krygowski, 1972)
# étendue aux liaisons hétéroatomiques via les paramètres alpha/R_opt
# des tables internes. Réutilise les cycles déjà validés pour LDM.
# ============================================================
# ============================================================

HOMA_PARAMS = {
    "CC": {"Ropt": 1.392, "alpha": 153.37, "source": "HOMAc"},
    "CN": {"Ropt": 1.333, "alpha": 111.83, "source": "HOMAc"},
    "NN": {"Ropt": 1.318, "alpha": 98.99, "source": "HOMAc"},
    "CO": {"Ropt": 1.315, "alpha": 335.16, "source": "HOMAc"},
    "CS": {"Ropt": 1.672, "alpha": 74.40, "source": "HOMHED-provisional"},
}

HOMA_BOND_ORDER = {"C": 1, "N": 2, "O": 3, "S": 4, "P": 5, "F": 6, "Cl": 7, "H": 8}
HOMA_Z_TO_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl"}


def homa_symbole_atome(z: int) -> str:
    return HOMA_Z_TO_SYMBOL.get(z, f"Z{z}")


def homa_cle_liaison(z1: int, z2: int) -> str:
    s1 = homa_symbole_atome(z1)
    s2 = homa_symbole_atome(z2)
    s1, s2 = sorted([s1, s2], key=lambda s: HOMA_BOND_ORDER.get(s, 999))
    return s1 + s2


def homa_extraire_liaisons_cycle(cycle: List[int]) -> List[Tuple[int, int]]:
    liaisons = []
    n = len(cycle)
    for i in range(n):
        liaisons.append((cycle[i], cycle[(i + 1) % n]))
    return liaisons


def homa_calculer_en_geo_generalise(details: List[Dict[str, object]]) -> Dict[str, object]:
    n_total = len(details)
    groupes: Dict[str, List[Dict[str, object]]] = {}
    for d in details:
        groupes.setdefault(d["bond_type"], []).append(d)

    en_by_type: Dict[str, float] = {}
    geo_by_type: Dict[str, float] = {}
    r_av_by_type: Dict[str, float] = {}
    count_by_type: Dict[str, int] = {}

    for bond_type, items in groupes.items():
        n_t = len(items)
        alpha = float(items[0]["alpha"])
        r_opt = float(items[0]["Ropt"])
        distances = [float(x["distance"]) for x in items]
        r_av_t = sum(distances) / n_t
        en_t = (n_t / n_total) * alpha * (r_av_t - r_opt) ** 2
        geo_t = (alpha / n_total) * sum((r - r_av_t) ** 2 for r in distances)
        en_by_type[bond_type] = float(en_t)
        geo_by_type[bond_type] = float(geo_t)
        r_av_by_type[bond_type] = float(r_av_t)
        count_by_type[bond_type] = n_t

    en = float(sum(en_by_type.values()))
    geo = float(sum(geo_by_type.values()))
    ratio = en / geo if geo != 0 else float("inf")
    homa = 1.0 - en - geo
    r_av = sum(float(d["distance"]) for d in details) / n_total

    return {
        "homa": float(homa),
        "penalty_total": float(en + geo),
        "en": en,
        "geo": geo,
        "ratio_en_geo": ratio,
        "r_av": float(r_av),
        "en_by_type": en_by_type,
        "geo_by_type": geo_by_type,
        "penalty_by_type": {k: en_by_type[k] + geo_by_type[k] for k in en_by_type},
        "r_av_by_type": r_av_by_type,
        "count_by_type": count_by_type,
    }


def homa_calculer_homa_generalise(
    numeros_atomiques: List[int],
    matrice_distances_angstrom: np.ndarray,
    indices_cycle: List[int],
) -> Dict[str, object]:
    liaisons = homa_extraire_liaisons_cycle(indices_cycle)
    details = []
    missing = []

    for a, b in liaisons:
        z_a = numeros_atomiques[a - 1]
        z_b = numeros_atomiques[b - 1]
        bond_type = homa_cle_liaison(z_a, z_b)
        distance = float(matrice_distances_angstrom[a - 1, b - 1])

        if bond_type not in HOMA_PARAMS:
            missing.append(bond_type)
            continue

        r_opt = HOMA_PARAMS[bond_type]["Ropt"]
        alpha = HOMA_PARAMS[bond_type]["alpha"]
        penalty = alpha * (distance - r_opt) ** 2
        details.append({
            "a": a,
            "b": b,
            "za": z_a,
            "zb": z_b,
            "bond_type": bond_type,
            "distance": distance,
            "Ropt": r_opt,
            "alpha": alpha,
            "penalty": float(penalty),
            "source": HOMA_PARAMS[bond_type]["source"],
        })

    if missing:
        return {"valid": False, "missing_bond_types": sorted(set(missing)), "details": details}

    decomp = homa_calculer_en_geo_generalise(details)
    return {
        "valid": True,
        **decomp,
        "details": details,
        "bond_types": [d["bond_type"] for d in details],
    }


def homa_label_depuis_cycle_info(cycle_info: Dict[str, object]) -> str:
    if cycle_info.get("origine_cycle") == ORIGINE_MANUEL:
        if cycle_info.get("type_cycle") == TYPE_CYCLE_PERIPHERAL:
            return "manual/peripheral"
        return "manual"
    type_cycle = cycle_info.get("type_cycle")
    if type_cycle == TYPE_CYCLE_ELEMENTAIRE:
        return "local"
    if type_cycle == TYPE_CYCLE_PERIPHERAL:
        return "peripheral"
    return str(type_cycle).lower()


def homa_preparer_resultats_cycles(
    numeros_atomiques: List[int],
    matrice_distances_angstrom: np.ndarray,
    cycles_classes: List[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    resultats_cycles: List[Dict[str, object]] = []
    skipped_cycles: List[Dict[str, object]] = []

    for cycle_info in cycles_classes:
        if cycle_info["statut_cycle"] != STATUT_CYCLE_TRAITE:
            continue
        indices_cycle = cycle_info["indices_cycle"]
        calc = homa_calculer_homa_generalise(numeros_atomiques, matrice_distances_angstrom, indices_cycle)
        if not calc["valid"]:
            skipped_cycles.append({
                "numero_cycle_ldm": cycle_info["numero_cycle"],
                "indices_cycle": indices_cycle,
                "missing_bond_types": calc["missing_bond_types"],
            })
            continue
        resultats_cycles.append({
            "numero_cycle": cycle_info["numero_cycle"],
            "label_cycle": homa_label_depuis_cycle_info(cycle_info),
            "taille_cycle": len(indices_cycle),
            "indices_cycle": indices_cycle,
            "origine_cycle": cycle_info["origine_cycle"],
            "statut_cycle": cycle_info["statut_cycle"],
            **calc,
        })

    return resultats_cycles, skipped_cycles


def homa_extraire_liaisons_uniques_cycles(resultats_cycles: List[Dict[str, object]]) -> List[Tuple[int, int]]:
    s: Set[Tuple[int, int]] = set()
    for res in resultats_cycles:
        for a, b in homa_extraire_liaisons_cycle(res["indices_cycle"]):
            s.add(tuple(sorted((a, b))))
    return sorted(s)


def homa_calculer_homa_totale_generalisee(
    numeros_atomiques: List[int],
    matrice_distances_angstrom: np.ndarray,
    resultats_cycles: List[Dict[str, object]],
) -> object:
    liaisons_uniques = homa_extraire_liaisons_uniques_cycles(resultats_cycles)
    if not liaisons_uniques:
        return None

    details = []
    missing = []
    for a, b in liaisons_uniques:
        z_a = numeros_atomiques[a - 1]
        z_b = numeros_atomiques[b - 1]
        bond_type = homa_cle_liaison(z_a, z_b)
        distance = float(matrice_distances_angstrom[a - 1, b - 1])
        if bond_type not in HOMA_PARAMS:
            missing.append(bond_type)
            continue
        r_opt = HOMA_PARAMS[bond_type]["Ropt"]
        alpha = HOMA_PARAMS[bond_type]["alpha"]
        details.append({
            "a": a,
            "b": b,
            "bond_type": bond_type,
            "distance": distance,
            "Ropt": r_opt,
            "alpha": alpha,
            "penalty": float(alpha * (distance - r_opt) ** 2),
            "source": HOMA_PARAMS[bond_type]["source"],
        })

    if missing:
        return {"valid": False, "missing_bond_types": sorted(set(missing))}

    decomp = homa_calculer_en_geo_generalise(details)
    return {
        "valid": True,
        "nombre_liaisons_uniques": len(details),
        "liaisons_uniques": liaisons_uniques,
        **decomp,
        "details": details,
    }


def homa_format_dict_float(d: Dict[str, float]) -> str:
    if not d:
        return "-"
    return "; ".join(f"{k}:{v:.6f}" for k, v in sorted(d.items()))


def homa_afficher_tableau_resume_cycles(
    resultats_cycles: List[Dict[str, object]],
    homa_totale_data: object = None,
    fichier=None,
) -> None:
    def ecrire(txt: str):
        if fichier is None:
            print(txt)
        else:
            fichier.write(txt + "\n")

    if not resultats_cycles:
        ecrire("Aucun cycle HOMA calculable.")
        return

    indices_textes = [",".join(str(x) for x in r["indices_cycle"]) for r in resultats_cycles]
    largeur_indices = max([40] + [len(x) + 2 for x in indices_textes])
    largeur_bonds = 22
    titre = (
        f" {'Cycle':<6} {'Type':<17} {'Size':<5} "
        f"{'Indices':<{largeur_indices}} {'Bonds':<{largeur_bonds}} "
        f"{'Rav(Å)':>10} {'EN':>12} {'GEO':>12} {'EN/GEO':>12} {'HOMA':>12}"
    )
    sep = " " + "=" * (len(titre) - 1)
    ecrire(sep)
    ecrire(" SUMMARY OF GENERALIZED HOMA ANALYSIS WITH EN/GEO")
    ecrire(sep)
    ecrire(titre)
    ecrire(" " + "-" * (len(titre) - 1))

    for r, indices_txt in zip(resultats_cycles, indices_textes):
        bonds_txt = ",".join(r["bond_types"])
        if len(bonds_txt) > largeur_bonds:
            bonds_txt = bonds_txt[:largeur_bonds - 3] + "..."
        ratio_txt = "inf" if math.isinf(r["ratio_en_geo"]) else f"{r['ratio_en_geo']:.6f}"
        ecrire(
            f" {r['numero_cycle']:<6d} {r['label_cycle']:<17s} {r['taille_cycle']:<5d} "
            f"{indices_txt:<{largeur_indices}} {bonds_txt:<{largeur_bonds}} "
            f"{r['r_av']:>10.6f} {r['en']:>12.6f} {r['geo']:>12.6f} {ratio_txt:>12} {r['homa']:>12.6f}"
        )

    if homa_totale_data and homa_totale_data.get("valid"):
        ecrire(" " + "-" * (len(titre) - 1))
        ratio_tot_txt = "inf" if math.isinf(homa_totale_data["ratio_en_geo"]) else f"{homa_totale_data['ratio_en_geo']:.6f}"
        ecrire(
            f" {'TOTAL':<6} {'global':<17} {homa_totale_data['nombre_liaisons_uniques']:<5d} "
            f"{'unique cycle bonds':<{largeur_indices}} {'mixed':<{largeur_bonds}} "
            f"{homa_totale_data['r_av']:>10.6f} {homa_totale_data['en']:>12.6f} "
            f"{homa_totale_data['geo']:>12.6f} {ratio_tot_txt:>12} {homa_totale_data['homa']:>12.6f}"
        )
    ecrire(sep)


def homa_ecrire_details(
    fichier,
    resultats_cycles: List[Dict[str, object]],
    homa_totale_data: object,
    skipped_cycles: List[Dict[str, object]],
) -> None:
    fichier.write("===============================================================\n")
    fichier.write("              GENERALIZED HOMA ANALYSIS REPORT\n")
    fichier.write("===============================================================\n")
    fichier.write(" Formula     : HOMA = 1 - EN - GEO ; EN/GEO generalized by bond type\n")
    fichier.write(" Cycle source: same validated AUTO/MANUAL cycle list used by LDM\n")
    fichier.write(" Manual rule : manual cycles are evaluated from their ordered contour even if connectivity is not respected\n")
    fichier.write("===============================================================\n\n")

    fichier.write(" Parameters used\n")
    fichier.write(" ------------------------------\n")
    for bt, p in HOMA_PARAMS.items():
        fichier.write(
            f" {bt:<3s} | Ropt = {p['Ropt']:>8.4f} Å | "
            f"alpha = {p['alpha']:>8.2f} Å^-2 | source = {p['source']}\n"
        )
    fichier.write("\n")

    if skipped_cycles:
        fichier.write(" Cycles skipped because of missing HOMA bond parameters\n")
        fichier.write(" ------------------------------\n")
        for item in skipped_cycles:
            fichier.write(
                f" Cycle {item['numero_cycle_ldm']} | indices = {item['indices_cycle']} | "
                f"missing = {item['missing_bond_types']}\n"
            )
        fichier.write("\n")

    fichier.write("===============================================================\n")
    fichier.write("              DETAILED GENERALIZED HOMA BY CYCLE\n")
    fichier.write("===============================================================\n\n")
    for r in resultats_cycles:
        fichier.write(" ------------------------------------------------------------\n")
        fichier.write(f" Cycle number        : {r['numero_cycle']}\n")
        fichier.write(f" Cycle type          : {r['label_cycle']}\n")
        fichier.write(f" Cycle origin        : {r['origine_cycle']}\n")
        fichier.write(f" Cycle size          : {r['taille_cycle']}\n")
        fichier.write(f" Atom indices        : {r['indices_cycle']}\n")
        fichier.write(f" Bond types          : {r['bond_types']}\n")
        fichier.write("\n")
        fichier.write(" Bond details\n")
        fichier.write(" ------------------------------\n")
        for d in r["details"]:
            fichier.write(
                f" Bond {d['a']:>4d}-{d['b']:<4d} | {d['bond_type']:<3s} | "
                f"R = {d['distance']:>10.6f} Å | "
                f"Ropt = {d['Ropt']:>8.4f} Å | "
                f"alpha = {d['alpha']:>8.2f} | "
                f"penalty = {d['penalty']:>12.6f} | "
                f"source = {d['source']}\n"
            )
        fichier.write("\n")
        ratio_txt = "inf" if math.isinf(r["ratio_en_geo"]) else f"{r['ratio_en_geo']:.6f}"
        fichier.write(" Generalized EN/GEO result\n")
        fichier.write(" ------------------------------\n")
        fichier.write(f" Rav global          = {r['r_av']:>12.6f} Å\n")
        fichier.write(f" EN                  = {r['en']:>12.6f}\n")
        fichier.write(f" GEO                 = {r['geo']:>12.6f}\n")
        fichier.write(f" EN/GEO ratio        = {ratio_txt:>12s}\n")
        fichier.write(f" HOMA generalized    = {r['homa']:>12.6f}\n")
        fichier.write(f" EN by type          = {homa_format_dict_float(r['en_by_type'])}\n")
        fichier.write(f" GEO by type         = {homa_format_dict_float(r['geo_by_type'])}\n")
        fichier.write(f" Rav by type         = {homa_format_dict_float(r['r_av_by_type'])}\n")
        fichier.write(" ------------------------------------------------------------\n\n")

    fichier.write("\n")
    homa_afficher_tableau_resume_cycles(resultats_cycles, homa_totale_data, fichier=fichier)
    fichier.write("\n")


def generer_nom_fichier_integre(nom_fichier_ldm: str) -> str:
    nom_simple = os.path.basename(nom_fichier_ldm)
    nom_sans_extension = os.path.splitext(nom_simple)[0]
    return nom_sans_extension + ".HOMA_LDM"


# ============================================================
# AroX v0.3.2 — STATIC HOMA+LDM + HOMA MD TRAJECTORY MODE
# ============================================================

PROGRAM_NAME = "AroX.v0.3.2"

SYMBOL_TO_Z_AROX = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9,
    "P": 15, "S": 16, "CL": 17, "Cl": 17, "cl": 17,
}


def demander_fichier_geometrie_arox() -> str:
    while True:
        nom_fichier = input("Entrez le fichier géométrie / distances / trajectoire (.xyz, .traj, .dat ou .dt) : ").strip()
        if os.path.exists(nom_fichier):
            return nom_fichier
        print("Erreur : fichier géométrie/distances introuvable. Réessayez.")


def demander_fichier_ldm_optionnel_arox() -> str:
    print()
    print("Fichier LDM optionnel.")
    print("- Appuyez sur Entrée pour faire seulement le calcul HOMA.")
    print("- Sinon, entrez le fichier LDM .dat.")
    while True:
        nom_fichier = input("Entrez le fichier LDM (.dat) [optionnel] : ").strip()
        if nom_fichier == "":
            return ""
        if os.path.exists(nom_fichier):
            return nom_fichier
        print("Erreur : fichier LDM introuvable. Réessayez, ou Entrée pour HOMA seul.")


def lire_fichier_xyz_arox(nom_fichier: str) -> Tuple[List[int], np.ndarray]:
    with open(nom_fichier, "r", encoding="utf-8") as fichier:
        lignes_brutes = [ligne.rstrip() for ligne in fichier if ligne.strip()]

    if not lignes_brutes:
        raise ValueError(f"Erreur : fichier XYZ vide : {nom_fichier}")

    debut = 0
    try:
        n_decl = int(lignes_brutes[0].split()[0])
        debut = 2
        lignes_atomiques = lignes_brutes[debut:debut + n_decl]
        if len(lignes_atomiques) < n_decl:
            raise ValueError("Erreur : nombre d'atomes XYZ insuffisant par rapport à l'en-tête.")
    except ValueError:
        # XYZ sans ligne de nombre d'atomes : on lit toutes les lignes comme lignes atomiques.
        lignes_atomiques = lignes_brutes

    numeros_atomiques: List[int] = []
    coordonnees: List[List[float]] = []

    for numero_ligne, ligne in enumerate(lignes_atomiques, start=1):
        morceaux = ligne.split()
        if len(morceaux) < 4:
            raise ValueError(f"Erreur XYZ ligne {numero_ligne} : format attendu = Symbole x y z.")
        symbole = morceaux[0]
        if symbole.isdigit():
            z = int(symbole)
        else:
            cle = symbole if symbole in SYMBOL_TO_Z_AROX else symbole.capitalize()
            if cle not in SYMBOL_TO_Z_AROX:
                raise ValueError(f"Erreur XYZ : symbole atomique non reconnu : {symbole}")
            z = SYMBOL_TO_Z_AROX[cle]
        try:
            x, y, zcoord = float(morceaux[1]), float(morceaux[2]), float(morceaux[3])
        except ValueError:
            raise ValueError(f"Erreur XYZ ligne {numero_ligne} : coordonnées non numériques.")
        numeros_atomiques.append(z)
        coordonnees.append([x, y, zcoord])

    coords = np.array(coordonnees, dtype=float)
    n = len(numeros_atomiques)
    matrice_distances = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            matrice_distances[i, j] = d
            matrice_distances[j, i] = d
    return numeros_atomiques, matrice_distances


def lire_geometrie_ou_distances_arox(nom_fichier: str) -> Tuple[List[int], np.ndarray, str]:
    extension = os.path.splitext(nom_fichier)[1].lower()
    if extension == ".xyz":
        numeros_atomiques, matrice_distances = lire_fichier_xyz_arox(nom_fichier)
        return numeros_atomiques, matrice_distances, "XYZ"

    # Compatibilité avec les anciens fichiers DIST : même format matrice .dat et distances en bohr.
    numeros_atomiques, matrice_distances = lire_fichier_dat(nom_fichier)
    matrice_distances = convertir_matrice_bohr_en_angstrom(matrice_distances)
    return numeros_atomiques, matrice_distances, "DIST"


def generer_nom_fichier_arox(nom_fichier_geometrie: str) -> str:
    nom_simple = os.path.basename(nom_fichier_geometrie)
    nom_sans_extension = os.path.splitext(nom_simple)[0]
    return nom_sans_extension + ".arx"


def enregistrer_fichier_integre(
    nom_fichier_sortie: str,
    nom_fichier_ldm: str,
    nom_fichier_geometrie: str,
    type_geometrie: str,
    rapport_preliminaire: str,
    parametres_reference: object,
    cycles_classes: List[Dict[str, object]],
    resultats_ldm_classes: List[Dict[str, object]],
    analyse_globale: object,
    resultats_homa: List[Dict[str, object]],
    homa_totale_data: object,
    skipped_homa: List[Dict[str, object]],
    resultats_entropie: List[Dict[str, object]] = None,
) -> None:
    mode_ldm = nom_fichier_ldm != ""
    with open(nom_fichier_sortie, "w", encoding="utf-8") as fichier:
        fichier.write("===============================================================\n")
        fichier.write(f"                    {PROGRAM_NAME} REPORT\n")
        fichier.write("===============================================================\n")
        fichier.write(f" Program name        : {PROGRAM_NAME}\n")
        fichier.write(f" Input geometry file : {nom_fichier_geometrie}\n")
        fichier.write(f" Geometry input type : {type_geometrie}\n")
        fichier.write(f" Input LDM file      : {nom_fichier_ldm if mode_ldm else 'NONE - HOMA ONLY MODE'}\n")
        fichier.write(f" Output file         : {nom_fichier_sortie}\n")
        fichier.write(" Shared cycles       : AUTO cycles + optional free MANUAL cycles + generated FUSED cycles\n")
        fichier.write(" Manual cycle rule   : accepted even without contour connectivity\n")
        fichier.write(" Fused cycle rule    : generated by symmetric difference of auto/manual cycle edges\n")
        fichier.write("===============================================================\n\n")

        fichier.write("===============================================================\n")
        fichier.write("                 SHARED PRELIMINARY CYCLE ANALYSIS\n")
        fichier.write("===============================================================\n")
        fichier.write(rapport_preliminaire)
        fichier.write("\n")

        if mode_ldm:
            fichier.write("===============================================================\n")
            fichier.write("                 LDM DETAILED ANALYSIS\n")
            fichier.write("===============================================================\n")
            fichier.write(f" Reference type      : {parametres_reference['reference_name']}\n")
            fichier.write(f" epsilon_O           : {parametres_reference['epsilon_o']:.6f}\n")
            fichier.write(f" epsilon_P           : {parametres_reference['epsilon_p']:.6f}\n")
            fichier.write(f" epsilon_M           : {parametres_reference['epsilon_m']:.6f}\n")
            fichier.write(f" N_A (default C)     : {parametres_reference['n_a']:.6f}\n")
            fichier.write(" N_A per site        : Z of each atom in the cycle (auto from geometry)\n")
            fichier.write(f" lambda (benzene EG) : {parametres_reference['lambda']:.6f}\n")
            fichier.write(" lambda formula      : lambda_k = Z_k - sum(off-diagonal row k)\n")
            fichier.write("===============================================================\n\n")

            for resultat_cycle in resultats_ldm_classes:
                ecrire_detail_cycle_traite(fichier=fichier, resultat_cycle=resultat_cycle)

            if analyse_globale is not None:
                ecrire_section_analyse_globale_atomes_lourds(fichier, analyse_globale)

            ecrire_summary_final_ldm(
                fichier=fichier,
                cycles_classes=cycles_classes,
                resultats_classes=resultats_ldm_classes,
                parametres_reference=parametres_reference,
                analyse_globale=analyse_globale,
            )
            fichier.write("\n\n")

            # Section entropique (H_LDM / H_Q / H_S) — uniquement si LDM fournie.
            if resultats_entropie:
                entropy_ecrire_section(fichier, resultats_entropie, cycles_classes)
                fichier.write("\n")
        else:
            fichier.write("===============================================================\n")
            fichier.write("                 LDM ANALYSIS\n")
            fichier.write("===============================================================\n")
            fichier.write(" LDM mode            : skipped because no LDM file was provided.\n")
            fichier.write(" HOMA mode           : active from geometry/distances only.\n")
            fichier.write("===============================================================\n\n")

        homa_ecrire_details(
            fichier=fichier,
            resultats_cycles=resultats_homa,
            homa_totale_data=homa_totale_data,
            skipped_cycles=skipped_homa,
        )


def afficher_resume_terminal_integre(
    resultats_ldm_classes: List[Dict[str, object]],
    analyse_globale: object,
    resultats_homa: List[Dict[str, object]],
    homa_totale_data: object,
    skipped_homa: List[Dict[str, object]],
    mode_ldm: bool,
    resultats_entropie: List[Dict[str, object]] = None,
) -> None:
    # Trois tableaux successifs dans le résumé final :
    #   1) LDM         (existant, uniquement si mode_ldm)
    #   2) ENTROPIE H  (nouveau, uniquement si mode_ldm)
    #   3) HOMA        (toujours)
    print()
    if mode_ldm:
        afficher_resume_terminal_cycles(resultats_classes=resultats_ldm_classes, analyse_globale=analyse_globale)
    else:
        print("=" * 80)
        print("RÉSUMÉ TERMINAL LDM")
        print("=" * 80)
        print("LDM ignoré : aucun fichier LDM fourni. Mode HOMA seul actif.")
        print("=" * 80)
    print()
    if mode_ldm and resultats_entropie:
        entropy_afficher_terminal(resultats_entropie)
        print()
    homa_afficher_tableau_resume_cycles(resultats_homa, homa_totale_data)
    if skipped_homa:
        print("\nCycles ignorés en HOMA car paramètres de liaison manquants :")
        for s in skipped_homa:
            print(f"  cycle {s['numero_cycle_ldm']} {s['indices_cycle']} -> {s['missing_bond_types']}")




# ============================================================
# BLOC 17A — MODE TRAJECTOIRE MD HOMA (issu de HOMAg-MD V12)
# ============================================================

def arox_est_ligne_nombre_atomes(ligne: str) -> bool:
    morceaux = ligne.strip().split()
    return len(morceaux) == 1 and morceaux[0].isdigit()


def arox_est_ligne_md_step(ligne: str) -> bool:
    return ligne.strip().lower().startswith("md step")


def arox_extraire_numero_step(ligne: str, fallback: int) -> int:
    try:
        return int(ligne.replace(":", " ").split()[-1])
    except Exception:
        return fallback


def arox_est_ligne_atome_traj(ligne: str) -> bool:
    morceaux = ligne.split()
    if len(morceaux) < 4:
        return False
    sym = morceaux[0]
    cle = sym if sym in SYMBOL_TO_Z_AROX else sym.capitalize()
    if cle not in SYMBOL_TO_Z_AROX and not sym.isdigit():
        return False
    try:
        float(morceaux[1]); float(morceaux[2]); float(morceaux[3])
        return True
    except ValueError:
        return False


def arox_lire_trajectoire_xyz(nom_fichier: str) -> List[Dict[str, object]]:
    """Lit un .xyz multi-frame ou .traj : N/commentaire/atomes, ou blocs MD Step."""
    with open(nom_fichier, "r", encoding="utf-8") as f:
        lignes = [ligne.rstrip() for ligne in f if ligne.strip()]

    frames: List[Dict[str, object]] = []
    i = 0
    frame_index = 0
    n_lignes = len(lignes)

    while i < n_lignes:
        ligne = lignes[i].strip()
        n_attendu = None
        step = frame_index

        if arox_est_ligne_nombre_atomes(ligne):
            n_attendu = int(ligne)
            i += 1
            if i < n_lignes:
                # Ligne commentaire XYZ : si elle contient MD Step, on extrait le step ; sinon on l'ignore.
                if arox_est_ligne_md_step(lignes[i]):
                    step = arox_extraire_numero_step(lignes[i], frame_index)
                i += 1
        elif arox_est_ligne_md_step(ligne):
            step = arox_extraire_numero_step(ligne, frame_index)
            i += 1
        else:
            i += 1
            continue

        symboles: List[str] = []
        numeros_atomiques: List[int] = []
        coords: List[List[float]] = []

        while i < n_lignes:
            ligne_atom = lignes[i].strip()
            if arox_est_ligne_nombre_atomes(ligne_atom) or arox_est_ligne_md_step(ligne_atom):
                break
            morceaux = ligne_atom.split()
            if arox_est_ligne_atome_traj(ligne_atom):
                symbole = morceaux[0]
                if symbole.isdigit():
                    z = int(symbole)
                    sym_norm = Z_TO_SYMBOL.get(z, str(z))
                else:
                    sym_norm = symbole if symbole in SYMBOL_TO_Z_AROX else symbole.capitalize()
                    z = SYMBOL_TO_Z_AROX[sym_norm]
                symboles.append(sym_norm)
                numeros_atomiques.append(z)
                coords.append([float(morceaux[1]), float(morceaux[2]), float(morceaux[3])])
            i += 1
            if n_attendu is not None and len(symboles) >= n_attendu:
                break

        if n_attendu is not None and len(symboles) != n_attendu:
            raise ValueError(
                f"Frame {frame_index} / MD step {step} incomplète : {len(symboles)} atomes lus au lieu de {n_attendu}."
            )

        if symboles:
            frames.append({
                "frame_index": frame_index,
                "step": step,
                "symboles": symboles,
                "numeros_atomiques": numeros_atomiques,
                "coords": np.array(coords, dtype=float),
            })
            frame_index += 1

    if not frames:
        raise ValueError("Aucune frame lisible dans la trajectoire.")

    ref_symboles = frames[0]["symboles"]
    for fr in frames[1:]:
        if fr["symboles"] != ref_symboles:
            raise ValueError("La liste des atomes change pendant la trajectoire ; suivi HOMA impossible.")
    return frames


def construire_matrice_distances_depuis_xyz(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=2)


def arox_demander_pas_temps_fs() -> float:
    while True:
        entree = input("Chaque pas MD correspond à combien de femtosecondes ? ").strip()
        try:
            dt = float(entree)
            if dt <= 0:
                print("Erreur : le pas de temps doit être positif.")
                continue
            return dt
        except ValueError:
            print("Erreur : entre une valeur numérique, par exemple 0.5 ou 1.0.")


def arox_generer_nom_csv_traj(nom_fichier_entree: str) -> str:
    nom_simple = os.path.basename(nom_fichier_entree)
    racine = os.path.splitext(nom_simple)[0]
    return racine + ".arx.csv"


def arox_calculer_stats(valeurs: List[float]) -> Dict[str, float]:
    arr = np.array(valeurs, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "first": float(arr[0]),
        "last": float(arr[-1]),
    }


def arox_ecrire_csv_trajectoire(nom_csv: str, lignes_resultats: List[Dict[str, object]]) -> None:
    colonnes = [
        "frame", "md_step", "time_fs", "cycle_id", "cycle_type", "cycle_size",
        "indices", "bond_types", "r_av", "EN", "GEO", "EN_GEO", "HOMA"
    ]
    with open(nom_csv, "w", encoding="utf-8") as f:
        f.write(",".join(colonnes) + "\n")
        for r in lignes_resultats:
            ratio = "inf" if math.isinf(r["ratio_en_geo"]) else f"{r['ratio_en_geo']:.10f}"
            valeurs = [
                str(r["frame_index"]),
                str(r["step"]),
                f"{r['time_fs']:.10f}",
                str(r["numero_cycle"]),
                r["label_cycle"],
                str(r["taille_cycle"]),
                "-".join(str(x) for x in r["indices_cycle"]),
                "-".join(r["bond_types"]),
                f"{r['r_av']:.10f}",
                f"{r['en']:.10f}",
                f"{r['geo']:.10f}",
                ratio,
                f"{r['homa']:.10f}",
            ]
            f.write(",".join(valeurs) + "\n")


def arox_ecrire_resume_evolution_par_cycle(fichier, resultats_par_cycle: Dict[int, List[Dict[str, object]]]) -> None:
    fichier.write("\n")
    fichier.write("===============================================================\n")
    fichier.write("             SUMMARY OF HOMA EVOLUTION BY CYCLE\n")
    fichier.write("===============================================================\n")
    fichier.write(
        f" {'Cycle':<6} {'Type':<13} {'Size':<5} {'Indices':<38} "
        f"{'HOMA_mean':>12} {'HOMA_std':>12} {'HOMA_min':>12} {'HOMA_max':>12} "
        f"{'EN_mean':>12} {'GEO_mean':>12} {'EN/GEO_mean':>14}\n"
    )
    fichier.write(" " + "-" * 155 + "\n")
    for numero_cycle in sorted(resultats_par_cycle):
        items = resultats_par_cycle[numero_cycle]
        if not items:
            continue
        ref = items[0]
        homa_stats = arox_calculer_stats([x["homa"] for x in items])
        en_stats = arox_calculer_stats([x["en"] for x in items])
        geo_stats = arox_calculer_stats([x["geo"] for x in items])
        ratios_finis = [x["ratio_en_geo"] for x in items if not math.isinf(x["ratio_en_geo"])]
        ratio_mean = float(np.mean(ratios_finis)) if ratios_finis else float("inf")
        ratio_txt = "inf" if math.isinf(ratio_mean) else f"{ratio_mean:.6f}"
        indices_txt = ",".join(str(i) for i in ref["indices_cycle"])
        if len(indices_txt) > 38:
            indices_txt = indices_txt[:35] + "..."
        fichier.write(
            f" {numero_cycle:<6d} {ref['label_cycle']:<13s} {ref['taille_cycle']:<5d} {indices_txt:<38} "
            f"{homa_stats['mean']:>12.6f} {homa_stats['std']:>12.6f} "
            f"{homa_stats['min']:>12.6f} {homa_stats['max']:>12.6f} "
            f"{en_stats['mean']:>12.6f} {geo_stats['mean']:>12.6f} {ratio_txt:>14}\n"
        )
    fichier.write(" " + "=" * 155 + "\n")


def arox_ecrire_fichier_homa_trajectoire(
    nom_sortie: str,
    nom_entree: str,
    dt_fs: float,
    frames: List[Dict[str, object]],
    rapport_preliminaire: str,
    cycles_classes: List[Dict[str, object]],
    resultats_tous: List[Dict[str, object]],
    resultats_par_cycle: Dict[int, List[Dict[str, object]]],
    skipped_reference: List[Dict[str, object]],
) -> None:
    with open(nom_sortie, "w", encoding="utf-8") as f:
        f.write("===============================================================\n")
        f.write(f"                    {PROGRAM_NAME} REPORT\n")
        f.write("===============================================================\n")
        f.write(f" Program name        : {PROGRAM_NAME}\n")
        f.write(" Mode                : HOMA MD trajectory\n")
        f.write(f" Input trajectory    : {nom_entree}\n")
        f.write(f" Output file         : {nom_sortie}\n")
        f.write(f" Number of frames    : {len(frames)}\n")
        f.write(f" Time step           : {dt_fs:.6f} fs per MD step\n")
        f.write(" Cycle tracking      : cycles detected/forced on the first frame, then followed by atom indices\n")
        f.write(" Shared cycles       : AUTO cycles + optional free MANUAL cycles + generated FUSED cycles\n")
        f.write(" Fused cycle rule    : symmetric difference of auto/manual cycle edges\n")
        f.write(" LDM mode            : disabled for trajectory input in v0.3.2\n")
        f.write("===============================================================\n\n")

        f.write(" Parameters used for HOMA\n")
        f.write(" ------------------------------\n")
        for bt, p in HOMA_PARAMS.items():
            f.write(
                f" {bt:<3s} | Ropt = {p['Ropt']:>8.4f} Å | "
                f"alpha = {p['alpha']:>8.2f} Å^-2 | source = {p['source']}\n"
            )
        f.write("\n")
        f.write(" Reference frame cycle analysis\n")
        f.write(" ------------------------------\n")
        f.write(f" Reference MD step : {frames[0]['step']}\n\n")
        f.write(rapport_preliminaire)
        f.write("\n")

        if skipped_reference:
            f.write(" Reference cycles skipped because of missing HOMA bond parameters\n")
            f.write(" ------------------------------\n")
            for s in skipped_reference:
                f.write(f" Cycle {s['numero_cycle_ldm']} | indices = {s['indices_cycle']} | missing = {s['missing_bond_types']}\n")
            f.write("\n")

        f.write("===============================================================\n")
        f.write("                  TIME SERIES BY FRAME\n")
        f.write("===============================================================\n")
        f.write(
            f" {'Frame':<7} {'MD_step':<9} {'time/fs':>12} {'Cycle':<6} {'Type':<13} "
            f"{'Size':<5} {'Indices':<38} {'Rav(Å)':>10} {'EN':>12} {'GEO':>12} {'EN/GEO':>12} {'HOMA':>12}\n"
        )
        f.write(" " + "-" * 150 + "\n")
        for r in resultats_tous:
            ratio_txt = "inf" if math.isinf(r["ratio_en_geo"]) else f"{r['ratio_en_geo']:.6f}"
            indices_txt = ",".join(str(i) for i in r["indices_cycle"])
            if len(indices_txt) > 38:
                indices_txt = indices_txt[:35] + "..."
            f.write(
                f" {r['frame_index']:<7d} {r['step']:<9d} {r['time_fs']:>12.6f} "
                f"{r['numero_cycle']:<6d} {r['label_cycle']:<13s} {r['taille_cycle']:<5d} "
                f"{indices_txt:<38} {r['r_av']:>10.6f} {r['en']:>12.6f} "
                f"{r['geo']:>12.6f} {ratio_txt:>12} {r['homa']:>12.6f}\n"
            )
        f.write(" " + "=" * 150 + "\n")
        arox_ecrire_resume_evolution_par_cycle(f, resultats_par_cycle)


def arox_analyser_trajectoire_homa(nom_fichier: str, dt_fs: float) -> None:
    frames = arox_lire_trajectoire_xyz(nom_fichier)
    numeros_atomiques = frames[0]["numeros_atomiques"]
    matrice_ref = construire_matrice_distances_depuis_xyz(frames[0]["coords"])

    connectivite_provisoire, connectivite_finale, rayons_adaptes = construire_connectivite_automatique(
        numeros_atomiques=numeros_atomiques,
        matrice_distances=matrice_ref,
        tolerance=TOLERANCE_CONNECTIVITE,
    )
    connectivite_cycles = filtrer_connectivite_sans_hydrogenes(numeros_atomiques, connectivite_finale)
    cycles_detectes_auto = trouver_cycles(connectivite_cycles, longueur_min=3, longueur_max=24)
    cycles_detectes_auto = filtrer_cycles_sans_contrainte_hybridation(cycles_detectes_auto)

    cycles_manuels = demander_cycles_manuels_optionnels(
        connectivite=connectivite_cycles,
        nombre_total_atomes=len(numeros_atomiques),
    )
    cycles_manuels = filtrer_cycles_sans_contrainte_hybridation(cycles_manuels)

    cycles_detectes = fusionner_cycles_auto_manuels_et_generer_fusionnes(
        cycles_automatiques=cycles_detectes_auto,
        cycles_manuels=cycles_manuels,
    )
    cycles_classes = preparer_cycles_classes(
        cycles_dicts=cycles_detectes,
        connectivite=connectivite_cycles,
        numeros_atomiques=numeros_atomiques,
    )

    rapport_preliminaire = generer_rapport_preliminaire(
        numeros_atomiques=numeros_atomiques,
        connectivite_provisoire=connectivite_provisoire,
        rayons_adaptes=rayons_adaptes,
        connectivite_finale=connectivite_finale,
        cycles_classes=cycles_classes,
    )

    cycles_homa_ref, skipped_reference = homa_preparer_resultats_cycles(
        numeros_atomiques=numeros_atomiques,
        matrice_distances_angstrom=matrice_ref,
        cycles_classes=cycles_classes,
    )
    if not cycles_homa_ref:
        print("Aucun cycle calculable en HOMA : paramètres de liaison manquants.")
        for s in skipped_reference:
            print(f"  cycle {s['indices_cycle']} -> {s['missing_bond_types']}")
        return

    cycles_calculables = [
        {
            "numero_cycle": r["numero_cycle"],
            "label_cycle": r["label_cycle"],
            "taille_cycle": r["taille_cycle"],
            "indices_cycle": r["indices_cycle"],
        }
        for r in cycles_homa_ref
    ]

    resultats_tous: List[Dict[str, object]] = []
    resultats_par_cycle: Dict[int, List[Dict[str, object]]] = {c["numero_cycle"]: [] for c in cycles_calculables}
    step0 = frames[0]["step"]

    for fr in frames:
        matrice = construire_matrice_distances_depuis_xyz(fr["coords"])
        time_fs = (fr["step"] - step0) * dt_fs
        for c in cycles_calculables:
            calc = homa_calculer_homa_generalise(numeros_atomiques, matrice, c["indices_cycle"])
            if not calc["valid"]:
                continue
            ligne = {
                "frame_index": fr["frame_index"],
                "step": fr["step"],
                "time_fs": time_fs,
                "numero_cycle": c["numero_cycle"],
                "label_cycle": c["label_cycle"],
                "taille_cycle": c["taille_cycle"],
                "indices_cycle": c["indices_cycle"],
                **calc,
            }
            resultats_tous.append(ligne)
            resultats_par_cycle[c["numero_cycle"]].append(ligne)

    nom_sortie = generer_nom_fichier_arox(nom_fichier)
    nom_csv = arox_generer_nom_csv_traj(nom_fichier)
    arox_ecrire_fichier_homa_trajectoire(
        nom_sortie=nom_sortie,
        nom_entree=nom_fichier,
        dt_fs=dt_fs,
        frames=frames,
        rapport_preliminaire=rapport_preliminaire,
        cycles_classes=cycles_classes,
        resultats_tous=resultats_tous,
        resultats_par_cycle=resultats_par_cycle,
        skipped_reference=skipped_reference,
    )
    arox_ecrire_csv_trajectoire(nom_csv, resultats_tous)

    print("=" * 100)
    print("RÉSUMÉ TERMINAL HOMA — TRAJECTOIRE MD")
    print("=" * 100)
    print(f"Frames lues : {len(frames)}")
    print(f"Cycles suivis : {len(cycles_calculables)}")
    if skipped_reference:
        print("Cycles ignorés dans la frame de référence car paramètres HOMA manquants :")
        for s in skipped_reference:
            print(f"  cycle {s['indices_cycle']} -> {s['missing_bond_types']}")
    print(f"Résultats trajectoire enregistrés dans : {nom_sortie}")
    print(f"Table CSV enregistrée dans : {nom_csv}")
    print("=" * 100)

# ============================================================
# BLOC 17 — MAIN INTÉGRÉ HOMA + LDM + ENTROPIE (v0.3)
# ============================================================
# Point d'entrée réellement exécuté par __main__. Enchaîne :
#   1. Lecture géométrie/distances + fichier LDM optionnel
#   2. Détection connectivité + cycles (auto/manuel/fusés)
#   3. Analyses LDM par cycle + globale atomes lourds
#   4. Descripteurs entropiques H_LDM / H_Q / H_S  (nouveau)
#   5. HOMA généralisé sur les mêmes cycles
#   6. Écriture du rapport .arx + résumé terminal
# ============================================================

def main_integre_homa_ldm() -> None:
    print("===============================================================")
    print(f"                    {PROGRAM_NAME}")
    print("        PROGRAMME D’ANALYSE AROMATICITÉ HOMA + LDM")
    print("===============================================================")
    print()

    nom_fichier_geometrie = demander_fichier_geometrie_arox()
    extension = os.path.splitext(nom_fichier_geometrie)[1].lower()
    basename_lower = os.path.basename(nom_fichier_geometrie).lower()
    if extension == ".traj" or "traj" in basename_lower:
        dt_fs = arox_demander_pas_temps_fs()
        arox_analyser_trajectoire_homa(nom_fichier_geometrie, dt_fs)
        return

    numeros_atomiques_geom, matrice_distances, type_geometrie = lire_geometrie_ou_distances_arox(nom_fichier_geometrie)

    nom_fichier_ldm = demander_fichier_ldm_optionnel_arox()
    mode_ldm = nom_fichier_ldm != ""

    matrice_ldm_complete = None
    parametres_reference = None

    if mode_ldm:
        numeros_atomiques_ldm, matrice_ldm_complete = lire_fichier_dat(nom_fichier_ldm)
        verifier_compatibilite_fichiers(
            numeros_atomiques_ldm=numeros_atomiques_ldm,
            matrice_ldm=matrice_ldm_complete,
            numeros_atomiques_dist=numeros_atomiques_geom,
            matrice_dist=matrice_distances,
        )
        parametres_reference = demander_parametres_reference()
    else:
        print()
        print("Mode HOMA seul : aucun paramètre LDM demandé.")

    connectivite_provisoire, connectivite_finale, rayons_adaptes = construire_connectivite_automatique(
        numeros_atomiques=numeros_atomiques_geom,
        matrice_distances=matrice_distances,
        tolerance=TOLERANCE_CONNECTIVITE,
    )

    connectivite_cycles = filtrer_connectivite_sans_hydrogenes(
        numeros_atomiques=numeros_atomiques_geom,
        connectivite=connectivite_finale,
    )

    cycles_detectes_auto = trouver_cycles(
        connectivite=connectivite_cycles,
        longueur_min=3,
        longueur_max=24,
    )
    cycles_detectes_auto = filtrer_cycles_sans_contrainte_hybridation(cycles_detectes_auto)

    cycles_manuels = demander_cycles_manuels_optionnels(
        connectivite=connectivite_cycles,
        nombre_total_atomes=len(numeros_atomiques_geom),
    )
    cycles_manuels = filtrer_cycles_sans_contrainte_hybridation(cycles_manuels)

    cycles_detectes = fusionner_cycles_auto_manuels_et_generer_fusionnes(
        cycles_automatiques=cycles_detectes_auto,
        cycles_manuels=cycles_manuels,
    )

    cycles_classes = preparer_cycles_classes(
        cycles_dicts=cycles_detectes,
        connectivite=connectivite_cycles,
        numeros_atomiques=numeros_atomiques_geom,
    )

    rapport_preliminaire = generer_rapport_preliminaire(
        numeros_atomiques=numeros_atomiques_geom,
        connectivite_provisoire=connectivite_provisoire,
        rayons_adaptes=rayons_adaptes,
        connectivite_finale=connectivite_finale,
        cycles_classes=cycles_classes,
    )

    resultats_ldm_classes: List[Dict[str, object]] = []
    analyse_globale = None
    resultats_entropie: List[Dict[str, object]] = []

    if mode_ldm:
        resultats_ldm = preparer_resultats_cycles_analyses(
            matrice_ldm_complete=matrice_ldm_complete,
            matrice_distances_complete=matrice_distances,
            cycles_classes=cycles_classes,
            parametres_reference=parametres_reference,
        )
        resultats_ldm_classes = preparer_classement_final(resultats_ldm)

        analyse_globale = preparer_analyse_globale_atomes_lourds(
            numeros_atomiques=numeros_atomiques_geom,
            matrice_ldm_complete=matrice_ldm_complete,
            matrice_distances_complete=matrice_distances,
            connectivite_finale=connectivite_finale,
        )

        # Descripteurs entropiques H_LDM / H_Q / H_S : locaux par cycle traité + global atomes lourds
        resultats_entropie = entropy_preparer_resultats(
            matrice_ldm=matrice_ldm_complete,
            matrice_distances=matrice_distances,
            numeros_atomiques=numeros_atomiques_geom,
            cycles_classes=cycles_classes,
        )

    resultats_homa, skipped_homa = homa_preparer_resultats_cycles(
        numeros_atomiques=numeros_atomiques_geom,
        matrice_distances_angstrom=matrice_distances,
        cycles_classes=cycles_classes,
    )
    homa_totale_data = homa_calculer_homa_totale_generalisee(
        numeros_atomiques=numeros_atomiques_geom,
        matrice_distances_angstrom=matrice_distances,
        resultats_cycles=resultats_homa,
    )

    nom_fichier_sortie = generer_nom_fichier_arox(nom_fichier_geometrie)
    enregistrer_fichier_integre(
        nom_fichier_sortie=nom_fichier_sortie,
        nom_fichier_ldm=nom_fichier_ldm,
        nom_fichier_geometrie=nom_fichier_geometrie,
        type_geometrie=type_geometrie,
        rapport_preliminaire=rapport_preliminaire,
        parametres_reference=parametres_reference,
        cycles_classes=cycles_classes,
        resultats_ldm_classes=resultats_ldm_classes,
        analyse_globale=analyse_globale,
        resultats_homa=resultats_homa,
        homa_totale_data=homa_totale_data,
        skipped_homa=skipped_homa,
        resultats_entropie=resultats_entropie,
    )

    afficher_resume_terminal_integre(
        resultats_ldm_classes=resultats_ldm_classes,
        analyse_globale=analyse_globale,
        resultats_homa=resultats_homa,
        homa_totale_data=homa_totale_data,
        skipped_homa=skipped_homa,
        mode_ldm=mode_ldm,
        resultats_entropie=resultats_entropie,
    )
    afficher_message_final(nom_fichier_sortie)


# ==============================
# LANCEMENT DU PROGRAMME INTÉGRÉ
# ==============================

if __name__ == "__main__":
    main_integre_homa_ldm()

