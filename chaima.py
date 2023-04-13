import streamlit as st
from scipy.optimize import linprog
import numpy as np

# interface utilisateur
st.title('Programmation linéaire avec la méthode des coupes de Gomory')
st.write("La méthode des coupes de Gomory est une technique de résolution de problèmes d'optimisation linéaire qui permet de renforcer les contraintes en ajoutant des inégalités. Cette technique est particulièrement utile pour résoudre des problèmes d'optimisation en nombres entiers en éliminant les solutions non entières. Le coupe de Gomory a été développé par Ralph E. Gomory dans les années 1950 et reste une technique importante dans la résolution de problèmes d'optimisation linéaire en nombres entiers.")

# Fonction pour trouver la plus petite fraction en utilisant la méthode du coup de Gomory
def find_gomory_cut(x):
    # Trouver l'indice de la plus grande fraction
    idx = np.argmax(x - np.floor(x))
    # Créer un tableau pour le nouveau coup de Gomory
    cut = np.zeros(len(x))
    cut[idx] = 1
    cut = np.array([cut])
    return cut

# Définir le problème linéaire en entier
c = [-1, -2, -3]
A = [[1, 1, 1], [3, 2, 1], [2, 5, 3]]
b = [4, 12, 18]
bounds = [(0, None), (0, None), (0, None)]

# Résoudre le problème linéaire en entier initial
res = linprog(c, A, b, bounds=bounds, method='highs')

# Tant qu'il y a une fraction dans la solution
while any(x % 1 != 0 for x in res.x):
    # Trouver le coup de Gomory
    cut = find_gomory_cut(res.x)
    # Ajouter le coup de Gomory au problème linéaire
    A = np.vstack([A, cut])
    b = np.append(b, np.floor(res.x.dot(cut.T)))
    # Résoudre le nouveau problème linéaire en entier
    res = linprog(c, A, b, bounds=bounds, method='highs')

# Afficher la solution finale
print("Solution optimale : ", res.x)

# Définir les variables et la fonction objectif
st.subheader("Définir la fonction objectif")
num_vars = st.number_input("Nombre de variables", min_value=1, value=2)
c = np.zeros(num_vars)
for i in range(num_vars):
    c[i] = st.number_input(f"Coefficient de x_{i+1}", key=f"c{i}")
op = st.radio("Type d'optimisation", ("Maximisation", "Minimisation"))

# Définir les contraintes
st.subheader("Définir les contraintes")
num_cons = st.number_input("Nombre de contraintes", min_value=1, value=2)
A = np.zeros((num_cons, num_vars))
b = np.zeros(num_cons)
for i in range(num_cons):
    for j in range(num_vars):
        A[i, j] = st.number_input(f"Coefficient de x_{j+1} dans la contrainte {i+1}", key=f"a{i}{j}")
    b[i] = st.number_input(f"Terme constant de la contrainte {i+1}", key=f"b{i}")
    op_cons = st.radio(f"Type de la contrainte {i+1}", ("<=", "=", ">="))
    if op_cons == "<=":
        A[i] = -A[i]
    elif op_cons == "=":
        A = np.vstack((A, -A[i]))
        b = np.hstack((b, -b[i]))
   
# Ajouter un bouton de calcul
if st.button("Calculer"):
    # Inverser la fonction objective si nécessaire
    if op == "Maximisation":
        c = -c
    # Résoudre le problème d'optimisation linéaire en nombres entiers avec la méthode des coupes de Gomory
    solution_gomory = res
    # Afficher la solution optimale
    st.write("La solution optimale est", solution_gomory)

   