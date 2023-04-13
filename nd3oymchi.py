import streamlit as st
import numpy as np
from scipy.optimize import linprog
import scipy as sp
from pulp import lpSum, LpProblem, LpStatus, LpVariable, LpMaximize, LpMinimize

# interface utilisateur
st.title('Programmation linéaire avec la méthode des coupes de Gomory')
st.write("La méthode des coupes de Gomory est une technique de résolution de problèmes d'optimisation linéaire qui permet de renforcer les contraintes en ajoutant des inégalités. Cette technique est particulièrement utile pour résoudre des problèmes d'optimisation en nombres entiers en éliminant les solutions non entières. Le coupe de Gomory a été développé par Ralph E. Gomory dans les années 1950 et reste une technique importante dans la résolution de problèmes d'optimisation linéaire en nombres entiers.")

# Résolution du problème de programmation linéaire
# Définition des bornes des variables
x_bounds = (0, None)
y_bounds = (0, None)

# Initialiser le problème de programmation linéaire
c_temp = c
method_temp = method
A_temp = A
b_temp = b

# Résoudre le problème de programmation linéaire avec le dual de simplexe
for sense in ["maximize", "minimize"]:
    if sense == "maximize":
        c_temp = [-i for i in c]  # Inversion des coefficients pour la maximisation
        method_temp = "dual_simplex"  # Choisir la méthode dual_simplex pour maximiser avec des coefficients négatifs
    for ineq in ["<=", ">=", "="]:
        if ineq == "<=":
            A_ub = A_temp
            b_ub = b_temp
            A_eq = None
            b_eq = None
        elif ineq == ">=":
            A_ub = -A_temp
            b_ub = [-i for i in b_temp]
            A_eq = None
            b_eq = None
        elif ineq == "=":
            A_ub = None
            b_ub = None
            A_eq = A_temp
            b_eq = b_temp

        # Boucle pour ajouter des coupes de Gomory jusqu'à ce que la solution soit entière
       # Boucle pour ajouter des coupes de Gomory jusqu'à ce que la solution soit entière
while True:
    # Résoudre le problème de programmation linéaire
    prob = LpProblem(name="Gomory's cutting plane", sense=LpMaximize)
    variables = [LpVariable(name=f'x{i}', lowBound=0) for i in range(len(c_temp))]
    prob += lpSum([c_temp[i] * variables[i] for i in range(len(c_temp))])

    for j in range(len(A_temp)):
        constraint = lpSum([A_temp[j][i] * variables[i] for i in range(len(c_temp))])
        if ineq == "<=":
            prob += constraint <= b_temp[j]
        elif ineq == ">=":
            prob += constraint >= b_temp[j]
        elif ineq == "=":
            prob += constraint == b_temp[j]

    prob.solve()
    res = {"status": LpStatus[prob.status]}
    for v in prob.variables():
        res[v.name] = v.value()

    # Vérifier si la solution est entière
    if np.allclose([res[v.name] for v in prob.variables()], np.round([res[v.name] for v in prob.variables()])):
        # Afficher la solution
        st.write("Solution optimale trouvée : ")
        st.write(f"x = {[res[v.name] for v in prob.variables()]}")
        st.write(f"Valeur de l'objectif : {res[prob.objective.name]}")
        break
    else:
        # Ajouter une nouvelle coupe de Gomory
        st.write("Solution non entière, ajout d'une nouvelle coupe de Gomory")
        gomory_row = [int(np.round(res[v.name])) for v in prob.variables()]
        gomory_rhs = np.dot(gomory_row, A_temp)
        A_temp = np.vstack([A_temp, gomory_row])
        b_temp = np.append(b_temp, gomory_rhs)



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

   