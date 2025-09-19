import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import psycopg2
import os

def connecter():
    user = entry_user.get()
    password = entry_pass.get()
    try:
        conn = psycopg2.connect(dbname="bridge", user=user, password=password)
        cur = conn.cursor()
        cur.execute("SELECT nom FROM tables;")
        noms = [row[0] for row in cur.fetchall()]
        conn.close()

        # Masquer les champs de connexion
        entry_user.pack_forget()
        entry_pass.pack_forget()
        btn_connect.pack_forget()
        label_user.pack_forget()
        label_pass.pack_forget()

        # Afficher les champs suivants
        label_nom.pack()
        combo_nom['values'] = noms
        combo_nom.pack()

        label_num.pack()
        entry_num.pack()
        btn_lancer.pack()

    except Exception as e:
        messagebox.showerror("Erreur de connexion", str(e))

def lancer_programme():
    nom = combo_nom.get()
    numero = entry_num.get()
    if not nom or not numero:
        messagebox.showwarning("Champs manquants", "Veuillez sélectionner un nom et saisir un numéro.")
        return
    try:
        # Lancer le programme en processus indépendant
        subprocess.Popen(
            ["./affichePlis", nom, numero],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setpgrp  # Détache du groupe de processus parent
        )
        root.destroy()  # Fermer la fenêtre immédiatement
    except Exception as e:
        messagebox.showerror("Erreur d'exécution", str(e))

# Interface
root = tk.Tk()
root.title("Interface PostgreSQL")

label_user = tk.Label(root, text="Nom d'utilisateur PostgreSQL :")
entry_user = tk.Entry(root)

label_pass = tk.Label(root, text="Mot de passe :")
entry_pass = tk.Entry(root, show="*")

btn_connect = tk.Button(root, text="Se connecter", command=connecter)

label_nom = tk.Label(root, text="Sélectionnez un nom :")
combo_nom = ttk.Combobox(root)

label_num = tk.Label(root, text="Saisissez un numéro :")
entry_num = tk.Entry(root)

btn_lancer = tk.Button(root, text="Lancer afficherPlis", command=lancer_programme)

# Affichage initial
label_user.pack()
entry_user.pack()
label_pass.pack()
entry_pass.pack()
btn_connect.pack()

root.mainloop()

