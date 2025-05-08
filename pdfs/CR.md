# Compte rendu des séances

## 12/02 - Premier RDV (visio)
+ Discussion autour du projet, de ses objectifs. 
+ Discussion sur l'intérêt de la synthèse d'image et des dernière avancées du domaine

## 28/02 - Deuxième RDV (présentiel)
+ Cours accéléré sur le fonctionnement des réseaux de neurones
+ Questions sur les spécificités des architectures choisies dans les deux papiers

## 28/02 - Séance en autonomie
+ Partage de connaissance sur les réseaux de neurones
+ compréhension commune et approfondie du fonctionnement décrit par les deux articles de recherche.
+ Début de lecture commune du code fourni et familiarisation avec pytorch

## 13/03 - 
+ Première implémenation du réseau effectué. L'entrainement du réseau ne produit aucune erreur mais les résultans sont inexploitable.
+ On décide alors de se pencher plus en profondeur sur quelle range de valeur chaque fonction/réseau fonctionne. On adapte alors le code. Modification notamment des fonctions dans utils.py qui traite les images/tensors avant de les envoyés dans le réseau.

## 20/03 - 
+ Le réseau ne converge toujours pas vers une sortie que l'on veut. Avec l'accès au machines, les tests deviennent plus simples. Cependant on se rend compte que l'on a beacoup de problème d'optimisation. On faisait par exemple des calculs de gradiants à des endroits inutiles. On n'avait aussi pas mis tout nos tensors sur le gpu. Une fois cela fait l'entrainement allait beacoup plus vite
+ On remarque aussi que nos valeurs de sorties divergent. On essaye dans un premier temps de les renormaliser. On opte finalement pour rajouter dans la loss une composantes qui privélégies les valeurs dans [-1,1]. 

## 02/04 - 
+ Finalement on se rend compte que le réseau ne convergeait pas vers la bonne texture les fonctions pytorch que l'on utilisait pour calculer les convulutions ne fonctionnait pas comme on le pensait. Une fois ce problème réglé le réseau fonctionne très bien.
+ On commence alors à s'intéresser au transfert de style.