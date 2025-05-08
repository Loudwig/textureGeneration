# Planning / Objectifs :

## 13/03
+ Récupérer dans le code fourni les fonctions connexes qui seront nécessaires (calcul de Gramm, etc)
+ Lire la documentation Pytorch et comprendre comment déclarer la structure de notre réseau de neurones


## 20/03
+ Implémentation complète de l'algorithme

## 26/03
+ Tests sur les divers paramètres et leur influence.
+ Reproduire les résultats du papier
+ Améliorer la loss (couleur, fréquences pour les lignes droites)
+ Tester diverses conditions au bord

## 02/04
+ Suite séance précédente
  
## 09/04
+ Transfert de style
+ Implémentation du papier qui fait le transfert de style

## 24/04
+ Suite séance précédente 

## 02/05
+ Transfert de style avec automate cellulaire
  (utilisation de l'architecture basée sur les automate cellulaire pour faire du transfert de style)
  Il s'agit peut-être d'un travail original (à confirmer)

## 07/05
+ Suite du transfert de style avec automate cellulaire (optimisation des paramètres, étude de leur influence)

## 15/05
+ Recherche d'autres algorithmes de génération de texture
+ Recherche d'extensions au projet


Le contenu des séances suivantes sera discuté avec l'encadrant une fois la partie principale du projet aura été finie et dépendra de la séance du 15/05
## 22/05
+ Entrainer récursivement notre modèle sur une même image avec différents niveau de précision.
+ Commencer à implémenter cela.

## 29/05
+ Continuer d'implémenter la feature précédente.

## 05/06

## 12/06

## Semaine du 24/06
+ Préparation de la présentation finale (diapositives, etc)



# Compte rendu des séances

## 12/02 - Premier RDV (visio)
+ Discussion autour du projet, de ses objectifs. (Romain,Kilian,Martin,Lucas,Yann Gousseau)
+ Discussion sur l'intérêt de la synthèse d'image et des dernière avancées du domaine (Romain,kilian,Martin,Lucas,Yann Gousseau)

## 28/02 - Deuxième RDV (présentiel)
+ Cours accéléré sur le fonctionnement des réseaux de neurones. (Romain,kilian,Martin,Lucas,Yann Gousseau)
+ Questions sur les spécificités des architectures choisies dans les deux papiers. (Romain,kilian,Martin,Lucas,Yann Gousseau)

## 28/02 - Séance en autonomie
+ Partage de connaissance sur les réseaux de neurones. (Romain,kilian,Martin,Lucas)
+ compréhension commune et approfondie du fonctionnement décrit par les deux articles de recherche. (Romain,kilian,Martin,Lucas)
+ Début de lecture commune du code fourni et familiarisation avec pytorch. (Romain,kilian,Martin,Lucas)

## 13/03 - 
+ Première implémenation du réseau effectué. L'entrainement du réseau ne produit aucune erreur mais les résultans sont inexploitable. (Chacun a fait des changements de son coté et même en mutualisant cela ne marche pas)
+ On décide alors de se pencher plus en profondeur sur quelle range de valeur chaque fonction/réseau fonctionne. On adapte alors le code. Modification notamment des fonctions dans utils.py qui traite les images/tensors avant de les envoyés dans le réseau. (Romain,kilian,Martin,Lucas)

## 20/03 - 
+ Le réseau ne converge toujours pas vers une sortie que l'on veut. Avec l'accès au machines, les tests deviennent plus simples. Cependant on se rend compte que l'on a beacoup de problème d'optimisation. On faisait par exemple des calculs de gradiants à des endroits inutiles. On n'avait aussi pas mis tout nos tensors sur le gpu. Une fois cela fait l'entrainement allait beacoup plus vite (Martin Lucas)
+ On remarque aussi que nos valeurs de sorties divergent. On essaye dans un premier temps de les renormaliser. On opte finalement pour rajouter dans la loss une composantes qui privélégies les valeurs dans [-1,1]. (Romain kilian)

## 02/04 - 
+ Finalement on se rend compte que le réseau ne convergeait pas vers la bonne texture les fonctions pytorch que l'on utilisait pour calculer les convulutions ne fonctionnait pas comme on le pensait. Une fois ce problème réglé le réseau fonctionne très bien. (Romain Lucas)
+ On commence alors à s'intéresser au transfert de style, lecture des pseudo codes fournis dans les articles de recherche et réflexion sur la nouvelle loss à utiliser. (kilian Martin)

## Semaine du 08/04
+ Commun : discution autour de la façon d'évaluer la qualité du transfert de style, conclusion : difficile de le faire autrement qu'à l'oeil humain car difficile de formaliser ce que l'on veut favoriser.


## Semaine du 06/05
+ Kilian : Début de rédaction du rapport environnemental, sur google doc pour faciliter la collaboration.
+ Lucas : amélioration du système d'enregistrement automatique des poids.
+ Martin : Rédaction de la section impact artistique du rapport


## Semaine du 13/05
+ Lucas : réorganisation du code en le restructurant avec des classes pour faciliter son adaptation et la documentation. Création d'un fichier .py plutôt qu'un notebook afin de faciliter l'exécution sur les Lames. Correction d'un bug qui parfois posait des problèmes de convergences. Il ne fallait pas initalisé tous les poids du réseau à zéro. Il fallait ne rien mettre et les poids du réseau s'auto-initialise avec du bruit gaussien.

+ Lucas : ajout d'une lience libre au dépot (avec accord de tous les membres du groupe)

+ Romain : calculation la consommation énergétique de l'entrainement de notre modèle.

## Semaine du 26/05 

+ Tout le monde : préparation présentation + présentation aux autres groupes. Discussion sur vers quel objectif final on voudrait que notre projet ce dirige. Cela semble être multi-résolution.

## Semaine 05/06 : 

+ Tout le monde : discussion sur comment implémenter la multirésolution
+ Martin,Lucas,Kylian : commence à implémenter leur achitecture multirésolution
+ Romain : corrige et modifie le fichier général pour qu'il soit ensuite compatible avec la multirésolution
## Semaine du 03/06

+ Lucas : Idée d'implémenter le multirésolution sous forme d'un seul réseau à 3x2 couches, avec une forward pass qui successivement itère dans les 3 "sous-réseaux", et on calcule la loss comme la somme des 3 loss, pour au final avoir un unique réseau qui est capable de partir de bruit et de générer une image avec une structure globale comme locale.

## Semaine du 12/06


+ Tout le monde : selection des éléments importants à faire figurer sur le poster.
+ Kilian : Prototypage de poster
+ Romain : Rework du générateur afin de faciliter son utilisation sur google collab afin de pouvoir lancer la génération de nombreuses images pour le poster.
+ Lucas / Martin : Avancée sur le code permetant la muti-résolution

## Semaine du 18/06 - 27/06: 
    + Tout le monde : prépration présentation pour notre encadrant. Préparation rendu écrit et réorganisation du git.
    + Romain Martin : implémentation multiresolution Parallel. Test et production d'images
    + Kilian : Comparaison de tout les resultats et analyse. Fabrication de tableau comparatif. Lancement de beacoup de synthèse avec paramètres différents.
    + Lucas : Implémentation du multiresolution en Série. Test et productions d'images
