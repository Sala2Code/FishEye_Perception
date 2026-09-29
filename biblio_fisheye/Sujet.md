# Réunion 29/09

Implémenter via OpenCV la détection d'objet (principalement humain) à travers des caméras FishEye. L'inférence se ferait par des modèles génériques, comme YOLO. Ce dernier est entraîné sur des images en perspective et non des fisheyes : c'est une **contrainte** à prendre en compte *(Selon la/les direction(s) prise(s) on pourra l'omettre)*.

Différentes vue fisheye existent. La plus commune est l'équidistante. Il existe équisolide, stéréographique et surement d'autre. Comprendre les différences de ces vues. 

Il faut savoir **étalonner** les caméras. 

Détailler, expliquer, comparer les différents **pré-traitements** à appliquer. Les principaux sont cyli

Deux sujets sont à étudier et développer :
+ **Tracking** d'un objet (voire plus si le projet avance). Une première étape est la détection d'objet. Appliquer filtre de Kalman.
+ **Banc stéréo fisheye** *(système stéréoscopique fisheye, paire stéréo de caméras fisheye, rig stéréo fisheye)* : deux caméras fisheye proches permettent par une technique similaire à la triangulation de reproduire la vue en 3D. Il y aurait deux approches : 
	+ **Reconstruction sparse** : seul certains points caractéristiques  sont projetés : peu couteux et robuste mais selon l'application cela peut être limité. 
	+ **Reconstruction dense** : chaque pixel de l'image a une correspondance dans le modèle 3D. Cette thématique a été abordée afin d'obtenir une **carte de profondeur**



# Sujet initial

**Mots clés** 
traitement des images, détection d'objets, deep learning

**Contexte du projet** 
Le projet se déroule au sein du département robotique du LAAS-CNRS qui utilise ces caméras fish eye pour détecter des obstacles à 360° au voisinage immédiat de robot mobile et ainsi anticiper les éventuelles collisions. Ces caméras sont donc embarquées sur robot lors de ses tâches de navigation. L'intérêt est que ces caméras  ont un très grand angle de vue mais les images acquises sont déformées (distorsion optique). L'idéal serait un groupe de 4-5 étudiants.

**Objectifs/attendus du projet** 
Le projet porte ici sur l’acquisition et traitement de flux vidéo par une caméra à grand angle de vue type « fish eye » qui est utilisée en robotique pour percevoir tout autour d’un système mobile (robot, véhicule) à l’instar des caméras GoPro. L’inconvénient de ces caméras réside dans la visualisation des images associées et la distorsion optique observée.

Il est alors pertinent de, hors ligne, étalonner la caméra afin de modéliser puis, en ligne : (1) corriger la distorsion optique, et (2) appliquer une transformation cylindrique ou sphérique pour améliorer le rendu. L’exploitation est alors simplifiée, par exemple pour détecter des cibles par réseau de neurones convolutifs.

Le projet porte sur l’implémentation d’un détecteur de cibles (obstacles) basé « deep learning » dans un flux de caméra « fish eye » après étalonnage et prétraitement comme indiqué précédemment. Une extension à la stéréovision (coopération entre deux caméras fish eye) sera envisagée si le timing le permet. Le développement s’effectuera en python et s’appuiera sur les bibliothèques openCV et PyTorch. 

Blog introductif : https://plaut.github.io/fisheye_tutorial/#monocular-3d-object-detection