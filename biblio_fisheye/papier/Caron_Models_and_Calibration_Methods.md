HAL Archive : https://hal.science/hal-04270351v1

TODO : read and write

Ahmad :

Une caméra classique utilise un modèle de perspective simple qui ne fonctionne plus dès que le champ de vision devient très large ou atteint 360 degrés, car les formules provoquent des divisions par zéro ou des incohérences géométriques. Ce texte présente les solutions pour modéliser mathématiquement la formation de l'image dans les caméras omnidirectionnelles et pour calibrer ces instruments.  Pour représenter le trajet de la lumière, trois grandes familles de modèles existent :  Les modèles ad hoc reposent sur la forme physique exacte des composants. Ils utilisent les équations géométriques des miroirs courbes, pour les systèmes catadioptriques, ou des formules polynomiales pour décrire la distorsion optique des lentilles fisheye.

Les modèles unifiés utilisent une passerelle géométrique abstraite. Au lieu de traiter chaque forme d'optique séparément, ils projettent d'abord la scène sur une sphère virtuelle, puis projettent cette sphère sur le capteur plat depuis un second point de vue. Ce principe est étendu avec deux sphères successives pour les fisheye très larges, ou appliqué à deux demi-sphères pour les caméras 360 à deux objectifs. 

Les modèles génériques renoncent aux formules globales pour les optiques trop complexes ou imparfaites. Il traitent la camera comme un tableau de correspondance où chaque pixel est simplement associé à son propre rayon lumineux dans l'espace. 

Enfin, l'étalonnage consiste à trouver numériquement les réglages internes de ces modèles, comme la focale ou les distorsions. Pour cela, on photographie des objets aux dimensions connues, comme des damiers ou des mires de points. Un algorithme d’apprentissage ajuste ensuite les paramètres en minimisant l'écart entre la position théorique calculée et la position réelle observée sur le capteur, ce qui permet ensuite de redresser l'image pour la robotique, les véhicules autonomes ou la 3D.


Papier recommandé : "modèle optique central unifié, équation 3.26"
