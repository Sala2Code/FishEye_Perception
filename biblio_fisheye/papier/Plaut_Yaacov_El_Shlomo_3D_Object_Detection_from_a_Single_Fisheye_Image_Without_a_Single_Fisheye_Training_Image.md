Date : 2021/05

Transformer le fisheye en projection cylindrique, appliquer un détecteur monoculaire 3D entraîné sur des images perspectives (aucune donnée fisheye), puis transformer géométriquement ses sorties d'un espace 3D virtuel vers l'espace 3D réel.

Bonne introduction pour dire ce qu'est brièvement les images perspectives, fisheyes et leurs limites.
Compare à un seul modèle existant au moment de la sortie du papier

Approximation non rigoureuses : page 4. Pour l'analyse des limites, quantification des incertitudes ou autre, ca peut être intéressant de détailler.

"Cylindrical" : 《the magnification is inversely proportional to thecylindrical radial distance ρ instead of Z [...]projected objects in cylindrical images become smaller as they become more distant, where distance is measured along the cylindrical ρ axis. [...] when training CNNs to detect 3D objects from cylindrical images, the geometrically meaningful parameter to predict is ρ and not Z.》

"Spherical" : 《the magnificationis inversely proportional to the Euclidean distance r [...] in addition to the requirement that the objects be not too large and not too close to the camera, theymust all have the same elevation relative to the camera. Under this assumption, the geometrically meaningful measureof depth to predict is the Euclidean distance.》

Limite des CNN pour le fish eye 《a shift-invariant CNN cannot be expected to learn to predictany measure of depth from a raw fisheye image, regardlessof the availability of labeled training data.》

Ils considèrent donc que la projection cylindrique car plus compatible avec les CNN que la sphériques. Aucune expérience avec le sphérique.





#recommande 