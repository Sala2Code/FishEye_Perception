# TODO

Bibliographie
+ Pré-traitement (cylindrique, sphérique, ERP (Equirectangular Projection), etc.)
+ Tracking 
+ Banc stéréo avec des fisheye caméra
+ Etalonnage

Théorie basique:
+ Comprendre le modèle de base : Perspective / rectilinéaire ; et expliquer la différence avec fisheye.

Où mettre cette dernière référence proposée ? https://github.com/isri-aist/undistortCalibFromDistort

# Suggestion 

Raphaël : 
TLDR : **Ne pas utiliser de CNN (donc YOLO) mais un modèle adapté à la structure propre des images fisheye. Etre équivariant non par translation mais par autre chose.**
La réunion a longtemps tourné autour du pré-traitement. L'idée actuelle du pré-traitement est de transformer l'image de sorte à ce que les modèles convolutifs, qui sont équivariant par translation, puisse traiter l'image comme si elle est était prise en perspective. Toutefois, ce n'est peut-être pas la meilleure approche car ces transformation ne règle jamais le problème de la distorsion propre à la méthode fisheye ET peut amener à de la perte d'information (rogner, étirement, distorsion etc.). Donc une proposition serait un **modèle adapté** aux images fisheye. C'est une idée à explorer voir ce qui s'est déjà fait et voir comment l'adopter : l'entraîner toujours sur des images en perspective ou considérer uniquement un ensemble d'images fisheye.