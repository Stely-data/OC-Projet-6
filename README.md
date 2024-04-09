# OC-Projet-6

Ce projet concerne l'amélioration de l'expérience utilisateur sur la marketplace e-commerce "Place de marché", à travers l'automatisation de la classification des articles en catégories basées sur leurs descriptions textuelles et leurs images. Actuellement, l'attribution manuelle des catégories par les vendeurs pose des problèmes de fiabilité et de scalabilité, particulièrement avec l'expansion prévue du volume des articles.

<b>Première mission</b> : explorer la faisabilité d'un moteur de classification automatique. Les objectifs principaux incluent :

- Prétraitement des Données : Nettoyer et préparer les textes des descriptions et les images des produits pour l'analyse.
- Extraction de Features : Utiliser des techniques avancées telles que SIFT, ORB, SURF pour les images, et des approches de traitement de texte comme Bag of Words et TF-IDF, ainsi que des embeddings pour capturer l'essence des descriptions.
- Réduction Dimensionnelle et Visualisation : Appliquer des techniques pour réduire les données traitées en deux dimensions et visualiser la distribution des produits dans l'espace de features, facilitant l'évaluation de la séparabilité des catégories.
- Analyse et Validation : Évaluer la faisabilité de la classification automatique à travers l'analyse visuelle et des mesures de similarité entre les catégories réelles et celles prédites par clustering.

<b>Seconde mission</b> : étendre le projet pour inclure une classification supervisée des images avec data augmentation pour optimiser le modèle. De plus, un nouvel objectif est d'explorer l'élargissement de la gamme de produits à l'épicerie fine, en commençant par une collecte de données sur les produits à base de "champagne".

Les étapes supplémentaires comprennent :

- Classification Supervisée : Mettre en place et évaluer un modèle de classification d'images avec augmentation des données.
- Expansion de la Gamme de Produits : Tester la collecte de données via API pour les produits d'épicerie fine et préparer un script pour l'extraction des informations produit dans un fichier CSV.
- Présentation des Résultats : Formaliser l'ensemble du processus et les analyses dans une présentation concise, maximisant à 30 slides au format PDF.

Ce projet vise à renforcer l'efficacité opérationnelle de "Place de marché" et à enrichir l'expérience de ses utilisateurs, en garantissant une classification précise et automatisée des produits, tout en explorant de nouvelles opportunités de marché
