# Dog_Breed_Classification
Construire un algorithme de prédiction de la race des chiens pour aider à l'archivage d'une association de protection des animaux

## Présentation

Vous êtes bénévole pour l'association de protection des animaux de votre quartier. C'est d'ailleurs ainsi que vous avez trouvé votre compagnon idéal, Snooky. Vous vous demandez donc ce que vous pouvez faire en retour pour aider l'association.

Vous apprenez, en discutant avec un bénévole, que leur base de données de pensionnaires commence à s'agrandir et qu'ils n'ont pas toujours le temps de référencer les images des animaux qu'ils ont accumulées depuis plusieurs années. Ils aimeraient donc obtenir un algorithme capable de classer les images en fonction de la race du chien présent sur l'image.

## Base de données

L'algorithme se basera sur la base de données Stanford Dogs Dataset disponible via ce lien http://vision.stanford.edu/aditya86/ImageNetDogs/

## Traitement et nettoyage de la base de données :

Jeu de donnée déjà clean

## DataViz

Le jeu de données contient 120 races.
Pour une question d'efficacité des premiers traitements, nous nous contenterons pour l'instant de 10 races de chiens

![image](https://user-images.githubusercontent.com/76253068/171597071-2dabfdae-f335-4776-a77c-8b988f58333d.png)

## Exploration

10 exemples d'images de chien :

![image](https://user-images.githubusercontent.com/76253068/171597434-abf41988-c632-4c65-9942-28793978ebae.png)

## Feature Engineering

### Fonction de preprocessing:

![image](https://user-images.githubusercontent.com/76253068/171597526-d267cc25-25ad-4892-974b-b42db32d8d79.png)

### Comparaison avant/après preprocessing:

![image](https://user-images.githubusercontent.com/76253068/171597602-a515245f-5f09-4ae4-a958-6b61fb8baeda.png)

## Pistes de modélisation

### Data Augmentation

![image](https://user-images.githubusercontent.com/76253068/171598059-62572d41-f89b-481d-9513-162c8f008813.png)

5 exemples avant/après:

![image](https://user-images.githubusercontent.com/76253068/171598170-4395b82b-6e9a-4c75-aba0-f82ba8b2b64f.png)

### VGG 16 avec Transfer Learning

Recherche d'hyperparamètres

![image](https://user-images.githubusercontent.com/76253068/171598585-4de2cf50-4502-47df-a148-a371a967e337.png)

### Lenet

![image](https://user-images.githubusercontent.com/76253068/171598687-a1f5cee6-03f5-41a1-b2bb-601508d4894a.png)

Résultats faibles:

![image](https://user-images.githubusercontent.com/76253068/171598760-2c542bc8-97d5-4b1b-9d5a-84d294c11412.png)


### Lenet avec SepConv

![image](https://user-images.githubusercontent.com/76253068/171598893-c74639e9-a728-40ea-8290-140a758a7e5b.png)

Résultats faibles:

![image](https://user-images.githubusercontent.com/76253068/171598934-6ea28bf1-9369-490c-b5f4-d42deb954ad3.png)

## Comparaison des modèles

VGG 16 est le modèle retenu

![image](https://user-images.githubusercontent.com/76253068/171599152-8552f604-f303-4ec6-957a-1fce10afd205.png)

### Entraînement final

![image](https://user-images.githubusercontent.com/76253068/171599299-ce6271e0-8bb3-4ac9-9ed0-06a205f6bd8d.png)

Matrice de confusion : quelles races sont les difficiles à identifier ? 

![image](https://user-images.githubusercontent.com/76253068/171599329-e49b4f86-6808-4c0c-aaa8-402ffe70b4c4.png)

- Races American Staffordshire confondues avec Pug
- Tableau à surveiller si augmentation du nombre de races
- Solution: augmenter la taille du jeu de données ou entraîner le modèle en OneVsOne

### Essai sur les 120 races

La val_accuracy tombe à 45%

![image](https://user-images.githubusercontent.com/76253068/171599987-0997e175-8701-4b58-b8da-2d60b126935a.png)

## Application d'identification de la race canine

Voir programme_cnn.py 
