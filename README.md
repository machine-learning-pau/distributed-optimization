<h1>meetup-pau</h1>
<a href="https://www.meetup.com/fr-FR/Meetup-Machine-Learning-Pau/" ><img src="img/meetup_ML_pau.png" style="float:center; max-width: 150px; display: inline"  alt="Meetup Machine Learning Pau"/></a>
<br>
<h2>Optimisation distribuée avec Apache Spark  (20 décembre 2017) </h2>

## Résumé 

Quand on utilise des algorithmes de Machine Learning on doit résoudre un problème d’optimisation pour estimer les paramètres du modèle. Ces méthodes d’optimisation sont souvent des boîtes noires pour les utilisateurs et leur fonctionnement reste invisible. L’objectif de ce meetup est de démystifier ces méthodes, en particulier lorsqu’on a énormément de données ou/et beaucoup de paramètres (_Big Data_ & _Big Model_).

Une première partie présentera comment on peut résoudre un problème d’optimisation en très grande taille en utilisant la puissance de nombreuses machines (_cluster_). Ensuite, nous développerons ensemble (en Python) une méthode d’optimisation distribuée à travers le _framework_ **Spark** sur un cas d’application pratique.

Quelques références :

[1] Spark - http://spark.apache.org

[2] Docker - http://www.docker.com

[3] [Algorithm Design: Parallel and Sequential](http://www.parallel-algorithms-book.com) by Umut A. Acar and Guy E. Blelloch.

[4] [Introduction to Algorithms](https://mitpress.mit.edu/sites/default/files/titles/sample/0262533057chap27.pdf) by Cormen, Leiserson, Rivest, Stein.

[5] [Convex Optimization](http://web.stanford.edu/~boyd/cvxbook/) by Boyd and Vandenberghe.

[6] [Learning Spark](http://github.com/databricks/learning-spark) by Holden Karau, Andy Konwinski, Patrick Wendell, Matei Zaharia.

## Quelques préparatifs pour la partie pratique du Meetup

Pour la partie pratique nous utilisons le framework _Spark_ en mode local (c’est-à-dire avec un seul noeud de stockage/calcul) sur un jeu de données de petite taille. Cependant, le code (Python) développé lors de la séance pourra être testé avec un gros jeu de données sur un cluster _Spark_ - _i.e._ avec un service cloud dédié (AWS, Microsoft Azure, Google Cloud, IBM CC Labs, …) ou un cluster privé.

Pour participer à cette partie pratique vous devez installer une image _Docker_ qui contient les outils et librairies nécessaires (Python, Notebook Jupyter, Spark …).

_Docker_ est un gestionnaire d’images qui permet de virtualiser une machine. Il utilise les librairies de votre système pour allouer dynamiquement des ressources pour une instance de l’image, appelée _container_. Il reste plus léger qu’une machine virtuelle standard. _Docker_ est disponible pour Windows, MacOS et Linux.

1 . **Installation de docker**

Premièrement, rendez-vous sur le site officiel de _Docker_ pour télécharger l'outil d'installation pour votre système :

- [Mac OS X](http://store.docker.com/editions/community/docker-ce-desktop-mac)

- [Windows](http://store.docker.com/editions/community/docker-ce-desktop-windows)

- [Linux](http://www.docker.com)

Lorsque l’installation est terminée, vérifiez votre installation en lançant la commande suivante (dans un _Terminal_) :

```docker run hello-world```

Si vous obtenez le message suivant c’est que l’installation est réussie et que _Docker_ peut charger et exécuter des images.

```
Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://cloud.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/engine/userguide/
```
2. **Téléchargement du Lab**

Télécharger (ou bien _cloner_) le projet _github_ qui contient le jeux de données et les _notebooks_ (fichiers avec l'extension `.ipynb`).

3. **Installation de l’image Docker**

À présent nous allons charger (il faut avoir internet !) et lancer l’image qui sera utilisée pour les travaux pratiques. Placez vous dans le répertoire _meetup-pau/_ où se trouve les données et les _notebooks_ et tapez la commande suivante dans le _Terminal_ :

```docker run -it --rm -p 8888:8888 -p 4040:4040 -v $(pwd)/:/home/jovyan/work jupyter/all-spark-notebook:latest```

Note - La première fois que vous utiliserez cette commande, l'image sera chargée, ce qui nécessite un certain temps de téléchargement (l'image fait environ 5Go).

Cette commande démarre un conteneur dans lequel le serveur _Jupyter Notebook_ est accessible sur le port 8888 (http://localhost:8888/?token=) avec un jeton d'authentification généré de manière aléatoire.

L'option `-v $(pwd)/:/home/jovyan/work` monte le répertoire courant (où sont les données et les _notebooks_) de l'ordinateur hôte en tant que dossier dans le conteneur (répertoire `/home/jovyan/work`). Utile lorsque vous souhaitez conserver vos _notebooks_ même après la destruction du conteneur.

Si vous souhaitez désactiver les mécanismes d'authentification tapez la commande suivante :

```docker run -it --rm -p 8888:8888 -p 4040:4040 -v $(pwd)/:/home/jovyan/work jupyter/all-spark-notebook:latest start-notebook.sh --NotebookApp.token=''```

4. **Utilisation des Notebooks Jupyter** - Ouvrez votre navigateur sur [http://localhost:8888/](http://localhost:8888/), vous devriez alors voir les données et les _notebooks_ téléchargés dans le répertoire `work/`, et être capable de les modifier et d'éxécuter le code à l'intérieur (vous êtes sur un serveur Jupyter qui tourne dans votre _container_ _Docker_ !). Pour finaliser votre installation ouvrez le _notebook_ `Lab0_Meetup_Pau.ipynb` et suivez les instructions.


