# EcoScan IA

Microservice d'intelligence artificielle d'EcoScan, construit avec FastAPI.
Il fournit l'assistant MAXI avec récupération documentaire (RAG) et données
métier en direct, l'extraction de champs de factures ou de compteurs dans des
images, la transcription vocale et la catégorisation de réclamations. Le
service est appelé par Django et ne doit pas être exposé directement aux
applications web et mobiles.

Ce guide décrit le fonctionnement implémenté dans ce dépôt. Les noms et
disponibilités des modèles distants évoluent : validez leur accès, quotas,
tarifs, limites multimodales et conditions de traitement dans la console du
fournisseur avant chaque mise en production. Les descriptions ci-dessous sont
des critères de choix, pas une promesse de qualité ou de disponibilité.

## Sommaire

- [Architecture et flux des données](#architecture-et-flux-des-données)
- [Fonctionnalités](#fonctionnalités)
- [Prérequis](#prérequis)
- [Configuration](#configuration)
- [Intégration avec le backend](#intégration-avec-le-backend)
- [Installation et lancement local](#installation-et-lancement-local)
- [Exécution avec Docker](#exécution-avec-docker)
- [Choix et routage des modèles](#choix-et-routage-des-modèles)
- [Inférences par modalité](#inférences-par-modalité)
- [RAG et données EcoScan](#rag-et-données-ecoscan)
- [API et contrats](#api-et-contrats)
- [Sécurité et confidentialité](#sécurité-et-confidentialité)
- [Diagnostic](#diagnostic)
- [Développement et validation](#développement-et-validation)
- [Limites connues](#limites-connues)

## Architecture et flux des données

```text
Web / mobile
    │ HTTPS + session utilisateur
    ▼
Django EcoScan
    ├── authentifie l'utilisateur et détermine son organisation
    ├── contrôle les autorisations et prépare le contexte métier
    └── X-Internal-Service-Token
            ▼
       FastAPI EcoScan IA
            ├── assistant MAXI
            │     ├── RAG local JSON, filtré par organisation
            │     ├── contexte d'organisation récupéré par GET depuis Django
            │     └── routeur texte Groq → Groq → OpenAI (configuration actuelle)
            ├── extraction d'image : Groq vision
            ├── transcription : Groq Whisper
            └── analyse de réclamation : Groq
```

Le navigateur et l'application mobile parlent à Django, pas directement à
FastAPI ni aux fournisseurs de modèles. Django authentifie l'utilisateur,
vérifie son organisation et relaie les requêtes internes. L'accès Django → IA
est protégé par `INTERNAL_TOKEN`; l'accès IA → Django est protégé séparément
par `DJANGO_INTERNAL_TOKEN`.

FastAPI expose le routeur à la racine (`/`) et aussi sous `/api/v1` pour les
routes compatibles. Le `health` est `GET /health`. En Docker, Django joint
FastAPI au nom de service `ai-service` sur le port interne 8001. Le service IA
n'a pas de port publié sur l'hôte par le Compose backend.

Les inférences distantes transmettent des prompts, documents pertinents,
questions et, pour l'extraction, des images aux fournisseurs configurés. Ce
flux externe doit être accepté par la politique de confidentialité d'EcoScan.

### Prototype Render Free

Le Blueprint `render.yaml` déploie le service IA comme web service gratuit
pour un prototype. Render Free ne fournit pas de réseau privé entrant entre
services : Django doit appeler l'URL HTTPS publique de l'IA et celle-ci doit
appeler l'URL HTTPS publique de Django. Les deux échanges restent protégés par
des jetons distincts configurés comme secrets dans Render (`AI_SERVICE_INTERNAL_TOKEN`
du backend doit correspondre à `INTERNAL_TOKEN` ici; `DJANGO_INTERNAL_TOKEN`
doit correspondre à celui du backend). Ne transmettez jamais ces valeurs dans
Git ou une conversation.

Le chemin `RAG_STORE_PATH` du Blueprint pointe vers `/tmp`; son contenu est
éphémère et peut disparaître lors d'un redémarrage ou d'une mise en veille.
N'indexez aucune donnée réelle ou document confidentiel sur cette instance.
Les limites de veille, d'heures et de trafic des offres Free peuvent également
interrompre l'accès; ce profil n'est pas destiné à la production.

## Fonctionnalités

| Fonction | Point d'entrée FastAPI | Inférence / stockage |
|---|---|---|
| Question texte générique | `POST /api/v1/query` | Routeur texte configurable |
| Assistant EcoScan/MAXI | `POST /internal/query` | RAG lexical local, contexte Django optionnel, routeur texte |
| Indexation documentaire | `POST /internal/documents` | JSON local dans `RAG_STORE_PATH` |
| Analyse de facture ou compteur | `POST /image/analyze` | Vision Groq configurée, JSON structuré |
| Transcription audio | `POST /audio/transcribe` | `whisper-large-v3` via Groq |
| Analyse de réclamation | `POST /claims/analyze` | GPT OSS 120B via Groq, JSON |
| Santé du processus | `GET /health` | Pas de vérification de quota fournisseur |

Les routes sont protégées par `verify_internal_token`, y compris les routes
qui sont décrites comme « publiques » dans le code : cela signifie qu'elles
sont sans JWT EcoScan utilisateur, mais pas sans jeton de service.

## Prérequis

- Python 3.11+ pour un environnement local; l'image Docker utilise Python 3.12.
- pip et un environnement virtuel.
- Une clé Groq pour l'assistant, la vision, l'audio et le scoring.
- Une clé OpenAI si le dernier maillon de la chaîne texte doit servir de
  secours.
- Une clé Hugging Face uniquement si un modèle `huggingface:` est ajouté à la
  chaîne de modèles.
- Le backend Django V2 pour charger le contexte métier MAXI.
- Docker Desktop pour une exécution conteneurisée.

Le RAG de ce dépôt est un index lexical JSON, sans base vectorielle externe ni
embeddings. Aucun compte GPU ou serveur de modèle local n'est requis par le
code actuel.

## Configuration

La classe de configuration se trouve dans `app/core/config.py`. Elle charge
les variables depuis l'environnement puis `.env` (encodage UTF-8); les champs
inconnus sont ignorés. `.env.example` est le modèle partageable. Ne lisez,
n'ajoutez ni ne copiez jamais les valeurs de `.env` dans Git, une image ou un
log.

### Variables principales

| Variable | Rôle | Valeur de développement / remarque |
|---|---|---|
| `APP_NAME` | Nom FastAPI et `/health` | `EcoScan AI/RAG` |
| `ENVIRONMENT` | Environnement annoncé par `/health` | `development` |
| `INTERNAL_TOKEN` | Authentifie Django vers IA | À remplacer par un secret fort |
| `DJANGO_API_URL` | Racine HTTP de Django | `http://localhost:8000` local; `http://backend:8000` en Compose |
| `DJANGO_HOST_HEADER` | En-tête Host optionnel pour Django | Vide sauf besoin de proxy |
| `DJANGO_INTERNAL_TOKEN` | Authentifie IA vers Django | Secret distinct, égal au jeton interne Django correspondant |
| `DJANGO_TIMEOUT` | Timeout des requêtes contexte Django | 15 secondes |
| `RAG_STORE_PATH` | Fichier d'index documentaire | `./data/rag_store.json` |
| `MAX_UPLOAD_MB` | Limite configurée d'upload | 50 Mo; vérifier les limites effectives dans les endpoints et proxy |
| `GROQ_API_KEY` | Fournisseur principal du code actuel | Nécessaire aux fonctions Groq |
| `OPENAI_API_KEY` | Secours texte OpenAI | Optionnel; quota et autorisation requis |
| `OPENAI_API_BASE` | Racine API compatible OpenAI | Vide = API OpenAI officielle |
| `HUGGINGFACE_API_KEY` | Appels Hugging Face | Optionnel; requis si utilisé |
| `TEXT_MODEL_CHAIN` | Chaîne texte ordonnée | `groq:openai/gpt-oss-120b,groq:qwen/qwen3.8-27b,openai:gpt-4.1-mini` |
| `VISION_MODEL_CHAIN` | Chaîne documentaire prévue | `groq:qwen/qwen3.6-27b`; le service vision actuel emploie son propre défaut |
| `AUDIO_TRANSCRIPTION_CHAIN` | Chaîne audio prévue | `groq:whisper-large-v3`; le service audio emploie Whisper directement |
| `MIN_BACKUPS_RECOMMANDES` | Seuil descriptif de redondance | 2; ne configure pas à lui seul de maillons |

Les deux jetons internes ne sont pas interchangeables. Générer des valeurs
fortes distinctes pour les deux usages; configurer chaque valeur à l'identique
des deux côtés de son échange, via le gestionnaire de secrets du déploiement.
Ne pas utiliser `change-me-in-production`.

### URL, ports et quotas

- En local, `DJANGO_API_URL=http://localhost:8000` si Django écoute sur la
  machine.
- Dans le réseau Docker partagé avec le backend V2,
  `DJANGO_API_URL=http://backend:8000`.
- Les requêtes modèles sortent vers les API des fournisseurs depuis le service
  IA; vérifier DNS, proxy, TLS et quota depuis ce conteneur.
- Les variables de modèle sont des identifiants fournisseur, pas des noms
  génériques garantis. Un modèle retiré ou non autorisé échoue et le routeur
  essaie le maillon suivant.

## Installation et lancement local

Depuis le dossier du dépôt, dans PowerShell :

```powershell
Set-Location 'C:\chemin\vers\ECOSCAN IA'
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Renseigner localement les clés et jetons dans `.env`, sans les afficher dans
le terminal. Puis démarrer :

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

Vérifier `http://127.0.0.1:8001/health`. L'option `--reload` est réservée au
développement. Pour intégrer à Django localement, faire pointer
`AI_SERVICE_URL` vers cette instance et configurer les jetons internes
correspondants dans les deux projets.

## Intégration avec le backend

Le guide [AI-INTEGRATION.md](AI-INTEGRATION.md) décrit le parcours MAXI entre
les applications web/mobile, Django et FastAPI, ainsi que les étapes
d'intégration locale. En environnement conteneurisé, le Compose du dépôt
backend V2 doit partager son réseau avec l'IA, joignable à l'adresse interne
`http://ai-service:8001`; Django est joignable à `http://backend:8000`.

Depuis WSL sans Docker, après création du virtualenv `venv` et installation des
dépendances, `bash run-local-wsl.sh` détermine la passerelle Windows, configure
l'URL Django et démarre Uvicorn. Par défaut, ce script configure deux modèles
Groq seulement (`gpt-oss-120b`, puis Qwen); il n'active donc pas le secours
OpenAI du défaut général. Une variable `TEXT_MODEL_CHAIN` déjà exportée est
conservée. Vérifiez la connectivité et les jetons avant d'envoyer une requête
d'assistant.

`requirements-ml.txt` contient les dépendances ML supplémentaires (notamment
les outils audio locaux). Ce fichier n'est pas requis pour les inférences
distantes Groq/OpenAI implémentées par défaut. Ne l'installez que si un
parcours local qui l'utilise a été explicitement mis en place.

## Exécution avec Docker

Le déploiement usuel est le Compose de
`ECOSCAN BACK-END/ecoscan-backend-v2`, qui construit ce service avec le
`Dockerfile` de ce dépôt. Le Compose backend fournit à l'IA notamment :

- `DJANGO_API_URL=http://backend:8000`;
- `RAG_STORE_PATH=/app/data/rag_store.json`;
- la chaîne texte `groq:openai/gpt-oss-120b,groq:qwen/qwen3.8-27b,openai:gpt-4.1-mini`.

Le Compose du backend doit recevoir `GROQ_API_KEY`, `OPENAI_API_KEY` si le
secours est retenu, `INTERNAL_TOKEN` et `DJANGO_INTERNAL_TOKEN` via la
configuration privée attendue par l'environnement. Le build Docker installe
les dépendances Python et les outils OCR système, copie le service, puis lance
Uvicorn sur le port conteneur 8001. Le service IA n'est pas publié
directement sur le réseau hôte.

Après configuration des secrets et du réseau requis par le backend V2 :

```powershell
Set-Location 'C:\chemin\vers\ECOSCAN BACK-END\ecoscan-backend-v2'
docker compose config --quiet
docker compose up -d --build backend ai-service
docker compose ps
docker compose logs --tail 100 ai-service
```

Ne passez pas les secrets en arguments de commande, qui peuvent apparaître dans
l'historique shell. N'exécutez pas `docker compose down --volumes` sur un
environnement ayant des données à conserver.

## Choix et routage des modèles

### Représentation de la chaîne

Chaque élément de `TEXT_MODEL_CHAIN` est `fournisseur:identifiant`. La chaîne
est parcourue dans l'ordre :

1. `groq:openai/gpt-oss-120b`;
2. `groq:qwen/qwen3.8-27b`;
3. `openai:gpt-4.1-mini`.

La configuration accepte aussi une liste séparée par des virgules ou une
représentation de tableau JSON. Une entrée sans `:` est interprétée comme
Hugging Face par défaut. Une génération réussie renvoie le texte et le
spécificateur du modèle utilisé. Si tous échouent, le routeur lève
`RuntimeError`; l'assistant interne traduit cet échec en HTTP 503.

Un HTTP 429 marque le fournisseur comme limité pour la requête en cours :
ses autres modèles sont sautés afin d'essayer un fournisseur différent. Les
autres erreurs d'appel passent au prochain maillon, puis sont consignées. Le
client utilise un timeout total de 30 secondes et un timeout de connexion de
10 secondes. Le routeur ne réessaie pas indéfiniment et ne fournit pas de
réponse IA de secours fictive.

### Comparaison et critères de sélection

| Option | Usage configuré | Points forts recherchés | Coûts / risques et quand la retenir |
|---|---|---|---|
| Groq avec GPT OSS 120B | Premier choix texte; scoring de réclamation | Modèle généraliste de grande taille pour raisonnement et réponses structurées | Dépend du catalogue, du quota, du coût et de la latence Groq; vérifier disponibilité exacte avant déploiement |
| Groq avec Qwen 3.8 27B | Second choix texte | Modèle de repli distinct, potentiellement plus léger/rapide; apporte une autre option fournisseur-modèle | Même dépendance fournisseur Groq : utile pour une panne ou disponibilité propre au modèle, pas pour une panne totale de Groq |
| OpenAI GPT-4.1 mini | Dernier secours texte | Fournisseur distinct; option de secours conçue pour un coût et une latence contenus par rapport à des modèles plus grands | Nécessite clé, quota, budget et autorisation de transférer les données vers OpenAI |
| Hugging Face Inference Providers | Option disponible dans le routeur, non prioritaire dans la chaîne par défaut | Sélection de modèles et fournisseurs par identifiant; choix flexible selon les fournisseurs activés | Disponibilité, routage, latence, conditions et coût varient selon le fournisseur choisi; ajouter seulement après test |
| Qwen 3.6 27B sur Groq | Vision facture/compteur par défaut | Modèle vision configuré pour analyser les images et rendre les champs structurés | Limites de taille/tokens et erreurs de lecture; validation humaine des montants et index obligatoire |
| Whisper large v3 sur Groq | Transcription audio | Moteur spécialisé parole/audio, distinct des modèles conversationnels | Coût selon durée/limites fournisseur; la transcription peut être inexacte, surtout avec bruit, noms ou accents |

Les propriétés comme « plus précis », « plus rapide » ou « moins coûteux » ne
sont pas garanties par un nom de famille. Comparez, sur un jeu représentatif
et non sensible : exactitude métier, taux de JSON valide, latence p50/p95,
taux d'échec, coût par tâche, quotas, limites contextuelles, disponibilité,
langues (dont français), qualité du sénégalais et conformité des transferts.
Documentez les versions, la date et les paramètres mesurés; répétez le test
après changement de modèle.

### Pourquoi une chaîne hybride

- Un second modèle Groq permet un secours si le premier identifiant est
  indisponible ou échoue, mais ne supprime pas le point de panne commun Groq.
- OpenAI est un fournisseur distinct, ce qui améliore la diversité du secours,
  à condition de configurer la clé, les quotas et l'accord de confidentialité.
- Un modèle spécialisé vision ou transcription est plus adapté à son entrée
  (image/audio) qu'un modèle texte ordinaire.
- Le scoring a besoin d'une réponse JSON cadrée; le chemin dédié utilise
  température basse et format JSON.
- Un modèle local peut réduire les transferts externes, mais requiert des
  ressources matérielles, une exploitation et des évaluations qui ne sont pas
  fournies par la configuration standard de ce dépôt.

Le routeur texte ne bascule pas automatiquement les chaînes vision et audio.
Le service vision appelle le client Groq et le modèle explicite ou son défaut;
le service audio appelle directement l'endpoint Groq Whisper. Le parsing des
variables `VISION_MODEL_CHAIN` et `AUDIO_TRANSCRIPTION_CHAIN` existe dans la
configuration, mais ces chaînes ne pilotent pas encore ces services.

## Inférences par modalité

### Assistant et réponses textuelles

`LLMRouterService.generate_text` construit les messages system/user et appelle
une API compatible OpenAI pour Groq, Hugging Face ou OpenAI. Les appels texte
utilisent température 0.7. Le résultat conserve le modèle effectivement
employé pour permettre à l'appelant de l'afficher ou de le journaliser.

MAXI utilise en plus un prompt système spécialisé : il doit s'appuyer sur le
contexte fourni, signaler les données manquantes, ne pas prétendre avoir
commandé un appareil, distinguer économies déclarées et mesurées, et proposer
une prochaine étape concrète. Ces consignes réduisent les erreurs mais ne
garantissent pas l'absence d'hallucination; les décisions restent à valider
par les parcours métier et les opérateurs.

Les limites de contexte sont actuellement gérées dans le service assistant :
question 4 000 caractères, documents 6 000, données live 8 000 et historique
2 400 caractères. L'historique conserve au plus quatre tours, avec au plus 600
caractères par entrée. Quand le JSON métier est trop long, un marqueur
`_tronque` indique au modèle que l'extrait est partiel.

### Vision des factures

`POST /image/analyze` accepte les extensions `.jpg`, `.jpeg`, `.png`, `.webp`.
Le service encode l'image en data URL, appelle le modèle vision avec
`temperature=0`, un format JSON demandé et une limite de 700 tokens de
complétion. La limite image déclarée est 20 MiB. Il demande les champs de
facture/compteur définis dans `CHAMPS_FACTURE` et une liste
`champs_non_lisibles`; les champs inconnus sont écartés.

Une extraction vide, un JSON invalide ou un échec fournisseur produit une
erreur explicite; aucun nombre n'est inventé par le service. L'extraction
n'est pas une validation comptable : vérifiez les valeurs et leur cohérence
avant de les enregistrer dans Django.

### Audio

`POST /audio/transcribe` accepte `.mp3`, `.wav`, `.m4a`, `.ogg`, `.webm`,
`.flac`. L'API transmet le fichier multipart à Groq
`/audio/transcriptions` avec `whisper-large-v3` et un paramètre de langue
optionnel. Le service renvoie le texte, la langue détectée, le moteur et la
durée d'exécution. Un résultat vide est rejeté au niveau HTTP.

### Réclamations

`POST /claims/analyze` demande catégorie, priorité, sentiment, urgence,
résumé et recommandation selon les catégories et priorités autorisées du
contrat. Le modèle configuré est `openai/gpt-oss-120b` sur le client Groq;
la température est 0.1 et le mode JSON est demandé. Le JSON brut est parsé
puis projeté vers le contrat de réponse. Une erreur fournisseur ou JSON
invalide est remontée, pas transformée en décision par défaut. Le score
d'urgence est une analyse assistée et ne remplace pas le triage humain des
situations dangereuses.

## RAG et données EcoScan

### Index documentaire lexical

`RagStore` charge `RAG_STORE_PATH` comme une liste JSON de documents. Chaque
document comporte au minimum `event_id`, `organisation_id`, `document_id`,
`texte` et `source`; des métadonnées et un `import_id` peuvent compléter
l'entrée. L'indexation interne est idempotente par `event_id`; une nouvelle
version de document pour la même organisation remplace l'ancienne. La
suppression est limitée au couple document/organisation.

La recherche est lexicale :

1. normalisation Unicode et retrait des accents;
2. tokenisation en mots;
3. suppression des mots vides français;
4. filtrage strict sur `organisation_id`;
5. score = nombre de termes communs / nombre de termes de la question;
6. tri par score décroissant et retour des premiers résultats (5 par défaut).

Il n'y a ni embeddings ni moteur vectoriel dans cette implémentation. Le score
est une mesure de recouvrement de mots, pas une probabilité de pertinence.
L'ordre des égalités suit l'ordre existant dans l'index.

### Contexte métier vivant

MAXI peut appeler Django en GET sur
`/api/internal/organisations/{organisation_id}/ai-context/`, avec
`X-Internal-Service-Token`. Django détermine et limite le contexte par
organisation. Si l'appel échoue, MAXI reçoit un marqueur d'indisponibilité et
doit l'indiquer; il ne doit pas présenter cette absence comme des données
réelles. `include_live_data=false` évite cet appel, sans éviter l'inférence
distante.

### Persistance et volumes

Le RAG est sauvegardé en JSON via un fichier temporaire puis remplacement
atomique. Le verrou est local au processus : plusieurs réplicas écrivant
simultanément dans le même fichier ne sont pas un stockage partagé sûr. En
Docker, montez un volume persistant vers le chemin configuré, sauvegardez le
fichier et contrôlez les permissions. La perte du fichier supprime l'index
documentaire, mais pas les originaux gérés par Django.

## API et contrats

Les contrats Pydantic sont définis dans `app/core/contracts.py`. Les routes
FastAPI visibles et leurs protections sont dans `app/api/v1/endpoints.py`.
Consultez `/docs` uniquement depuis un réseau de développement/protégé.

| Méthode et route | Accès | Fonction |
|---|---|---|
| `GET /health` | Sans jeton | Santé du processus, pas des fournisseurs |
| `POST /internal/query` | `INTERNAL_TOKEN` | Assistant RAG + données live |
| `POST /internal/documents` | `INTERNAL_TOKEN` | Indexer un document validé par Django |
| `POST /api/v1/query` | `INTERNAL_TOKEN` | Génération texte sans contexte RAG |
| `POST /audio/transcribe` | `INTERNAL_TOKEN` | Transcrire un fichier audio |
| `POST /image/analyze` | `INTERNAL_TOKEN` | Extraire facture/compteur d'une image |
| `POST /claims/analyze` | `INTERNAL_TOKEN` | Scorer une réclamation |

Le même routeur est également monté à la racine pour compatibilité; les
endpoints qui ne sont pas marqués internes sont tout de même protégés par le
jeton de service. Les réponses d'erreur HTTP sont définies dans les endpoints;
une clé manquante ou un fournisseur indisponible ne doit pas être interprété
comme un succès.

## Sécurité et confidentialité

- Ne publiez jamais `.env`, des clés fournisseur, des jetons internes, des
  prompts contenant des données personnelles, ni des images/audio réels.
- Utilisez TLS et des secrets forts, distincts pour les deux directions de
  communication interne.
- N'exposez pas le port FastAPI à Internet; faites passer les requêtes métiers
  par Django et son contrôle d'organisation.
- Les utilisateurs, documents, historiques et données métier transmis aux
  modèles peuvent être transférés à Groq puis à OpenAI selon le failover.
  Vérifiez contrats, régions, rétention et consentements avant d'activer ces
  fournisseurs.
- Limitez et surveillez les téléversements au proxy et à l'application; la
  limite `MAX_UPLOAD_MB` déclarée dans les réglages n'est pas à elle seule une
  preuve que chaque endpoint l'applique.
- Une image peut contenir plus de données que les champs extraits. Minimisez
  les fichiers et durées de conservation.
- Le RAG applique un filtrage organisationnel; gardez ce filtre lors de toute
  évolution vers une base vectorielle.
- Ne traitez pas le prompt comme un mécanisme d'autorisation. Les permissions
  doivent être appliquées par Django et les services avant l'appel du modèle.
- Une réponse générée, un score, une transcription ou une extraction ne sont
  pas des validations humaines ni une commande d'équipement.

## Diagnostic

| Symptôme | Contrôles |
|---|---|
| Le conteneur démarre mais `GET /health` échoue | Port interne 8001, logs Uvicorn, configuration Compose et démarrage d'import |
| MAXI répond HTTP 503 | Clé Groq/quota, identifiants de modèles actifs, clé/quota OpenAI si secours, DNS/TLS sortants |
| MAXI manque les données récentes | `DJANGO_API_URL`, réseau Docker, `DJANGO_INTERNAL_TOKEN`, accès au endpoint Django et affiliation de l'utilisateur |
| Erreur 401 entre Django et IA | Comparer localement `AI_SERVICE_INTERNAL_TOKEN` Django et `INTERNAL_TOKEN` IA sans imprimer les valeurs |
| Erreur contexte Django | Vérifier distinctement le `DJANGO_INTERNAL_TOKEN` côté IA et Django, le header Host si proxy, puis le timeout |
| RAG ne retrouve aucun document | Contrôler le chemin/volume `RAG_STORE_PATH`, le format JSON, l'`organisation_id` et les termes réellement présents |
| Vision renvoie extraction invalide | Extension, MIME, taille <= 20 MiB, clé/quota, modèle actif; vérifier manuellement la facture |
| Audio renvoie 400/422/502/503 | Extension autorisée, fichier non vide, langue, clé Groq, quota et réponse Whisper |
| Scoring renvoie JSON invalide | Quota et disponibilité du modèle; vérifier le format, sans convertir l'échec en décision |

Les journaux indiquent la progression et les modèles essayés, mais ne journalisez
pas les clés, les données d'authentification, le contenu intégral des factures,
les enregistrements audio, ni les questions métier identifiantes.

## Développement et validation

Installer les dépendances de base :

```powershell
pip install -r requirements.txt
```

La vérification syntaxique suivante ne nécessite pas d'appel réel à un
fournisseur :

```powershell
python -m compileall app
```

Le dépôt ne contient pas actuellement de suite de tests automatisés; `pytest`
n'est pas une dépendance de `requirements.txt`. `compileall` valide uniquement
la syntaxe. N'envoyez pas de données de production à un endpoint de modèle pour
tester.

Pour modifier une chaîne de modèles, commencer par vérifier son parsing,
l'identifiant dans le fournisseur et la région/quota. Effectuer un test
fonctionnel avec des entrées synthétiques, mesurer qualité, erreurs, latence et
coût, contrôler le fournisseur réellement indiqué par `model_used`, puis
déployer progressivement. Conserver une configuration précédente pour le
retour arrière.

## Limites connues

- `/health` confirme seulement que FastAPI répond; il ne valide pas les clés,
  le quota, Django ou le RAG.
- La recherche RAG est lexicale, avec un seul fichier JSON et un verrou
  uniquement au sein du processus.
- La chaîne de failover est principalement pour le texte; vision, audio et
  scoring ont des modèles définis dans leur propre service.
- `VISION_MODEL_CHAIN`, `AUDIO_TRANSCRIPTION_CHAIN` et
  `MIN_BACKUPS_RECOMMANDES` existent dans la configuration, mais ne pilotent
  pas à eux seuls de failover/validation de redondance.
- Les API de fournisseurs, noms de modèles, tarifs et quotas changent; le
  dépôt n'embarque ni benchmarks indépendants ni garantie de disponibilité.
- Les erreurs de réponse modèle ne sont pas une validation métier; les sorties
  structurées nécessitent validation côté consommateur et contrôle humain.
