# Integration IA EcoScan

## Parcours

Le web et le mobile envoient les questions de MAXI à l'endpoint authentifié
`POST /api/analyses/assistant/interroger/` de Django. Django transmet la
question, son historique et le contexte de l'organisation au service FastAPI
EcoScan IA. Les réponses utilisent Groq et, si disponible, les documents
validés indexés dans le RAG. Les clés restent côté serveur ; elles ne doivent
pas être ajoutées au front web ou à l'application mobile.
La génération essaie successivement deux modèles Groq, puis OpenAI si Groq est
indisponible ou limité. Le secours OpenAI nécessite une clé et un quota actifs.

Les simulations et commandes de démonstration de MAXI dans le front web restent
locales et ne pilotent pas des appareils. Une erreur de connexion ou de
configuration est affichée comme une erreur, jamais comme une réponse IA
simulée.

## Configuration locale

1. Créer `.env` à la racine du dépôt IA à partir de `.env.example`, puis
   renseigner les clés `GROQ_API_KEY` et `OPENAI_API_KEY`. Groq est essayé en
   premier; OpenAI sert de secours. Vérifier que les comptes fournisseurs
   disposent de quota.
2. Dans le `.env` Django du backend, configurer
   `AI_SERVICE_INTERNAL_TOKEN` avec la même valeur que `INTERNAL_TOKEN` du
   service IA.
3. Configurer `DJANGO_INTERNAL_TOKEN` avec la même valeur dans les `.env` Django
   et IA. Ces deux jetons sont distincts l'un de l'autre.
4. Depuis le dossier `ecoscan-backend-v2` du dépôt backend, lancer les seuls
   services nécessaires :

   ```powershell
   docker compose up -d --build backend ai-service
   ```

Le compose configure `AI_SERVICE_URL` pour le réseau Docker (`ai-service:8001`)
et l'URL Django interne (`backend:8000`). Le port IA n'est pas publié sur la
machine hôte ; le front continue à joindre Django, pas Groq ni FastAPI.

Pour le front web, `NEXT_PUBLIC_API_URL` doit viser l'API Django, par défaut
`http://localhost:8000/api`. Le mobile doit également être configuré pour
joindre cette API et disposer d'une session authentifiée.

## Vérification et limites

Le point de santé FastAPI est `GET /health`. Une requête MAXI nécessite les deux
jetons internes valides, un compte autorisé avec une organisation et au moins
un fournisseur IA accessible avec un quota disponible. Si tous les modèles
échouent, Django renvoie une erreur explicite (notamment 503). Les clés ne sont
pas incluses dans les exemples et aucune clé de production ne doit être commitée.

Les questions et le contexte métier transmis aux modèles quittent
l'infrastructure EcoScan. Ils sont envoyés à Groq en premier et peuvent être
transmis à OpenAI en cas d'échec Groq. N'activer ce secours que si les deux
fournisseurs sont autorisés par votre politique de confidentialité.
