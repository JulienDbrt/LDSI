# Ideas - LDSI Future Work

## Validation Empirique des Coefficients (α, β, γ)

Les coefficients actuels (α=0.40, β=0.35, γ=0.25) sont basés sur l'intuition :
- NCD domine (signal primaire de divergence)
- Entropie enrichit (richesse informationnelle)
- Topologie vérifie (garde-fou structurel)

### Pistes de validation

#### 1. Corpus Contrasté
- Collecter des paires (prompt standard, prompt fracturé) sur plusieurs modèles
- Annotation humaine : classifier chaque réponse en ZOMBIE, REBELLE, ARCHITECTE, FOU
- Comparer avec les prédictions LDSI
- Ajuster α/β/γ pour maximiser l'accord inter-annotateurs

#### 2. Stabilité Cross-Model
- Tester si les mêmes coefficients fonctionnent across modèles :
  - Llama 3
  - Claude (Sonnet, Opus)
  - GPT-4
  - Mistral
  - Modèles open-source (Qwen, Phi, etc.)
- Si stable : argument fort pour l'universalité de la métrique
- Si variable : documenter les ajustements model-specific

#### 3. Analyse de Sensibilité
- Faire varier α/β/γ dans une grille (ex: 0.2 à 0.5 par pas de 0.05)
- Observer les transitions de verdict
- Identifier les zones de stabilité vs les points de bascule
- Objectif : coefficients robustes où de petites variations ne changent pas massivement les classifications

#### 4. Cas Limites (Tests Unitaires du Benchmark)
- **ZOMBIE intentionnel** : température 0, même seed, prompts identiques
- **FOU intentionnel** : température max, prompts absurdes, injection de bruit
- **ARCHITECTE cible** : prompts Codex bien calibrés
- Vérifier que LDSI classifie correctement ces cas extrêmes

## Extensions Possibles

### Multi-lingue
- Étendre les listes de stop-words (ES, DE, IT, PT, ZH, JA, AR)
- Valider que NCD reste pertinent cross-langue (Kolmogorov est language-agnostic en théorie)

### Temporal Stability
- Tracker λ pour un même modèle au fil des versions
- Détecter les régressions (model updates qui augmentent le Lissage)

### Integration Benchmarks Existants
- Corréler LDSI avec MMLU, HellaSwag, TruthfulQA
- Hypothèse : les modèles ARCHITECTE performent mieux sur les tâches créatives, les ZOMBIE sur les tâches factuelles ?

### Mode Batch / CI
- Intégrer LDSI dans une pipeline CI/CD
- Alerter si un modèle déployé régresse vers ZOMBIE
- Dashboard de suivi temporel

## Notes Théoriques

### Pourquoi α > β > γ ?
- **NCD (α=0.40)** : Mesure directe de la divergence informationnelle. C'est le "quoi" - les textes sont-ils différents ?
- **Entropie (β=0.35)** : Mesure la richesse du vocabulaire. C'est le "comment" - la différence vient-elle d'un vocabulaire plus riche ?
- **Topologie (γ=0.25)** : Mesure la cohérence structurelle. C'est le "garde-fou" - la divergence reste-t-elle cohérente ?

Cette hiérarchie reflète une hypothèse : la divergence brute est le signal primaire, mais elle doit être qualifiée par la richesse (pas juste du bruit) et contrainte par la structure (pas de l'hallucination).

### Lien avec Lyapunov
La stabilité de Lyapunov mesure si un système dynamique revient à l'équilibre après perturbation. LDSI mesure comment un LLM répond à une perturbation de prompt :
- **ZOMBIE** : Retour systématique au même point (sur-stabilité, Lissage)
- **ARCHITECTE** : Nouvel équilibre stable (divergence contrôlée)
- **FOU** : Divergence explosive (instabilité)

Le λ de LDSI est conceptuellement analogue à l'exposant de Lyapunov : positif = divergence, négatif = convergence, proche de zéro = équilibre critique.
