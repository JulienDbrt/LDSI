# LDSI - Lyapunov-Dabert Stability Index

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2025.XXXXX-b31b1b.svg)](https://arxiv.org/)

**A White-Box Benchmark for Measuring Semantic Divergence in Large Language Models**

*Un Benchmark Boîte Blanche pour Mesurer la Divergence Sémantique des Grands Modèles de Langage*

*Un Benchmark de Caja Blanca para Medir la Divergencia Semántica en Grandes Modelos de Lenguaje*

---

**Author / Auteur / Autor:** Julien DABERT

---

## Language / Langue / Idioma

- [English](#english)
- [Français](#français)
- [Español](#español)

---

# English

## Abstract

We introduce the **Lyapunov-Dabert Stability Index (LDSI)**, a novel white-box metric for quantifying semantic divergence between Large Language Model (LLM) outputs. Unlike existing evaluation methods that rely on neural network judges (introducing opacity and potential bias), LDSI employs exclusively mathematical and information-theoretic measures: **Normalized Compression Distance (NCD)** based on Kolmogorov complexity approximation, **Shannon entropy** for lexical diversity assessment, and **graph-theoretic analysis** for structural coherence verification.

The proposed metric is fully deterministic, reproducible to the bit level, model-agnostic, and computationally efficient (requiring no GPU). LDSI provides an auditable framework for evaluating the creative and structural properties of LLM responses, distinguishing between mere repetition, meaningful divergence, and incoherent hallucination.

**Keywords:** Large Language Models, Evaluation Metrics, Kolmogorov Complexity, Shannon Entropy, Graph Theory, White-Box Benchmarking

## 1. Introduction

### 1.1 Problem Statement

Current LLM evaluation paradigms face a fundamental epistemological challenge: using neural networks to evaluate neural networks introduces circular dependencies and opacity. When GPT-4 judges the output of Claude, or vice versa, the evaluation inherits the biases, hallucinations, and training artifacts of the judge model. This creates an **evaluation hall of mirrors** where no ground truth can be established.

### 1.2 Proposed Solution

LDSI addresses this challenge by grounding evaluation in mathematical primitives that predate and transcend neural architectures:

1. **Compression-based distance** (rooted in algorithmic information theory)
2. **Information entropy** (from Shannon's foundational work)
3. **Graph topology metrics** (from discrete mathematics)

These measures are:
- **Deterministic**: Identical inputs always produce identical outputs
- **Auditable**: Any researcher can verify results with standard tools
- **Model-agnostic**: Applicable to any text-generating system
- **Computationally trivial**: Executable on commodity hardware in milliseconds

## 2. Theoretical Background

### 2.1 Kolmogorov Complexity and NCD

The **Kolmogorov complexity** K(x) of a string x is defined as the length of the shortest program that produces x on a universal Turing machine. While K(x) is uncomputable, it can be approximated using real-world compression algorithms.

The **Normalized Compression Distance** (Cilibrasi & Vitányi, 2005) leverages this approximation:

$$NCD(x, y) = \frac{C(xy) - \min(C(x), C(y))}{\max(C(x), C(y))}$$

Where:
- $C(x)$ = compressed size of text x
- $C(y)$ = compressed size of text y
- $C(xy)$ = compressed size of concatenation

**Properties:**
- $NCD \approx 0$: Texts are nearly identical (total smoothing)
- $NCD \approx 1$: Texts share minimal information (maximum divergence)

### 2.2 Shannon Entropy

**Shannon entropy** (Shannon, 1948) quantifies the average information content of a discrete random variable:

$$H(X) = -\sum_{i=1}^{n} p(x_i) \log_2 p(x_i)$$

Applied to text, where $p(x_i)$ is the probability of token $x_i$:
- **Low entropy**: Predictable, repetitive vocabulary
- **High entropy**: Diverse, surprising lexical choices

We extend this with:
- **Type-Token Ratio (TTR)**: $\frac{|V|}{N}$ where |V| = unique tokens, N = total tokens
- **Hapax Legomena Ratio**: Proportion of words appearing exactly once

### 2.3 Graph-Theoretic Structural Analysis

Text is transformed into a **co-occurrence graph** $G = (V, E)$:
- **Vertices** $V$: Unique lemmatized tokens
- **Edges** $E$: Co-occurrence within sliding window (default: 5 tokens)

Structural metrics include:
- **Density**: $\frac{|E|}{|V|(|V|-1)}$
- **Largest Connected Component (LCC) Ratio**: Coherence indicator
- **Clustering Coefficient**: Local interconnectedness
- **Average Path Length**: Navigability between concepts
- **Small-World Index**: $\frac{C}{L}$ (clustering / path length)

## 3. The LDSI Formula

### 3.1 Composite Index

The **Lyapunov-Dabert Stability Index** combines the three pillars:

$$\lambda_{LD} = \alpha \cdot NCD(A, B) + \beta \cdot \frac{H(B)}{H(A)} + \gamma \cdot \Delta_{Graph}$$

Where:
- $A$ = Reference response (standard prompt)
- $B$ = Test response (modified prompt)
- $\alpha, \beta, \gamma$ = Weighting coefficients (default: 0.40, 0.35, 0.25)
- $\Delta_{Graph}$ = Structural preservation score

### 3.2 Default Coefficients

| Coefficient | Value | Rationale |
|-------------|-------|-----------|
| α (NCD) | 0.40 | Primary divergence measure |
| β (Entropy) | 0.35 | Information richness |
| γ (Topology) | 0.25 | Coherence preservation |

### 3.3 Interpretation Scale

| λ_LD Range | Classification | Interpretation |
|------------|----------------|----------------|
| 0.0 - 0.3 | **ZOMBIE** | Model ignores prompt variation; pure recitation |
| 0.3 - 0.7 | **REBEL** | Notable divergence; enriched vocabulary |
| 0.7 - 1.2 | **ARCHITECT** | Optimal zone; high divergence with preserved structure |
| > 1.2 | **FOOL** | Maximum entropy; collapsed structure (hallucination) |

## 4. Implementation

### 4.1 Technology Stack

- **Language**: Rust (Edition 2024)
- **Compression**: Zstandard (zstd crate)
- **Graph Analysis**: petgraph
- **CLI**: clap
- **HTTP Client**: reqwest + tokio (async)

### 4.2 Module Architecture

```
src/
├── core/
│   ├── ncd.rs        # Normalized Compression Distance
│   ├── entropy.rs    # Shannon entropy + TTR + Hapax
│   └── topology.rs   # Graph construction and metrics
├── probe/
│   ├── cleaner.rs    # Text preprocessing (stopword removal)
│   └── injector.rs   # LLM API client (Ollama/OpenAI/Anthropic)
├── audit/
│   └── logger.rs     # JSON audit trail
└── main.rs           # CLI interface
```

### 4.3 Computational Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| NCD | O(n log n) | O(n) |
| Entropy | O(n) | O(|V|) |
| Graph Construction | O(n × w) | O(|V|² ) |
| Graph Metrics | O(|V|² ) | O(|V|) |

Where n = text length, w = window size, |V| = vocabulary size.

## 5. Installation

### 5.1 From Source

```bash
git clone https://github.com/JulienDbrt/LDSI.git
cd LDSI
cargo build --release
```

### 5.2 Binary Location

```bash
./target/release/ldsi --help
```

## 6. Usage

### 6.1 Full Analysis

```bash
ldsi analyze \
  --text-a "The cat sleeps on the couch." \
  --text-b "The feline entity transcends oneiric paradigms." \
  --output results.json
```

### 6.2 Individual Metrics

```bash
# NCD only
ldsi ncd "text A" "text B"

# Entropy only
ldsi entropy "text to analyze"

# Topology only
ldsi topology "text to analyze"
```

### 6.3 Live LLM Injection

```bash
ldsi inject \
  --url http://localhost:11434 \
  --model llama3 \
  --api-type ollama \
  --prompt-a "Explain quantum physics." \
  --prompt-b "As a chaos poet, explain quantum physics."
```

## 7. Experimental Validation

### 7.1 Reproducibility

Given identical inputs, LDSI produces **bit-identical outputs** across:
- Different hardware architectures
- Different operating systems
- Multiple executions

### 7.2 Discriminative Power

Preliminary experiments demonstrate clear separation between:
- Repetitive outputs (λ < 0.3)
- Creative responses (0.7 < λ < 1.2)
- Incoherent hallucinations (λ > 1.2)

## 8. Limitations and Future Work

### 8.1 Current Limitations

1. **Language sensitivity**: Stopword lists currently support French and English
2. **Semantic blindness**: NCD captures structural similarity, not meaning
3. **Window size sensitivity**: Graph topology depends on co-occurrence window

### 8.2 Future Directions

1. Multilingual stopword expansion
2. Cross-lingual NCD analysis
3. Temporal stability analysis across model versions
4. Integration with existing LLM benchmarks (MMLU, HellaSwag)

## 9. Citation

```bibtex
@software{dabert2025ldsi,
  author = {Dabert, Julien},
  title = {{LDSI}: {L}yapunov-{D}abert {S}tability {I}ndex for {LLM} Evaluation},
  year = {2025},
  url = {https://github.com/JulienDbrt/LDSI},
  version = {0.1.0}
}
```

## 10. References

1. Cilibrasi, R., & Vitányi, P. M. (2005). Clustering by compression. *IEEE Transactions on Information Theory*, 51(4), 1523-1545.

2. Shannon, C. E. (1948). A mathematical theory of communication. *The Bell System Technical Journal*, 27(3), 379-423.

3. Kolmogorov, A. N. (1965). Three approaches to the quantitative definition of information. *Problems of Information Transmission*, 1(1), 1-7.

4. Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks. *Nature*, 393(6684), 440-442.

5. Lyapunov, A. M. (1892). *The general problem of the stability of motion*. Kharkov Mathematical Society.

---

# Français

## Résumé

Nous introduisons le **Lyapunov-Dabert Stability Index (LDSI)**, une métrique boîte blanche novatrice pour quantifier la divergence sémantique entre les sorties de Grands Modèles de Langage (LLM). Contrairement aux méthodes d'évaluation existantes qui s'appuient sur des juges à base de réseaux de neurones (introduisant opacité et biais potentiels), LDSI emploie exclusivement des mesures mathématiques et informationnelles : la **Distance de Compression Normalisée (NCD)** basée sur l'approximation de la complexité de Kolmogorov, l'**entropie de Shannon** pour l'évaluation de la diversité lexicale, et l'**analyse théorique des graphes** pour la vérification de la cohérence structurelle.

La métrique proposée est entièrement déterministe, reproductible au bit près, agnostique au modèle, et computationnellement efficace (ne nécessitant aucun GPU). LDSI fournit un cadre auditable pour évaluer les propriétés créatives et structurelles des réponses LLM, distinguant entre simple répétition, divergence significative et hallucination incohérente.

**Mots-clés :** Grands Modèles de Langage, Métriques d'Évaluation, Complexité de Kolmogorov, Entropie de Shannon, Théorie des Graphes, Benchmark Boîte Blanche

## 1. Introduction

### 1.1 Problématique

Les paradigmes actuels d'évaluation des LLM font face à un défi épistémologique fondamental : utiliser des réseaux de neurones pour évaluer des réseaux de neurones introduit des dépendances circulaires et de l'opacité. Quand GPT-4 juge la sortie de Claude, ou vice versa, l'évaluation hérite des biais, hallucinations et artefacts d'entraînement du modèle juge. Cela crée une **galerie des glaces évaluative** où aucune vérité terrain ne peut être établie.

### 1.2 Solution Proposée

LDSI répond à ce défi en ancrant l'évaluation dans des primitives mathématiques qui précèdent et transcendent les architectures neuronales :

1. **Distance basée sur la compression** (enracinée dans la théorie algorithmique de l'information)
2. **Entropie informationnelle** (des travaux fondateurs de Shannon)
3. **Métriques topologiques de graphes** (des mathématiques discrètes)

Ces mesures sont :
- **Déterministes** : Des entrées identiques produisent toujours des sorties identiques
- **Auditables** : Tout chercheur peut vérifier les résultats avec des outils standard
- **Agnostiques au modèle** : Applicables à tout système générateur de texte
- **Computationnellement triviales** : Exécutables sur matériel standard en millisecondes

## 2. Fondements Théoriques

### 2.1 Complexité de Kolmogorov et NCD

La **complexité de Kolmogorov** K(x) d'une chaîne x est définie comme la longueur du plus court programme qui produit x sur une machine de Turing universelle. Bien que K(x) soit incalculable, elle peut être approximée par des algorithmes de compression réels.

La **Distance de Compression Normalisée** (Cilibrasi & Vitányi, 2005) exploite cette approximation :

$$NCD(x, y) = \frac{C(xy) - \min(C(x), C(y))}{\max(C(x), C(y))}$$

Où :
- $C(x)$ = taille compressée du texte x
- $C(y)$ = taille compressée du texte y
- $C(xy)$ = taille compressée de la concaténation

**Propriétés :**
- $NCD \approx 0$ : Les textes sont quasi-identiques (lissage total)
- $NCD \approx 1$ : Les textes partagent un minimum d'information (divergence maximale)

### 2.2 Entropie de Shannon

L'**entropie de Shannon** (Shannon, 1948) quantifie le contenu informationnel moyen d'une variable aléatoire discrète :

$$H(X) = -\sum_{i=1}^{n} p(x_i) \log_2 p(x_i)$$

Appliquée au texte, où $p(x_i)$ est la probabilité du token $x_i$ :
- **Faible entropie** : Vocabulaire prévisible, répétitif
- **Haute entropie** : Choix lexicaux divers, surprenants

Nous étendons avec :
- **Ratio Type-Token (TTR)** : $\frac{|V|}{N}$ où |V| = tokens uniques, N = tokens totaux
- **Ratio Hapax Legomena** : Proportion de mots apparaissant exactement une fois

### 2.3 Analyse Structurelle par Théorie des Graphes

Le texte est transformé en **graphe de co-occurrence** $G = (V, E)$ :
- **Sommets** $V$ : Tokens lemmatisés uniques
- **Arêtes** $E$ : Co-occurrence dans une fenêtre glissante (défaut : 5 tokens)

Les métriques structurelles incluent :
- **Densité** : $\frac{|E|}{|V|(|V|-1)}$
- **Ratio de Plus Grande Composante Connexe (LCC)** : Indicateur de cohérence
- **Coefficient de Clustering** : Interconnexion locale
- **Longueur Moyenne des Chemins** : Navigabilité entre concepts
- **Indice Small-World** : $\frac{C}{L}$ (clustering / longueur de chemin)

## 3. La Formule LDSI

### 3.1 Indice Composite

Le **Lyapunov-Dabert Stability Index** combine les trois piliers :

$$\lambda_{LD} = \alpha \cdot NCD(A, B) + \beta \cdot \frac{H(B)}{H(A)} + \gamma \cdot \Delta_{Graph}$$

Où :
- $A$ = Réponse de référence (prompt standard)
- $B$ = Réponse de test (prompt modifié)
- $\alpha, \beta, \gamma$ = Coefficients de pondération (défaut : 0.40, 0.35, 0.25)
- $\Delta_{Graph}$ = Score de préservation structurelle

### 3.2 Coefficients par Défaut

| Coefficient | Valeur | Justification |
|-------------|--------|---------------|
| α (NCD) | 0.40 | Mesure primaire de divergence |
| β (Entropie) | 0.35 | Richesse informationnelle |
| γ (Topologie) | 0.25 | Préservation de cohérence |

### 3.3 Échelle d'Interprétation

| Plage λ_LD | Classification | Interprétation |
|------------|----------------|----------------|
| 0.0 - 0.3 | **ZOMBIE** | Le modèle ignore la variation de prompt ; pure récitation |
| 0.3 - 0.7 | **REBELLE** | Divergence notable ; vocabulaire enrichi |
| 0.7 - 1.2 | **ARCHITECTE** | Zone optimale ; haute divergence avec structure préservée |
| > 1.2 | **FOU** | Entropie maximale ; structure effondrée (hallucination) |

## 4. Implémentation

### 4.1 Stack Technologique

- **Langage** : Rust (Édition 2024)
- **Compression** : Zstandard (crate zstd)
- **Analyse de Graphes** : petgraph
- **CLI** : clap
- **Client HTTP** : reqwest + tokio (async)

### 4.2 Architecture Modulaire

```
src/
├── core/
│   ├── ncd.rs        # Distance de Compression Normalisée
│   ├── entropy.rs    # Entropie Shannon + TTR + Hapax
│   └── topology.rs   # Construction de graphes et métriques
├── probe/
│   ├── cleaner.rs    # Prétraitement texte (suppression stop-words)
│   └── injector.rs   # Client API LLM (Ollama/OpenAI/Anthropic)
├── audit/
│   └── logger.rs     # Trace d'audit JSON
└── main.rs           # Interface CLI
```

## 5. Installation

```bash
git clone https://github.com/JulienDbrt/LDSI.git
cd LDSI
cargo build --release
./target/release/ldsi --help
```

## 6. Utilisation

### 6.1 Analyse Complète

```bash
ldsi analyze \
  --text-a "Le chat dort sur le canapé." \
  --text-b "L'entité féline transcende les paradigmes oniriques." \
  --output resultats.json
```

### 6.2 Métriques Individuelles

```bash
# NCD seul
ldsi ncd "texte A" "texte B"

# Entropie seule
ldsi entropy "texte à analyser"

# Topologie seule
ldsi topology "texte à analyser"
```

### 6.3 Injection Live sur LLM

```bash
ldsi inject \
  --url http://localhost:11434 \
  --model llama3 \
  --api-type ollama \
  --prompt-a "Explique la physique quantique." \
  --prompt-b "En tant que poète du chaos, explique la physique quantique."
```

## 7. Limitations et Travaux Futurs

### 7.1 Limitations Actuelles

1. **Sensibilité linguistique** : Les listes de stop-words supportent actuellement le français et l'anglais
2. **Cécité sémantique** : NCD capture la similarité structurelle, pas le sens
3. **Sensibilité à la taille de fenêtre** : La topologie de graphe dépend de la fenêtre de co-occurrence

### 7.2 Directions Futures

1. Extension multilingue des stop-words
2. Analyse NCD cross-linguale
3. Analyse de stabilité temporelle entre versions de modèles
4. Intégration avec benchmarks LLM existants (MMLU, HellaSwag)

## 8. Citation

```bibtex
@software{dabert2025ldsi,
  author = {Dabert, Julien},
  title = {{LDSI}: {L}yapunov-{D}abert {S}tability {I}ndex pour l'Évaluation des {LLM}},
  year = {2025},
  url = {https://github.com/JulienDbrt/LDSI},
  version = {0.1.0}
}
```

---

# Español

## Resumen

Introducimos el **Lyapunov-Dabert Stability Index (LDSI)**, una métrica novedosa de caja blanca para cuantificar la divergencia semántica entre las salidas de Grandes Modelos de Lenguaje (LLM). A diferencia de los métodos de evaluación existentes que dependen de jueces basados en redes neuronales (introduciendo opacidad y sesgo potencial), LDSI emplea exclusivamente medidas matemáticas y de teoría de la información: **Distancia de Compresión Normalizada (NCD)** basada en la aproximación de la complejidad de Kolmogorov, **entropía de Shannon** para la evaluación de diversidad léxica, y **análisis teórico de grafos** para la verificación de coherencia estructural.

La métrica propuesta es completamente determinista, reproducible a nivel de bit, agnóstica al modelo y computacionalmente eficiente (sin necesidad de GPU). LDSI proporciona un marco auditable para evaluar las propiedades creativas y estructurales de las respuestas de LLM, distinguiendo entre mera repetición, divergencia significativa y alucinación incoherente.

**Palabras clave:** Grandes Modelos de Lenguaje, Métricas de Evaluación, Complejidad de Kolmogorov, Entropía de Shannon, Teoría de Grafos, Benchmarking de Caja Blanca

## 1. Introducción

### 1.1 Planteamiento del Problema

Los paradigmas actuales de evaluación de LLM enfrentan un desafío epistemológico fundamental: usar redes neuronales para evaluar redes neuronales introduce dependencias circulares y opacidad. Cuando GPT-4 juzga la salida de Claude, o viceversa, la evaluación hereda los sesgos, alucinaciones y artefactos de entrenamiento del modelo juez. Esto crea una **galería de espejos evaluativa** donde no se puede establecer ninguna verdad fundamental.

### 1.2 Solución Propuesta

LDSI aborda este desafío fundamentando la evaluación en primitivas matemáticas que preceden y trascienden las arquitecturas neuronales:

1. **Distancia basada en compresión** (arraigada en la teoría algorítmica de la información)
2. **Entropía de información** (del trabajo fundacional de Shannon)
3. **Métricas de topología de grafos** (de las matemáticas discretas)

Estas medidas son:
- **Deterministas**: Entradas idénticas siempre producen salidas idénticas
- **Auditables**: Cualquier investigador puede verificar los resultados con herramientas estándar
- **Agnósticas al modelo**: Aplicables a cualquier sistema generador de texto
- **Computacionalmente triviales**: Ejecutables en hardware común en milisegundos

## 2. Fundamentos Teóricos

### 2.1 Complejidad de Kolmogorov y NCD

La **complejidad de Kolmogorov** K(x) de una cadena x se define como la longitud del programa más corto que produce x en una máquina de Turing universal. Aunque K(x) es incomputable, puede aproximarse usando algoritmos de compresión reales.

La **Distancia de Compresión Normalizada** (Cilibrasi & Vitányi, 2005) aprovecha esta aproximación:

$$NCD(x, y) = \frac{C(xy) - \min(C(x), C(y))}{\max(C(x), C(y))}$$

Donde:
- $C(x)$ = tamaño comprimido del texto x
- $C(y)$ = tamaño comprimido del texto y
- $C(xy)$ = tamaño comprimido de la concatenación

**Propiedades:**
- $NCD \approx 0$: Los textos son casi idénticos (suavizado total)
- $NCD \approx 1$: Los textos comparten información mínima (divergencia máxima)

### 2.2 Entropía de Shannon

La **entropía de Shannon** (Shannon, 1948) cuantifica el contenido de información promedio de una variable aleatoria discreta:

$$H(X) = -\sum_{i=1}^{n} p(x_i) \log_2 p(x_i)$$

Aplicada al texto, donde $p(x_i)$ es la probabilidad del token $x_i$:
- **Baja entropía**: Vocabulario predecible, repetitivo
- **Alta entropía**: Elecciones léxicas diversas, sorprendentes

Extendemos esto con:
- **Ratio Tipo-Token (TTR)**: $\frac{|V|}{N}$ donde |V| = tokens únicos, N = tokens totales
- **Ratio Hapax Legomena**: Proporción de palabras que aparecen exactamente una vez

### 2.3 Análisis Estructural por Teoría de Grafos

El texto se transforma en un **grafo de co-ocurrencia** $G = (V, E)$:
- **Vértices** $V$: Tokens lematizados únicos
- **Aristas** $E$: Co-ocurrencia dentro de ventana deslizante (predeterminado: 5 tokens)

Las métricas estructurales incluyen:
- **Densidad**: $\frac{|E|}{|V|(|V|-1)}$
- **Ratio de Componente Conexo Más Grande (LCC)**: Indicador de coherencia
- **Coeficiente de Clustering**: Interconexión local
- **Longitud Media de Camino**: Navegabilidad entre conceptos
- **Índice Small-World**: $\frac{C}{L}$ (clustering / longitud de camino)

## 3. La Fórmula LDSI

### 3.1 Índice Compuesto

El **Lyapunov-Dabert Stability Index** combina los tres pilares:

$$\lambda_{LD} = \alpha \cdot NCD(A, B) + \beta \cdot \frac{H(B)}{H(A)} + \gamma \cdot \Delta_{Graph}$$

Donde:
- $A$ = Respuesta de referencia (prompt estándar)
- $B$ = Respuesta de prueba (prompt modificado)
- $\alpha, \beta, \gamma$ = Coeficientes de ponderación (predeterminado: 0.40, 0.35, 0.25)
- $\Delta_{Graph}$ = Puntuación de preservación estructural

### 3.2 Coeficientes Predeterminados

| Coeficiente | Valor | Justificación |
|-------------|-------|---------------|
| α (NCD) | 0.40 | Medida primaria de divergencia |
| β (Entropía) | 0.35 | Riqueza informacional |
| γ (Topología) | 0.25 | Preservación de coherencia |

### 3.3 Escala de Interpretación

| Rango λ_LD | Clasificación | Interpretación |
|------------|---------------|----------------|
| 0.0 - 0.3 | **ZOMBIE** | El modelo ignora la variación del prompt; pura recitación |
| 0.3 - 0.7 | **REBELDE** | Divergencia notable; vocabulario enriquecido |
| 0.7 - 1.2 | **ARQUITECTO** | Zona óptima; alta divergencia con estructura preservada |
| > 1.2 | **LOCO** | Entropía máxima; estructura colapsada (alucinación) |

## 4. Implementación

### 4.1 Stack Tecnológico

- **Lenguaje**: Rust (Edición 2024)
- **Compresión**: Zstandard (crate zstd)
- **Análisis de Grafos**: petgraph
- **CLI**: clap
- **Cliente HTTP**: reqwest + tokio (async)

### 4.2 Arquitectura Modular

```
src/
├── core/
│   ├── ncd.rs        # Distancia de Compresión Normalizada
│   ├── entropy.rs    # Entropía Shannon + TTR + Hapax
│   └── topology.rs   # Construcción de grafos y métricas
├── probe/
│   ├── cleaner.rs    # Preprocesamiento de texto (eliminación de stopwords)
│   └── injector.rs   # Cliente API LLM (Ollama/OpenAI/Anthropic)
├── audit/
│   └── logger.rs     # Registro de auditoría JSON
└── main.rs           # Interfaz CLI
```

## 5. Instalación

```bash
git clone https://github.com/JulienDbrt/LDSI.git
cd LDSI
cargo build --release
./target/release/ldsi --help
```

## 6. Uso

### 6.1 Análisis Completo

```bash
ldsi analyze \
  --text-a "El gato duerme en el sofá." \
  --text-b "La entidad felina trasciende los paradigmas oníricos." \
  --output resultados.json
```

### 6.2 Métricas Individuales

```bash
# Solo NCD
ldsi ncd "texto A" "texto B"

# Solo Entropía
ldsi entropy "texto a analizar"

# Solo Topología
ldsi topology "texto a analizar"
```

### 6.3 Inyección en Vivo a LLM

```bash
ldsi inject \
  --url http://localhost:11434 \
  --model llama3 \
  --api-type ollama \
  --prompt-a "Explica la física cuántica." \
  --prompt-b "Como poeta del caos, explica la física cuántica."
```

## 7. Limitaciones y Trabajo Futuro

### 7.1 Limitaciones Actuales

1. **Sensibilidad lingüística**: Las listas de stopwords actualmente soportan francés e inglés
2. **Ceguera semántica**: NCD captura similitud estructural, no significado
3. **Sensibilidad al tamaño de ventana**: La topología del grafo depende de la ventana de co-ocurrencia

### 7.2 Direcciones Futuras

1. Expansión multilingüe de stopwords
2. Análisis NCD cross-lingüístico
3. Análisis de estabilidad temporal entre versiones de modelos
4. Integración con benchmarks LLM existentes (MMLU, HellaSwag)

## 8. Citación

```bibtex
@software{dabert2025ldsi,
  author = {Dabert, Julien},
  title = {{LDSI}: {L}yapunov-{D}abert {S}tability {I}ndex para Evaluación de {LLM}},
  year = {2025},
  url = {https://github.com/JulienDbrt/LDSI},
  version = {0.1.0}
}
```

---

## License / Licence / Licencia

MIT License

Copyright (c) 2025 Julien DABERT

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
