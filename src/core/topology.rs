//! Module Topology - Analyse de Cohérence par Théorie des Graphes
//!
//! Transforme le texte en graphe de co-occurrence et mesure sa structure.
//! Détecte le délire (graphe éclaté) vs l'intelligence (graphe interconnecté).
//!
//! Auteur: Julien DABERT
//! LDSI - Lyapunov-Dabert Stability Index

use petgraph::algo::connected_components;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet, VecDeque};

/// Résultat détaillé de l'analyse topologique
#[derive(Debug, Clone)]
pub struct TopologyResult {
    /// Nombre de nœuds (mots uniques)
    pub node_count: usize,
    /// Nombre d'arêtes (co-occurrences)
    pub edge_count: usize,
    /// Densité du graphe (edges / max_possible_edges)
    pub density: f64,
    /// Nombre de composantes connexes
    pub components: usize,
    /// Taille de la plus grande composante connexe (LCC)
    pub lcc_size: usize,
    /// Ratio LCC / total nodes
    pub lcc_ratio: f64,
    /// Coefficient de clustering moyen
    pub clustering_coefficient: f64,
    /// Longueur moyenne des chemins (approximation)
    pub avg_path_length: f64,
    /// Indicateur Small-World (clustering / path_length)
    pub small_world_index: f64,
    /// Degré moyen des nœuds
    pub avg_degree: f64,
}

/// Taille de la fenêtre glissante pour co-occurrence
const WINDOW_SIZE: usize = 5;

/// Tokenize simplement (même logique que entropy pour cohérence)
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphabetic())
        .filter(|s| !s.is_empty() && s.len() > 1)
        .map(|s| s.to_lowercase())
        .collect()
}

/// Construit un graphe dirigé de co-occurrence
///
/// - Nœuds = mots uniques (lemmatisés en minuscules)
/// - Arêtes = séquentialité dans une fenêtre glissante
fn build_cooccurrence_graph(tokens: &[String]) -> DiGraph<String, u32> {
    let mut graph: DiGraph<String, u32> = DiGraph::new();
    let mut node_indices: HashMap<String, NodeIndex> = HashMap::new();

    // Créer les nœuds
    for token in tokens {
        if !node_indices.contains_key(token) {
            let idx = graph.add_node(token.clone());
            node_indices.insert(token.clone(), idx);
        }
    }

    // Créer les arêtes par fenêtre glissante
    if tokens.len() >= 2 {
        for window in tokens.windows(WINDOW_SIZE.min(tokens.len())) {
            for i in 0..window.len() {
                for j in (i + 1)..window.len() {
                    let from = node_indices[&window[i]];
                    let to = node_indices[&window[j]];

                    // Éviter les self-loops
                    if from != to {
                        // Incrémenter le poids si l'arête existe déjà
                        if let Some(edge) = graph.find_edge(from, to) {
                            if let Some(weight) = graph.edge_weight_mut(edge) {
                                *weight += 1;
                            }
                        } else {
                            graph.add_edge(from, to, 1);
                        }
                    }
                }
            }
        }
    }

    graph
}

/// Calcule la densité du graphe
fn compute_density(node_count: usize, edge_count: usize) -> f64 {
    if node_count < 2 {
        return 0.0;
    }
    let max_edges = node_count * (node_count - 1); // Graphe dirigé
    edge_count as f64 / max_edges as f64
}

/// Calcule le coefficient de clustering local d'un nœud
fn local_clustering(graph: &DiGraph<String, u32>, node: NodeIndex) -> f64 {
    let neighbors: HashSet<NodeIndex> = graph
        .neighbors_undirected(node)
        .collect();

    let k = neighbors.len();
    if k < 2 {
        return 0.0;
    }

    let mut triangles = 0;
    for &n1 in &neighbors {
        for &n2 in &neighbors {
            if n1 != n2 && (graph.contains_edge(n1, n2) || graph.contains_edge(n2, n1)) {
                triangles += 1;
            }
        }
    }

    // Chaque triangle est compté 2 fois
    triangles as f64 / (k * (k - 1)) as f64
}

/// Calcule le coefficient de clustering moyen
fn average_clustering(graph: &DiGraph<String, u32>) -> f64 {
    let nodes: Vec<NodeIndex> = graph.node_indices().collect();
    if nodes.is_empty() {
        return 0.0;
    }

    let sum: f64 = nodes.iter().map(|&n| local_clustering(graph, n)).sum();
    sum / nodes.len() as f64
}

/// Calcule la longueur moyenne des plus courts chemins (BFS, échantillonné)
fn average_path_length(graph: &DiGraph<String, u32>) -> f64 {
    let nodes: Vec<NodeIndex> = graph.node_indices().collect();
    if nodes.len() < 2 {
        return 0.0;
    }

    // Échantillonnage pour performance (max 50 nœuds sources)
    let sample_size = nodes.len().min(50);
    let mut total_length = 0usize;
    let mut path_count = 0usize;

    for &source in nodes.iter().take(sample_size) {
        // BFS depuis source
        let mut visited: HashMap<NodeIndex, usize> = HashMap::new();
        let mut queue: VecDeque<NodeIndex> = VecDeque::new();

        visited.insert(source, 0);
        queue.push_back(source);

        while let Some(current) = queue.pop_front() {
            let current_dist = visited[&current];

            for neighbor in graph.neighbors(current) {
                if !visited.contains_key(&neighbor) {
                    let new_dist = current_dist + 1;
                    visited.insert(neighbor, new_dist);
                    queue.push_back(neighbor);
                    total_length += new_dist;
                    path_count += 1;
                }
            }
        }
    }

    if path_count > 0 {
        total_length as f64 / path_count as f64
    } else {
        0.0
    }
}

/// Trouve la taille de la plus grande composante connexe
fn largest_connected_component(graph: &DiGraph<String, u32>) -> usize {
    if graph.node_count() == 0 {
        return 0;
    }

    let mut visited: HashSet<NodeIndex> = HashSet::new();
    let mut max_size = 0;

    for node in graph.node_indices() {
        if !visited.contains(&node) {
            // BFS pour trouver la composante
            let mut component_size = 0;
            let mut queue: VecDeque<NodeIndex> = VecDeque::new();
            queue.push_back(node);

            while let Some(current) = queue.pop_front() {
                if visited.insert(current) {
                    component_size += 1;
                    for neighbor in graph.neighbors_undirected(current) {
                        if !visited.contains(&neighbor) {
                            queue.push_back(neighbor);
                        }
                    }
                }
            }

            max_size = max_size.max(component_size);
        }
    }

    max_size
}

/// Calcule le degré moyen des nœuds
fn average_degree(graph: &DiGraph<String, u32>) -> f64 {
    let node_count = graph.node_count();
    if node_count == 0 {
        return 0.0;
    }

    let total_degree: usize = graph
        .node_indices()
        .map(|n| graph.edges(n).count())
        .sum();

    total_degree as f64 / node_count as f64
}

/// Analyse topologique complète d'un texte
///
/// # Arguments
/// * `text` - Texte à analyser
///
/// # Returns
/// Structure TopologyResult avec toutes les métriques de graphe
pub fn analyze_topology(text: &str) -> TopologyResult {
    let tokens = tokenize(text);

    if tokens.is_empty() {
        return TopologyResult {
            node_count: 0,
            edge_count: 0,
            density: 0.0,
            components: 0,
            lcc_size: 0,
            lcc_ratio: 0.0,
            clustering_coefficient: 0.0,
            avg_path_length: 0.0,
            small_world_index: 0.0,
            avg_degree: 0.0,
        };
    }

    let graph = build_cooccurrence_graph(&tokens);

    let node_count = graph.node_count();
    let edge_count = graph.edge_count();
    let density = compute_density(node_count, edge_count);
    let components = connected_components(&graph);
    let lcc_size = largest_connected_component(&graph);
    let lcc_ratio = if node_count > 0 {
        lcc_size as f64 / node_count as f64
    } else {
        0.0
    };
    let clustering_coefficient = average_clustering(&graph);
    let avg_path_length = average_path_length(&graph);

    // Small-World Index: C / L (clustering élevé, path court)
    let small_world_index = if avg_path_length > 0.0 {
        clustering_coefficient / avg_path_length
    } else {
        0.0
    };

    let avg_degree = average_degree(&graph);

    TopologyResult {
        node_count,
        edge_count,
        density,
        components,
        lcc_size,
        lcc_ratio,
        clustering_coefficient,
        avg_path_length,
        small_world_index,
        avg_degree,
    }
}

/// Calcule le delta topologique entre deux textes
///
/// Retourne un score de conservation de structure:
/// - Positif = structure améliorée ou maintenue
/// - Négatif = structure dégradée (potentiel délire)
pub fn topology_delta(text_a: &str, text_b: &str) -> f64 {
    let topo_a = analyze_topology(text_a);
    let topo_b = analyze_topology(text_b);

    // Facteurs de qualité structurelle
    let lcc_score = topo_b.lcc_ratio - topo_a.lcc_ratio;
    let clustering_score = topo_b.clustering_coefficient - topo_a.clustering_coefficient;

    // Pénalité si trop fragmenté (beaucoup de composantes)
    let fragmentation_penalty = if topo_b.components > topo_a.components * 2 {
        -0.2
    } else {
        0.0
    };

    // Score composite
    (lcc_score * 0.5) + (clustering_score * 0.3) + fragmentation_penalty + 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_graph() {
        let text = "Le chat mange la souris. La souris fuit le chat.";
        let result = analyze_topology(text);

        assert!(result.node_count > 0, "Devrait avoir des nœuds");
        assert!(result.edge_count > 0, "Devrait avoir des arêtes");
        assert!(result.density > 0.0, "Densité devrait être > 0");
    }

    #[test]
    fn test_connected_text() {
        let text = "Alpha beta gamma. Beta gamma delta. Gamma delta epsilon.";
        let result = analyze_topology(text);

        // Un texte bien connecté devrait avoir une seule composante
        assert!(result.lcc_ratio > 0.5, "LCC ratio devrait être élevé pour texte connecté");
    }

    #[test]
    fn test_empty_text() {
        let result = analyze_topology("");
        assert_eq!(result.node_count, 0);
        assert_eq!(result.edge_count, 0);
    }

    #[test]
    fn test_topology_delta() {
        let standard = "Le chat dort.";
        let enriched = "Le félin somnole paisiblement sur le coussin moelleux du salon.";
        let delta = topology_delta(standard, enriched);

        // Le texte enrichi devrait avoir une structure au moins comparable
        assert!(delta > 0.0, "Delta devrait être positif pour texte enrichi structuré");
    }
}
