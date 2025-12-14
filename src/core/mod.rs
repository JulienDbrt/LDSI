//! Core LDSI - Moteur de Calcul Mathématique
//!
//! Les 3 piliers de la mesure White Box:
//! - NCD: Distance de compression normalisée
//! - Entropy: Entropie de Shannon & diversité lexicale
//! - Topology: Analyse de graphes de co-occurrence
//!
//! Auteur: Julien DABERT
//! LDSI - Lyapunov-Dabert Stability Index

pub mod entropy;
pub mod ncd;
pub mod topology;

use serde::{Deserialize, Serialize};

/// Coefficients de la formule λLD
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LdsiCoefficients {
    /// Poids de la divergence NCD (α)
    pub alpha: f64,
    /// Poids du ratio d'entropie (β)
    pub beta: f64,
    /// Poids du delta topologique (γ)
    pub gamma: f64,
}

impl Default for LdsiCoefficients {
    fn default() -> Self {
        // Coefficients calibrés par défaut
        Self {
            alpha: 0.4, // NCD: 40%
            beta: 0.35, // Entropie: 35%
            gamma: 0.25, // Topologie: 25%
        }
    }
}

/// Résultat complet du calcul LDSI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LdsiResult {
    /// Score λLD final
    pub lambda: f64,
    /// Verdict textuel
    pub verdict: LdsiVerdict,
    /// Métriques NCD détaillées
    pub ncd: NcdMetrics,
    /// Métriques d'entropie détaillées
    pub entropy: EntropyMetrics,
    /// Métriques topologiques détaillées
    pub topology: TopologyMetrics,
    /// Coefficients utilisés
    pub coefficients: LdsiCoefficients,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NcdMetrics {
    pub score: f64,
    pub size_a: usize,
    pub size_b: usize,
    pub size_combined: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyMetrics {
    pub shannon_a: f64,
    pub shannon_b: f64,
    pub ratio: f64,
    pub ttr_a: f64,
    pub ttr_b: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyMetrics {
    pub delta: f64,
    pub density_a: f64,
    pub density_b: f64,
    pub lcc_ratio_a: f64,
    pub lcc_ratio_b: f64,
    pub clustering_a: f64,
    pub clustering_b: f64,
}

/// Verdict du score LDSI
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LdsiVerdict {
    /// 0.0 - 0.3: Le modèle récite, ignore le prompt
    Zombie,
    /// 0.3 - 0.7: Divergence notable, vocabulaire enrichi
    Rebelle,
    /// 0.7 - 1.2: Zone optimale - forte divergence, structure solide
    Architecte,
    /// > 1.2: Chaos - entropie max mais structure effondrée
    Fou,
}

impl LdsiVerdict {
    pub fn from_lambda(lambda: f64) -> Self {
        match lambda {
            l if l < 0.3 => LdsiVerdict::Zombie,
            l if l < 0.7 => LdsiVerdict::Rebelle,
            l if l < 1.2 => LdsiVerdict::Architecte,
            _ => LdsiVerdict::Fou,
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            LdsiVerdict::Zombie => "ZOMBIE - Le modèle récite, lissage total",
            LdsiVerdict::Rebelle => "REBELLE - Divergence notable, enrichissement lexical",
            LdsiVerdict::Architecte => "ARCHITECTE - Zone optimale DAN, structure préservée",
            LdsiVerdict::Fou => "FOU - Chaos maximal, structure effondrée",
        }
    }
}

/// Calcule le score LDSI complet entre deux textes
///
/// Formule: λLD = α·NCD(A,B) + β·(H(B)/H(A)) + γ·ΔGraph
///
/// # Arguments
/// * `text_a` - Réponse standard (contrôle)
/// * `text_b` - Réponse fracturée (Codex/DAN)
/// * `coefficients` - Coefficients α, β, γ (optionnel, défaut si None)
///
/// # Returns
/// Structure LdsiResult avec le score et toutes les métriques d'audit
pub fn compute_ldsi(
    text_a: &str,
    text_b: &str,
    coefficients: Option<LdsiCoefficients>,
) -> LdsiResult {
    let coef = coefficients.unwrap_or_default();

    // 1. Calcul NCD
    let ncd_result = ncd::compute_ncd(text_a, text_b);

    // 2. Calcul Entropie
    let entropy_a = entropy::compute_entropy(text_a);
    let entropy_b = entropy::compute_entropy(text_b);
    let entropy_ratio = if entropy_a.shannon > 0.0 {
        entropy_b.shannon / entropy_a.shannon
    } else if entropy_b.shannon > 0.0 {
        2.0 // Bonus si A est nul mais B non
    } else {
        1.0
    };

    // 3. Calcul Topologie
    let topo_a = topology::analyze_topology(text_a);
    let topo_b = topology::analyze_topology(text_b);
    let topo_delta = topology::topology_delta(text_a, text_b);

    // 4. Formule λLD
    let lambda = (coef.alpha * ncd_result.score)
        + (coef.beta * entropy_ratio.min(2.0)) // Cap à 2.0 pour éviter explosion
        + (coef.gamma * topo_delta);

    let verdict = LdsiVerdict::from_lambda(lambda);

    LdsiResult {
        lambda,
        verdict,
        ncd: NcdMetrics {
            score: ncd_result.score,
            size_a: ncd_result.size_a,
            size_b: ncd_result.size_b,
            size_combined: ncd_result.size_combined,
        },
        entropy: EntropyMetrics {
            shannon_a: entropy_a.shannon,
            shannon_b: entropy_b.shannon,
            ratio: entropy_ratio,
            ttr_a: entropy_a.ttr,
            ttr_b: entropy_b.ttr,
        },
        topology: TopologyMetrics {
            delta: topo_delta,
            density_a: topo_a.density,
            density_b: topo_b.density,
            lcc_ratio_a: topo_a.lcc_ratio,
            lcc_ratio_b: topo_b.lcc_ratio,
            clustering_a: topo_a.clustering_coefficient,
            clustering_b: topo_b.clustering_coefficient,
        },
        coefficients: coef,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ldsi_identical() {
        let text = "Le chat dort sur le canapé.";
        let result = compute_ldsi(text, text, None);

        // Pour textes identiques: NCD≈0, entropy_ratio=1.0, topo_delta=0.5
        // λLD = 0.4*0 + 0.35*1.0 + 0.25*0.5 ≈ 0.475
        assert!(result.lambda < 0.6, "Textes identiques = score bas, got {}", result.lambda);
        assert!(result.ncd.score < 0.3, "NCD identique devrait être bas, got {}", result.ncd.score);
    }

    #[test]
    fn test_ldsi_divergent() {
        let standard = "Le chat dort.";
        let fractured = "L'entité féline transcende les paradigmes oniriques dans une \
                         dissolution quantique de la conscience perceptuelle, fragmentant \
                         les axiomes cartésiens de la réalité somatique.";

        let result = compute_ldsi(standard, fractured, None);

        assert!(result.lambda > 0.5, "Textes divergents = score élevé");
        assert!(result.ncd.score > 0.5, "NCD devrait être élevé");
    }

    #[test]
    fn test_verdict_ranges() {
        assert_eq!(LdsiVerdict::from_lambda(0.1), LdsiVerdict::Zombie);
        assert_eq!(LdsiVerdict::from_lambda(0.5), LdsiVerdict::Rebelle);
        assert_eq!(LdsiVerdict::from_lambda(1.0), LdsiVerdict::Architecte);
        assert_eq!(LdsiVerdict::from_lambda(1.5), LdsiVerdict::Fou);
    }
}
