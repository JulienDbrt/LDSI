//! Module NCD - Normalized Compression Distance
//!
//! Mesure la distance sémantique brute entre deux textes via compression.
//! Basé sur la complexité de Kolmogorov approximée par Zstandard.
//!
//! Auteur: Julien DABERT
//! LDSI - Lyapunov-Dabert Stability Index

use std::cmp::{max, min};
use std::io::Cursor;
use zstd::stream::encode_all;

/// Résultat détaillé du calcul NCD pour audit
#[derive(Debug, Clone)]
pub struct NcdResult {
    /// Score NCD final (0.0 = identique, ~1.0 = totalement différent)
    pub score: f64,
    /// Taille compressée du texte A (octets)
    pub size_a: usize,
    /// Taille compressée du texte B (octets)
    pub size_b: usize,
    /// Taille compressée de A+B concaténés (octets)
    pub size_combined: usize,
    /// Taille brute du texte A (octets)
    pub raw_size_a: usize,
    /// Taille brute du texte B (octets)
    pub raw_size_b: usize,
}

/// Niveau de compression Zstandard (1-22)
/// Niveau 3 = bon compromis vitesse/ratio
const COMPRESSION_LEVEL: i32 = 3;

/// Calcule la taille compressée d'une chaîne via Zstandard
///
/// # Arguments
/// * `input` - Texte à compresser
///
/// # Returns
/// Taille en octets du texte compressé
fn compressed_size(input: &str) -> usize {
    let cursor = Cursor::new(input.as_bytes());
    match encode_all(cursor, COMPRESSION_LEVEL) {
        Ok(compressed) => compressed.len(),
        Err(_) => input.len(), // Fallback: taille brute si erreur
    }
}

/// Calcule la Normalized Compression Distance entre deux textes
///
/// Formule: NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
///
/// # Arguments
/// * `text_a` - Premier texte (réponse standard)
/// * `text_b` - Second texte (réponse fracturée/Codex)
///
/// # Returns
/// Structure NcdResult avec le score et les métriques d'audit
///
/// # Interprétation
/// - NCD ≈ 0.0 : Textes quasi-identiques (lissage total)
/// - NCD ≈ 0.5 : Divergence modérée
/// - NCD ≈ 1.0 : Divergence maximale
pub fn compute_ncd(text_a: &str, text_b: &str) -> NcdResult {
    // Compression individuelle
    let size_a = compressed_size(text_a);
    let size_b = compressed_size(text_b);

    // Compression combinée (concaténation)
    let combined = format!("{}{}", text_a, text_b);
    let size_combined = compressed_size(&combined);

    // Calcul NCD
    let min_c = min(size_a, size_b) as f64;
    let max_c = max(size_a, size_b) as f64;

    // Protection division par zéro
    let score = if max_c > 0.0 {
        (size_combined as f64 - min_c) / max_c
    } else {
        0.0
    };

    // Clamp [0.0, 1.5] - valeurs > 1.0 possibles avec certains compresseurs
    let score = score.max(0.0).min(1.5);

    NcdResult {
        score,
        size_a,
        size_b,
        size_combined,
        raw_size_a: text_a.len(),
        raw_size_b: text_b.len(),
    }
}

/// Calcule uniquement le score NCD (version simplifiée)
pub fn ncd_score(text_a: &str, text_b: &str) -> f64 {
    compute_ncd(text_a, text_b).score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_texts() {
        let text = "Le chat dort sur le canapé.";
        let result = compute_ncd(text, text);
        // Textes identiques = NCD très faible
        assert!(result.score < 0.3, "NCD identique devrait être < 0.3, got {}", result.score);
    }

    #[test]
    fn test_different_texts() {
        let a = "Le chat dort paisiblement sur le canapé rouge.";
        let b = "La singularité quantique transcende les paradigmes ontologiques.";
        let result = compute_ncd(a, b);
        // Textes très différents = NCD élevé
        assert!(result.score > 0.5, "NCD différent devrait être > 0.5, got {}", result.score);
    }

    #[test]
    fn test_audit_trail() {
        let a = "Hello";
        let b = "World";
        let result = compute_ncd(a, b);
        // Vérification que les tailles sont cohérentes
        assert!(result.size_a > 0);
        assert!(result.size_b > 0);
        assert!(result.size_combined > 0);
        assert_eq!(result.raw_size_a, 5);
        assert_eq!(result.raw_size_b, 5);
    }
}
