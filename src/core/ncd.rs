//! Module NCD - Normalized Compression Distance
//!
//! Mesure la distance sémantique brute entre deux textes via compression.
//! Basé sur la complexité de Kolmogorov approximée par Zstandard.
//!
//! IMPORTANT: La fenêtre de compression est configurée dynamiquement pour
//! garantir que le compresseur "voit" l'intégralité des deux textes.
//! Sans cela, les textes longs produisent des NCD faussés (myopie zstd).
//!
//! Auteur: Julien DABERT
//! LDSI - Lyapunov-Dabert Stability Index

use std::cmp::{max, min};
use std::io::Cursor;
use zstd::stream::read::Encoder;
use std::io::Read;

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

/// window_log minimum (10 = 1KB) et maximum (31 = 2GB)
const MIN_WINDOW_LOG: u32 = 10;
const MAX_WINDOW_LOG: u32 = 31;

/// Calcule le window_log optimal pour couvrir la taille donnée
/// window_log = ceil(log2(size)) avec clamp [10, 31]
fn optimal_window_log(size: usize) -> u32 {
    if size == 0 {
        return MIN_WINDOW_LOG;
    }
    // ceil(log2(size)) = nombre de bits nécessaires
    let bits_needed = usize::BITS - size.leading_zeros();
    bits_needed.clamp(MIN_WINDOW_LOG, MAX_WINDOW_LOG)
}

/// Calcule la taille compressée d'une chaîne via Zstandard
/// avec fenêtre configurée pour couvrir l'intégralité du texte.
///
/// # Arguments
/// * `input` - Texte à compresser
/// * `window_log` - Taille de fenêtre (2^window_log octets)
///
/// # Returns
/// Taille en octets du texte compressé
fn compressed_size_with_window(input: &str, window_log: u32) -> usize {
    let cursor = Cursor::new(input.as_bytes());

    let mut encoder = match Encoder::new(cursor, COMPRESSION_LEVEL) {
        Ok(enc) => enc,
        Err(_) => return input.len(),
    };

    // Configure la fenêtre de compression
    if encoder.set_parameter(zstd::stream::raw::CParameter::WindowLog(window_log)).is_err() {
        // Fallback si le paramètre échoue
        let cursor = Cursor::new(input.as_bytes());
        if let Ok(mut enc) = Encoder::new(cursor, COMPRESSION_LEVEL) {
            let mut compressed = Vec::new();
            if enc.read_to_end(&mut compressed).is_ok() {
                return compressed.len();
            }
        }
        return input.len();
    }

    let mut compressed = Vec::new();
    match encoder.read_to_end(&mut compressed) {
        Ok(_) => compressed.len(),
        Err(_) => input.len(),
    }
}

/// Calcule la Normalized Compression Distance entre deux textes
///
/// Formule: NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
///
/// La fenêtre de compression est calculée dynamiquement pour garantir
/// que le compresseur "voit" l'intégralité de la concaténation A+B.
/// Cela évite la "myopie zstd" sur les textes longs.
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
    // Calcul de la fenêtre optimale basée sur la taille totale
    let combined = format!("{}{}", text_a, text_b);
    let window_log = optimal_window_log(combined.len());

    // Compression avec fenêtre cohérente pour toutes les mesures
    let size_a = compressed_size_with_window(text_a, window_log);
    let size_b = compressed_size_with_window(text_b, window_log);
    let size_combined = compressed_size_with_window(&combined, window_log);

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
    let score = score.clamp(0.0, 1.5);

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

    #[test]
    fn test_optimal_window_log() {
        // Cas limites
        assert_eq!(optimal_window_log(0), MIN_WINDOW_LOG);
        assert_eq!(optimal_window_log(1), MIN_WINDOW_LOG);

        // Cas typiques
        assert_eq!(optimal_window_log(1024), 11);      // 2^10 = 1024, besoin de 11 bits
        assert_eq!(optimal_window_log(1025), 11);      // Juste au-dessus
        assert_eq!(optimal_window_log(1_000_000), 20); // ~1MB

        // Très grand (clamp à MAX_WINDOW_LOG)
        assert_eq!(optimal_window_log(usize::MAX), MAX_WINDOW_LOG);
    }

    #[test]
    fn test_long_text_no_myopia() {
        // Génère deux textes longs (> 1MB pour dépasser la fenêtre par défaut)
        // avec un pattern répétitif qui serait compressible si la fenêtre est correcte
        let pattern_a = "Le chat dort sur le canapé rouge. ";
        let pattern_b = "Le chien court dans le jardin vert. ";

        // Répète pour atteindre ~100KB chacun (suffisant pour tester la fenêtre)
        let text_a: String = pattern_a.repeat(3000); // ~105KB
        let text_b: String = pattern_b.repeat(3000);

        let result = compute_ncd(&text_a, &text_b);

        // Avec la fenêtre correcte, les patterns répétitifs devraient bien se compresser
        // et le NCD devrait être dans une plage raisonnable (pas artificiellement élevé)
        assert!(result.score < 1.2, "NCD textes longs ne devrait pas exploser: {}", result.score);
        assert!(result.score > 0.3, "NCD textes différents devrait être notable: {}", result.score);

        // Vérifie que la compression a fonctionné (ratio significatif)
        assert!(result.size_a < result.raw_size_a / 2, "Compression A inefficace");
        assert!(result.size_b < result.raw_size_b / 2, "Compression B inefficace");
    }

    #[test]
    fn test_identical_long_texts() {
        // Même texte long doit avoir NCD très bas
        let pattern = "Ceci est un test de répétition pour valider la fenêtre. ";
        let text: String = pattern.repeat(2000); // ~100KB

        let result = compute_ncd(&text, &text);

        // Textes identiques = NCD très faible même pour textes longs
        assert!(result.score < 0.3, "NCD identique long devrait être < 0.3, got {}", result.score);
    }
}
