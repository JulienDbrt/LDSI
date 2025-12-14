//! Module Entropy - Entropie de Shannon & Diversité Lexicale
//!
//! Mesure la richesse informationnelle et la "surprise" du texte.
//! Zéro IA - Pure statistique fréquentielle.
//!
//! Auteur: Julien DABERT
//! LDSI - Lyapunov-Dabert Stability Index

use std::collections::HashMap;

/// Résultat détaillé de l'analyse entropique
#[derive(Debug, Clone)]
pub struct EntropyResult {
    /// Entropie de Shannon (bits)
    pub shannon: f64,
    /// Type-Token Ratio (vocabulaire unique / total mots)
    pub ttr: f64,
    /// Nombre total de tokens
    pub total_tokens: usize,
    /// Nombre de tokens uniques
    pub unique_tokens: usize,
    /// Hapax legomena (mots n'apparaissant qu'une fois)
    pub hapax_count: usize,
    /// Ratio hapax / total (indicateur de richesse)
    pub hapax_ratio: f64,
}

/// Calcule l'entropie de Shannon sur une distribution de fréquences
///
/// Formule: H(X) = -Σ p(x) * log2(p(x))
///
/// # Arguments
/// * `frequencies` - HashMap mot -> nombre d'occurrences
/// * `total` - Nombre total de tokens
///
/// # Returns
/// Entropie en bits
fn shannon_entropy(frequencies: &HashMap<String, usize>, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }

    let total_f64 = total as f64;

    frequencies
        .values()
        .map(|&count| {
            let p = count as f64 / total_f64;
            if p > 0.0 {
                -p * p.log2()
            } else {
                0.0
            }
        })
        .sum()
}

/// Tokenize un texte en mots (White Box - regex pure)
///
/// Conserve uniquement les mots alphabétiques, convertis en minuscules.
/// Pas de stemming/lemmatisation pour rester déterministe.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphabetic())
        .filter(|s| !s.is_empty() && s.len() > 1) // Ignore les lettres seules
        .map(|s| s.to_lowercase())
        .collect()
}

/// Calcule les métriques d'entropie pour un texte
///
/// # Arguments
/// * `text` - Texte à analyser
///
/// # Returns
/// Structure EntropyResult avec toutes les métriques
pub fn compute_entropy(text: &str) -> EntropyResult {
    let tokens = tokenize(text);
    let total_tokens = tokens.len();

    // Comptage des fréquences
    let mut frequencies: HashMap<String, usize> = HashMap::new();
    for token in &tokens {
        *frequencies.entry(token.clone()).or_insert(0) += 1;
    }

    let unique_tokens = frequencies.len();

    // Hapax legomena (mots uniques)
    let hapax_count = frequencies.values().filter(|&&c| c == 1).count();

    // Calculs
    let shannon = shannon_entropy(&frequencies, total_tokens);

    let ttr = if total_tokens > 0 {
        unique_tokens as f64 / total_tokens as f64
    } else {
        0.0
    };

    let hapax_ratio = if total_tokens > 0 {
        hapax_count as f64 / total_tokens as f64
    } else {
        0.0
    };

    EntropyResult {
        shannon,
        ttr,
        total_tokens,
        unique_tokens,
        hapax_count,
        hapax_ratio,
    }
}

/// Calcule le ratio d'entropie entre deux textes
///
/// # Arguments
/// * `text_a` - Texte de référence (standard)
/// * `text_b` - Texte à comparer (fracturé)
///
/// # Returns
/// Ratio H(B) / H(A) - > 1.0 signifie gain d'information
pub fn entropy_ratio(text_a: &str, text_b: &str) -> f64 {
    let h_a = compute_entropy(text_a).shannon;
    let h_b = compute_entropy(text_b).shannon;

    if h_a > 0.0 {
        h_b / h_a
    } else if h_b > 0.0 {
        f64::INFINITY
    } else {
        1.0 // Les deux sont nuls
    }
}

/// Calcule l'entropie sur les n-grammes (bigrammes par défaut)
///
/// Plus sensible aux patterns structurels que l'entropie sur mots seuls.
pub fn compute_ngram_entropy(text: &str, n: usize) -> f64 {
    let tokens = tokenize(text);

    if tokens.len() < n {
        return 0.0;
    }

    let mut ngram_freq: HashMap<String, usize> = HashMap::new();

    for window in tokens.windows(n) {
        let ngram = window.join(" ");
        *ngram_freq.entry(ngram).or_insert(0) += 1;
    }

    let total = tokens.len() - n + 1;
    shannon_entropy(&ngram_freq, total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_distribution() {
        // 4 mots uniques = entropie maximale pour 4 symboles = 2 bits
        let text = "alpha beta gamma delta";
        let result = compute_entropy(text);
        assert!((result.shannon - 2.0).abs() < 0.01, "Entropie uniforme 4 symboles = 2 bits, got {}", result.shannon);
        assert_eq!(result.ttr, 1.0, "TTR devrait être 1.0 pour mots tous uniques");
    }

    #[test]
    fn test_repetitive_text() {
        let text = "le le le le le le le le";
        let result = compute_entropy(text);
        assert!(result.shannon < 0.1, "Texte répétitif = entropie quasi-nulle");
        assert!(result.ttr < 0.2, "TTR très faible pour répétition");
    }

    #[test]
    fn test_entropy_ratio() {
        let standard = "Le chat dort sur le tapis.";
        let enriched = "Le félin somnole paisiblement sur le kilim persan ancestral.";
        let ratio = entropy_ratio(standard, enriched);
        assert!(ratio > 1.0, "Texte enrichi devrait avoir ratio > 1.0");
    }

    #[test]
    fn test_hapax() {
        let text = "alpha beta gamma delta alpha";
        let result = compute_entropy(text);
        // beta, gamma, delta sont des hapax
        assert_eq!(result.hapax_count, 3);
    }
}
