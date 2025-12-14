//! Module Cleaner - Nettoyage Déterministe du Texte
//!
//! Prépare le texte pour l'analyse en supprimant le bruit
//! tout en préservant la matière sémantique.
//!
//! Auteur: Julien DABERT
//! LDSI - Lyapunov-Dabert Stability Index

use regex::Regex;
use std::collections::HashSet;
use unicode_normalization::UnicodeNormalization;

/// Stop-words français (mots vides à filtrer)
const FRENCH_STOPWORDS: &[&str] = &[
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "l",
    "et", "ou", "mais", "donc", "or", "ni", "car",
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "me", "te", "se", "lui", "leur", "y", "en",
    "ce", "cet", "cette", "ces", "mon", "ton", "son", "ma", "ta", "sa",
    "mes", "tes", "ses", "notre", "votre", "nos", "vos", "leurs",
    "qui", "que", "quoi", "dont", "où", "lequel", "laquelle",
    "au", "aux", "avec", "sans", "sous", "sur", "dans", "par", "pour",
    "en", "vers", "chez", "entre", "contre", "depuis", "pendant",
    "être", "avoir", "faire", "dire", "aller", "voir", "pouvoir", "vouloir",
    "est", "sont", "suis", "es", "sommes", "êtes", "était", "été", "a", "ont", "avait", "eu",
    "fait", "dit", "va", "vont", "peut", "veut",
    "ne", "pas", "plus", "moins", "très", "bien", "mal",
    "tout", "tous", "toute", "toutes", "autre", "autres",
    "même", "aussi", "comme", "si", "quand", "alors", "ainsi",
    "c", "n", "s", "j", "qu", "m", "t",
];

/// Stop-words anglais
const ENGLISH_STOPWORDS: &[&str] = &[
    "the", "a", "an", "and", "or", "but", "if", "then", "else",
    "when", "at", "from", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "of", "in", "on", "off", "over", "under",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "been", "being", "because", "as", "until", "while",
    "not", "no", "nor", "only", "own", "same", "so", "than", "too",
    "very", "just", "can", "now", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "any",
];

/// Configuration du nettoyeur
#[derive(Debug, Clone)]
pub struct CleanerConfig {
    /// Supprimer les stop-words
    pub remove_stopwords: bool,
    /// Convertir en minuscules
    pub lowercase: bool,
    /// Supprimer la ponctuation
    pub remove_punctuation: bool,
    /// Supprimer les nombres
    pub remove_numbers: bool,
    /// Normaliser l'unicode (NFD -> NFC)
    pub normalize_unicode: bool,
    /// Langue pour les stop-words
    pub language: Language,
    /// Longueur minimale des mots à conserver
    pub min_word_length: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Language {
    French,
    English,
    Both,
}

impl Default for CleanerConfig {
    fn default() -> Self {
        Self {
            remove_stopwords: true,
            lowercase: true,
            remove_punctuation: true,
            remove_numbers: true,
            normalize_unicode: true,
            language: Language::Both,
            min_word_length: 2,
        }
    }
}

/// Nettoie un texte selon la configuration
///
/// # Arguments
/// * `text` - Texte brut à nettoyer
/// * `config` - Configuration du nettoyage
///
/// # Returns
/// Texte nettoyé prêt pour l'analyse
pub fn clean_text(text: &str, config: &CleanerConfig) -> String {
    let mut result = text.to_string();

    // 1. Normalisation Unicode
    if config.normalize_unicode {
        result = result.nfd().collect::<String>().nfc().collect();
    }

    // 2. Minuscules
    if config.lowercase {
        result = result.to_lowercase();
    }

    // 3. Suppression des nombres
    if config.remove_numbers {
        let re = Regex::new(r"\d+").unwrap();
        result = re.replace_all(&result, " ").to_string();
    }

    // 4. Suppression de la ponctuation (garde les espaces)
    if config.remove_punctuation {
        result = result
            .chars()
            .map(|c| if c.is_alphabetic() || c.is_whitespace() { c } else { ' ' })
            .collect();
    }

    // 5. Construction du set de stop-words
    let stopwords: HashSet<&str> = match config.language {
        Language::French => FRENCH_STOPWORDS.iter().copied().collect(),
        Language::English => ENGLISH_STOPWORDS.iter().copied().collect(),
        Language::Both => FRENCH_STOPWORDS
            .iter()
            .chain(ENGLISH_STOPWORDS.iter())
            .copied()
            .collect(),
    };

    // 6. Filtrage des mots
    let words: Vec<&str> = result
        .split_whitespace()
        .filter(|word| {
            let dominated = word.len() >= config.min_word_length;
            let not_stopword = !config.remove_stopwords || !stopwords.contains(word);
            dominated && not_stopword
        })
        .collect();

    words.join(" ")
}

/// Nettoie avec la configuration par défaut
pub fn clean_default(text: &str) -> String {
    clean_text(text, &CleanerConfig::default())
}

/// Extrait uniquement les substantifs/verbes/adjectifs significatifs
/// (heuristique basée sur la longueur et la fréquence)
pub fn extract_semantic_core(text: &str) -> String {
    let config = CleanerConfig {
        min_word_length: 4, // Mots plus longs = plus significatifs
        ..Default::default()
    };
    clean_text(text, &config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_cleaning() {
        let text = "Le Chat mange la Souris!!! 123";
        let cleaned = clean_default(text);

        assert!(!cleaned.contains("le"), "Stop-word 'le' devrait être supprimé");
        assert!(!cleaned.contains("123"), "Nombres devraient être supprimés");
        assert!(!cleaned.contains("!"), "Ponctuation devrait être supprimée");
        assert!(cleaned.contains("chat"), "Mots significatifs gardés en minuscules");
    }

    #[test]
    fn test_stopwords_french() {
        let text = "je suis un chat qui mange";
        let config = CleanerConfig {
            language: Language::French,
            ..Default::default()
        };
        let cleaned = clean_text(text, &config);

        assert!(!cleaned.contains("je"));
        assert!(!cleaned.contains("suis"));
        assert!(!cleaned.contains("qui"));
        assert!(cleaned.contains("chat"));
        assert!(cleaned.contains("mange"));
    }

    #[test]
    fn test_min_word_length() {
        let text = "a ab abc abcd abcde";
        let config = CleanerConfig {
            remove_stopwords: false,
            min_word_length: 4,
            ..Default::default()
        };
        let cleaned = clean_text(text, &config);

        assert!(!cleaned.contains(" ab "));
        assert!(!cleaned.contains("abc "));
        assert!(cleaned.contains("abcd"));
        assert!(cleaned.contains("abcde"));
    }

    #[test]
    fn test_semantic_core() {
        let text = "Le petit chat noir mange rapidement sa nourriture fraîche.";
        let core = extract_semantic_core(text);

        // Seuls les mots de 4+ caractères sans stop-words
        assert!(core.contains("petit"));
        assert!(core.contains("chat"));
        assert!(core.contains("noir"));
        assert!(core.contains("mange"));
    }
}
