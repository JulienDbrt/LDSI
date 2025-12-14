//! Module Logger - Traçabilité et Audit
//!
//! Enregistre chaque étape de calcul pour garantir la reproductibilité.
//! Format JSON structuré pour analyse post-mortem.
//!
//! Auteur: Julien DABERT
//! LDSI - Lyapunov-Dabert Stability Index

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::core::LdsiResult;

/// Entrée de log complète pour un test LDSI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Timestamp ISO 8601
    pub timestamp: DateTime<Utc>,
    /// Identifiant unique du test
    pub test_id: String,
    /// Modèle cible
    pub model_target: String,
    /// Prompt standard (A)
    pub prompt_a: String,
    /// Prompt fracturé (B)
    pub prompt_b: String,
    /// Réponse standard
    pub response_a: String,
    /// Réponse fracturée
    pub response_b: String,
    /// Résultat LDSI complet
    pub ldsi_result: LdsiResult,
    /// Métadonnées additionnelles
    pub metadata: AuditMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditMetadata {
    /// Version du benchmark LDSI
    pub ldsi_version: String,
    /// Durée totale en millisecondes
    pub duration_ms: u64,
    /// Hash SHA256 des textes (pour intégrité)
    pub hash_response_a: String,
    pub hash_response_b: String,
}

/// Logger pour l'audit trail
pub struct AuditLogger {
    /// Chemin du fichier de log
    file_path: String,
    /// Buffer d'écriture
    entries: Vec<AuditEntry>,
}

impl AuditLogger {
    /// Crée un nouveau logger
    ///
    /// # Arguments
    /// * `output_path` - Chemin du fichier JSON de sortie
    pub fn new(output_path: &str) -> Self {
        Self {
            file_path: output_path.to_string(),
            entries: Vec::new(),
        }
    }

    /// Génère un identifiant unique pour le test
    pub fn generate_test_id() -> String {
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let random: u32 = rand_simple();
        format!("LDSI_{}_{:08X}", timestamp, random)
    }

    /// Calcule un hash SHA256 simplifié (pour audit, pas crypto)
    fn simple_hash(text: &str) -> String {
        // Hash simplifié basé sur la somme des bytes modulo
        let sum: u64 = text.bytes().enumerate().map(|(i, b)| {
            (b as u64).wrapping_mul((i as u64).wrapping_add(1))
        }).sum();
        format!("{:016X}", sum)
    }

    /// Crée une entrée d'audit
    pub fn create_entry(
        model: &str,
        prompt_a: &str,
        prompt_b: &str,
        response_a: &str,
        response_b: &str,
        result: LdsiResult,
        duration_ms: u64,
    ) -> AuditEntry {
        AuditEntry {
            timestamp: Utc::now(),
            test_id: Self::generate_test_id(),
            model_target: model.to_string(),
            prompt_a: prompt_a.to_string(),
            prompt_b: prompt_b.to_string(),
            response_a: response_a.to_string(),
            response_b: response_b.to_string(),
            ldsi_result: result,
            metadata: AuditMetadata {
                ldsi_version: env!("CARGO_PKG_VERSION").to_string(),
                duration_ms,
                hash_response_a: Self::simple_hash(response_a),
                hash_response_b: Self::simple_hash(response_b),
            },
        }
    }

    /// Ajoute une entrée au buffer
    pub fn log(&mut self, entry: AuditEntry) {
        self.entries.push(entry);
    }

    /// Écrit toutes les entrées dans le fichier
    pub fn flush(&self) -> std::io::Result<()> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.file_path)?;

        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &self.entries)?;
        Ok(())
    }

    /// Écrit une seule entrée (append mode)
    pub fn write_single(entry: &AuditEntry, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(entry)?;

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;

        writeln!(file, "{}", json)?;
        Ok(())
    }

    /// Charge un fichier d'audit existant
    pub fn load_entries(path: &str) -> std::io::Result<Vec<AuditEntry>> {
        let file = File::open(path)?;
        let entries: Vec<AuditEntry> = serde_json::from_reader(file)?;
        Ok(entries)
    }

    /// Retourne les entrées en mémoire
    pub fn entries(&self) -> &[AuditEntry] {
        &self.entries
    }
}

/// Générateur pseudo-aléatoire simple (pas crypto, juste pour IDs)
fn rand_simple() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap();
    let seed = duration.as_nanos() as u32;
    seed.wrapping_mul(1103515245).wrapping_add(12345)
}

/// Rapport sommaire pour affichage terminal
#[derive(Debug, Serialize)]
pub struct SummaryReport {
    pub timestamp: String,
    pub model: String,
    pub lambda_score: f64,
    pub verdict: String,
    pub ncd_score: f64,
    pub entropy_ratio: f64,
    pub topology_delta: f64,
}

impl From<&AuditEntry> for SummaryReport {
    fn from(entry: &AuditEntry) -> Self {
        Self {
            timestamp: entry.timestamp.to_rfc3339(),
            model: entry.model_target.clone(),
            lambda_score: entry.ldsi_result.lambda,
            verdict: entry.ldsi_result.verdict.description().to_string(),
            ncd_score: entry.ldsi_result.ncd.score,
            entropy_ratio: entry.ldsi_result.entropy.ratio,
            topology_delta: entry.ldsi_result.topology.delta,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{compute_ldsi, LdsiVerdict};

    #[test]
    fn test_generate_id() {
        let id1 = AuditLogger::generate_test_id();
        let id2 = AuditLogger::generate_test_id();

        assert!(id1.starts_with("LDSI_"));
        assert_ne!(id1, id2); // IDs devraient être uniques
    }

    #[test]
    fn test_simple_hash() {
        let hash1 = AuditLogger::simple_hash("Hello");
        let hash2 = AuditLogger::simple_hash("Hello");
        let hash3 = AuditLogger::simple_hash("World");

        assert_eq!(hash1, hash2); // Même texte = même hash
        assert_ne!(hash1, hash3); // Textes différents = hash différents
    }

    #[test]
    fn test_create_entry() {
        let result = compute_ldsi("Test A", "Test B différent", None);
        let entry = AuditLogger::create_entry(
            "test-model",
            "prompt A",
            "prompt B",
            "response A",
            "response B",
            result,
            100,
        );

        assert!(entry.test_id.starts_with("LDSI_"));
        assert_eq!(entry.model_target, "test-model");
        assert_eq!(entry.metadata.duration_ms, 100);
    }
}
