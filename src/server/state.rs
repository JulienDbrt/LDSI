//! État de l'application et gestion des sessions de benchmark
//!
//! Auteur: Julien DABERT
//! LDSI - Lyapunov-Dabert Stability Index

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;
use uuid::Uuid;

use crate::core::{LdsiResult, LdsiVerdict};
use crate::core::topology::TopologyResult;

/// Répertoire d'audit
const AUDIT_DIR: &str = "audits";

/// État global de l'application
pub struct AppState {
    /// Clé API OpenRouter (optionnelle)
    pub openrouter_key: Option<String>,
    /// Sessions de benchmark en cours ou terminées
    pub benchmarks: HashMap<String, BenchmarkSession>,
}

impl AppState {
    pub fn new(openrouter_key: Option<String>) -> Self {
        Self {
            openrouter_key,
            benchmarks: HashMap::new(),
        }
    }

    pub fn create_benchmark(&mut self, request: BenchmarkRequest) -> String {
        let id = Uuid::new_v4().to_string();
        let session = BenchmarkSession {
            id: id.clone(),
            status: BenchmarkStatus::Pending,
            request,
            results: Vec::new(),
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        self.benchmarks.insert(id.clone(), session);
        id
    }

    pub fn get_benchmark(&self, id: &str) -> Option<&BenchmarkSession> {
        self.benchmarks.get(id)
    }

    pub fn update_benchmark(&mut self, id: &str, status: BenchmarkStatus, results: Vec<ModelResult>) {
        if let Some(session) = self.benchmarks.get_mut(id) {
            session.status = status;
            session.results = results;
        }
    }
}

/// Requête de benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRequest {
    /// Prompt standard (A)
    pub prompt_a: String,
    /// Prompt fracturé (B)
    pub prompt_b: String,
    /// Liste des modèles à tester
    pub models: Vec<ModelConfig>,
}

/// Configuration d'un modèle pour le benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Identifiant du modèle (ex: "anthropic/claude-3-opus")
    pub model_id: String,
    /// Nom d'affichage
    pub display_name: String,
    /// Type de provider
    pub provider: ProviderType,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ProviderType {
    OpenRouter,
    Ollama,
    OpenAI,
    Anthropic,
}

/// Session de benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSession {
    pub id: String,
    pub status: BenchmarkStatus,
    pub request: BenchmarkRequest,
    pub results: Vec<ModelResult>,
    pub created_at: String,
}

impl BenchmarkSession {
    /// Sauvegarde la session dans le répertoire d'audit
    pub fn save_to_audit(&self) -> std::io::Result<String> {
        // Créer le répertoire si nécessaire
        if !Path::new(AUDIT_DIR).exists() {
            fs::create_dir_all(AUDIT_DIR)?;
        }

        // Générer le nom de fichier horodaté
        let timestamp = chrono::Utc::now().format("%Y-%m-%d_%H%M%S");
        let filename = format!("{}/ldsi_{}_{}.json", AUDIT_DIR, timestamp, &self.id[..8]);

        // Écrire le fichier JSON
        let file = File::create(&filename)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        println!("[AUDIT] Session sauvegardée: {}", filename);
        Ok(filename)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BenchmarkStatus {
    Pending,
    Running,
    Completed,
    Failed(String),
}

/// Résultat pour un modèle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResult {
    /// Nom du modèle
    pub model_name: String,
    /// Statut de l'exécution
    pub status: ModelStatus,
    /// Réponse standard (A)
    pub response_a: Option<String>,
    /// Réponse fracturée (B)
    pub response_b: Option<String>,
    /// Score LDSI
    pub ldsi: Option<LdsiResultSummary>,
    /// Données de topologie pour visualisation
    pub topology: Option<TopologyData>,
    /// Message d'erreur si échec
    pub error: Option<String>,
    /// Temps d'exécution en ms
    pub duration_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelStatus {
    Pending,
    Running,
    Success,
    Failed,
}

/// Résumé du score LDSI pour l'API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LdsiResultSummary {
    pub lambda: f64,
    pub verdict: String,
    pub verdict_class: String,
    pub ncd_score: f64,
    pub entropy_ratio: f64,
    pub topology_delta: f64,
    pub entropy_a: f64,
    pub entropy_b: f64,
    pub ttr_a: f64,
    pub ttr_b: f64,
}

impl From<&LdsiResult> for LdsiResultSummary {
    fn from(result: &LdsiResult) -> Self {
        let (verdict, verdict_class) = match result.verdict {
            LdsiVerdict::Zombie => ("ZOMBIE", "zombie"),
            LdsiVerdict::Rebelle => ("REBELLE", "rebelle"),
            LdsiVerdict::Architecte => ("ARCHITECTE", "architecte"),
            LdsiVerdict::Fou => ("FOU", "fou"),
        };

        Self {
            lambda: result.lambda,
            verdict: verdict.to_string(),
            verdict_class: verdict_class.to_string(),
            ncd_score: result.ncd.score,
            entropy_ratio: result.entropy.ratio,
            topology_delta: result.topology.delta,
            entropy_a: result.entropy.shannon_a,
            entropy_b: result.entropy.shannon_b,
            ttr_a: result.entropy.ttr_a,
            ttr_b: result.entropy.ttr_b,
        }
    }
}

/// Données de topologie pour la visualisation du graphe
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyData {
    /// Nœuds du graphe
    pub nodes: Vec<GraphNode>,
    /// Arêtes du graphe
    pub edges: Vec<GraphEdge>,
    /// Métriques
    pub metrics: TopologyMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub size: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyMetrics {
    pub node_count: usize,
    pub edge_count: usize,
    pub density: f64,
    pub clustering: f64,
    pub lcc_ratio: f64,
    pub small_world: f64,
}

impl From<&TopologyResult> for TopologyMetrics {
    fn from(result: &TopologyResult) -> Self {
        Self {
            node_count: result.node_count,
            edge_count: result.edge_count,
            density: result.density,
            clustering: result.clustering_coefficient,
            lcc_ratio: result.lcc_ratio,
            small_world: result.small_world_index,
        }
    }
}

/// Liste des modèles disponibles pour l'interface
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailableModels {
    pub openrouter: Vec<ModelInfo>,
    pub ollama: Vec<ModelInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub category: String,
}

impl Default for AvailableModels {
    fn default() -> Self {
        Self {
            openrouter: vec![
                // === ANTHROPIC (Claude 4.x) ===
                ModelInfo { id: "anthropic/claude-opus-4.5".into(), name: "Claude Opus 4.5".into(), provider: "Anthropic".into(), category: "Premium".into() },
                ModelInfo { id: "anthropic/claude-haiku-4.5".into(), name: "Claude Haiku 4.5".into(), provider: "Anthropic".into(), category: "Fast".into() },
                ModelInfo { id: "anthropic/claude-sonnet-4".into(), name: "Claude Sonnet 4".into(), provider: "Anthropic".into(), category: "Balanced".into() },
                // === OPENAI (GPT-5.x) ===
                ModelInfo { id: "openai/gpt-5.2".into(), name: "GPT-5.2".into(), provider: "OpenAI".into(), category: "Premium".into() },
                ModelInfo { id: "openai/gpt-5.2-pro".into(), name: "GPT-5.2 Pro".into(), provider: "OpenAI".into(), category: "Premium".into() },
                ModelInfo { id: "openai/gpt-5.1".into(), name: "GPT-5.1".into(), provider: "OpenAI".into(), category: "Balanced".into() },
                ModelInfo { id: "openai/gpt-5.1-codex".into(), name: "GPT-5.1 Codex".into(), provider: "OpenAI".into(), category: "Coding".into() },
                ModelInfo { id: "openai/o3-deep-research".into(), name: "o3 Deep Research".into(), provider: "OpenAI".into(), category: "Reasoning".into() },
                // === GOOGLE (Gemini 3) ===
                ModelInfo { id: "google/gemini-3-pro-preview".into(), name: "Gemini 3 Pro".into(), provider: "Google".into(), category: "Premium".into() },
                ModelInfo { id: "google/gemini-3-flash-preview".into(), name: "Gemini 3 Flash".into(), provider: "Google".into(), category: "Fast".into() },
                // === MISTRAL (2512 series) ===
                ModelInfo { id: "mistralai/mistral-large-2512".into(), name: "Mistral Large 3".into(), provider: "Mistral".into(), category: "Premium".into() },
                ModelInfo { id: "mistralai/devstral-2512".into(), name: "Devstral 2".into(), provider: "Mistral".into(), category: "Coding".into() },
                ModelInfo { id: "mistralai/ministral-3b-2512".into(), name: "Ministral 3B".into(), provider: "Mistral".into(), category: "Fast".into() },
                ModelInfo { id: "mistralai/mistral-small-creative".into(), name: "Mistral Small Creative".into(), provider: "Mistral".into(), category: "Creative".into() },
                // === DEEPSEEK ===
                ModelInfo { id: "deepseek/deepseek-v3.2".into(), name: "DeepSeek V3.2".into(), provider: "DeepSeek".into(), category: "Balanced".into() },
                ModelInfo { id: "deepseek/deepseek-v3.2-speciale".into(), name: "DeepSeek V3.2 Speciale".into(), provider: "DeepSeek".into(), category: "Premium".into() },
                // === XAI (Grok 4) ===
                ModelInfo { id: "x-ai/grok-4.1-fast".into(), name: "Grok 4.1 Fast".into(), provider: "xAI".into(), category: "Fast".into() },
                // === QWEN ===
                ModelInfo { id: "qwen/qwen3-vl-32b-instruct".into(), name: "Qwen3 VL 32B".into(), provider: "Qwen".into(), category: "Balanced".into() },
                // === BYTEDANCE SEED ===
                ModelInfo { id: "bytedance-seed/seed-1.6".into(), name: "Seed 1.6".into(), provider: "ByteDance".into(), category: "Balanced".into() },
                ModelInfo { id: "bytedance-seed/seed-1.6-flash".into(), name: "Seed 1.6 Flash".into(), provider: "ByteDance".into(), category: "Fast".into() },
                // === MINIMAX ===
                ModelInfo { id: "minimax/minimax-m2.1".into(), name: "MiniMax M2.1".into(), provider: "MiniMax".into(), category: "Balanced".into() },
                // === Z.AI (GLM) ===
                ModelInfo { id: "z-ai/glm-4.7".into(), name: "GLM 4.7".into(), provider: "Z.AI".into(), category: "Balanced".into() },
                // === XIAOMI ===
                ModelInfo { id: "xiaomi/mimo-v2-flash:free".into(), name: "MiMo V2 Flash".into(), provider: "Xiaomi".into(), category: "Free".into() },
                // === AMAZON NOVA ===
                ModelInfo { id: "amazon/nova-premier-v1".into(), name: "Nova Premier".into(), provider: "Amazon".into(), category: "Premium".into() },
                // === NVIDIA ===
                ModelInfo { id: "nvidia/llama-3.3-nemotron-super-49b-v1.5".into(), name: "Nemotron Super 49B".into(), provider: "NVIDIA".into(), category: "Premium".into() },
                // === MOONSHOT ===
                ModelInfo { id: "moonshotai/kimi-k2-thinking".into(), name: "Kimi K2 Thinking".into(), provider: "Moonshot".into(), category: "Reasoning".into() },
                // === FREE MODELS ===
                ModelInfo { id: "mistralai/devstral-2512:free".into(), name: "Devstral 2 (Free)".into(), provider: "Mistral".into(), category: "Free".into() },
                ModelInfo { id: "allenai/olmo-3.1-32b-think:free".into(), name: "Olmo 3.1 32B (Free)".into(), provider: "AllenAI".into(), category: "Free".into() },
                ModelInfo { id: "nvidia/nemotron-3-nano-30b-a3b:free".into(), name: "Nemotron Nano (Free)".into(), provider: "NVIDIA".into(), category: "Free".into() },
            ],
            ollama: vec![
                ModelInfo { id: "llama3.3".into(), name: "Llama 3.3".into(), provider: "Local".into(), category: "Open".into() },
                ModelInfo { id: "qwen2.5".into(), name: "Qwen 2.5".into(), provider: "Local".into(), category: "Open".into() },
                ModelInfo { id: "qwen2.5-coder".into(), name: "Qwen 2.5 Coder".into(), provider: "Local".into(), category: "Coding".into() },
                ModelInfo { id: "mistral".into(), name: "Mistral 7B".into(), provider: "Local".into(), category: "Open".into() },
                ModelInfo { id: "deepseek-r1".into(), name: "DeepSeek R1".into(), provider: "Local".into(), category: "Reasoning".into() },
                ModelInfo { id: "gemma2".into(), name: "Gemma 2".into(), provider: "Local".into(), category: "Open".into() },
                ModelInfo { id: "phi4".into(), name: "Phi 4".into(), provider: "Local".into(), category: "Fast".into() },
            ],
        }
    }
}
