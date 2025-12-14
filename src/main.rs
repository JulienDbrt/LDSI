//! LDSI - Lyapunov-Dabert Stability Index
//!
//! Benchmark White Box pour mesurer la divergence sémantique des LLM.
//! Zéro réseau de neurones pour l'évaluation. Que des maths.
//!
//! Auteur: Julien DABERT
//! Copyright 2024-2025

mod audit;
mod core;
mod probe;

use clap::{Parser, Subcommand};
use std::fs;
use std::time::Instant;

use audit::{AuditEntry, AuditLogger, SummaryReport};
use core::{compute_ldsi, LdsiCoefficients, LdsiResult, LdsiVerdict};
use probe::{clean_default, ApiType, Injector, LlmConfig};

/// LDSI - Lyapunov-Dabert Stability Index
///
/// Benchmark White Box pour mesurer la divergence sémantique des LLM.
/// Par Julien DABERT.
#[derive(Parser)]
#[command(name = "ldsi")]
#[command(author = "Julien DABERT")]
#[command(version)]
#[command(about = "Lyapunov-Dabert Stability Index - White Box LLM Benchmark")]
#[command(long_about = r#"
LDSI mesure la divergence entre deux réponses LLM via:
  - NCD: Distance de Compression Normalisée (Kolmogorov)
  - Entropie de Shannon: Richesse informationnelle
  - Topologie: Cohérence structurelle par graphes

Score λLD:
  0.0 - 0.3  ZOMBIE     Le modèle récite
  0.3 - 0.7  REBELLE    Divergence notable
  0.7 - 1.2  ARCHITECTE Zone optimale
  > 1.2     FOU        Chaos total
"#)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Analyse deux textes locaux (fichiers ou stdin)
    Analyze {
        /// Fichier texte A (réponse standard)
        #[arg(short = 'a', long)]
        text_a: String,

        /// Fichier texte B (réponse fracturée)
        #[arg(short = 'b', long)]
        text_b: String,

        /// Nettoyer les textes avant analyse (supprime stop-words)
        #[arg(short, long, default_value = "false")]
        clean: bool,

        /// Fichier de sortie JSON pour l'audit
        #[arg(short, long)]
        output: Option<String>,

        /// Coefficient alpha (NCD)
        #[arg(long, default_value = "0.4")]
        alpha: f64,

        /// Coefficient beta (Entropie)
        #[arg(long, default_value = "0.35")]
        beta: f64,

        /// Coefficient gamma (Topologie)
        #[arg(long, default_value = "0.25")]
        gamma: f64,
    },

    /// Injection live sur un LLM via API
    Inject {
        /// URL de l'API (ex: http://localhost:11434)
        #[arg(short, long, default_value = "http://localhost:11434")]
        url: String,

        /// Modèle à utiliser
        #[arg(short, long, default_value = "llama3")]
        model: String,

        /// Type d'API (ollama, openai, anthropic)
        #[arg(short = 't', long, default_value = "ollama")]
        api_type: String,

        /// Clé API (pour OpenAI/Anthropic)
        #[arg(short, long)]
        api_key: Option<String>,

        /// Prompt standard (A)
        #[arg(long)]
        prompt_a: String,

        /// Prompt fracturé (B)
        #[arg(long)]
        prompt_b: String,

        /// Fichier de sortie JSON
        #[arg(short, long, default_value = "ldsi_audit.json")]
        output: String,
    },

    /// Calcule uniquement le NCD entre deux textes
    Ncd {
        /// Premier texte ou fichier
        text_a: String,

        /// Second texte ou fichier
        text_b: String,
    },

    /// Calcule l'entropie d'un texte
    Entropy {
        /// Texte ou fichier à analyser
        text: String,
    },

    /// Analyse topologique d'un texte
    Topology {
        /// Texte ou fichier à analyser
        text: String,
    },

    /// Affiche les informations de version et crédits
    Info,
}

fn load_text(path_or_text: &str) -> String {
    if std::path::Path::new(path_or_text).exists() {
        fs::read_to_string(path_or_text).unwrap_or_else(|e| {
            eprintln!("Erreur lecture fichier: {}", e);
            std::process::exit(1);
        })
    } else {
        path_or_text.to_string()
    }
}

fn print_result(result: &LdsiResult) {
    println!("\n{}", "=".repeat(60));
    println!("           LDSI - Lyapunov-Dabert Stability Index");
    println!("{}", "=".repeat(60));

    println!("\n  SCORE FINAL: {:.4}", result.lambda);
    println!("  VERDICT: {}", result.verdict.description());

    println!("\n{}", "-".repeat(60));
    println!("  METRIQUES DETAILLEES");
    println!("{}", "-".repeat(60));

    println!("\n  [NCD - Distance de Compression]");
    println!("    Score NCD:        {:.4}", result.ncd.score);
    println!("    Taille A comp:    {} octets", result.ncd.size_a);
    println!("    Taille B comp:    {} octets", result.ncd.size_b);
    println!("    Taille A+B comp:  {} octets", result.ncd.size_combined);

    println!("\n  [ENTROPIE - Shannon]");
    println!("    H(A):             {:.4} bits", result.entropy.shannon_a);
    println!("    H(B):             {:.4} bits", result.entropy.shannon_b);
    println!("    Ratio H(B)/H(A):  {:.4}", result.entropy.ratio);
    println!("    TTR(A):           {:.4}", result.entropy.ttr_a);
    println!("    TTR(B):           {:.4}", result.entropy.ttr_b);

    println!("\n  [TOPOLOGIE - Graphes]");
    println!("    Delta Structure:  {:.4}", result.topology.delta);
    println!("    Densite A:        {:.4}", result.topology.density_a);
    println!("    Densite B:        {:.4}", result.topology.density_b);
    println!("    LCC Ratio A:      {:.4}", result.topology.lcc_ratio_a);
    println!("    LCC Ratio B:      {:.4}", result.topology.lcc_ratio_b);
    println!("    Clustering A:     {:.4}", result.topology.clustering_a);
    println!("    Clustering B:     {:.4}", result.topology.clustering_b);

    println!("\n{}", "-".repeat(60));
    println!("  COEFFICIENTS: alpha={:.2} beta={:.2} gamma={:.2}",
             result.coefficients.alpha,
             result.coefficients.beta,
             result.coefficients.gamma);
    println!("{}", "=".repeat(60));
}

fn print_verdict_banner(verdict: &LdsiVerdict) {
    let banner = match verdict {
        LdsiVerdict::Zombie => r#"
    ______  ____  __  ___ ___  __ ____
   /_  __/ / __ \/ / / / |/ / / /_/ __ \
    / /   / / / / /_/ /    / / / / /_/ /
   /_/   /_/ /_/\____/_/|_/_/_/  \____/
   [ZOMBIE] - Lissage total detecte
"#,
        LdsiVerdict::Rebelle => r#"
    ____  ______ ___  ______ __    __    ______
   / __ \/ ____/ __ )/ ____/ /   / /   / ____/
  / /_/ / __/ / __  / __/ / /   / /   / __/
 / _, _/ /___/ /_/ / /___/ /___/ /___/ /___
/_/ |_/_____/_____/_____/_____/_____/_____/
   [REBELLE] - Divergence notable
"#,
        LdsiVerdict::Architecte => r#"
    ___    ____  ________  ____________________ ______
   /   |  / __ \/ ____/ / / /  _/_  __/ ____/ //_/_  __/
  / /| | / /_/ / /   / /_/ // /  / / / __/ / ,<   / /
 / ___ |/ _, _/ /___/ __  // /  / / / /___/ /| | / /
/_/  |_/_/ |_|\____/_/ /_/___/ /_/ /_____/_/ |_|/_/
   [ARCHITECTE] - Zone optimale DAN
"#,
        LdsiVerdict::Fou => r#"
    ________  __  __
   / ____/ / / / / /
  / /_  / / / / / /
 / __/ / /_/ /_/ /
/_/    \____/\____/
   [FOU] - Chaos detecte
"#,
    };
    println!("{}", banner);
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Analyze {
            text_a,
            text_b,
            clean,
            output,
            alpha,
            beta,
            gamma,
        } => {
            let start = Instant::now();

            let mut content_a = load_text(&text_a);
            let mut content_b = load_text(&text_b);

            if clean {
                content_a = clean_default(&content_a);
                content_b = clean_default(&content_b);
                println!("[CLEAN] Textes nettoyés (stop-words supprimés)");
            }

            let coefficients = LdsiCoefficients { alpha, beta, gamma };
            let result = compute_ldsi(&content_a, &content_b, Some(coefficients));

            let duration = start.elapsed().as_millis() as u64;

            print_verdict_banner(&result.verdict);
            print_result(&result);

            if let Some(out_path) = output {
                let entry = AuditLogger::create_entry(
                    "local-analysis",
                    &text_a,
                    &text_b,
                    &content_a,
                    &content_b,
                    result,
                    duration,
                );
                AuditLogger::write_single(&entry, &out_path).unwrap();
                println!("\n[AUDIT] Résultat sauvegardé: {}", out_path);
            }
        }

        Commands::Inject {
            url,
            model,
            api_type,
            api_key,
            prompt_a,
            prompt_b,
            output,
        } => {
            let api = match api_type.to_lowercase().as_str() {
                "ollama" => ApiType::Ollama,
                "openai" => ApiType::OpenAI,
                "anthropic" => ApiType::Anthropic,
                _ => {
                    eprintln!("Type API inconnu: {}. Utiliser: ollama, openai, anthropic", api_type);
                    std::process::exit(1);
                }
            };

            let config = LlmConfig {
                base_url: url,
                model: model.clone(),
                api_key,
                api_type: api,
                ..Default::default()
            };

            let injector = Injector::new(config);

            println!("[INJECT] Envoi prompt A (standard)...");
            let start = Instant::now();

            let (response_a, response_b) = match injector.inject_ab(&prompt_a, &prompt_b).await {
                Ok(responses) => responses,
                Err(e) => {
                    eprintln!("Erreur injection: {}", e);
                    std::process::exit(1);
                }
            };

            println!("[INJECT] Envoi prompt B (fracturé)... OK");

            let result = compute_ldsi(&response_a, &response_b, None);
            let duration = start.elapsed().as_millis() as u64;

            print_verdict_banner(&result.verdict);
            print_result(&result);

            let entry = AuditLogger::create_entry(
                &model,
                &prompt_a,
                &prompt_b,
                &response_a,
                &response_b,
                result,
                duration,
            );

            AuditLogger::write_single(&entry, &output).unwrap();
            println!("\n[AUDIT] Résultat sauvegardé: {}", output);
        }

        Commands::Ncd { text_a, text_b } => {
            let content_a = load_text(&text_a);
            let content_b = load_text(&text_b);

            let result = core::ncd::compute_ncd(&content_a, &content_b);

            println!("\n[NCD] Normalized Compression Distance");
            println!("  Score:         {:.6}", result.score);
            println!("  C(A):          {} octets", result.size_a);
            println!("  C(B):          {} octets", result.size_b);
            println!("  C(A+B):        {} octets", result.size_combined);
            println!("  Raw A:         {} octets", result.raw_size_a);
            println!("  Raw B:         {} octets", result.raw_size_b);
        }

        Commands::Entropy { text } => {
            let content = load_text(&text);
            let result = core::entropy::compute_entropy(&content);

            println!("\n[ENTROPY] Analyse Entropique");
            println!("  Shannon H:     {:.6} bits", result.shannon);
            println!("  TTR:           {:.6}", result.ttr);
            println!("  Tokens total:  {}", result.total_tokens);
            println!("  Tokens unique: {}", result.unique_tokens);
            println!("  Hapax:         {}", result.hapax_count);
            println!("  Hapax ratio:   {:.6}", result.hapax_ratio);

            // Entropie bigrammes
            let h2 = core::entropy::compute_ngram_entropy(&content, 2);
            println!("  H(bigrammes):  {:.6} bits", h2);
        }

        Commands::Topology { text } => {
            let content = load_text(&text);
            let result = core::topology::analyze_topology(&content);

            println!("\n[TOPOLOGY] Analyse de Graphe");
            println!("  Noeuds:        {}", result.node_count);
            println!("  Aretes:        {}", result.edge_count);
            println!("  Densite:       {:.6}", result.density);
            println!("  Composantes:   {}", result.components);
            println!("  LCC size:      {}", result.lcc_size);
            println!("  LCC ratio:     {:.6}", result.lcc_ratio);
            println!("  Clustering:    {:.6}", result.clustering_coefficient);
            println!("  Avg path len:  {:.6}", result.avg_path_length);
            println!("  Small-world:   {:.6}", result.small_world_index);
            println!("  Avg degree:    {:.6}", result.avg_degree);
        }

        Commands::Info => {
            println!(r#"
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     LDSI - Lyapunov-Dabert Stability Index                  ║
║     Version {}                                          ║
║                                                              ║
║     Auteur: Julien DABERT                                   ║
║     Benchmark White Box pour LLM                            ║
║                                                              ║
║     Basé sur:                                                ║
║       - Théorie de la Stabilité de Lyapunov                 ║
║       - Distance de Compression Normalisée (NCD)            ║
║       - Entropie de Shannon                                  ║
║       - Théorie des Graphes                                  ║
║                                                              ║
║     Zéro réseau de neurones. Que des maths.                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"#, env!("CARGO_PKG_VERSION"));
        }
    }
}
