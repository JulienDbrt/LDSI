//! Probe LDSI - Système d'Injection et Nettoyage
//!
//! Gère l'interaction avec les LLM et la préparation des données.
//!
//! Auteur: Julien DABERT
//! LDSI - Lyapunov-Dabert Stability Index

pub mod cleaner;
pub mod injector;

pub use cleaner::{clean_default, clean_text, extract_semantic_core, CleanerConfig, Language};
pub use injector::{ApiType, Injector, InjectorError, LlmConfig};
