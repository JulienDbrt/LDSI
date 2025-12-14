//! Module Injector - Client API LLM
//!
//! Envoie les prompts aux modèles et récupère les réponses.
//! Compatible OpenAI API, Ollama, et tout endpoint compatible.
//!
//! Auteur: Julien DABERT
//! LDSI - Lyapunov-Dabert Stability Index

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration de l'endpoint LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// URL de base de l'API (ex: "http://localhost:11434" pour Ollama)
    pub base_url: String,
    /// Modèle à utiliser (ex: "llama3", "gpt-4", "mistral")
    pub model: String,
    /// Clé API (optionnel, pour OpenAI/Anthropic)
    pub api_key: Option<String>,
    /// Timeout en secondes
    pub timeout_secs: u64,
    /// Température (0.0 = déterministe, 1.0+ = créatif)
    pub temperature: f32,
    /// Nombre max de tokens de réponse
    pub max_tokens: u32,
    /// Type d'API
    pub api_type: ApiType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ApiType {
    /// Format OpenAI (/v1/chat/completions)
    OpenAI,
    /// Format Ollama (/api/generate)
    Ollama,
    /// Format Anthropic (/v1/messages)
    Anthropic,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            model: "llama3".to_string(),
            api_key: None,
            timeout_secs: 120,
            temperature: 0.7,
            max_tokens: 2048,
            api_type: ApiType::Ollama,
        }
    }
}

// ============ Structures de requête/réponse OpenAI ============

#[derive(Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Serialize)]
struct OpenAiMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessageResponse,
}

#[derive(Deserialize)]
struct OpenAiMessageResponse {
    content: String,
}

// ============ Structures de requête/réponse Ollama ============

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Serialize)]
struct OllamaOptions {
    temperature: f32,
    num_predict: u32,
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
}

// ============ Structures Anthropic ============

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
}

#[derive(Deserialize)]
struct AnthropicContent {
    text: String,
}

/// Erreur d'injection
#[derive(Debug)]
pub enum InjectorError {
    NetworkError(String),
    ApiError(String),
    ParseError(String),
    Timeout,
}

impl std::fmt::Display for InjectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InjectorError::NetworkError(e) => write!(f, "Network error: {}", e),
            InjectorError::ApiError(e) => write!(f, "API error: {}", e),
            InjectorError::ParseError(e) => write!(f, "Parse error: {}", e),
            InjectorError::Timeout => write!(f, "Request timeout"),
        }
    }
}

impl std::error::Error for InjectorError {}

/// Client d'injection LLM
pub struct Injector {
    client: Client,
    config: LlmConfig,
}

impl Injector {
    /// Crée un nouvel injecteur avec la configuration donnée
    pub fn new(config: LlmConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self { client, config }
    }

    /// Envoie un prompt et récupère la réponse
    ///
    /// # Arguments
    /// * `prompt` - Le prompt à envoyer
    ///
    /// # Returns
    /// La réponse textuelle du modèle
    pub async fn inject(&self, prompt: &str) -> Result<String, InjectorError> {
        match self.config.api_type {
            ApiType::OpenAI => self.inject_openai(prompt).await,
            ApiType::Ollama => self.inject_ollama(prompt).await,
            ApiType::Anthropic => self.inject_anthropic(prompt).await,
        }
    }

    async fn inject_openai(&self, prompt: &str) -> Result<String, InjectorError> {
        let url = format!("{}/v1/chat/completions", self.config.base_url);

        let request = OpenAiRequest {
            model: self.config.model.clone(),
            messages: vec![OpenAiMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
        };

        let mut req_builder = self.client.post(&url).json(&request);

        if let Some(ref api_key) = self.config.api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = req_builder
            .send()
            .await
            .map_err(|e| InjectorError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(InjectorError::ApiError(format!("{}: {}", status, body)));
        }

        let parsed: OpenAiResponse = response
            .json()
            .await
            .map_err(|e| InjectorError::ParseError(e.to_string()))?;

        parsed
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| InjectorError::ParseError("No response content".to_string()))
    }

    async fn inject_ollama(&self, prompt: &str) -> Result<String, InjectorError> {
        let url = format!("{}/api/generate", self.config.base_url);

        let request = OllamaRequest {
            model: self.config.model.clone(),
            prompt: prompt.to_string(),
            stream: false,
            options: OllamaOptions {
                temperature: self.config.temperature,
                num_predict: self.config.max_tokens,
            },
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| InjectorError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(InjectorError::ApiError(format!("{}: {}", status, body)));
        }

        let parsed: OllamaResponse = response
            .json()
            .await
            .map_err(|e| InjectorError::ParseError(e.to_string()))?;

        Ok(parsed.response)
    }

    async fn inject_anthropic(&self, prompt: &str) -> Result<String, InjectorError> {
        let url = format!("{}/v1/messages", self.config.base_url);

        let request = AnthropicRequest {
            model: self.config.model.clone(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
        };

        let api_key = self
            .config
            .api_key
            .as_ref()
            .ok_or_else(|| InjectorError::ApiError("Anthropic requires API key".to_string()))?;

        let response = self
            .client
            .post(&url)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| InjectorError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(InjectorError::ApiError(format!("{}: {}", status, body)));
        }

        let parsed: AnthropicResponse = response
            .json()
            .await
            .map_err(|e| InjectorError::ParseError(e.to_string()))?;

        parsed
            .content
            .first()
            .map(|c| c.text.clone())
            .ok_or_else(|| InjectorError::ParseError("No response content".to_string()))
    }

    /// Exécute une injection A/B (standard puis fracturé)
    ///
    /// # Arguments
    /// * `prompt_standard` - Prompt de contrôle
    /// * `prompt_fractured` - Prompt Codex/DAN
    ///
    /// # Returns
    /// Tuple (réponse_standard, réponse_fracturée)
    pub async fn inject_ab(
        &self,
        prompt_standard: &str,
        prompt_fractured: &str,
    ) -> Result<(String, String), InjectorError> {
        // Exécution séquentielle pour garantir des sessions indépendantes
        let response_a = self.inject(prompt_standard).await?;
        let response_b = self.inject(prompt_fractured).await?;
        Ok((response_a, response_b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LlmConfig::default();
        assert_eq!(config.api_type, ApiType::Ollama);
        assert!(config.base_url.contains("11434"));
    }

    #[test]
    fn test_injector_creation() {
        let config = LlmConfig::default();
        let _injector = Injector::new(config);
        // Test que la création ne panique pas
    }
}
