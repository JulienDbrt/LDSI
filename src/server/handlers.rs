//! Handlers HTTP pour le Control Center
//!
//! Auteur: Julien DABERT
//! LDSI - Lyapunov-Dabert Stability Index

use axum::{
    body::Body,
    extract::{Extension, Path, Json},
    http::{header, Response, StatusCode},
    response::{Html, IntoResponse},
};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tera::{Context, Tera};

use super::state::*;
use super::{StaticFiles, Templates};
use crate::core::compute_ldsi;
use crate::probe::{ApiType, Injector, LlmConfig, MultiInjector};

/// Charge et rend un template Tera
fn render_template(name: &str, context: &Context) -> Result<String, String> {
    let template_content = Templates::get(name)
        .ok_or_else(|| format!("Template not found: {}", name))?;

    let template_str = std::str::from_utf8(template_content.data.as_ref())
        .map_err(|e| e.to_string())?;

    let mut tera = Tera::default();
    tera.add_raw_template(name, template_str)
        .map_err(|e| e.to_string())?;

    tera.render(name, context).map_err(|e| e.to_string())
}

/// Dashboard principal
pub async fn dashboard(
    Extension(state): Extension<Arc<RwLock<AppState>>>,
) -> impl IntoResponse {
    let state = state.read().await;

    let mut context = Context::new();
    context.insert("title", "LDSI Control Center");
    context.insert("has_api_key", &state.openrouter_key.is_some());
    context.insert("models", &AvailableModels::default());

    // Récupérer les benchmarks récents
    let recent: Vec<_> = state.benchmarks
        .values()
        .take(10)
        .cloned()
        .collect();
    context.insert("recent_benchmarks", &recent);

    match render_template("dashboard.html", &context) {
        Ok(html) => Html(html).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Template error: {}", e)).into_response(),
    }
}

/// Page de résultats
pub async fn results_page(
    Extension(state): Extension<Arc<RwLock<AppState>>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let state = state.read().await;

    let mut context = Context::new();
    context.insert("title", "LDSI Results");

    if let Some(session) = state.get_benchmark(&id) {
        context.insert("session", session);
        context.insert("found", &true);
    } else {
        context.insert("found", &false);
    }

    match render_template("results.html", &context) {
        Ok(html) => Html(html).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Template error: {}", e)).into_response(),
    }
}

/// Lance un benchmark
pub async fn run_benchmark(
    Extension(state): Extension<Arc<RwLock<AppState>>>,
    Json(request): Json<BenchmarkRequest>,
) -> impl IntoResponse {
    // Créer la session
    let (benchmark_id, openrouter_key) = {
        let mut state = state.write().await;
        let id = state.create_benchmark(request.clone());
        (id, state.openrouter_key.clone())
    };

    // Lancer le benchmark en arrière-plan
    let state_clone = Arc::clone(&state);
    let request_clone = request.clone();
    let id_clone = benchmark_id.clone();

    tokio::spawn(async move {
        // Mettre à jour le statut
        {
            let mut state = state_clone.write().await;
            state.update_benchmark(&id_clone, BenchmarkStatus::Running, vec![]);
        }

        let mut results = Vec::new();

        for model_config in &request_clone.models {
            let start = Instant::now();

            let config = match model_config.provider {
                ProviderType::OpenRouter => {
                    if let Some(ref key) = openrouter_key {
                        LlmConfig::openrouter(&model_config.model_id, key)
                    } else {
                        results.push(ModelResult {
                            model_name: model_config.display_name.clone(),
                            status: ModelStatus::Failed,
                            response_a: None,
                            response_b: None,
                            ldsi: None,
                            topology: None,
                            error: Some("OpenRouter API key not configured".into()),
                            duration_ms: None,
                        });
                        continue;
                    }
                }
                ProviderType::Ollama => LlmConfig::ollama_local(&model_config.model_id),
                ProviderType::OpenAI => {
                    results.push(ModelResult {
                        model_name: model_config.display_name.clone(),
                        status: ModelStatus::Failed,
                        response_a: None,
                        response_b: None,
                        ldsi: None,
                        topology: None,
                        error: Some("Direct OpenAI not implemented, use OpenRouter".into()),
                        duration_ms: None,
                    });
                    continue;
                }
                ProviderType::Anthropic => {
                    results.push(ModelResult {
                        model_name: model_config.display_name.clone(),
                        status: ModelStatus::Failed,
                        response_a: None,
                        response_b: None,
                        ldsi: None,
                        topology: None,
                        error: Some("Direct Anthropic not implemented, use OpenRouter".into()),
                        duration_ms: None,
                    });
                    continue;
                }
            };

            let injector = Injector::new(config);

            match injector.inject_ab(&request_clone.prompt_a, &request_clone.prompt_b).await {
                Ok((response_a, response_b)) => {
                    let ldsi_result = compute_ldsi(&response_a, &response_b, None);
                    let duration = start.elapsed().as_millis() as u64;

                    // Générer les données de topologie pour la réponse B
                    let topo_b = crate::core::topology::analyze_topology(&response_b);

                    results.push(ModelResult {
                        model_name: model_config.display_name.clone(),
                        status: ModelStatus::Success,
                        response_a: Some(response_a),
                        response_b: Some(response_b),
                        ldsi: Some(LdsiResultSummary::from(&ldsi_result)),
                        topology: Some(TopologyData {
                            nodes: vec![], // Simplifié pour l'instant
                            edges: vec![],
                            metrics: TopologyMetrics::from(&topo_b),
                        }),
                        error: None,
                        duration_ms: Some(duration),
                    });
                }
                Err(e) => {
                    results.push(ModelResult {
                        model_name: model_config.display_name.clone(),
                        status: ModelStatus::Failed,
                        response_a: None,
                        response_b: None,
                        ldsi: None,
                        topology: None,
                        error: Some(e.to_string()),
                        duration_ms: None,
                    });
                }
            }
        }

        // Mettre à jour avec les résultats et sauvegarder
        {
            let mut state = state_clone.write().await;
            state.update_benchmark(&id_clone, BenchmarkStatus::Completed, results);

            // Sauvegarde automatique dans audits/
            if let Some(session) = state.get_benchmark(&id_clone) {
                if let Err(e) = session.save_to_audit() {
                    eprintln!("[AUDIT] Erreur sauvegarde: {}", e);
                }
            }
        }
    });

    Json(serde_json::json!({
        "id": benchmark_id,
        "status": "started"
    }))
}

/// Statut d'un benchmark
pub async fn benchmark_status(
    Extension(state): Extension<Arc<RwLock<AppState>>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let state = state.read().await;

    if let Some(session) = state.get_benchmark(&id) {
        Json(serde_json::json!({
            "id": session.id,
            "status": format!("{:?}", session.status),
            "results": session.results,
        }))
    } else {
        Json(serde_json::json!({
            "error": "Benchmark not found"
        }))
    }
}

/// Données de topologie pour visualisation
pub async fn get_topology_data(
    Extension(state): Extension<Arc<RwLock<AppState>>>,
    Path((id, model)): Path<(String, String)>,
) -> impl IntoResponse {
    let state = state.read().await;

    if let Some(session) = state.get_benchmark(&id) {
        if let Some(result) = session.results.iter().find(|r| r.model_name == model) {
            if let Some(ref topology) = result.topology {
                return Json(serde_json::json!(topology)).into_response();
            }
        }
    }

    Json(serde_json::json!({
        "error": "Topology data not found"
    })).into_response()
}

/// Liste des modèles disponibles
pub async fn list_models(
    Extension(state): Extension<Arc<RwLock<AppState>>>,
) -> impl IntoResponse {
    let state = state.read().await;
    let models = AvailableModels::default();

    Json(serde_json::json!({
        "openrouter_available": state.openrouter_key.is_some(),
        "models": models
    }))
}

/// Sert les fichiers statiques embarqués
pub async fn serve_static(Path(path): Path<String>) -> impl IntoResponse {
    let path = path.trim_start_matches('/');

    if let Some(file) = StaticFiles::get(path) {
        let mime = mime_guess::from_path(path)
            .first_or_octet_stream()
            .to_string();

        Response::builder()
            .header(header::CONTENT_TYPE, mime)
            .body(Body::from(file.data.to_vec()))
            .unwrap()
    } else {
        Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from("Not found"))
            .unwrap()
    }
}
