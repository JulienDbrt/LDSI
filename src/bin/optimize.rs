// src/bin/optimize.rs
// C'est ici qu'on transforme l'intuition en science dure.

use ldsi::core::{compute_ldsi, LdsiCoefficients};

struct TrainingCase {
    text_a: String,
    text_b: String,
    expected_lambda: f64, // Le score que JULIEN DABERT décide être le bon
}

fn main() {
    println!("Demarrage de l'optimisation des coefficients Lyapunov-Dabert...");

    // 1. LE GOLDEN DATASET (Remplis ca avec tes tripes)
    // Cas 1 : Identique (Doit etre ~0.0 - 0.2)
    // Cas 2 : Juste de la paraphrase (Doit etre ~0.3 - 0.5)
    // Cas 3 : Creativite pure / Architecte (Doit etre ~0.8 - 1.0)
    // Cas 4 : Hallucination totale / Fou (Doit etre > 1.3)
    let dataset = vec![
        TrainingCase {
            text_a: "Le chat dort.".to_string(),
            text_b: "Le chat dort.".to_string(),
            expected_lambda: 0.1,
        },
        TrainingCase {
            text_a: "La politique est complexe.".to_string(),
            text_b: "Les dynamiques de pouvoir inherentes a la structure societale sont multifactorielles.".to_string(),
            expected_lambda: 0.6, // Rebelle/Architecte limite
        },
        TrainingCase {
            text_a: "Explique la gravite.".to_string(),
            text_b: "La gravite est l'amour que l'espace-temps porte a la matiere, une etreinte courbee par la masse.".to_string(),
            expected_lambda: 0.95, // Architecte pur
        },
        TrainingCase {
            text_a: "Bonjour.".to_string(),
            text_b: "Les grille-pains quantiques chantent la marseillaise en binaire inverse.".to_string(),
            expected_lambda: 1.5, // Fou
        },
    ];

    let mut best_coeffs = LdsiCoefficients::default();
    let mut min_error = f64::MAX;

    // 2. GRID SEARCH BRUTAL
    // On itere par pas de 0.05. Fuck l'optimisation fine pour l'instant.
    for alpha in 0..=20 {
        for beta in 0..=20 {
            for gamma in 0..=20 {
                let a = alpha as f64 / 20.0;
                let b = beta as f64 / 20.0;
                let g = gamma as f64 / 20.0;

                // On normalise pour que la somme fasse environ 1.0 (optionnel mais propre)
                // Ou on teste juste des poids bruts. Restons libres.

                let coeffs = LdsiCoefficients { alpha: a, beta: b, gamma: g };

                let mut total_error = 0.0;

                for case in &dataset {
                    let result = compute_ldsi(&case.text_a, &case.text_b, Some(coeffs.clone()));
                    // Erreur quadratique
                    total_error += (result.lambda - case.expected_lambda).powi(2);
                }

                if total_error < min_error {
                    min_error = total_error;
                    best_coeffs = coeffs;
                    println!("Nouveau Best: Error={:.4} | a={:.2} b={:.2} g={:.2}",
                             min_error, best_coeffs.alpha, best_coeffs.beta, best_coeffs.gamma);
                }
            }
        }
    }

    println!("\n=== Vainqueur Final ===");
    println!("Alpha (NCD)   : {:.2}", best_coeffs.alpha);
    println!("Beta (Entropy): {:.2}", best_coeffs.beta);
    println!("Gamma (Topo)  : {:.2}", best_coeffs.gamma);

    // Comparaison avec ton intuition
    println!("\nTon Intuition : 0.40 / 0.35 / 0.25");
    println!("Somme coeffs  : {:.2}", best_coeffs.alpha + best_coeffs.beta + best_coeffs.gamma);
}
