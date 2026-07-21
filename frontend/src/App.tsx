import { FormEvent, useState } from "react";

import { analyzeTicket } from "./api";
import type { TicketAnalysisResponse } from "./types";

const initialSubject = "Cannot log in";
const initialDescription =
  "The password reset link is not working for my account.";

function formatLabel(value: string): string {
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export default function App() {
  const [subject, setSubject] = useState(initialSubject);
  const [description, setDescription] = useState(initialDescription);
  const [result, setResult] =
    useState<TicketAnalysisResponse | null>(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setResult(null);
    setIsLoading(true);

    try {
      setResult(await analyzeTicket({ subject, description }));
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "An unexpected error occurred.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="page-shell">
      <header className="hero">
        <p className="eyebrow">
          Portfolio project · ML integration v0.2
        </p>
        <h1>SupportLens</h1>
        <p className="hero-copy">
          Classify customer-support requests with a reproducible
          TF-IDF and logistic-regression model. Low-confidence
          predictions are flagged for human review.
        </p>
      </header>

      <section className="workspace" aria-label="Ticket analysis workspace">
        <form className="panel ticket-form" onSubmit={handleSubmit}>
          <div>
            <p className="section-kicker">New ticket</p>
            <h2>Analyse a support request</h2>
          </div>

          <label>
            Subject
            <input
              value={subject}
              onChange={(event) => setSubject(event.target.value)}
              minLength={3}
              maxLength={200}
              required
            />
          </label>

          <label>
            Description
            <textarea
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              minLength={10}
              maxLength={5000}
              rows={8}
              required
            />
          </label>

          <button type="submit" disabled={isLoading}>
            {isLoading ? "Analysing…" : "Analyse ticket"}
          </button>
        </form>

        <section className="panel result-panel" aria-live="polite">
          <div>
            <p className="section-kicker">Triage result</p>
            <h2>Prediction details</h2>
          </div>

          {!result && !error && !isLoading && (
            <div className="empty-state">
              Submit a ticket to view the predicted intent,
              confidence, alternatives, and priority.
            </div>
          )}

          {isLoading && (
            <div className="empty-state">Running the model…</div>
          )}

          {error && <div className="error-state">{error}</div>}

          {result && (
            <div className="result-content">
              <div className="metric-grid">
                <article>
                  <span>Intent</span>
                  <strong>{formatLabel(result.intent)}</strong>
                </article>
                <article>
                  <span>Priority</span>
                  <strong>{formatLabel(result.priority)}</strong>
                </article>
                <article>
                  <span>Confidence</span>
                  <strong>
                    {Math.round(result.confidence * 100)}%
                  </strong>
                </article>
              </div>

              {result.warnings.map((warning) => (
                <div className="warning-state" key={warning}>
                  {warning}
                </div>
              ))}

              <div className="prediction-list">
                <h3>Top predictions</h3>
                {result.top_predictions.map((prediction) => (
                  <div
                    className="prediction-row"
                    key={prediction.intent}
                  >
                    <span>{formatLabel(prediction.intent)}</span>
                    <strong>
                      {Math.round(prediction.probability * 100)}%
                    </strong>
                  </div>
                ))}
              </div>

              <div className="explanation">
                <h3>Priority explanation</h3>
                <ul>
                  {result.priority_reasons.map((reason) => (
                    <li key={reason}>{reason}</li>
                  ))}
                </ul>
              </div>

              <p className="model-label">
                Model: <code>{result.model_version}</code>
              </p>
            </div>
          )}
        </section>
      </section>
    </main>
  );
}
