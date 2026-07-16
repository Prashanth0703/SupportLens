import { FormEvent, useState } from "react";

import { analyzeTicket } from "./api";
import type { TicketAnalysisResponse } from "./types";

const initialSubject = "Urgent login issue";
const initialDescription =
  "I cannot access my account after changing my password.";

function formatLabel(value: string): string {
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export default function App() {
  const [subject, setSubject] = useState(initialSubject);
  const [description, setDescription] = useState(initialDescription);
  const [result, setResult] = useState<TicketAnalysisResponse | null>(null);
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
        <p className="eyebrow">Portfolio project · Vertical Slice v0.1</p>
        <h1>SupportLens</h1>
        <p className="hero-copy">
          Classify customer-support tickets and explain every prediction.
          The current classifier is a transparent keyword baseline that
          will later be replaced by a reproducible ML pipeline.
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
              Submit the ticket to view its suggested category,
              priority, confidence, and explanation.
            </div>
          )}

          {isLoading && <div className="empty-state">Running the baseline…</div>}
          {error && <div className="error-state">{error}</div>}

          {result && (
            <div className="result-content">
              <div className="metric-grid">
                <article>
                  <span>Category</span>
                  <strong>{formatLabel(result.category)}</strong>
                </article>
                <article>
                  <span>Priority</span>
                  <strong>{formatLabel(result.priority)}</strong>
                </article>
                <article>
                  <span>Confidence</span>
                  <strong>{Math.round(result.confidence * 100)}%</strong>
                </article>
              </div>

              <div className="explanation">
                <h3>Why this result?</h3>
                <ul>
                  {result.reasons.map((reason) => (
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
