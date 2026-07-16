import type {
  TicketAnalysisRequest,
  TicketAnalysisResponse,
} from "./types";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export async function analyzeTicket(
  ticket: TicketAnalysisRequest,
): Promise<TicketAnalysisResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/v1/tickets/analyze`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(ticket),
    },
  );

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with status ${response.status}`);
  }

  return response.json() as Promise<TicketAnalysisResponse>;
}
