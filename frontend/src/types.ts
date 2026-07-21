export type TicketAnalysisRequest = {
  subject: string;
  description: string;
};

export type TopPrediction = {
  intent: string;
  probability: number;
};

export type TicketAnalysisResponse = {
  intent: string;
  confidence: number;
  top_predictions: TopPrediction[];
  priority: "low" | "medium" | "high";
  model_version: string;
  warnings: string[];
  priority_reasons: string[];
};
