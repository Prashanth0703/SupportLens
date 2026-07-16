export type TicketAnalysisRequest = {
  subject: string;
  description: string;
};

export type TicketAnalysisResponse = {
  category:
    | "billing"
    | "account_access"
    | "technical_issue"
    | "feature_request"
    | "general";
  priority: "low" | "medium" | "high";
  confidence: number;
  model_version: string;
  reasons: string[];
};
