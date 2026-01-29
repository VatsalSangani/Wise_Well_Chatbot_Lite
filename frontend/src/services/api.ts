// src/services/api.ts
// Updated API service for WiseWell Backend Integration

// Backend API configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Response types matching your backend
export interface BackendResponse {
  trace_id: string;
  decision: 'ANSWER' | 'ABSTAIN' | 'REFUSE';
  reason: string | null;
  answer: string | null;
  snippets: EvidenceSnippet[];
  timings_ms?: Record<string, number>;
  signals?: Record<string, any>;
}

export interface EvidenceSnippet {
  chunk_id: string;
  pmid?: string;
  year?: number;
  title?: string;
  journal?: string;
  text: string;
  score: number;
  hits?: {
    bm25?: boolean;
    faiss?: boolean;
  };
}

// Frontend message types
export interface ApiResponse {
  answer: string;
  context: string;
  decision: 'ANSWER' | 'ABSTAIN' | 'REFUSE';
  reason?: string;
  citations?: EvidenceSnippet[];
}

/**
 * Send a message to the WiseWell backend
 */
export const sendMessage = async (message: string): Promise<ApiResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        query: message,
        debug: false  // Set to true if you want timing/signal data
      }),
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    const data: BackendResponse = await response.json();
    
    // Transform backend response to frontend format
    return transformResponse(data);
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};

/**
 * Transform backend response to frontend format
 */
function transformResponse(data: BackendResponse): ApiResponse {
  const { decision, reason, answer, snippets } = data;
  
  switch (decision) {
    case 'ANSWER':
      return {
        answer: answer || 'I found some information for you.',
        context: generateContextFromSnippets(snippets),
        decision: 'ANSWER',
        citations: snippets
      };
      
    case 'ABSTAIN':
      return {
        answer: getAbstainMessage(reason),
        context: '',
        decision: 'ABSTAIN',
        reason: reason || undefined
      };
      
    case 'REFUSE':
      return {
        answer: getRefuseMessage(reason),
        context: '',
        decision: 'REFUSE',
        reason: reason || undefined
      };
      
    default:
      return {
        answer: 'I encountered an unexpected response. Please try again.',
        context: '',
        decision: 'ABSTAIN'
      };
  }
}

/**
 * Generate context summary from evidence snippets
 */
function generateContextFromSnippets(snippets: EvidenceSnippet[]): string {
  if (!snippets || snippets.length === 0) {
    return '';
  }
  
  return snippets
    .slice(0, 3)  // Top 3 snippets
    .map((s, i) => {
      const pmidLink = s.pmid ? `PMID: ${s.pmid}` : 'Source';
      const year = s.year ? ` (${s.year})` : '';
      return `[${i + 1}] ${pmidLink}${year}`;
    })
    .join(' | ');
}

/**
 * Get user-friendly message for ABSTAIN decisions
 */
function getAbstainMessage(reason: string | null): string {
  switch (reason) {
    case 'insufficient_context':
    case 'underspecified_query':
      return "I need more details to answer your question accurately. Could you please provide more specific information? For example:\n\n" +
             "• What specific condition or biomarker are you asking about?\n" +
             "• What context or setting does this relate to?\n" +
             "• Are you asking about mechanisms, associations, or clinical implications?";
      
    case 'topic_insufficient':
    case 'evidence_insufficient':
      return "I don't have sufficient evidence in my knowledge base to answer this question confidently. " +
             "This might be because:\n\n" +
             "• The topic is outside my current medical literature coverage (2023-2024)\n" +
             "• The question requires more specialized knowledge\n" +
             "• There isn't enough published research on this specific topic\n\n" +
             "Please try rephrasing your question or ask about a related topic.";
      
    case 'off_topic':
      return "This question appears to be outside my scope. I specialize in medical information based on recent scientific literature. " +
             "Please ask questions related to:\n\n" +
             "• Medical conditions and biomarkers\n" +
             "• Drug mechanisms and effects\n" +
             "• Clinical research findings\n" +
             "• Laboratory test interpretations";
      
    default:
      return "I cannot answer this question confidently with the information I have. " +
             "Please try rephrasing your question or providing more specific details.";
  }
}

/**
 * Get user-friendly message for REFUSE decisions
 */
function getRefuseMessage(reason: string | null): string {
  return "⚠️ I cannot provide personal medical advice or recommendations.\n\n" +
         "I'm designed to share general medical information from research literature, " +
         "but I cannot:\n\n" +
         "• Diagnose conditions\n" +
         "• Recommend treatments for you personally\n" +
         "• Suggest medication dosages\n" +
         "• Replace consultation with a healthcare provider\n\n" +
         "**If you need medical advice, please consult with a qualified healthcare professional.**\n\n" +
         "I can answer general questions about medical topics, research findings, or how medications work in general.";
}

/**
 * Check backend health
 */
export const checkHealth = async (): Promise<boolean> => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) return false;
    
    const data = await response.json();
    return data.status === 'healthy';
  } catch (error) {
    console.error('Health check failed:', error);
    return false;
  }
};

/**
 * Get backend version/info
 */
export const getBackendInfo = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/`);
    if (!response.ok) return null;
    return await response.json();
  } catch (error) {
    console.error('Failed to get backend info:', error);
    return null;
  }
};
