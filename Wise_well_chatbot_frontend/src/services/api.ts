import { createClient } from '@supabase/supabase-js';

interface ApiResponse {
  answer: string;
  context: string;
}

export const sendMessage = async (message: string): Promise<ApiResponse> => {
  try {
    const response = await fetch('https://brendvat-wise-well-chatbot-lite.hf.space/ask', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question: message }),
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};