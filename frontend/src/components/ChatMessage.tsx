// src/components/ChatMessage.tsx
// Enhanced with citation support for ANSWER responses

import React, { useState } from 'react';
import { EvidenceSnippet } from '../services/api';

export interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  decision?: 'ANSWER' | 'ABSTAIN' | 'REFUSE';
  citations?: EvidenceSnippet[];
}

interface ChatMessageProps {
  message: Message;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const [showCitations, setShowCitations] = useState(false);
  const isUser = message.sender === 'user';
  const formattedTime = message.timestamp.toLocaleTimeString([], { 
    hour: '2-digit', 
    minute: '2-digit' 
  });
  
  const hasCitations = message.citations && message.citations.length > 0;
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className="max-w-[85%]">
        <div 
          className={`
            px-4 py-3 rounded-2xl 
            ${isUser 
              ? 'bg-teal-600 text-white rounded-br-none' 
              : 'bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-bl-none'
            }
          `}
        >
          {/* Message text */}
          <div className="text-sm sm:text-base break-words whitespace-pre-line">
            {message.text}
          </div>
          
          {/* Decision badge (for bot messages) */}
          {!isUser && message.decision && (
            <div className="mt-2 flex items-center gap-2">
              <span 
                className={`
                  text-xs px-2 py-1 rounded-full font-medium
                  ${message.decision === 'ANSWER' 
                    ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300' 
                    : message.decision === 'REFUSE'
                    ? 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'
                    : 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300'
                  }
                `}
              >
                {message.decision}
              </span>
              
              {hasCitations && (
                <button
                  onClick={() => setShowCitations(!showCitations)}
                  className="text-xs text-teal-600 dark:text-teal-400 hover:underline font-medium"
                >
                  {showCitations ? 'Hide' : 'Show'} {message.citations!.length} source{message.citations!.length > 1 ? 's' : ''}
                </button>
              )}
            </div>
          )}
          
          {/* Timestamp */}
          <div 
            className={`
              text-xs mt-1 
              ${isUser 
                ? 'text-teal-100' 
                : 'text-gray-500 dark:text-gray-400'
              }
            `}
          >
            {formattedTime}
          </div>
        </div>
        
        {/* Citations panel */}
        {!isUser && hasCitations && showCitations && (
          <div className="mt-2 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-3 space-y-3">
            <div className="text-xs font-semibold text-gray-700 dark:text-gray-300 uppercase">
              Sources from Medical Literature
            </div>
            {message.citations!.map((citation, idx) => (
              <div 
                key={idx}
                className="pb-3 border-b border-gray-200 dark:border-gray-700 last:border-0 last:pb-0"
              >
                {/* Citation header */}
                <div className="flex items-start justify-between gap-2 mb-1">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-medium text-gray-600 dark:text-gray-400">
                      [{idx + 1}]
                    </span>
                    {citation.pmid && (
                      <a
                        href={`https://pubmed.ncbi.nlm.nih.gov/${citation.pmid}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs font-medium text-teal-600 dark:text-teal-400 hover:underline"
                      >
                        PMID: {citation.pmid}
                      </a>
                    )}
                    {citation.year && (
                      <span className="text-xs text-gray-500 dark:text-gray-500">
                        ({citation.year})
                      </span>
                    )}
                  </div>
                  <div className="flex gap-1">
                    {citation.hits?.bm25 && (
                      <span className="text-xs px-1.5 py-0.5 bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300 rounded">
                        BM25
                      </span>
                    )}
                    {citation.hits?.faiss && (
                      <span className="text-xs px-1.5 py-0.5 bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300 rounded">
                        FAISS
                      </span>
                    )}
                  </div>
                </div>
                
                {/* Title */}
                {citation.title && (
                  <div className="text-xs font-medium text-gray-800 dark:text-gray-200 mb-1">
                    {citation.title}
                  </div>
                )}
                
                {/* Journal */}
                {citation.journal && (
                  <div className="text-xs text-gray-600 dark:text-gray-400 italic mb-2">
                    {citation.journal}
                  </div>
                )}
                
                {/* Snippet text */}
                <div className="text-xs text-gray-700 dark:text-gray-300 leading-relaxed">
                  {citation.text.length > 300 
                    ? `${citation.text.substring(0, 300)}...` 
                    : citation.text
                  }
                </div>
                
                {/* Score */}
                <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                  Relevance: {(citation.score * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;
