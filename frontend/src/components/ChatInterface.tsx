// src/components/ChatInterface.tsx
// Enhanced with backend integration and decision handling

import React, { useState, useRef, useEffect } from 'react';
import ChatMessage, { Message } from './ChatMessage';
import ChatInput from './ChatInput';
import Disclaimer from './Disclaimer';
import ThemeToggle from './ThemeToggle';
import { sendMessage, checkHealth } from '../services/api';

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: "Hello! I'm Wise Well, your medical information assistant. I provide evidence-based information from recent medical literature (2023-2024).\n\nHow can I help you today?",
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  
  const [isLoading, setIsLoading] = useState(false);
  const [backendHealthy, setBackendHealthy] = useState<boolean | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Check backend health on mount
  useEffect(() => {
    const checkBackendHealth = async () => {
      const healthy = await checkHealth();
      setBackendHealthy(healthy);
      
      if (!healthy) {
        const errorMessage: Message = {
          id: 'health-error',
          text: "⚠️ Unable to connect to the medical information service. Please check that the backend is running at https://portfolio.vatsalsangani.in/wiswell-ui/api",
          sender: 'bot',
          timestamp: new Date(),
          decision: 'ABSTAIN'
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    };
    
    checkBackendHealth();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async (text: string) => {
    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      text,
      sender: 'user',
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    
    try {
      // Call backend API
      const response = await sendMessage(text);
      
      // Create bot message with decision and citations
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: response.answer,
        sender: 'bot',
        timestamp: new Date(),
        decision: response.decision,
        citations: response.citations
      };
      
      setMessages(prev => [...prev, botMessage]);
      
      // Update backend health status
      if (backendHealthy === false) {
        setBackendHealthy(true);
      }
    } catch (error) {
      console.error('Error getting response:', error);
      
      // Determine error message based on error type
      let errorText = "I'm having trouble connecting to the medical information service right now. Please try again later.";
      
      if (error instanceof Error) {
        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
          errorText = "⚠️ Cannot connect to the backend service. Please ensure:\n\n" +
                     "• The backend is running (uvicorn backend.main:app --reload)\n" +
                     "• Backend is accessible at https://portfolio.vatsalsangani.in/wiswell-ui/api\n" +
                     "• No firewall is blocking the connection";
        } else if (error.message.includes('400')) {
          errorText = "Your question couldn't be processed. Please try rephrasing it.";
        } else if (error.message.includes('500')) {
          errorText = "The service encountered an error. Please try again or rephrase your question.";
        }
      }
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: errorText,
        sender: 'bot',
        timestamp: new Date(),
        decision: 'ABSTAIN'
      };
      
      setMessages(prev => [...prev, errorMessage]);
      setBackendHealthy(false);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <img src="/Wise_Well_logo.png" alt="Wise Well" className="h-8 w-auto mr-2" />
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Wise Well</h1>
            {backendHealthy !== null && (
              <div className="flex items-center gap-1 text-xs mt-0.5">
                <div 
                  className={`w-2 h-2 rounded-full ${
                    backendHealthy 
                      ? 'bg-green-500 animate-pulse' 
                      : 'bg-red-500'
                  }`}
                />
                <span className="text-gray-600 dark:text-gray-400">
                  {backendHealthy ? 'Connected' : 'Disconnected'}
                </span>
              </div>
            )}
          </div>
        </div>
        <ThemeToggle />
      </div>
      
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto mb-4 space-y-4 scrollbar-thin scrollbar-thumb-gray-300 dark:scrollbar-thumb-gray-600 pr-2">
        {messages.map(message => (
          <ChatMessage key={message.id} message={message} />
        ))}
        
        {/* Loading indicator */}
        {isLoading && (
          <div className="flex justify-start mb-4">
            <div className="bg-gray-200 dark:bg-gray-700 px-4 py-3 rounded-2xl rounded-bl-none max-w-[80%]">
              <div className="flex space-x-2">
                <div className="w-2 h-2 rounded-full bg-gray-400 dark:bg-gray-500 animate-bounce" style={{ animationDelay: '0ms' }}></div>
                <div className="w-2 h-2 rounded-full bg-gray-400 dark:bg-gray-500 animate-bounce" style={{ animationDelay: '150ms' }}></div>
                <div className="w-2 h-2 rounded-full bg-gray-400 dark:bg-gray-500 animate-bounce" style={{ animationDelay: '300ms' }}></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      {/* Input area */}
      <div className="mt-auto space-y-4">
        <ChatInput onSendMessage={handleSendMessage} disabled={isLoading || backendHealthy === false} />
        <Disclaimer />
      </div>
    </div>
  );
};

export default ChatInterface;
