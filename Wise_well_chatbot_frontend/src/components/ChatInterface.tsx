import React, { useState, useRef, useEffect } from 'react';
import ChatMessage, { Message } from './ChatMessage';
import ChatInput from './ChatInput';
import Disclaimer from './Disclaimer';
import ThemeToggle from './ThemeToggle';
import { sendMessage } from '../services/api';

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: "Hello! I'm Wise Well, your health assistant. How can I help you today?",
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async (text: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      text,
      sender: 'user',
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    
    try {
      const response = await sendMessage(text);
      
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: response.answer,
        sender: 'bot',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error getting response:', error);
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: "I'm having trouble connecting right now. Please try again later.",
        sender: 'bot',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <img src="/Wise_Well_logo.png" alt="Wise Well" className="h-8 w-auto mr-2" />
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Wise Well</h1>
        </div>
        <ThemeToggle />
      </div>
      
      <div className="flex-1 overflow-y-auto mb-4 space-y-4 scrollbar-thin scrollbar-thumb-gray-300 dark:scrollbar-thumb-gray-600 pr-2">
        {messages.map(message => (
          <ChatMessage key={message.id} message={message} />
        ))}
        {isLoading && (
          <div className="flex justify-start mb-4">
            <div className="bg-gray-200 dark:bg-gray-700 px-4 py-3 rounded-2xl rounded-bl-none max-w-[80%]">
              <div className="flex space-x-2">
                <div className="w-2 h-2 rounded-full bg-gray-400 dark:bg-gray-500 animate-bounce\" style={{ animationDelay: '0ms' }}></div>
                <div className="w-2 h-2 rounded-full bg-gray-400 dark:bg-gray-500 animate-bounce\" style={{ animationDelay: '150ms' }}></div>
                <div className="w-2 h-2 rounded-full bg-gray-400 dark:bg-gray-500 animate-bounce" style={{ animationDelay: '300ms' }}></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <div className="mt-auto space-y-4">
        <ChatInput onSendMessage={handleSendMessage} disabled={isLoading} />
        <Disclaimer />
      </div>
    </div>
  );
};

export default ChatInterface;