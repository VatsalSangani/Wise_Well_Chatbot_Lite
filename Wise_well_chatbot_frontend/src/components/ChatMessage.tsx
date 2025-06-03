import React from 'react';

export interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

interface ChatMessageProps {
  message: Message;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.sender === 'user';
  const formattedTime = message.timestamp.toLocaleTimeString([], { 
    hour: '2-digit', 
    minute: '2-digit' 
  });
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div 
        className={`
          max-w-[80%] px-4 py-3 rounded-2xl 
          ${isUser 
            ? 'bg-teal-600 text-white rounded-br-none' 
            : 'bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-bl-none'
          }
        `}
      >
        <div className="text-sm sm:text-base break-words">{message.text}</div>
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
    </div>
  );
};

export default ChatMessage;