import React, { useState } from 'react';
import { Send } from 'lucide-react';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
}

const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage, disabled = false }) => {
  const [message, setMessage] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !disabled) {
      onSendMessage(message);
      setMessage('');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex items-center gap-2">
      <div className="relative flex-grow">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Type your message here..."
          disabled={disabled}
          className="w-full py-3 px-4 pr-12 rounded-full border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-teal-500 disabled:opacity-60 disabled:cursor-not-allowed"
        />
      </div>
      <button
        type="submit"
        disabled={!message.trim() || disabled}
        className="bg-teal-600 hover:bg-teal-700 text-white p-3 rounded-full transition-colors disabled:opacity-60 disabled:cursor-not-allowed disabled:hover:bg-teal-600"
        aria-label="Send message"
      >
        <Send size={20} />
      </button>
    </form>
  );
};

export default ChatInput;