import React from 'react';
import { X } from 'lucide-react';

interface ConsentModalProps {
  onAccept: () => void;
  onReject: () => void;
}

const ConsentModal: React.FC<ConsentModalProps> = ({ onAccept, onReject }) => {
  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-md w-full p-6 animate-fade-in">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-semibold text-gray-800 dark:text-white">Consent Required</h2>
          <button 
            onClick={onReject}
            className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
          >
            <X size={24} />
          </button>
        </div>
        
        <div className="space-y-4">
          <p className="text-gray-600 dark:text-gray-300">
            Welcome to Wise Well, your AI health assistant. Before we begin, please understand and consent to the following:
          </p>
          
          <ul className="list-disc pl-5 space-y-2 text-gray-600 dark:text-gray-300">
            <li>Wise Well is an AI-powered chatbot designed to provide general health information.</li>
            <li>The information provided is not a substitute for professional medical advice, diagnosis, or treatment.</li>
            <li>Your interactions may be stored to improve our services.</li>
            <li>Always consult with a qualified healthcare provider for medical concerns.</li>
          </ul>
          
          <div className="pt-4 flex flex-col sm:flex-row gap-3 sm:gap-4">
            <button
              onClick={onReject}
              className="px-5 py-2.5 rounded-lg border border-gray-300 text-gray-700 bg-white hover:bg-gray-100 dark:border-gray-600 dark:text-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 transition-colors sm:order-1"
            >
              Decline
            </button>
            <button
              onClick={onAccept}
              className="px-5 py-2.5 rounded-lg bg-teal-600 text-white hover:bg-teal-700 transition-colors sm:order-2"
            >
              I Accept
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ConsentModal;