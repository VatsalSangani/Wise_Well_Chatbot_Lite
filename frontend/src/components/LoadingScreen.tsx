import React from 'react';

const LoadingScreen: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center h-full space-y-6">
      <div className="relative">
        <div className="w-16 h-16 border-4 border-teal-200 border-t-teal-600 rounded-full animate-spin"></div>
      </div>
      <div className="text-center space-y-2">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200">
          Initializing Wise Well
        </h2>
        <p className="text-gray-600 dark:text-gray-400">
          Please wait while we wake up the AI...
        </p>
      </div>
    </div>
  );
};

export default LoadingScreen;