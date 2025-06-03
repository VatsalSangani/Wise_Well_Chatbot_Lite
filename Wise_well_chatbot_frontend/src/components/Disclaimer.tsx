import React from 'react';
import { AlertTriangle } from 'lucide-react';

const Disclaimer: React.FC = () => {
  return (
    <div className="bg-amber-50 dark:bg-amber-900/20 border-l-4 border-amber-500 p-4 rounded-r-md mb-4">
      <div className="flex">
        <div className="flex-shrink-0">
          <AlertTriangle className="h-5 w-5 text-amber-500" aria-hidden="true" />
        </div>
        <div className="ml-3">
          <p className="text-sm text-amber-800 dark:text-amber-200">
            <strong>Medical Disclaimer:</strong> Wise Well runs on AI technology and is intended for informational purposes only. 
            Always consult with a qualified healthcare professional for medical advice, diagnosis, or treatment.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Disclaimer;