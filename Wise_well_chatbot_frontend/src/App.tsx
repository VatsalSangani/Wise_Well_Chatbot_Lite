import React, { useState, useEffect } from 'react';
import ConsentModal from './components/ConsentModal';
import ChatInterface from './components/ChatInterface';
import LoadingScreen from './components/LoadingScreen';
import { ThemeProvider } from './contexts/ThemeContext';

function App() {
  const [hasConsented, setHasConsented] = useState<boolean>(() => {
    const storedConsent = sessionStorage.getItem('wiseWellConsent');
    return storedConsent === 'true';
  });

  const [isInitializing, setIsInitializing] = useState(false);

  const handleAcceptConsent = () => {
    sessionStorage.setItem('wiseWellConsent', 'true');
    setHasConsented(true);
  };

  const handleRejectConsent = () => {
    window.close();
    alert('Consent is required to use Wise Well. Please close this tab.');
  };

  useEffect(() => {
    if (hasConsented) {
      setIsInitializing(true);
      fetch('https://brendvat-wise-well-chatbot-lite.hf.space/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: 'test' }),
      })
        .then(() => {
          setIsInitializing(false);
        })
        .catch((error) => {
          console.error('Backend initialization error:', error);
          setIsInitializing(false);
        });
    }
  }, [hasConsented]);

  return (
    <ThemeProvider>
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 text-gray-900 dark:text-gray-100 transition-colors duration-300">
        <div className="h-screen flex flex-col overflow-hidden">
          {!hasConsented && (
            <ConsentModal 
              onAccept={handleAcceptConsent} 
              onReject={handleRejectConsent} 
            />
          )}
          {hasConsented && isInitializing && <LoadingScreen />}
          {hasConsented && !isInitializing && <ChatInterface />}
        </div>
      </div>
    </ThemeProvider>
  );
}

export default App;