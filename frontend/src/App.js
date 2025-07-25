import React, { useState } from 'react';
import LeftPanel from './components/LeftPanel';
import CenterPanel from './components/CenterPanel';
import RightPanel from './components/RightPanel';

function App() {
  const [selectedPersona, setSelectedPersona] = useState('financial_analyst');
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [currentQuery, setCurrentQuery] = useState('');
  const [queryResult, setQueryResult] = useState(null);
  const [processingFlow, setProcessingFlow] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [chatMetadata, setChatMetadata] = useState(null);

  const handleFileUpload = async (files) => {
    // files should be an array of File objects from the input event
    for (const file of files) {
      if (!(file instanceof File)) {
        console.log("file is not a File object");
        console.warn('Skipping non-File object:', file);
        continue;
      }
      const formData = new FormData();
      formData.append('file', file);
      console.log("form Data " ,formData);
      try {
        const response = await fetch('http://localhost:8000/upload', {
          method: 'POST',
          body: formData // Do NOT set Content-Type manually!
        });
        const result = await response.json();
        if (result.status === 'success') {
          setUploadedFiles([
            {
              name: file.name,
              type: file.type,
              size: file.size,
              id: Date.now() + Math.random(),
              file_type: result.file_type,
              backend_message: result.message
            }
          ]);
        } else {
          alert(`Failed to upload ${file.name}: ${result.message}`);
        }
      } catch (error) {
        console.error('File upload failed:', error);
        alert(`File upload failed for ${file.name}`);
      }
    }
  };

  const handleQuerySubmit = async (query) => {
    setCurrentQuery(query);
    setIsProcessing(true);
    setProcessingFlow([]);

    try {
      // Simulate processing flow
      const flowSteps = [
        { step: 'persona_selection', status: 'processing', timestamp: new Date() },
        { step: 'routing', status: 'processing', timestamp: new Date() },
        { step: 'document_processing', status: 'processing', timestamp: new Date() },
        { step: 'suggestion_generation', status: 'processing', timestamp: new Date() },
        { step: 'answer_formatting', status: 'processing', timestamp: new Date() }
      ];

      // Update flow steps progressively
      for (let i = 0; i < flowSteps.length; i++) {
        await new Promise(resolve => setTimeout(resolve, 500));
        setProcessingFlow(prev => [...prev, { ...flowSteps[i], status: 'completed' }]);
      }

      // Make API call to backend
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          persona: selectedPersona,
          context: {
            files: uploadedFiles.map(f => f.name)
          }
        })
      });

      const result = await response.json();
      setQueryResult(result);
      setChatMetadata(result.metadata || null);
    } catch (error) {
      console.error('Query failed:', error);
      setQueryResult({
        error: 'Failed to process query',
        formatted_response: {
          response: 'An error occurred while processing your query.',
          suggestions: []
        }
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handlePersonaChange = (persona) => {
    setSelectedPersona(persona);
  };

  return (
    <div className="h-screen bg-dark-primary flex">
      {/* Left Panel - KB Sources and Persona Management */}
      <div className="w-80 min-w-80 max-w-80 p-4">
        <LeftPanel
          selectedPersona={selectedPersona}
          onPersonaChange={handlePersonaChange}
          uploadedFiles={uploadedFiles}
          onFileUpload={handleFileUpload}
        />
      </div>

      {/* Center Panel - Chat + Answer + Suggested Queries */}
      <div className="flex-1 p-4">
        <CenterPanel
          currentQuery={currentQuery}
          queryResult={queryResult}
          isProcessing={isProcessing}
          onQuerySubmit={handleQuerySubmit}
          onMetadataUpdate={setChatMetadata}
          selectedPersona={selectedPersona}
        />
      </div>

      {/* Right Panel - Metadata and Source Preview */}
      <div className="w-80 min-w-80 max-w-80 p-4">
        <RightPanel
          queryResult={queryResult}
          processingFlow={processingFlow}
          uploadedFiles={uploadedFiles}
          chatMetadata={chatMetadata}
        />
      </div>
    </div>
  );
}

export default App; 