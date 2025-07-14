import React, { useState } from 'react';
import { 
  UserOutlined, 
  DollarOutlined, 
  RobotOutlined,
  UploadOutlined,
  FileTextOutlined,
  DatabaseOutlined,
  SettingOutlined,
  PlusOutlined,
  DeleteOutlined,
  ApiOutlined
} from '@ant-design/icons';

const LeftPanel = ({ selectedPersona, onPersonaChange, uploadedFiles, onFileUpload }) => {
  const [activeTab, setActiveTab] = useState('personas');
  const [showLLMConfig, setShowLLMConfig] = useState(false);
  const [llmProviders, setLLMProviders] = useState([
    { name: 'OpenAI', status: 'connected', model: 'gpt-4' },
    { name: 'Anthropic', status: 'disconnected', model: 'claude-3' },
    { name: 'Cohere', status: 'disconnected', model: 'command' }
  ]);

  const personas = [
    { id: 'financial_analyst', name: 'Financial Analyst', icon: DollarOutlined, color: 'text-green-400' },
    { id: 'legal_advisor', name: 'Legal Advisor', icon: UserOutlined, color: 'text-blue-400' },
    { id: 'general', name: 'General', icon: RobotOutlined, color: 'text-yellow-400' }
  ];

  const handleFileUpload = (event) => {
    console.log("handleFileUpload");
    const files = Array.from(event.target.files); // This is an array of File objects
    try
    {
     onFileUpload(files); // Pass the real File objects directly
      console.log("File uploaded successfully");
    }
    catch(error)
    {
      console.error("Error in handleFileUpload", error);
    }
};

  const handleRemoveFile = (fileId) => {
    // Implementation would remove file from parent state
    console.log('Remove file:', fileId);
  };

  return (
    <div className="h-full flex flex-col space-y-4">
      {/* Header */}
      <div className="neo-panel p-4">
        <h2 className="text-xl font-bold text-gradient mb-4">Control Panel</h2>
        
        {/* Tab Navigation */}
        <div className="flex neo-panel-inset rounded-lg p-1">
          <button
            onClick={() => setActiveTab('personas')}
            className={`flex-1 py-2 px-3 rounded-md transition-all ${
              activeTab === 'personas' 
                ? 'neo-button-small text-dark-info' 
                : 'text-dark-textSecondary hover:text-dark-text'
            }`}
          >
            <UserOutlined className="mr-2" />
            Personas
          </button>
          <button
            onClick={() => setActiveTab('sources')}
            className={`flex-1 py-2 px-3 rounded-md transition-all ${
              activeTab === 'sources' 
                ? 'neo-button-small text-dark-info' 
                : 'text-dark-textSecondary hover:text-dark-text'
            }`}
          >
            <DatabaseOutlined className="mr-2" />
            Sources
          </button>

        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-hidden">
        {activeTab === 'personas' && (
          <div className="neo-panel p-4 h-full flex flex-col">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">AI Personas</h3>
              <button
                onClick={() => setShowLLMConfig(!showLLMConfig)}
                className="neo-button-small p-2"
              >
                <SettingOutlined className="text-dark-textSecondary" />
              </button>
            </div>

            {/* LLM Configuration */}
            {showLLMConfig && (
              <div className="neo-panel-inset p-3 mb-4 animate-slide-down">
                <h4 className="text-sm font-semibold mb-2 text-dark-textSecondary">LLM Providers</h4>
                {llmProviders.map(provider => (
                  <div key={provider.name} className="flex items-center justify-between py-2">
                    <div className="flex items-center">
                      <div className={`w-2 h-2 rounded-full mr-2 ${
                        provider.status === 'connected' ? 'bg-dark-success' : 'bg-dark-error'
                      }`} />
                      <span className="text-sm">{provider.name}</span>
                    </div>
                    <span className="text-xs text-dark-textSecondary">{provider.model}</span>
                  </div>
                ))}
                <button className="neo-button-small p-2 w-full mt-2 text-sm">
                  <PlusOutlined className="mr-2" />
                  Add Provider
                </button>
              </div>
            )}

            {/* Persona Selection */}
            <div className="space-y-3 flex-1">
              {personas.map(persona => {
                const IconComponent = persona.icon;
                return (
                  <div
                    key={persona.id}
                    onClick={() => onPersonaChange(persona.id)}
                    className={`neo-card cursor-pointer transition-all hover:shadow-neo-hover ${
                      selectedPersona === persona.id 
                        ? 'ring-2 ring-dark-info ring-opacity-50' 
                        : ''
                    }`}
                  >
                    <div className="flex items-center">
                      <IconComponent className={`text-xl mr-3 ${persona.color}`} />
                      <div>
                        <h4 className="font-semibold text-sm">{persona.name}</h4>
                        <p className="text-xs text-dark-textSecondary">
                          {persona.id === 'financial_analyst' && 'Financial analysis & insights'}
                          {persona.id === 'legal_advisor' && 'Legal review & compliance'}
                          {persona.id === 'data_scientist' && 'Data analysis & modeling'}
                          {persona.id === 'business_consultant' && 'Strategy & consulting'}
                        </p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {activeTab === 'sources' && (
          <div className="neo-panel p-4 h-full flex flex-col">
            <h3 className="text-lg font-semibold mb-4">Knowledge Sources</h3>
            
            {/* File Upload */}
            <div className="neo-panel-inset p-4 mb-4">
              <label className="neo-button w-full p-3 flex items-center justify-center cursor-pointer">
                <UploadOutlined className="mr-2" />
                Upload Files
                <input
                  type="file"
                  multiple
                  accept=".pdf,.csv,.xlsx,.json"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
              <p className="text-xs text-dark-textSecondary mt-2 text-center">
                Supports PDF, CSV, Excel, JSON
              </p>
            </div>

            {/* Database Connections */}
            <div className="neo-panel-inset p-4 mb-4">
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-sm font-semibold">Database Connections</h4>
                <button className="neo-button-small p-1">
                  <PlusOutlined className="text-xs" />
                </button>
              </div>
              <div className="text-xs text-dark-textSecondary">
                No databases connected
              </div>
            </div>

            {/* Uploaded Files */}
            <div className="flex-1 overflow-y-auto">
              <h4 className="text-sm font-semibold mb-3">Uploaded Files</h4>
              <div className="space-y-2">
                {uploadedFiles.map(file => (
                  <div key={file.id} className="neo-card-flat p-3 flex items-center justify-between">
                    <div className="flex items-center">
                      <FileTextOutlined className="text-dark-textSecondary mr-2" />
                      <div>
                        <div className="text-sm truncate max-w-40">{file.name}</div>
                        <div className="text-xs text-dark-textSecondary">
                          {(file.size / 1024).toFixed(1)} KB
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={() => handleRemoveFile(file.id)}
                      className="neo-button-small p-1 text-dark-error hover:text-red-400"
                    >
                      <DeleteOutlined className="text-xs" />
                    </button>
                  </div>
                ))}
                {uploadedFiles.length === 0 && (
                  <div className="text-center text-dark-textSecondary py-8">
                    <FileTextOutlined className="text-2xl mb-2" />
                    <p className="text-sm">No files uploaded yet</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default LeftPanel; 