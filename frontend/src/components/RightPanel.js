import React, { useState } from 'react';
import { 
  InfoCircleOutlined, 
  FileTextOutlined, 
  ClockCircleOutlined,
  CheckCircleOutlined,
  LoadingOutlined,
  EyeOutlined,
  DownloadOutlined,
  BranchesOutlined,
  UserOutlined,
  RobotOutlined,
  DatabaseOutlined,
  ThunderboltOutlined,
  BarChartOutlined
} from '@ant-design/icons';

const RightPanel = ({ queryResult, processingFlow, uploadedFiles }) => {
  const [activeTab, setActiveTab] = useState('metadata');
  const [selectedFile, setSelectedFile] = useState(null);

  const getStepIcon = (step) => {
    const icons = {
      persona_selection: UserOutlined,
      routing: BranchesOutlined,
      document_processing: FileTextOutlined,
      database_processing: DatabaseOutlined,
      math_processing: ThunderboltOutlined,
      suggestion_generation: RobotOutlined,
      answer_formatting: BarChartOutlined
    };
    return icons[step] || CheckCircleOutlined;
  };

  const getStepLabel = (step) => {
    const labels = {
      persona_selection: 'Persona Selection',
      routing: 'Query Routing',
      document_processing: 'Document Processing',
      database_processing: 'Database Processing',
      math_processing: 'Math Processing',
      suggestion_generation: 'Suggestion Generation',
      answer_formatting: 'Answer Formatting'
    };
    return labels[step] || step;
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="h-full flex flex-col space-y-4">
      {/* Header */}
      <div className="neo-panel p-4">
        <h2 className="text-xl font-bold text-gradient mb-4">Analysis Panel</h2>
        
        {/* Tab Navigation */}
        <div className="flex neo-panel-inset rounded-lg p-1">
          <button
            onClick={() => setActiveTab('metadata')}
            className={`flex-1 py-2 px-3 rounded-md transition-all text-xs ${
              activeTab === 'metadata' 
                ? 'neo-button-small text-dark-info' 
                : 'text-dark-textSecondary hover:text-dark-text'
            }`}
          >
            <InfoCircleOutlined className="mr-1" />
            Metadata
          </button>
          <button
            onClick={() => setActiveTab('flow')}
            className={`flex-1 py-2 px-3 rounded-md transition-all text-xs ${
              activeTab === 'flow' 
                ? 'neo-button-small text-dark-info' 
                : 'text-dark-textSecondary hover:text-dark-text'
            }`}
          >
            <BranchesOutlined className="mr-1" />
            Flow
          </button>
          <button
            onClick={() => setActiveTab('sources')}
            className={`flex-1 py-2 px-3 rounded-md transition-all text-xs ${
              activeTab === 'sources' 
                ? 'neo-button-small text-dark-info' 
                : 'text-dark-textSecondary hover:text-dark-text'
            }`}
          >
            <FileTextOutlined className="mr-1" />
            Sources
          </button>
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-hidden">
        {activeTab === 'metadata' && (
          <div className="neo-panel p-4 h-full">
            <h3 className="text-lg font-semibold mb-4">Query Metadata</h3>
            
            {queryResult ? (
              <div className="space-y-4">
                {/* System Status */}
                <div className="neo-panel-inset p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-semibold text-dark-textSecondary">System Mode</span>
                    <div className={`px-2 py-1 rounded-full text-xs ${
                      queryResult.metadata?.system_mode === 'actual_graph' 
                        ? 'bg-dark-success bg-opacity-20 text-dark-success' 
                        : 'bg-dark-warning bg-opacity-20 text-dark-warning'
                    }`}>
                      {queryResult.metadata?.system_mode === 'actual_graph' ? 'Live' : 'Mock'}
                    </div>
                  </div>
                </div>

                {/* Persona Info */}
                <div className="neo-panel-inset p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-semibold text-dark-textSecondary">Selected Persona</span>
                    <UserOutlined className="text-dark-info" />
                  </div>
                  <p className="text-sm capitalize">
                    {queryResult.metadata?.persona_selected?.replace('_', ' ') || 'Unknown'}
                  </p>
                </div>

                {/* Route Info */}
                <div className="neo-panel-inset p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-semibold text-dark-textSecondary">Processing Route</span>
                    <BranchesOutlined className="text-dark-info" />
                  </div>
                  <p className="text-sm">{queryResult.metadata?.route_selected || 'Unknown'}</p>
                  <div className="flex items-center mt-2">
                    <span className="text-xs text-dark-textSecondary mr-2">Confidence:</span>
                    <div className="flex-1 bg-dark-primary rounded-full h-2">
                      <div
                        className="bg-dark-success h-2 rounded-full transition-all duration-300"
                        style={{ width: `${(queryResult.metadata?.route_confidence || 0) * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-dark-textSecondary ml-2">
                      {Math.round((queryResult.metadata?.route_confidence || 0) * 100)}%
                    </span>
                  </div>
                </div>

                {/* Performance */}
                <div className="neo-panel-inset p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-semibold text-dark-textSecondary">Performance</span>
                    <ClockCircleOutlined className="text-dark-info" />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Processing Time:</span>
                      <span>{queryResult.metadata?.processing_time?.toFixed(2)}s</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Timestamp:</span>
                      <span>{new Date(queryResult.metadata?.timestamp * 1000).toLocaleTimeString()}</span>
                    </div>
                  </div>
                </div>

                {/* Response Stats */}
                <div className="neo-panel-inset p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-semibold text-dark-textSecondary">Response Stats</span>
                    <BarChartOutlined className="text-dark-info" />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Response Length:</span>
                      <span>{queryResult.formatted_response?.response?.length || 0} chars</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Suggestions:</span>
                      <span>{queryResult.formatted_response?.suggestions?.length || 0}</span>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center text-dark-textSecondary py-12">
                <InfoCircleOutlined className="text-4xl mb-4" />
                <p className="text-lg">No query processed yet</p>
                <p className="text-sm">Submit a query to see metadata</p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'flow' && (
          <div className="neo-panel p-4 h-full">
            <h3 className="text-lg font-semibold mb-4">Processing Flow</h3>
            
            {processingFlow.length > 0 ? (
              <div className="space-y-3">
                {processingFlow.map((step, index) => {
                  const StepIcon = getStepIcon(step.step);
                  return (
                    <div key={index} className="neo-panel-inset p-3 animate-slide-down">
                      <div className="flex items-center">
                        <div className={`p-2 rounded-full mr-3 ${
                          step.status === 'completed' 
                            ? 'bg-dark-success bg-opacity-20' 
                            : 'bg-dark-warning bg-opacity-20'
                        }`}>
                          {step.status === 'completed' ? (
                            <CheckCircleOutlined className="text-dark-success" />
                          ) : (
                            <LoadingOutlined className="text-dark-warning animate-spin" />
                          )}
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-semibold flex items-center">
                              <StepIcon className="mr-2 text-dark-info" />
                              {getStepLabel(step.step)}
                            </span>
                            <span className="text-xs text-dark-textSecondary">
                              {step.timestamp.toLocaleTimeString()}
                            </span>
                          </div>
                          <div className={`text-xs mt-1 ${
                            step.status === 'completed' ? 'text-dark-success' : 'text-dark-warning'
                          }`}>
                            {step.status === 'completed' ? 'Completed' : 'Processing...'}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-center text-dark-textSecondary py-12">
                <BranchesOutlined className="text-4xl mb-4" />
                <p className="text-lg">No processing flow yet</p>
                <p className="text-sm">Submit a query to see the processing steps</p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'sources' && (
          <div className="neo-panel p-4 h-full flex flex-col">
            <h3 className="text-lg font-semibold mb-4">Source Files</h3>
            
            {uploadedFiles.length > 0 ? (
              <div className="flex-1 overflow-y-auto space-y-3">
                {uploadedFiles.map(file => (
                  <div key={file.id} className="neo-panel-inset p-3">
                    <div className="flex items-start justify-between">
                      <div className="flex items-start flex-1">
                        <FileTextOutlined className="text-dark-info mr-3 mt-1" />
                        <div className="flex-1">
                          <h4 className="text-sm font-semibold truncate">{file.name}</h4>
                          <p className="text-xs text-dark-textSecondary mt-1">
                            {formatFileSize(file.size)} • {file.type}
                          </p>
                        </div>
                      </div>
                      <div className="flex space-x-2">
                        <button
                          onClick={() => setSelectedFile(file)}
                          className="neo-button-small p-2 text-dark-textSecondary hover:text-dark-text"
                          title="Preview"
                        >
                          <EyeOutlined className="text-xs" />
                        </button>
                        <button
                          className="neo-button-small p-2 text-dark-textSecondary hover:text-dark-text"
                          title="Download"
                        >
                          <DownloadOutlined className="text-xs" />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center text-dark-textSecondary py-12">
                <FileTextOutlined className="text-4xl mb-4" />
                <p className="text-lg">No source files</p>
                <p className="text-sm">Upload files to see them here</p>
              </div>
            )}

            {/* File Preview Modal */}
            {selectedFile && (
              <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                <div className="neo-panel p-6 max-w-2xl max-h-96 overflow-y-auto">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold">File Preview</h3>
                    <button
                      onClick={() => setSelectedFile(null)}
                      className="neo-button-small p-2 text-dark-textSecondary hover:text-dark-text"
                    >
                      ×
                    </button>
                  </div>
                  <div className="neo-panel-inset p-4">
                    <p className="text-sm text-dark-textSecondary">
                      Preview for {selectedFile.name}
                    </p>
                    <p className="text-xs text-dark-textSecondary mt-2">
                      File preview functionality would be implemented here
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default RightPanel; 