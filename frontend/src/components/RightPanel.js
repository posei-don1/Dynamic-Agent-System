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

const RightPanel = ({ queryResult, processingFlow, uploadedFiles, chatMetadata }) => {
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
          <div className="neo-panel p-4 h-full overflow-y-auto">
            <h3 className="text-lg font-semibold mb-4">Query Metadata</h3>
            
            {chatMetadata ? (
              <div className="space-y-4">
                {/* Processing Details */}
                <div className="neo-panel-inset p-3">
                  <h4 className="text-sm font-semibold mb-2 text-dark-textSecondary">Processing Details</h4>
                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between">
                      <span className="text-dark-textSecondary">Status:</span>
                      <span className={`${chatMetadata.success ? 'text-green-400' : 'text-red-400'}`}>
                        {chatMetadata.success ? 'Success' : 'Failed'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-dark-textSecondary">Persona:</span>
                      <span className="text-dark-text capitalize">{chatMetadata.persona_selected?.replace('_', ' ') || 'Unknown'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-dark-textSecondary">Route:</span>
                      <span className="text-dark-text">{chatMetadata.route_selected || 'Unknown'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-dark-textSecondary">Confidence:</span>
                      <span className="text-dark-text">{(chatMetadata.route_confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-dark-textSecondary">Time:</span>
                      <span className="text-dark-text">{chatMetadata.processing_time?.toFixed(2)}s</span>
                    </div>
                  </div>
                </div>

                {/* System Mode */}
                <div className="neo-panel-inset p-3">
                  <h4 className="text-sm font-semibold mb-2 text-dark-textSecondary">System Mode</h4>
                  <div className="text-xs text-dark-textSecondary">
                    {chatMetadata.system_mode || 'Unknown'}
                  </div>
                </div>

                {/* Document Information */}
                {chatMetadata.sections && (
                  <div className="neo-panel-inset p-3">
                    <h4 className="text-sm font-semibold mb-2 text-dark-textSecondary">Document Information</h4>
                    {(() => {
                      const docInfo = chatMetadata.sections.find(section => section.title === 'Document Information');
                      if (docInfo) {
                        return (
                          <div className="space-y-1 text-xs">
                            <div className="flex justify-between">
                              <span className="text-dark-textSecondary">Document:</span>
                              <span className="text-dark-text">{docInfo.details?.document_used || 'Unknown'}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-dark-textSecondary">Chunks Found:</span>
                              <span className="text-dark-text">{docInfo.details?.relevant_chunks_found || 0}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-dark-textSecondary">Query:</span>
                              <span className="text-dark-text truncate max-w-32">{docInfo.details?.query || 'N/A'}</span>
                            </div>
                          </div>
                        );
                      }
                      return <div className="text-xs text-dark-textSecondary">No document information</div>;
                    })()}
                  </div>
                )}

                {/* Action Plan */}
                {chatMetadata.sections && (() => {
                  const actionPlan = chatMetadata.sections.find(section => section.type === 'action_plan')?.actions;
                  if (actionPlan && actionPlan.length > 0) {
                    return (
                      <div className="neo-panel-inset p-3">
                        <h4 className="text-sm font-semibold mb-2 text-dark-textSecondary">Action Plan</h4>
                        <div className="space-y-1">
                          {actionPlan.map((action, i) => (
                            <div key={i} className="text-xs text-dark-textSecondary">
                              • {action}
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  }
                  return null;
                })()}
              </div>
            ) : (
              <div className="text-center text-dark-textSecondary py-12">
                <InfoCircleOutlined className="text-4xl mb-4" />
                <p className="text-lg">No metadata available</p>
                <p className="text-sm">Send a query to see system information</p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'flow' && (
          <div className="neo-panel p-4 h-full overflow-y-auto">
            <h3 className="text-lg font-semibold mb-4">System Flow</h3>
            
            {chatMetadata?.graph_flow ? (
              <div className="space-y-4">
                {/* Graph Flow Visualization */}
                <div className="neo-panel-inset p-4">
                  <h4 className="text-sm font-semibold mb-3 text-dark-textSecondary">Graph Flow</h4>
                  <div className="text-xs text-dark-textSecondary leading-relaxed">
                    {chatMetadata.graph_flow}
                  </div>
                </div>

                {/* Flow Steps */}
                <div className="neo-panel-inset p-4">
                  <h4 className="text-sm font-semibold mb-3 text-dark-textSecondary">Processing Steps</h4>
                  <div className="space-y-2">
                    {chatMetadata.graph_flow.split(' → ').map((step, index) => (
                      <div key={index} className="flex items-center">
                        <div className="p-1 rounded-full mr-2 bg-dark-success bg-opacity-20">
                          <CheckCircleOutlined className="text-dark-success text-xs" />
                        </div>
                        <span className="text-xs text-dark-text capitalize">
                          {step.replace('_', ' ')}
                        </span>
                        {index < chatMetadata.graph_flow.split(' → ').length - 1 && (
                          <div className="mx-2 text-dark-textSecondary">→</div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Flow Details */}
                <div className="neo-panel-inset p-4">
                  <h4 className="text-sm font-semibold mb-3 text-dark-textSecondary">Flow Details</h4>
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-dark-textSecondary">Total Steps:</span>
                      <span className="text-dark-text">{chatMetadata.graph_flow.split(' → ').length}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-dark-textSecondary">Processing Time:</span>
                      <span className="text-dark-text">{chatMetadata.processing_time?.toFixed(2)}s</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-dark-textSecondary">Route Confidence:</span>
                      <span className="text-dark-text">{(chatMetadata.route_confidence * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center text-dark-textSecondary py-12">
                <BranchesOutlined className="text-4xl mb-4" />
                <p className="text-lg">No flow information</p>
                <p className="text-sm">Send a query to see the system flow</p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'sources' && (
          <div className="neo-panel p-4 h-full overflow-y-auto">
            <h3 className="text-lg font-semibold mb-4">Sources & References</h3>
            
            {chatMetadata?.sections ? (
              <div className="space-y-4">
                {/* Document Sources */}
                {(() => {
                  const docInfo = chatMetadata.sections.find(section => section.title === 'Document Information');
                  if (docInfo) {
                    return (
                      <div className="neo-panel-inset p-4">
                        <h4 className="text-sm font-semibold mb-3 text-dark-textSecondary">Document Source</h4>
                        <div className="space-y-2 text-xs">
                          <div className="flex justify-between">
                            <span className="text-dark-textSecondary">File:</span>
                            <span className="text-dark-text">{docInfo.details?.document_used || 'Unknown'}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-dark-textSecondary">Chunks Found:</span>
                            <span className="text-dark-text">{docInfo.details?.relevant_chunks_found || 0}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-dark-textSecondary">Query:</span>
                            <span className="text-dark-text">{docInfo.details?.query || 'N/A'}</span>
                          </div>
                        </div>
                      </div>
                    );
                  }
                  return null;
                })()}

                {/* Pinecone References */}
                {(() => {
                  const references = chatMetadata.sections.find(section => section.type === 'references')?.references;
                  if (references && references.length > 0) {
                    return (
                      <div className="neo-panel-inset p-4">
                        <h4 className="text-sm font-semibold mb-3 text-dark-textSecondary">Pinecone References ({references.length})</h4>
                        <div className="space-y-3 max-h-60 overflow-y-auto">
                          {references.map((ref, i) => (
                            <div key={i} className="border-l-2 border-dark-info pl-3">
                              <div className="flex justify-between items-start mb-1">
                                <div className="font-medium text-xs text-dark-text">
                                  Source {i + 1}
                                </div>
                                <div className="text-xs text-dark-textSecondary">
                                  Score: {ref.relevance_score?.toFixed(3) || 'N/A'}
                                </div>
                              </div>
                              <div className="text-xs text-dark-textSecondary leading-relaxed">
                                {ref.text_preview || 'No preview available'}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  }
                  return null;
                })()}

                {/* Uploaded Files */}
                {uploadedFiles.length > 0 && (
                  <div className="neo-panel-inset p-4">
                    <h4 className="text-sm font-semibold mb-3 text-dark-textSecondary">Uploaded Files</h4>
                    <div className="space-y-2">
                      {uploadedFiles.map((file, index) => (
                        <div key={index} className="flex items-center justify-between text-xs">
                          <div className="flex items-center">
                            <FileTextOutlined className="text-dark-info mr-2" />
                            <span className="text-dark-text">{file.name}</span>
                          </div>
                          <span className="text-dark-textSecondary">
                            {formatFileSize(file.size)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center text-dark-textSecondary py-12">
                <FileTextOutlined className="text-4xl mb-4" />
                <p className="text-lg">No sources available</p>
                <p className="text-sm">Send a query to see document sources and references</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default RightPanel; 