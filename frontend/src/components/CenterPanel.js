import React, { useState, useRef, useEffect } from 'react';
import { 
  SendOutlined, 
  LoadingOutlined, 
  BulbOutlined, 
  MessageOutlined,
  RobotOutlined,
  UserOutlined,
  ClearOutlined,
  CopyOutlined
} from '@ant-design/icons';

const CenterPanel = ({ currentQuery, queryResult, isProcessing, onQuerySubmit }) => {
  const [inputValue, setInputValue] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const chatEndRef = useRef(null);
  const textareaRef = useRef(null);

  const suggestedQueries = [
    "Analyze Microsoft's revenue trends from the uploaded data",
    "Review the contract terms for potential risks",
    "Calculate the year-over-year growth rate",
    "Summarize key financial metrics",
    "What are the main legal considerations?",
    "Perform statistical analysis on the dataset"
  ];

  useEffect(() => {
    if (queryResult) {
      setChatHistory(prev => [...prev, {
        type: 'bot',
        message: queryResult.formatted_response?.response || 'No response received',
        suggestions: queryResult.formatted_response?.suggestions || [],
        timestamp: new Date(),
        metadata: queryResult.metadata
      }]);
    }
  }, [queryResult]);

  useEffect(() => {
    if (currentQuery) {
      setChatHistory(prev => [...prev, {
        type: 'user',
        message: currentQuery,
        timestamp: new Date()
      }]);
    }
  }, [currentQuery]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, isProcessing]);

  const handleSubmit = (query = inputValue) => {
    if (query.trim() && !isProcessing) {
      onQuerySubmit(query.trim());
      setInputValue('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSuggestedQuery = (query) => {
    setInputValue(query);
    handleSubmit(query);
  };

  const handleClearChat = () => {
    setChatHistory([]);
  };

  const handleCopyMessage = (message) => {
    navigator.clipboard.writeText(message);
  };

  const adjustTextareaHeight = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }
  };

  useEffect(() => {
    adjustTextareaHeight();
  }, [inputValue]);

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="neo-panel p-4 mb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <MessageOutlined className="text-dark-info mr-2" />
            <h2 className="text-xl font-bold text-gradient">Query Interface</h2>
          </div>
          <button
            onClick={handleClearChat}
            className="neo-button-small p-2 text-dark-textSecondary hover:text-dark-text"
            title="Clear chat"
          >
            <ClearOutlined />
          </button>
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 neo-panel p-4 flex flex-col">
        <div className="flex-1 overflow-y-auto scrollbar-thin space-y-4 mb-4">
          {chatHistory.length === 0 && (
            <div className="text-center text-dark-textSecondary py-12">
              <RobotOutlined className="text-4xl mb-4" />
              <p className="text-lg">Ask me anything!</p>
              <p className="text-sm">Try one of the suggested queries below.</p>
            </div>
          )}

          {chatHistory.map((chat, index) => (
            <div
              key={index}
              className={`flex ${chat.type === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}
            >
              <div className={`max-w-2xl flex ${chat.type === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                {/* Avatar */}
                <div className={`flex-shrink-0 ${chat.type === 'user' ? 'ml-3' : 'mr-3'}`}>
                  <div className="neo-button-small p-2 rounded-full">
                    {chat.type === 'user' ? (
                      <UserOutlined className="text-dark-info" />
                    ) : (
                      <RobotOutlined className="text-dark-success" />
                    )}
                  </div>
                </div>

                {/* Message */}
                <div className={`neo-card-flat ${chat.type === 'user' ? 'bg-dark-tertiary' : 'bg-dark-secondary'}`}>
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <p className="text-sm leading-relaxed">{chat.message}</p>
                      
                      {/* Suggestions */}
                      {chat.suggestions && chat.suggestions.length > 0 && (
                        <div className="mt-3 space-y-2">
                          <div className="flex items-center text-xs text-dark-textSecondary">
                            <BulbOutlined className="mr-1" />
                            Suggestions:
                          </div>
                          {chat.suggestions.map((suggestion, i) => (
                            <div key={i} className="neo-panel-inset p-2 text-xs">
                              • {suggestion}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                    
                    <button
                      onClick={() => handleCopyMessage(chat.message)}
                      className="neo-button-small p-1 ml-2 text-dark-textSecondary hover:text-dark-text"
                      title="Copy message"
                    >
                      <CopyOutlined className="text-xs" />
                    </button>
                  </div>
                  
                  <div className="text-xs text-dark-textSecondary mt-2">
                    {chat.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            </div>
          ))}

          {/* Processing Indicator */}
          {isProcessing && (
            <div className="flex justify-start animate-fade-in">
              <div className="max-w-2xl flex flex-row">
                <div className="flex-shrink-0 mr-3">
                  <div className="neo-button-small p-2 rounded-full">
                    <LoadingOutlined className="text-dark-info animate-spin" />
                  </div>
                </div>
                <div className="neo-card-flat bg-dark-secondary">
                  <div className="flex items-center space-x-2">
                    <LoadingOutlined className="animate-spin text-dark-info" />
                    <span className="text-sm">Processing your query...</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        {/* Input Area */}
        <div className="neo-panel-inset p-4">
          <div className="flex items-end space-x-3">
            <div className="flex-1">
              <textarea
                ref={textareaRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your query here..."
                className="neo-input w-full resize-none min-h-12 max-h-32"
                disabled={isProcessing}
                rows={1}
              />
            </div>
            <button
              onClick={() => handleSubmit()}
              disabled={!inputValue.trim() || isProcessing}
              className={`neo-button p-3 ${
                inputValue.trim() && !isProcessing
                  ? 'text-dark-info hover:text-blue-400'
                  : 'text-dark-textSecondary cursor-not-allowed'
              }`}
            >
              {isProcessing ? (
                <LoadingOutlined className="animate-spin" />
              ) : (
                <SendOutlined />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Suggested Queries */}
      <div className="neo-panel p-4 mt-4">
        <h3 className="text-sm font-semibold mb-3 text-dark-textSecondary">Suggested Queries</h3>
        <div className="grid grid-cols-1 gap-2">
          {suggestedQueries.map((query, index) => (
            <button
              key={index}
              onClick={() => handleSuggestedQuery(query)}
              disabled={isProcessing}
              className="neo-button-small p-3 text-left text-sm hover:text-dark-info transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <BulbOutlined className="mr-2 text-dark-warning" />
              {query}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default CenterPanel; 