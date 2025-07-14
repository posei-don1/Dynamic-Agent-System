import React, { useState, useRef, useEffect } from 'react';
import { 
  SendOutlined, 
  LoadingOutlined, 
  BulbOutlined, 
  MessageOutlined,
  RobotOutlined,
  UserOutlined,
  ClearOutlined,
  CopyOutlined,
  FileTextOutlined
} from '@ant-design/icons';

const CenterPanel = ({ onMetadataUpdate }) => {
  const [inputValue, setInputValue] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const chatEndRef = useRef(null);
  const textareaRef = useRef(null);

 
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, isProcessing]);

  const handleSendMessage = async (userMessage) => {
    if (!userMessage || typeof userMessage !== 'string' || !userMessage.trim()) {
      setChatHistory(prev => [
        ...prev,
        { type: 'bot', message: 'Please enter a valid question.', timestamp: new Date() }
      ]);
      return;
    }
    setChatHistory(prev => [
      ...prev,
      { type: 'user', message: userMessage, timestamp: new Date() }
    ]);
    setIsProcessing(true);
    try {
      // Backend expects: { query: <str>, persona: <str>, context: <dict|null> }
      const payload = {
        query: userMessage.trim(),
        persona: 'default', // You can make this dynamic if you add persona selection
        context: null // Extend this if you want to pass extra context
      };
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!response.ok) {
        let errorMsg = 'Backend error';
        try {
          const err = await response.json();
          errorMsg = err.detail || JSON.stringify(err);
        } catch {}
        setChatHistory(prev => [
          ...prev,
          { type: 'bot', message: `Error: ${errorMsg}`, timestamp: new Date() }
        ]);
        setIsProcessing(false);
        return;
      }
      const data = await response.json();
      console.log(data)
      
      // Parse the formatted response
      const formattedResponse = data.formatted_response || {};
      const sections = formattedResponse.sections || [];
      
      // Debug: Log the sections to see the structure
      console.log('Sections:', sections);
      
      // Extract main answer and suggestions
      const answer = sections.find(section => section.type === 'answer')?.content || '';
      const suggestions = sections.find(section => section.type === 'suggestions')?.suggestions || [];
      
      // Look for structured data results in different possible sections
      let structuredResult = '';
      
      // First, try to find "Query Results" section
      const queryResults = sections.find(section => section.title === 'Query Results');
      if (queryResults) {
        structuredResult = queryResults.formatted_result || queryResults.content || '';
      }
      
      // If not found, look for "Processed Data" section (which contains the actual result)
      if (!structuredResult) {
        const processedData = sections.find(section => section.title === 'Processed Data');
        if (processedData && processedData.data && processedData.data.result) {
          const result = processedData.data.result;
          if (result.formatted_result) {
            structuredResult = result.formatted_result;
          } else if (result.description) {
            structuredResult = result.description;
          }
        }
      }
      
      // Also check for any section with formatted_result
      if (!structuredResult) {
        for (const section of sections) {
          if (section.data && section.data.result && section.data.result.formatted_result) {
            structuredResult = section.data.result.formatted_result;
            break;
          }
        }
      }
      
      // Debug: Log the extracted structured result
      console.log('Extracted structured result:', structuredResult);
      
      // Update metadata in parent component (includes all detailed info)
      if (onMetadataUpdate) {
        onMetadataUpdate({
          ...data.metadata,
          sections: sections,
          formattedResponse: formattedResponse,
          success: data.status === 'success'
        });
      }
      
      // Show the actual result or success/failure message
      let displayMessage = '';
      if (data.status === 'success') {
        if (structuredResult) {
          // Format the result in a clean way
          const processedData = sections.find(section => section.title === 'Processed Data');
          if (processedData && processedData.data && processedData.data.result) {
            const result = processedData.data.result;
            const column = processedData.data.used_column;
            const operation = result.type;
            const value = result.result;
            
            // Format: "mean value of agricultural land is 159488060.27030674"
            displayMessage = `${operation} value of ${column} is ${value}`;
          } else {
            displayMessage = structuredResult;
          }
        } else if (answer) {
          displayMessage = answer;
        } else if (formattedResponse.title) {
          displayMessage = formattedResponse.title;
        } else {
          displayMessage = 'Query processed successfully';
        }
      } else {
        displayMessage = 'Query failed. Please try again.';
      }
      
      setChatHistory(prev => [
        ...prev,
        {
          type: 'bot',
          message: displayMessage,
          suggestions: suggestions,
          timestamp: new Date(),
          success: data.status === 'success',
          rawData: data // Store the full response data for debugging
        }
      ]);
    } catch (error) {
      setChatHistory(prev => [
        ...prev,
        { type: 'bot', message: 'Error: Could not reach backend.', timestamp: new Date() }
      ]);
    }
    setIsProcessing(false);
  };

  const handleSubmit = (query = inputValue) => {
    if (query.trim() && !isProcessing) {
      handleSendMessage(query.trim());
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
      <div className="flex-1 neo-panel p-4 flex flex-col min-h-0">
        <div className="flex-1 overflow-y-auto scrollbar-thin space-y-4 mb-4 max-h-full" style={{ minHeight: 0 }}>
          {chatHistory.length === 0 && (
            <div className="text-center text-dark-textSecondary py-12">
              <RobotOutlined className="text-4xl mb-4" />
              <p className="text-lg">Ask me anything!</p>
              <p className="text-sm">Try one of the suggested queries below.</p>
            </div>
          )}

          {/* Group user-bot pairs */}
          {(() => {
            const pairs = [];
            for (let i = 0; i < chatHistory.length; i++) {
              const chat = chatHistory[i];
              if (chat.type === 'user') {
                // Find the next bot message (if any)
                const bot = chatHistory[i + 1] && chatHistory[i + 1].type === 'bot' ? chatHistory[i + 1] : null;
                pairs.push(
                  <div key={i} className="space-y-2">
                    {/* User message */}
                    <div className="flex justify-end animate-fade-in">
                      <div className="max-w-2xl flex flex-row-reverse">
                        <div className="flex-shrink-0 ml-3">
                          <div className="neo-button-small p-2 rounded-full">
                            <UserOutlined className="text-dark-info" />
                          </div>
                        </div>
                        <div className="neo-card-flat bg-dark-tertiary border border-dark-info">
                          <p className="text-sm leading-relaxed font-semibold">{chat.message}</p>
                          <div className="text-xs text-dark-textSecondary mt-2">{chat.timestamp.toLocaleTimeString()}</div>
                        </div>
                      </div>
                    </div>
                    {/* Bot message (if any) */}
                    {bot && (
                      <div className="flex justify-start animate-fade-in">
                        <div className="max-w-2xl flex flex-row">
                          <div className="flex-shrink-0 mr-3">
                            <div className="neo-button-small p-2 rounded-full">
                              <RobotOutlined className="text-dark-success" />
                            </div>
                          </div>
                          <div className={`neo-card-flat ${bot.success ? 'bg-dark-secondary' : 'bg-red-900/20'}`}> 
                            <div className="flex items-start justify-between">
                              <div className="flex-1">
                                {bot.message && (
                                  <div>
                                    <p className="text-sm leading-relaxed">{bot.message}</p>
                                    {bot.rawData && bot.rawData.formatted_response && (
                                      <div className="mt-2 text-xs text-dark-textSecondary">
                                        {bot.rawData.formatted_response.sections?.map((section, idx) => {
                                          if (section.title === 'Processed Data' && section.data && section.data.used_file) {
                                            return (
                                              <div key={idx} className="text-[10px] opacity-70">
                                                Source: {section.data.used_file}
                                              </div>
                                            );
                                          }
                                          return null;
                                        })}
                                      </div>
                                    )}
                                  </div>
                                )}
                                {bot.suggestions && bot.suggestions.length > 0 && (
                                  <div className="mt-3 space-y-2">
                                    <div className="flex items-center text-[11px] text-dark-textSecondary font-semibold">
                                      <BulbOutlined className="mr-1 text-[12px]" />
                                      Suggested Questions:
                                    </div>
                                    <div className="space-y-1">
                                      {bot.suggestions.slice(0, 3).map((suggestion, i) => (
                                        <button
                                          key={i}
                                          onClick={() => handleSuggestedQuery(suggestion.title)}
                                          className="block w-full text-left neo-panel-inset p-2 rounded text-[11px] text-dark-textSecondary hover:text-dark-text hover:bg-dark-tertiary/50 transition-colors"
                                        >
                                          {suggestion.title}
                                        </button>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </div>
                            </div>
                            <div className="text-xs text-dark-textSecondary mt-2">{bot.timestamp.toLocaleTimeString()}</div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                );
                if (bot) i++; // Skip the bot message in the next iteration
              }
            }
            return pairs;
          })()}

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
    </div>
  );
};

export default CenterPanel; 