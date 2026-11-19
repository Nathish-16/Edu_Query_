import React from 'react';

const SUGGESTIONS = [
  "Explain Unit 3 in detail",
  "Give important exam questions",
  "Summarize this study material",
  "Generate a 5-question multiple choice quiz"
];

export default function QuickPrompts({ onSelectPrompt, disabled }) {
  return (
    <div className="prompts-container">
      <span className="prompts-label">💡 Suggestions:</span>
      {SUGGESTIONS.map((prompt, idx) => (
        <button
          key={idx}
          className="prompt-chip"
          onClick={() => onSelectPrompt(prompt)}
          disabled={disabled}
        >
          {prompt}
        </button>
      ))}
    </div>
  );
}
